"""Prompt-cache TTL detection and the 5m-vs-1h counterfactual.

Two questions live here, and they are not the same question:

1. **What did idling cost me?**  A pause longer than the cache TTL between
   two API requests means the next request re-writes the whole prefix at the
   cache-*write* rate instead of reading it at the cache-*read* rate.  That
   premium is real money already spent.
2. **Would a different TTL have been cheaper?**  Anthropic charges 1.25x
   input for a 5m cache write and 2x input for a 1h write.  A 1h TTL saves
   the rebuilds in the 5-60 minute band but pays a 60% surcharge on *every*
   incremental write for the whole session.  Answering it needs a two-policy
   simulation, not a waste number.

Question 2 is why the "wasted on cache misses" figure alone can't drive the
``promptCacheTtl`` decision, and why gaps longer than an hour are excluded
from it: no setting recovers those, so counting them inflates the apparent
upside of switching.

Model
-----
Every Anthropic API request re-sends the whole conversation prefix.  The
provider splits it into what it could reuse (``cache_read_input_tokens``)
and what it had to write (``cache_creation.*``).  Their sum,

    prefix_total = cache_read + cache_creation

is a property of the *conversation*, not of the cache policy: the same
messages get re-sent whichever TTL is configured.  That invariant is what
makes the counterfactual honest — only the read/write *split* moves, and
the split is decided by one predicate:

    warm(T) = gap_seconds <= T AND NOT structural_invalidation AND seq > 1

``gap_seconds`` is measured from the end of the previous response to the
moment the next request was triggered (a human prompt *or* a tool result —
a tool that runs for six minutes expires the cache exactly like a coffee
break does).  Anthropic's TTL refreshes on every hit, so the gap since the
*previous request* is the right clock, not the gap since the cache was
first written.

Assumptions, stated so they can be argued with
----------------------------------------------
* **Prefix invariance.**  ``prefix_total`` is unchanged by the TTL setting.
* **Common prefix.**  When a request was observed warm, its actual
  ``cache_read_tokens`` *is* the reusable overlap (it already accounts for
  cache-breakpoint granularity).  When it was observed cold we fall back to
  ``min(prev_prefix_total, prefix_total)``, which is exact for the ordinary
  append-only case.
* **Structural invalidations are excluded.**  A request that read ~nothing
  despite a sub-5-minute gap didn't miss on time — the prefix itself changed
  (model or effort switch, ``/compact``, a tool-set change).  It costs the
  same under both policies, so attributing it to pausing would be wrong.
* **Gaps over an hour are breaks, not waste.**  Neither setting recovers
  them; they are reported separately.
* **Subscription dollars are API-equivalent.**  Costs here are list API
  prices.  Whether a subscription plan discounts them by some coefficient is
  not recorded in the transcripts, so the comparison is a ratio statement as
  much as a dollar one.
* **Sidechains are simulated separately.**  Subagents have their own
  ``subagentPromptCacheTtl`` (5m for everyone).  Concurrent subagents
  interleave in wall-clock order within a session, so their per-request gaps
  are noisier than the main chain's — they are diagnostics, never merged
  into the main-chain recommendation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from introspect.pricing import (
    CACHE_TTL_SECONDS,
    PRICING_CACHE_READ_RATE_SQL,
    PRICING_CACHE_WRITE_1H_RATE_SQL,
    PRICING_CACHE_WRITE_5M_RATE_SQL,
    PRICING_INPUT_RATE_SQL,
    PRICING_OUTPUT_RATE_SQL,
)

if TYPE_CHECKING:
    import duckdb


log = logging.getLogger(__name__)

# The two policies Claude Code can be configured with. 5m is Anthropic's
# default ephemeral TTL (re-exported from pricing so there is one definition);
# 1h is what `promptCacheTtl: "1h"` buys.
TTL_5M_SECONDS = CACHE_TTL_SECONDS
TTL_1H_SECONDS = 3600

# A gap wider than this is a break, not a cache miss: no TTL Claude Code
# offers would have kept the prefix warm across it.
MAX_RECOVERABLE_GAP_SECONDS = TTL_1H_SECONDS

# A request that reused less than this fraction of the previous prefix,
# despite a gap short enough to be warm under *both* policies, did not miss
# on time — the prefix changed underneath it.
STRUCTURAL_READ_RATIO = 0.1

# Below this many prefix tokens the read/write split is noise (a tiny
# first turn, a synthetic probe); such requests are never called structural.
STRUCTURAL_MIN_PREFIX = 1_000

# Recommend switching only when the margin clears this share of total spend.
# A 2% edge is inside the modelling error and must not read as a decision.
RECOMMENDATION_MARGIN_PCT = 2.0

# Gap histogram edges, in seconds, paired with the labels the UI renders.
GAP_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("<5m", TTL_5M_SECONDS),
    ("5-15m", 15 * 60),
    ("15-60m", TTL_1H_SECONDS),
    (">60m", None),
)

# Effective cache-write tokens for one ``assistant_message_costs`` row.
# Legacy transcripts set ``usage.cache_creation_input_tokens`` without the
# ``ephemeral_{5m,1h}`` split; those bill at the 5m rate and their total is
# the only signal we have. Where the split *is* present the two agree
# exactly (verified against live transcripts), so this is a strict superset
# of ``cache_creation_5m + cache_creation_1h``.
CACHE_CREATION_EFFECTIVE_SQL = (
    "(CASE WHEN cache_creation_5m + cache_creation_1h > 0"
    " THEN cache_creation_5m + cache_creation_1h"
    " ELSE cache_creation_tokens END)"
)

# The TTL a row was actually billed at, derived from which ephemeral bucket
# the provider filled. Claude Code hands out 1h on subscription-within-plan
# for the main conversation and 5m otherwise, so this is the only way to
# know which regime a row came from.
TTL_OBSERVED_SQL = """
    CASE
        WHEN cache_creation_5m > 0 AND cache_creation_1h > 0 THEN 'mixed'
        WHEN cache_creation_1h > 0 THEN '1h'
        WHEN cache_creation_5m > 0 THEN '5m'
        ELSE 'unknown'
    END
"""


def _warm_sql(ttl_seconds: int) -> str:
    """Would this request have hit the cache under a ``ttl_seconds`` policy?

    The whole counterfactual turns on this one predicate — everything else
    (``prefix_total``, ``common_prefix``) is invariant to the setting.
    """
    return f"(seq > 1 AND NOT structural_invalidation AND gap_seconds <= {ttl_seconds})"


def _policy_cost_sql(ttl_seconds: int, write_rate_sql: str) -> str:
    """Simulated USD cost of one request under a fixed TTL policy.

    ``prefix_total`` is invariant to the policy; only the read/write split
    moves, so the counterfactual is one CASE over ``warm(T)``.
    """
    read = f"(CASE WHEN {_warm_sql(ttl_seconds)} THEN common_prefix ELSE 0 END)"
    return (
        f"({read} * ({PRICING_CACHE_READ_RATE_SQL})"
        f" + (prefix_total - {read}) * ({write_rate_sql})"
        f" + input_tokens * ({PRICING_INPUT_RATE_SQL})"
        f" + output_tokens * ({PRICING_OUTPUT_RATE_SQL})) / 1000000.0"
    )


# Seconds of TTL a row was actually billed under. 'mixed' and 'unknown' fall
# back to 5m, matching how the legacy no-split case is billed everywhere else.
OBSERVED_TTL_SECONDS_SQL = (
    f"(CASE WHEN ttl_observed = '1h' THEN {TTL_1H_SECONDS} ELSE {TTL_5M_SECONDS} END)"
)

# Cache-write rate the row was actually billed at, for the miss premium.
_OBSERVED_WRITE_RATE_SQL = (
    f"(CASE WHEN ttl_observed = '1h' THEN ({PRICING_CACHE_WRITE_1H_RATE_SQL})"
    f" ELSE ({PRICING_CACHE_WRITE_5M_RATE_SQL}) END)"
)


def _gap_bucket_sql() -> str:
    branches = []
    for label, upper in GAP_BUCKETS:
        if upper is None:
            branches.append(f"ELSE '{label}'")
        else:
            branches.append(f"WHEN gap_seconds <= {upper} THEN '{label}'")
    return "CASE WHEN gap_seconds IS NULL THEN NULL " + " ".join(branches) + " END"


# The observed (actually billed) cost of one request. Kept separate from the
# shared ``COST_EXPR_SQL`` fragment because ``cache_requests`` has already
# resolved the legacy split into ``cache_creation_effective``.
# Everything that is not a 1h write bills at the 5m rate — the 5m bucket
# itself and, for legacy rows, the unsplit remainder of cache_creation.
_OBSERVED_COST_SQL = (
    f"(cache_read_tokens * ({PRICING_CACHE_READ_RATE_SQL})"
    f" + (cache_creation_effective - cache_creation_1h)"
    f"   * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
    f" + cache_creation_1h * ({PRICING_CACHE_WRITE_1H_RATE_SQL})"
    f" + input_tokens * ({PRICING_INPUT_RATE_SQL})"
    f" + output_tokens * ({PRICING_OUTPUT_RATE_SQL})) / 1000000.0"
)


# One row per main-conversation (or sidechain) API request, chain-ordered,
# carrying everything both the detector and the counterfactual need.
#
# Ordering is by wall-clock timestamp within (session_id, is_sidechain).
# `parent_uuid` would be the more principled chain, but real transcripts
# break it: parallel tool calls and harness rewrites leave dangling parents,
# so timestamp order is the robust choice (and is what the previous
# detection query used).
CACHE_REQUESTS_BODY = f"""
WITH response_spans AS (
    -- assistant_message_costs holds one row per API *request* (deduped on
    -- message.id, earliest copy wins). A streamed response is logged as
    -- several raw rows sharing that id, so the response *end* is the last
    -- of them. Duplicate logging of one response can only stretch this
    -- later, which shrinks gaps — biased against over-reporting waste.
    SELECT
        session_id,
        json_extract_string(message, '$.id') AS message_id,
        MAX(timestamp) AS response_end_ts
    FROM raw_messages
    WHERE type = 'assistant'
      AND json_extract_string(message, '$.id') IS NOT NULL
    GROUP BY 1, 2
),
requests AS (
    SELECT
        a.session_id,
        a.uuid,
        a.message_id,
        a.timestamp,
        COALESCE(a.is_sidechain, FALSE) AS is_sidechain,
        a.model,
        a.input_tokens,
        a.output_tokens,
        a.cache_read_tokens,
        a.cache_creation_tokens,
        a.cache_creation_5m,
        a.cache_creation_1h,
        a.ttl_observed,
        {CACHE_CREATION_EFFECTIVE_SQL} AS cache_creation_effective,
        COALESCE(rs.response_end_ts, a.timestamp) AS response_end_ts
    FROM assistant_message_costs a
    LEFT JOIN response_spans rs
           ON rs.session_id = a.session_id
          AND rs.message_id = a.message_id
    WHERE a.model IS DISTINCT FROM '<synthetic>'
      AND a.cache_read_tokens + a.cache_creation_tokens + a.input_tokens > 0
),
-- Interleave the user-side records (human prompts *and* tool results) with
-- the requests so each request can pick up the timestamp of whatever
-- triggered it, in one ordered pass rather than a correlated subquery.
-- Tie-break puts the user record before the response it triggered when the
-- two share a timestamp.
timeline AS (
    SELECT
        session_id,
        COALESCE(is_sidechain, FALSE) AS is_sidechain,
        timestamp,
        0 AS tie_break,
        timestamp AS user_ts,
        NULL AS req_message_id
    FROM raw_messages
    WHERE type = 'user'
    UNION ALL
    SELECT
        session_id,
        is_sidechain,
        timestamp,
        1 AS tie_break,
        NULL AS user_ts,
        message_id AS req_message_id
    FROM requests
),
triggers AS (
    SELECT
        session_id,
        req_message_id,
        last_value(user_ts IGNORE NULLS) OVER (
            PARTITION BY session_id, is_sidechain
            ORDER BY timestamp, tie_break
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS trigger_ts
    FROM timeline
),
seqd AS (
    SELECT
        r.*,
        COALESCE(t.trigger_ts, r.timestamp) AS trigger_ts,
        r.cache_read_tokens + r.cache_creation_effective AS prefix_total,
        ROW_NUMBER() OVER w AS seq,
        LAG(r.cache_read_tokens + r.cache_creation_effective)
            OVER w AS prev_prefix_total,
        LAG(r.response_end_ts) OVER w AS prev_response_end_ts
    FROM requests r
    LEFT JOIN triggers t
           ON t.session_id = r.session_id
          AND t.req_message_id = r.message_id
    WINDOW w AS (
        PARTITION BY r.session_id, r.is_sidechain
        ORDER BY r.timestamp, r.message_id
    )
),
gapped AS (
    SELECT
        *,
        CASE
            WHEN prev_response_end_ts IS NULL THEN NULL
            ELSE GREATEST(
                0, date_diff('second', prev_response_end_ts, trigger_ts)
            )
        END AS gap_seconds
    FROM seqd
),
classified AS (
    SELECT
        *,
        -- Read ~nothing back despite a gap short enough to be warm under
        -- *both* policies: the prefix itself changed (model/effort switch,
        -- /compact, tool-set change). Identical cost either way, so it is
        -- never attributed to pausing.
        COALESCE(
            seq > 1
            AND gap_seconds <= {TTL_5M_SECONDS}
            AND prev_prefix_total >= {STRUCTURAL_MIN_PREFIX}
            AND cache_read_tokens
                < {STRUCTURAL_READ_RATIO} * prev_prefix_total,
            FALSE
        ) AS structural_invalidation,
        COALESCE(seq > 1 AND prefix_total < prev_prefix_total, FALSE)
            AS prefix_shrank,
        -- The one detection rule. A request paid to rebuild a prefix it
        -- could have read: the gap outran the TTL it was actually billed
        -- under, and the prefix itself did not change underneath it.
        -- DuckDB resolves lateral aliases in the same SELECT list, so this
        -- reuses ``structural_invalidation`` above rather than restating the
        -- predicate — one copy of the rule, not two.
        COALESCE(
            seq > 1
            AND NOT structural_invalidation
            AND gap_seconds > {OBSERVED_TTL_SECONDS_SQL},
            FALSE
        ) AS cache_miss
    FROM gapped
),
priced AS (
    SELECT
        *,
        -- The reusable overlap is a property of the content, not the
        -- policy — but only a request that actually *hit* the cache reports
        -- it. When the request was warm, its own cache_read is the overlap
        -- exactly (cache-breakpoint granularity included), which makes the
        -- parity gate exact by construction. When it missed, its cache_read
        -- is a residue of whatever survived (often just the system-prompt
        -- block) and would badly understate what a warm cache would have
        -- reused, so fall back to min(prev, current) — exact for the
        -- ordinary append-only case.
        CASE
            WHEN NOT cache_miss AND cache_read_tokens > 0
                THEN cache_read_tokens
            ELSE LEAST(COALESCE(prev_prefix_total, 0), prefix_total)
        END AS common_prefix
    FROM classified
)
SELECT
    session_id,
    uuid,
    message_id,
    timestamp,
    trigger_ts,
    response_end_ts,
    is_sidechain,
    model,
    seq,
    input_tokens,
    output_tokens,
    cache_read_tokens,
    cache_creation_tokens,
    cache_creation_5m,
    cache_creation_1h,
    cache_creation_effective,
    ttl_observed,
    prefix_total,
    prev_prefix_total,
    common_prefix,
    gap_seconds,
    {_gap_bucket_sql()} AS gap_bucket,
    structural_invalidation,
    prefix_shrank,
    cache_miss,
    -- A miss a 1h TTL would have avoided. Always false on a session already
    -- billed at 1h: nothing is recoverable once you are on the longer TTL.
    COALESCE(
        cache_miss AND gap_seconds <= {MAX_RECOVERABLE_GAP_SECONDS}, FALSE
    ) AS gap_recoverable,
    -- A gap no setting recovers. Reported as a break, never as waste.
    COALESCE(
        cache_miss AND gap_seconds > {MAX_RECOVERABLE_GAP_SECONDS}, FALSE
    ) AS gap_unrecoverable,
    -- Premium already paid for this miss: the prefix that would have been
    -- read, billed at the write rate instead. Scoped to ``common_prefix``
    -- rather than the whole cache_creation, because the tokens the new
    -- message itself added would have been written under any policy.
    CASE
        WHEN cache_miss
        THEN common_prefix
             * ({_OBSERVED_WRITE_RATE_SQL} - ({PRICING_CACHE_READ_RATE_SQL}))
             / 1000000.0
        ELSE 0.0
    END AS miss_premium_usd,
    {_warm_sql(TTL_5M_SECONDS)} AS warm_5m,
    {_warm_sql(TTL_1H_SECONDS)} AS warm_1h,
    {_policy_cost_sql(TTL_5M_SECONDS, PRICING_CACHE_WRITE_5M_RATE_SQL)}
        AS cost_5m_usd,
    {_policy_cost_sql(TTL_1H_SECONDS, PRICING_CACHE_WRITE_1H_RATE_SQL)}
        AS cost_1h_usd,
    {_OBSERVED_COST_SQL} AS cost_observed_usd
FROM priced
"""  # noqa: S608


def _rollup_select(alias: str = "") -> str:
    """Aggregate list every TTL rollup shares, optionally table-qualified.

    ``_comparison_from_row`` unpacks these positionally, and
    ``session_cache_ttl`` is built from the same fragment, so the column
    order lives in exactly one place.
    """
    q = f"{alias}." if alias else ""
    return f"""
    SUM({q}cost_5m_usd) AS cost_5m,
    SUM({q}cost_1h_usd) AS cost_1h,
    COUNT(*) AS n_requests,
    COUNT(*) FILTER (WHERE {q}gap_recoverable) AS n_gaps_recoverable,
    COUNT(*) FILTER (WHERE {q}gap_unrecoverable) AS n_gaps_unrecoverable,
    COUNT(*) FILTER (WHERE {q}structural_invalidation) AS n_structural,
    mode({q}ttl_observed) FILTER (WHERE {q}ttl_observed <> 'unknown')
        AS ttl_observed_dominant,
    COALESCE(
        SUM({q}miss_premium_usd) FILTER (WHERE {q}gap_recoverable), 0
    ) AS recoverable_waste_usd,
    COALESCE(
        SUM({q}miss_premium_usd) FILTER (WHERE {q}gap_unrecoverable), 0
    ) AS unrecoverable_break_usd
"""


# Per-session rollup of the counterfactual, exposed for ad-hoc SQL.
# Diagnostics only: ``promptCacheTtl`` is set per user/project, so a
# per-session "you should have used 1h here" is not actionable on its own —
# the project and global rollups are what you act on. Built from the same
# ``_rollup_select`` fragment as those, so the aggregate definitions cannot
# drift between the view and the Python API.
def _session_cache_ttl_body() -> str:
    return f"""
SELECT
    cr.session_id,
    cr.is_sidechain,
    {_rollup_select("cr")},
    SUM(cr.cost_1h_usd) - SUM(cr.cost_5m_usd) AS delta,
    SUM(cr.cost_observed_usd) AS cost_observed,
    COALESCE(SUM(cr.prefix_total) FILTER (WHERE cr.gap_recoverable), 0)
        AS recoverable_prefix_tokens
FROM cache_requests cr
GROUP BY cr.session_id, cr.is_sidechain
"""  # noqa: S608


SESSION_CACHE_TTL_BODY = _session_cache_ttl_body()


class TtlComparison(NamedTuple):
    """One 5m-vs-1h verdict over a set of sessions."""

    cost_5m: float
    cost_1h: float
    delta: float
    """``cost_1h - cost_5m``. Negative means 1h is cheaper."""
    margin_pct: float
    """``|delta|`` as a percent of the cheaper policy's cost."""
    recommendation: str
    """``'1h'``, ``'5m'``, or ``'either'`` when the margin is within noise."""
    n_requests: int
    n_gaps_recoverable: int
    n_gaps_unrecoverable: int
    n_structural: int
    ttl_observed_dominant: str | None
    recoverable_waste_usd: float = 0.0
    """Premium already paid on misses a 1h TTL would have avoided."""
    unrecoverable_break_usd: float = 0.0
    """Premium paid on gaps over an hour. No setting recovers these."""

    @property
    def decisive(self) -> bool:
        """True when the margin is wide enough to act on."""
        return self.recommendation != "either"

    @property
    def savings(self) -> float:
        """USD the recommended policy saves over the other one."""
        return abs(self.delta)


def compare_ttl(  # noqa: PLR0913
    *,
    cost_5m: float,
    cost_1h: float,
    n_requests: int = 0,
    n_gaps_recoverable: int = 0,
    n_gaps_unrecoverable: int = 0,
    n_structural: int = 0,
    ttl_observed_dominant: str | None = None,
    recoverable_waste_usd: float = 0.0,
    unrecoverable_break_usd: float = 0.0,
) -> TtlComparison:
    """Turn a pair of simulated totals into a verdict with an honest margin.

    The margin gate matters more than the sign: the simulation carries
    modelling error (estimated common prefixes on cold requests, gaps
    measured from log timestamps), so a delta inside
    ``RECOMMENDATION_MARGIN_PCT`` is reported as ``'either'`` rather than
    dressed up as a decision.
    """
    delta = cost_1h - cost_5m
    cheaper = min(cost_5m, cost_1h)
    margin_pct = 100.0 * abs(delta) / cheaper if cheaper > 0 else 0.0
    if margin_pct < RECOMMENDATION_MARGIN_PCT:
        recommendation = "either"
    else:
        recommendation = "1h" if delta < 0 else "5m"
    return TtlComparison(
        cost_5m=cost_5m,
        cost_1h=cost_1h,
        delta=delta,
        margin_pct=margin_pct,
        recommendation=recommendation,
        n_requests=n_requests,
        n_gaps_recoverable=n_gaps_recoverable,
        n_gaps_unrecoverable=n_gaps_unrecoverable,
        n_structural=n_structural,
        ttl_observed_dominant=ttl_observed_dominant,
        recoverable_waste_usd=recoverable_waste_usd,
        unrecoverable_break_usd=unrecoverable_break_usd,
    )


def _comparison_from_row(row: tuple | None) -> TtlComparison:
    if row is None or row[0] is None:
        return compare_ttl(cost_5m=0.0, cost_1h=0.0)
    return compare_ttl(
        cost_5m=float(row[0] or 0.0),
        cost_1h=float(row[1] or 0.0),
        n_requests=int(row[2] or 0),
        n_gaps_recoverable=int(row[3] or 0),
        n_gaps_unrecoverable=int(row[4] or 0),
        n_structural=int(row[5] or 0),
        ttl_observed_dominant=row[6],
        recoverable_waste_usd=float(row[7] or 0.0),
        unrecoverable_break_usd=float(row[8] or 0.0),
    )


def _sidechain_and_window(
    *, sidechain: bool, window: tuple[str, str] | None, alias: str = ""
) -> tuple[str, list[Any]]:
    """WHERE clause shared by the global and per-project rollups.

    ``alias`` qualifies the column names for callers that join
    ``cache_requests`` against another relation.
    """
    prefix = f"{alias}." if alias else ""
    clause = f"WHERE {prefix}is_sidechain = ?"
    params: list[Any] = [sidechain]
    if window is not None:
        clause += f" AND {prefix}timestamp >= ? AND {prefix}timestamp < ?"
        params.extend(window)
    return clause, params


def global_ttl_comparison(
    db: duckdb.DuckDBPyConnection,
    *,
    sidechain: bool = False,
    window: tuple[str, str] | None = None,
) -> TtlComparison:
    """Portfolio-wide 5m-vs-1h verdict.

    ``sidechain=True`` scores subagent traffic, which carries its own
    ``subagentPromptCacheTtl`` setting — never merge the two.
    """
    clause, params = _sidechain_and_window(sidechain=sidechain, window=window)
    row = db.execute(
        f"SELECT {_rollup_select()} FROM cache_requests {clause}",  # noqa: S608
        params,
    ).fetchone()
    return _comparison_from_row(row)


def project_ttl_comparisons(
    db: duckdb.DuckDBPyConnection,
    *,
    sidechain: bool = False,
    window: tuple[str, str] | None = None,
) -> list[tuple[str, TtlComparison]]:
    """Per-project verdicts, most spend first.

    The setting is per user/project, so this — not the per-session rollup —
    is the actionable granularity.
    """
    clause, params = _sidechain_and_window(
        sidechain=sidechain, window=window, alias="cr"
    )
    rows = db.execute(
        f"""
        SELECT COALESCE(ls.project, '?') AS project, {_rollup_select("cr")}
        FROM cache_requests cr
        LEFT JOIN logical_sessions ls ON ls.session_id = cr.session_id
        {clause}
        GROUP BY 1
        ORDER BY GREATEST(SUM(cr.cost_5m_usd), SUM(cr.cost_1h_usd)) DESC
        """,  # noqa: S608
        params,
    ).fetchall()
    return [(row[0], _comparison_from_row(row[1:])) for row in rows]


def cache_miss_event_rows(
    db: duckdb.DuckDBPyConnection,
    *,
    session_id: str | None = None,
    timestamp_window: tuple[str, str] | None = None,
    sidechain: bool = False,
) -> list[dict[str, Any]]:
    """Every cache miss in scope, straight off ``cache_requests``.

    The single detection rule, read by the session-detail divider, the
    tokenscape event track and the cost-overview panel alike — there is no
    second threshold anywhere for them to disagree over.

    Each row carries ``recoverable``: ``True`` when a 1h TTL would have kept
    the prefix warm, ``False`` when the gap ran past an hour and no setting
    would have. Callers must keep the two apart; summing them back together
    reintroduces exactly the inflation this replaced.
    """
    clauses = ["cache_miss", "is_sidechain = ?"]
    params: list[Any] = [sidechain]
    if session_id is not None:
        clauses.append("session_id = ?")
        params.append(session_id)
    if timestamp_window is not None:
        clauses.append("timestamp >= ? AND timestamp < ?")
        params.extend(timestamp_window)
    relation = db.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'cache_requests'"
    ).fetchone()
    if relation is None:
        # A hot-reloaded server can retain a database materialized by an older
        # release. Keep session details usable until the next materialization
        # upgrades the derived schema.
        log.warning("cache_requests is unavailable; cache-loss events are hidden")
        return []
    rows = db.execute(
        f"""
        SELECT uuid, session_id, timestamp, gap_seconds, gap_bucket, model,
               ttl_observed, prefix_total, common_prefix, miss_premium_usd,
               gap_recoverable, prefix_shrank
        FROM cache_requests
        WHERE {" AND ".join(clauses)}
        ORDER BY timestamp
        """,  # noqa: S608
        params,
    ).fetchall()
    return [
        {
            "uuid": str(row[0]),
            "session_id": str(row[1]),
            "timestamp": row[2],
            "gap_seconds": int(row[3] or 0),
            "gap_bucket": row[4],
            "model": row[5],
            "ttl_observed": row[6],
            "prefix_total": int(row[7] or 0),
            "common_prefix": int(row[8] or 0),
            "miss_premium_usd": float(row[9] or 0.0),
            "recoverable": bool(row[10]),
            "prefix_shrank": bool(row[11]),
        }
        for row in rows
    ]


class MissSummary(NamedTuple):
    """Cache misses split by whether any TTL setting would have avoided them.

    Kept as a pair rather than one total: adding them back together is
    exactly the conflation this replaced.
    """

    recoverable_count: int
    recoverable_usd: float
    break_count: int
    break_usd: float

    @property
    def count(self) -> int:
        """Every miss, recoverable or not — for diagnostics, not display."""
        return self.recoverable_count + self.break_count


def summarize_misses(events: list[dict[str, Any]]) -> MissSummary:
    """Split a miss list into recoverable waste and unrecoverable breaks.

    Kept next to the detection rule so every surface renders the same two
    numbers instead of one conflated total.
    """
    recoverable = [e for e in events if e["recoverable"]]
    breaks = [e for e in events if not e["recoverable"]]
    return MissSummary(
        recoverable_count=len(recoverable),
        recoverable_usd=sum(e["miss_premium_usd"] for e in recoverable),
        break_count=len(breaks),
        break_usd=sum(e["miss_premium_usd"] for e in breaks),
    )


def gap_histogram(
    db: duckdb.DuckDBPyConnection,
    *,
    project: str | None = None,
    sidechain: bool = False,
) -> list[dict[str, Any]]:
    """Gap-size histogram with the prefix tokens at stake in each bucket.

    Token counts are what makes the histogram actionable: fifty short gaps
    over a 2k prefix are noise, three 20-minute gaps over a 200k prefix are
    the whole bill.
    """
    params: list[Any] = [sidechain]
    project_clause = ""
    if project is not None:
        project_clause = (
            " AND cr.session_id IN"
            " (SELECT session_id FROM logical_sessions WHERE project = ?)"
        )
        params.append(project)
    rows = db.execute(
        f"""
        SELECT cr.gap_bucket, COUNT(*), COALESCE(SUM(cr.prefix_total), 0)
        FROM cache_requests cr
        WHERE cr.is_sidechain = ? AND cr.gap_bucket IS NOT NULL
        {project_clause}
        GROUP BY 1
        """,  # noqa: S608
        params,
    ).fetchall()
    counts = {row[0]: (int(row[1]), int(row[2])) for row in rows}
    return [
        {
            "bucket": label,
            "count": counts.get(label, (0, 0))[0],
            "prefix_tokens": counts.get(label, (0, 0))[1],
            # Derived from the bucket's own upper bound, so renaming a label
            # can't silently turn the column all-False. Note this is the
            # *policy* band (what 1h rescues over 5m), not a claim about any
            # particular project: on a project already billed at 1h these
            # gaps are already warm, which is why the summary line reports
            # recoverable misses separately.
            "recoverable": upper is not None
            and TTL_5M_SECONDS < upper <= MAX_RECOVERABLE_GAP_SECONDS,
        }
        for label, upper in GAP_BUCKETS
    ]


def parity_residuals(
    db: duckdb.DuckDBPyConnection, *, sidechain: bool = False
) -> list[dict[str, Any]]:
    """Per-session simulated-vs-observed residuals on uniform-TTL sessions.

    The gate on the whole counterfactual: if simulating the TTL a session
    was *actually* billed at doesn't reproduce its bill, the gap definition
    or the prefix invariant is wrong and nothing built on top means anything.
    Sessions with mixed or unknown TTLs are excluded — there is no single
    policy to reproduce.
    """
    rows = db.execute(
        """
        WITH uniform AS (
            -- Uniformity is tested over *every* request in the session, not
            -- only the 5m/1h ones: pre-filtering would let a 'mixed' or
            -- 'unknown' row hide from the test and then be simulated at a
            -- TTL it was never billed under, reporting a residual against a
            -- model that is in fact correct.
            SELECT session_id,
                   ANY_VALUE(ttl_observed) AS ttl
            FROM cache_requests
            WHERE is_sidechain = ?
            GROUP BY session_id
            HAVING COUNT(*) FILTER (
                       WHERE ttl_observed NOT IN ('5m', '1h')
                   ) = 0
               AND COUNT(DISTINCT ttl_observed) = 1
        )
        SELECT
            cr.session_id,
            u.ttl,
            COUNT(*) AS n_requests,
            SUM(cr.cost_observed_usd) AS observed,
            SUM(CASE WHEN u.ttl = '1h' THEN cr.cost_1h_usd
                     ELSE cr.cost_5m_usd END) AS simulated
        FROM cache_requests cr
        JOIN uniform u ON u.session_id = cr.session_id
        WHERE cr.is_sidechain = ?
        GROUP BY cr.session_id, u.ttl
        """,
        [sidechain, sidechain],
    ).fetchall()
    out: list[dict[str, Any]] = []
    for session_id, ttl, n_requests, observed, simulated in rows:
        obs, sim = float(observed or 0.0), float(simulated or 0.0)
        out.append(
            {
                "session_id": str(session_id),
                "ttl_observed": ttl,
                "n_requests": int(n_requests),
                "observed_usd": obs,
                "simulated_usd": sim,
                "residual_usd": sim - obs,
                "residual_pct": (100.0 * (sim - obs) / obs) if obs > 0 else 0.0,
            }
        )
    out.sort(key=lambda r: abs(r["residual_pct"]), reverse=True)
    return out


def split_coverage(db: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Per-month coverage of the nested ``cache_creation`` 5m/1h split.

    Step-0 diagnostic: ``cache_creation_5m + cache_creation_1h`` must equal
    ``cache_creation_tokens`` wherever the split is present, and rows that
    lack it fall back to billing the total at the 5m rate.
    """
    rows = db.execute(
        """
        SELECT
            strftime(timestamp, '%Y-%m') AS month,
            COUNT(*) AS n_requests,
            COUNT(*) FILTER (WHERE cache_creation_tokens > 0) AS n_with_writes,
            COUNT(*) FILTER (
                WHERE cache_creation_tokens > 0
                  AND cache_creation_5m + cache_creation_1h = 0
            ) AS n_missing_split,
            COUNT(*) FILTER (
                WHERE cache_creation_5m + cache_creation_1h > 0
                  AND cache_creation_5m + cache_creation_1h
                      <> cache_creation_tokens
            ) AS n_split_mismatch
        FROM assistant_message_costs
        WHERE model IS DISTINCT FROM '<synthetic>'
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    return [
        {
            "month": row[0],
            "n_requests": int(row[1]),
            "n_with_writes": int(row[2]),
            "n_missing_split": int(row[3]),
            "n_split_mismatch": int(row[4]),
            "pct_missing_split": (
                100.0 * int(row[3]) / int(row[2]) if int(row[2]) else 0.0
            ),
        }
        for row in rows
    ]
