"""Cache-TTL detection and the 5m-vs-1h counterfactual.

The fixtures below are deliberately arithmetic-checkable: prefix sizes and
gaps are round numbers so an expected cost can be worked out by hand from
``pricing._PRICING`` rather than snapshotted from the implementation.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from introspect.cache_ttl import (
    MAX_RECOVERABLE_GAP_SECONDS,
    RECOMMENDATION_MARGIN_PCT,
    TTL_5M_SECONDS,
    cache_miss_event_rows,
    compare_ttl,
    gap_histogram,
    global_ttl_comparison,
    parity_residuals,
    project_ttl_comparisons,
    split_coverage,
    summarize_misses,
)
from introspect.db import materialize_views
from introspect.pricing import rates_for

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

_MODEL = "claude-opus-4-6"
_RATES = rates_for(_MODEL)
_T0 = datetime.fromisoformat("2026-04-21T09:00:00+00:00")


def _ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _usage(
    *, read: int, create: int, ttl: str | None = "5m", inp: int = 10, out: int = 20
) -> dict:
    """Usage block with an optional nested 5m/1h split.

    ``ttl=None`` reproduces the legacy schema — ``cache_creation_input_tokens``
    with no ``cache_creation`` sub-object — which bills at the 5m rate.
    """
    usage: dict = {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": create,
    }
    if ttl is not None:
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": create if ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": create if ttl == "1h" else 0,
        }
    return usage


def _turn(
    session_id: str,
    n: int,
    at: datetime,
    *,
    prompt: str = "go",
    read: int,
    create: int,
    ttl: str | None = "5m",
    inp: int = 10,
    out: int = 20,
) -> list[dict]:
    """One user prompt plus the assistant reply it triggered.

    The first prompt carries a ``toolUseResult`` so ``read_json_auto``
    infers that column; without it the raw-messages load fails to bind.
    """
    return [
        make_user_message(
            session_id,
            f"u{n}",
            f"a{n - 1}" if n > 1 else None,
            _ts(at),
            prompt,
            tool_use_result={"content": "seed"} if n == 1 else None,
        ),
        make_assistant_message(
            session_id,
            f"a{n}",
            f"u{n}",
            _ts(at + timedelta(seconds=1)),
            [{"type": "text", "text": f"reply {n}"}],
            model=_MODEL,
            msg_id=f"msg{n}",
            usage=_usage(read=read, create=create, ttl=ttl, inp=inp, out=out),
        ),
    ]


def _materialize(tmp: Path, session_id: str, lines: list[dict]):
    write_jsonl(tmp, session_id, lines)
    conn = duckdb.connect(":memory:")
    materialize_views(conn, glob_pattern(tmp), 0, resolve_projects=False)
    return conn


def _requests(conn, session_id: str) -> list[dict]:
    cols = [
        "seq",
        "gap_seconds",
        "ttl_observed",
        "prefix_total",
        "common_prefix",
        "structural_invalidation",
        "prefix_shrank",
        "cache_miss",
        "gap_recoverable",
        "gap_unrecoverable",
        "warm_5m",
        "warm_1h",
        "cost_5m_usd",
        "cost_1h_usd",
        "cost_observed_usd",
        "miss_premium_usd",
    ]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM cache_requests "
        "WHERE session_id = ? AND NOT is_sidechain ORDER BY seq",
        [session_id],
    ).fetchall()
    return [dict(zip(cols, row, strict=True)) for row in rows]


# --------------------------------------------------------------------------
# The headline fixture: gaps of 2 / 20 / 90 minutes plus a /compact.
# --------------------------------------------------------------------------

_MIXED_SID = "11111111-1111-1111-1111-111111111111"


def _mixed_gap_lines() -> list[dict]:
    """Five turns: cold start, 2 min gap, 20 min gap, 90 min gap, /compact.

    Prefix grows 10k -> 20k -> 30k -> 40k, then /compact truncates it to 5k.
    """
    lines: list[dict] = []
    lines += _turn(_MIXED_SID, 1, _T0, read=0, create=10_000)
    lines += _turn(
        _MIXED_SID, 2, _T0 + timedelta(minutes=2), read=10_000, create=10_000
    )
    lines += _turn(
        _MIXED_SID, 3, _T0 + timedelta(minutes=22), read=20_000, create=10_000
    )
    lines += _turn(
        _MIXED_SID, 4, _T0 + timedelta(minutes=112), read=30_000, create=10_000
    )
    # /compact: the prefix collapses and almost nothing is reused, but the
    # next request follows immediately.
    lines += _turn(
        _MIXED_SID,
        5,
        _T0 + timedelta(minutes=113),
        prompt="/compact",
        read=0,
        create=5_000,
    )
    return lines


@pytest.fixture
def mixed_gaps():
    with tempfile.TemporaryDirectory() as tmp_str:
        conn = _materialize(Path(tmp_str), _MIXED_SID, _mixed_gap_lines())
        yield conn, _requests(conn, _MIXED_SID)
        conn.close()


def test_gap_seconds_measured_from_previous_response_end(mixed_gaps):
    """Gaps run from the end of the previous reply to the next trigger."""
    _, reqs = mixed_gaps
    assert reqs[0]["gap_seconds"] is None  # nothing precedes the first request
    # Turn 2's prompt lands at T0+2min; turn 1's reply ended at T0+1s.
    assert reqs[1]["gap_seconds"] == 2 * 60 - 1
    assert reqs[2]["gap_seconds"] == 20 * 60 - 1
    assert reqs[3]["gap_seconds"] == 90 * 60 - 1


def test_warm_flags_differ_only_in_the_five_to_sixty_minute_band(mixed_gaps):
    """The whole counterfactual reduces to this disagreement."""
    _, reqs = mixed_gaps
    warm = [(r["warm_5m"], r["warm_1h"]) for r in reqs]
    assert warm[0] == (False, False)  # seq 1 is cold under any policy
    assert warm[1] == (True, True)  # 2 min: warm either way
    assert warm[2] == (False, True)  # 20 min: only 1h saves it
    assert warm[3] == (False, False)  # 90 min: neither
    assert warm[4] == (False, False)  # /compact: structural, cold either way


def test_twenty_minute_gap_is_cheaper_under_1h(mixed_gaps):
    """The 20-min turn is exactly the case a 1h TTL exists to fix."""
    _, reqs = mixed_gaps
    turn = reqs[2]
    assert turn["cost_1h_usd"] < turn["cost_5m_usd"]
    # Warm under 1h: read the 20k common prefix, write only the 10k delta.
    expected_1h = (
        20_000 * _RATES.cache_read
        + 10_000 * _RATES.cache_write_1h
        + 10 * _RATES.input
        + 20 * _RATES.output
    ) / 1_000_000
    assert turn["cost_1h_usd"] == pytest.approx(expected_1h)
    # Cold under 5m: the whole 30k prefix is rewritten.
    expected_5m = (
        30_000 * _RATES.cache_write_5m + 10 * _RATES.input + 20 * _RATES.output
    ) / 1_000_000
    assert turn["cost_5m_usd"] == pytest.approx(expected_5m)


def test_ninety_minute_gap_is_unrecoverable_under_both(mixed_gaps):
    """No TTL recovers it, so it is a break — never counted as waste."""
    _, reqs = mixed_gaps
    turn = reqs[3]
    assert turn["cache_miss"] is True
    assert turn["gap_recoverable"] is False
    assert turn["gap_unrecoverable"] is True
    # Identical bill under both policies apart from the write rate itself.
    assert turn["warm_5m"] is False
    assert turn["warm_1h"] is False


def test_compact_is_structural_not_a_pause(mixed_gaps):
    """A short-gap turn that reused nothing changed prefix, not TTL."""
    _, reqs = mixed_gaps
    turn = reqs[4]
    assert turn["structural_invalidation"] is True
    assert turn["prefix_shrank"] is True
    assert turn["cache_miss"] is False
    assert turn["miss_premium_usd"] == 0.0


def test_no_gap_session_is_cheaper_under_5m():
    """1h pays 2x on every incremental write; with no gaps that is pure loss."""
    sid = "22222222-2222-2222-2222-222222222222"
    lines = _turn(sid, 1, _T0, read=0, create=10_000)
    for n in range(2, 8):
        lines += _turn(
            sid,
            n,
            _T0 + timedelta(seconds=30 * (n - 1)),
            read=10_000 * (n - 1),
            create=10_000,
        )
    with tempfile.TemporaryDirectory() as tmp_str:
        conn = _materialize(Path(tmp_str), sid, lines)
        try:
            verdict = global_ttl_comparison(conn)
            assert verdict.cost_1h > verdict.cost_5m
            assert verdict.recommendation == "5m"
            assert verdict.n_gaps_recoverable == 0
        finally:
            conn.close()


def test_recoverable_and_unrecoverable_are_reported_apart(mixed_gaps):
    """The cap is the point: only the 20-min gap counts as avoidable waste."""
    conn, _ = mixed_gaps
    misses = summarize_misses(cache_miss_event_rows(conn, session_id=_MIXED_SID))
    assert misses["recoverable_count"] == 1
    assert misses["break_count"] == 1
    assert misses["recoverable_usd"] > 0
    assert misses["break_usd"] > 0
    # A caller that adds them back together gets the old, inflated number.
    assert misses["count"] == 2


def test_tool_result_triggered_break_is_detected():
    """A tool that runs past the TTL expires the cache like a pause does.

    The previous human-prompt-anchored rule could not see this at all.
    """
    sid = "33333333-3333-3333-3333-333333333333"
    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            _ts(_T0),
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            _ts(_T0 + timedelta(seconds=1)),
            [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            model=_MODEL,
            msg_id="msg1",
            usage=_usage(read=0, create=10_000),
        ),
        # The tool returns 20 minutes later — no human involved.
        make_user_message(
            sid,
            "u2",
            "a1",
            _ts(_T0 + timedelta(minutes=20)),
            [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}],
        ),
        make_assistant_message(
            sid,
            "a2",
            "u2",
            _ts(_T0 + timedelta(minutes=20, seconds=1)),
            [{"type": "text", "text": "ok"}],
            model=_MODEL,
            msg_id="msg2",
            usage=_usage(read=0, create=20_000),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        conn = _materialize(Path(tmp_str), sid, lines)
        try:
            events = cache_miss_event_rows(conn, session_id=sid)
            assert len(events) == 1
            assert events[0]["recoverable"] is True
            assert events[0]["gap_seconds"] == pytest.approx(20 * 60 - 1, abs=2)
        finally:
            conn.close()


# --------------------------------------------------------------------------
# Parity: simulating the TTL a session was actually billed at must reproduce
# its bill. Without this gate nothing built on the simulation means anything.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_uniform_ttl_session_simulates_to_its_observed_bill(ttl):
    sid = f"44444444-4444-4444-4444-4444444444{'55' if ttl == '5m' else '11'}"
    lines = _turn(sid, 1, _T0, read=0, create=10_000, ttl=ttl)
    for n in range(2, 6):
        lines += _turn(
            sid,
            n,
            _T0 + timedelta(minutes=n - 1),
            read=10_000 * (n - 1),
            create=10_000,
            ttl=ttl,
        )
    with tempfile.TemporaryDirectory() as tmp_str:
        conn = _materialize(Path(tmp_str), sid, lines)
        try:
            residuals = parity_residuals(conn)
            assert len(residuals) == 1
            assert residuals[0]["ttl_observed"] == ttl
            assert residuals[0]["residual_pct"] == pytest.approx(0.0, abs=1e-9)
        finally:
            conn.close()


def test_legacy_rows_without_the_split_still_simulate():
    """No nested cache_creation → ttl_observed 'unknown', simulation runs."""
    sid = "55555555-5555-5555-5555-555555555555"
    lines = _turn(sid, 1, _T0, read=0, create=10_000, ttl=None)
    lines += _turn(sid, 2, _T0 + timedelta(minutes=20), read=0, create=20_000, ttl=None)
    with tempfile.TemporaryDirectory() as tmp_str:
        conn = _materialize(Path(tmp_str), sid, lines)
        try:
            reqs = _requests(conn, sid)
            assert [r["ttl_observed"] for r in reqs] == ["unknown", "unknown"]
            # Legacy rows bill at 5m, so a 20-min gap is a recoverable miss.
            assert reqs[1]["cache_miss"] is True
            assert reqs[1]["gap_recoverable"] is True
            assert reqs[1]["cost_1h_usd"] < reqs[1]["cost_5m_usd"]
            # Every month is reported as lacking the split.
            coverage = split_coverage(conn)
            assert coverage[0]["pct_missing_split"] == pytest.approx(100.0)
            assert coverage[0]["n_split_mismatch"] == 0
        finally:
            conn.close()


def test_session_already_on_1h_has_nothing_recoverable():
    """You cannot recover a gap by switching to the TTL you already have."""
    sid = "66666666-6666-6666-6666-666666666666"
    lines = _turn(sid, 1, _T0, read=0, create=10_000, ttl="1h")
    lines += _turn(
        sid, 2, _T0 + timedelta(minutes=20), read=10_000, create=10_000, ttl="1h"
    )
    with tempfile.TemporaryDirectory() as tmp_str:
        conn = _materialize(Path(tmp_str), sid, lines)
        try:
            reqs = _requests(conn, sid)
            assert reqs[1]["cache_miss"] is False
            assert reqs[1]["gap_recoverable"] is False
        finally:
            conn.close()


# --------------------------------------------------------------------------
# Rollups and the recommendation gate.
# --------------------------------------------------------------------------


def test_gap_histogram_carries_tokens_at_stake(mixed_gaps):
    conn, _ = mixed_gaps
    buckets = {b["bucket"]: b for b in gap_histogram(conn)}
    assert buckets["<5m"]["count"] == 2  # the 2-min gap and the /compact turn
    assert buckets["15-60m"]["count"] == 1
    assert buckets[">60m"]["count"] == 1
    assert buckets["15-60m"]["prefix_tokens"] == 30_000
    assert buckets["15-60m"]["recoverable"] is True
    assert buckets[">60m"]["recoverable"] is False


def test_project_rollup_carries_the_verdict(mixed_gaps):
    conn, _ = mixed_gaps
    per_project = project_ttl_comparisons(conn)
    assert len(per_project) == 1
    _, verdict = per_project[0]
    assert verdict.n_requests == 5
    assert verdict.n_gaps_recoverable == 1
    assert verdict.n_gaps_unrecoverable == 1
    assert verdict.n_structural == 1


def test_sidechains_are_never_merged_into_the_main_verdict(mixed_gaps):
    """Subagents carry their own setting, so they roll up separately."""
    conn, _ = mixed_gaps
    assert global_ttl_comparison(conn, sidechain=True).n_requests == 0
    assert global_ttl_comparison(conn, sidechain=False).n_requests == 5


def test_margin_inside_the_noise_band_is_not_a_recommendation():
    """A 1% edge is modelling error, not a decision."""
    verdict = compare_ttl(cost_5m=100.0, cost_1h=101.0)
    assert verdict.recommendation == "either"
    assert verdict.decisive is False
    assert verdict.margin_pct == pytest.approx(1.0)


def test_margin_beyond_the_noise_band_recommends():
    verdict = compare_ttl(cost_5m=100.0, cost_1h=80.0)
    assert verdict.recommendation == "1h"
    assert verdict.decisive is True
    assert verdict.savings == pytest.approx(20.0)
    assert verdict.margin_pct == pytest.approx(25.0)
    assert verdict.margin_pct > RECOMMENDATION_MARGIN_PCT


def test_zero_spend_is_not_a_recommendation():
    verdict = compare_ttl(cost_5m=0.0, cost_1h=0.0)
    assert verdict.recommendation == "either"
    assert verdict.margin_pct == 0.0


def test_constants_bound_the_recoverable_band():
    """The two thresholds the whole feature hangs on."""
    assert TTL_5M_SECONDS == 300
    assert MAX_RECOVERABLE_GAP_SECONDS == 3600
