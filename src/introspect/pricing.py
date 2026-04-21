"""Hardcoded Anthropic API pricing (USD per 1M tokens).

Snapshot fetched 2026-04-21 from Anthropic's pricing page. No live fetch.

The module exposes both:

* Python helpers (``rates_for``, ``compute_cost_usd``) for per-row cost
  computation in handlers.
* SQL ``CASE`` expressions (``PRICING_INPUT_RATE_SQL``,
  ``PRICING_OUTPUT_RATE_SQL``) for use in the sessions-list cost subquery,
  where DuckDB needs per-row pricing so mixed-model sessions sort correctly.

Cache rates derive from input * {1.25, 2.0, 0.1} for write-5m / write-1h /
read; the named-tuple stores the absolute rates so the SQL strings can be
generated mechanically.
"""

from __future__ import annotations

import functools
import logging
from typing import NamedTuple

log = logging.getLogger(__name__)

_PER_MILLION = 1_000_000


class Rates(NamedTuple):
    """Pricing for one model family — USD per 1M tokens for each token type."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


# Each entry maps a model-name *prefix* to its rate table.  Prefix matching
# (not equality) lets us cover dated suffixes like "claude-haiku-4-5-20251001".
_PRICING: dict[str, Rates] = {
    # Opus current generation
    "claude-opus-4-7": Rates(5, 6.25, 10, 0.50, 25),
    "claude-opus-4-6": Rates(5, 6.25, 10, 0.50, 25),
    "claude-opus-4-5": Rates(5, 6.25, 10, 0.50, 25),
    # Opus legacy generation
    "claude-opus-4-1": Rates(15, 18.75, 30, 1.50, 75),
    "claude-opus-4": Rates(15, 18.75, 30, 1.50, 75),
    "claude-opus-3": Rates(15, 18.75, 30, 1.50, 75),
    # Sonnet
    "claude-sonnet-4-6": Rates(3, 3.75, 6, 0.30, 15),
    "claude-sonnet-4-5": Rates(3, 3.75, 6, 0.30, 15),
    "claude-sonnet-4": Rates(3, 3.75, 6, 0.30, 15),
    "claude-sonnet-3-7": Rates(3, 3.75, 6, 0.30, 15),
    # Haiku
    "claude-haiku-4-5": Rates(1, 1.25, 2, 0.10, 5),
    "claude-haiku-3-5": Rates(0.80, 1, 1.60, 0.08, 4),
}

_ZERO_RATES = Rates(0, 0, 0, 0, 0)
_SYNTHETIC = "<synthetic>"


# Sort once at module load: CASE evaluates branches in order, and LIKE
# 'claude-opus-4%' would otherwise swallow 'claude-opus-4-7'.  Sort longest
# prefix first so the most specific match wins.  Reused by ``rates_for`` so
# Python and SQL share the same prefix-resolution order.
_PRICING_BY_PREFIX_LEN = sorted(
    _PRICING.items(), key=lambda item: len(item[0]), reverse=True
)

# Defensive: every prefix is interpolated into a SQL LIKE literal in
# ``_build_case_sql``.  Fail loud at module load if a key contains a single
# quote (would break the SQL string) or a ``%`` wildcard (would broaden the
# CASE branch silently).  Today's keys are clean; this exists so a future
# contributor adding e.g. ``"claude-foo'bar"`` gets a clear error instead of
# a SQL syntax exception thousands of lines away.
for _prefix in _PRICING:
    if "'" in _prefix or "%" in _prefix:
        msg = f"_PRICING key {_prefix!r} contains a SQL-unsafe character"
        raise ValueError(msg)


@functools.lru_cache(maxsize=128)
def _warn_unknown_model_once(model: str) -> None:
    """Emit one WARNING per unknown model name (bounded LRU)."""
    log.warning("unknown model for pricing: %r — billing as $0", model)


def rates_for(model: str | None) -> Rates:
    """Look up rates for a model, matching by prefix.

    Returns zero rates for ``<synthetic>``, ``None``, the empty string, or any
    unrecognized model. Unknown models are logged once at WARNING level
    (bounded by an LRU cache so a flood of distinct unknown names can't grow
    memory unboundedly).
    """
    if not model or model == _SYNTHETIC:
        return _ZERO_RATES
    # Prefer the longest matching prefix (so "claude-opus-4-1" beats
    # "claude-opus-4" if both are present).  Reuse the same sorted list the
    # SQL builder uses to keep the two lookup strategies consistent.
    for prefix, rates in _PRICING_BY_PREFIX_LEN:
        if model.startswith(prefix):
            return rates
    _warn_unknown_model_once(model)
    return _ZERO_RATES


def compute_cost_usd(  # noqa: PLR0913
    *,
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_5m: int = 0,
    cache_creation_1h: int = 0,
) -> float:
    """Compute the USD cost of a single API call (or aggregate of same-model calls).

    Tokens are token counts; rates are USD per 1M tokens.
    """
    r = rates_for(model)
    return (
        (input_tokens or 0) * r.input
        + (output_tokens or 0) * r.output
        + (cache_read_tokens or 0) * r.cache_read
        + (cache_creation_5m or 0) * r.cache_write_5m
        + (cache_creation_1h or 0) * r.cache_write_1h
    ) / _PER_MILLION


def _build_case_sql(rate_attr: str) -> str:
    """Build a DuckDB CASE expression that returns the per-1M rate for a model.

    Used by the sessions-list cost subquery so per-row pricing happens in SQL
    (avoids materializing every assistant message in Python just to sort the
    sessions table).
    """
    branches = [
        f"WHEN model LIKE '{prefix}%' THEN {getattr(rates, rate_attr)}"
        for prefix, rates in _PRICING_BY_PREFIX_LEN
    ]
    return "CASE " + " ".join(branches) + " ELSE 0 END"


# SQL CASE expressions for per-row pricing in DuckDB. Keep in lockstep with
# _PRICING — the test suite checks SQL totals match Python totals.
PRICING_INPUT_RATE_SQL = _build_case_sql("input")
PRICING_OUTPUT_RATE_SQL = _build_case_sql("output")
PRICING_CACHE_READ_RATE_SQL = _build_case_sql("cache_read")
PRICING_CACHE_WRITE_5M_RATE_SQL = _build_case_sql("cache_write_5m")
PRICING_CACHE_WRITE_1H_RATE_SQL = _build_case_sql("cache_write_1h")
