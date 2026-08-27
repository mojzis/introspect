"""Table-driven tests for ``introspect.pricing``."""

from __future__ import annotations

import pytest

from introspect.pricing import (
    LONG_CONTEXT_INPUT_TOKENS,
    PRICING_CACHE_READ_RATE_SQL,
    PRICING_CACHE_WRITE_1H_RATE_SQL,
    PRICING_CACHE_WRITE_5M_RATE_SQL,
    PRICING_INPUT_RATE_SQL,
    PRICING_OUTPUT_RATE_SQL,
    Rates,
    compute_cost_usd,
    rates_for,
)

# (model_name, expected Rates) — covers every family + a dated suffix variant.
_RATE_CASES = [
    ("claude-fable-5", Rates(10, 12.50, 20, 1.00, 50)),
    ("claude-mythos-5", Rates(10, 12.50, 20, 1.00, 50)),
    ("claude-opus-5", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-8", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-7", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-6", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-5", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-1", Rates(15, 18.75, 30, 1.50, 75)),
    ("claude-opus-3", Rates(15, 18.75, 30, 1.50, 75)),
    ("claude-sonnet-5", Rates(2, 2.50, 4, 0.20, 10)),
    ("claude-sonnet-4-6", Rates(3, 3.75, 6, 0.30, 15)),
    ("claude-sonnet-3-7", Rates(3, 3.75, 6, 0.30, 15)),
    ("claude-haiku-4-5-20251001", Rates(1, 1.25, 2, 0.10, 5)),
    ("claude-haiku-3-5", Rates(0.80, 1, 1.60, 0.08, 4)),
    ("gpt-5.6-sol", Rates(4.00, 5.00, 8.00, 0.40, 20.00)),
    ("gpt-5.6-terra", Rates(2.00, 2.50, 4.00, 0.20, 12.00)),
    ("gpt-5.6-luna", Rates(0.20, 0.25, 0.40, 0.02, 1.20)),
    ("<synthetic>", Rates(0, 0, 0, 0, 0)),
    ("totally-made-up-model", Rates(0, 0, 0, 0, 0)),
    (None, Rates(0, 0, 0, 0, 0)),
    ("", Rates(0, 0, 0, 0, 0)),
]


@pytest.mark.parametrize(("model", "expected"), _RATE_CASES)
def test_rates_for(model, expected):
    """Each model family resolves to its expected rate table."""
    assert rates_for(model) == expected


def test_every_registered_model_resolves_to_its_own_rates():
    """No entry is shadowed by another prefix (which would silently mis-bill)."""
    from introspect.pricing import _PRICING  # noqa: PLC0415

    for prefix, rates in _PRICING.items():
        assert rates_for(prefix) == rates, prefix


def test_rates_for_prefix_match_picks_longest():
    """Dated suffix variants resolve via prefix match without overshadowing."""
    # opus-4-1 should match the legacy table, not the current opus-4 one
    assert rates_for("claude-opus-4-1-some-date").input == 15
    # opus-4 alone (legacy) should not silently bind to opus-4-7 etc.
    assert rates_for("claude-opus-4-99").input == 15


def test_compute_cost_usd_zero_for_no_tokens():
    """Empty usage yields zero cost."""
    assert compute_cost_usd(model="claude-opus-4-7") == 0


def test_compute_cost_usd_opus_current():
    """Per-1M math works for opus current generation."""
    # 1M input, 1M output, 1M cache_read, 1M cache_5m, 1M cache_1h
    cost = compute_cost_usd(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_5m=1_000_000,
        cache_creation_1h=1_000_000,
    )
    assert cost == pytest.approx(5 + 25 + 0.50 + 6.25 + 10)


def test_compute_cost_usd_haiku():
    """Per-1M math works for the Haiku 4.5 family with dated suffix."""
    cost = compute_cost_usd(
        model="claude-haiku-4-5-20251001",
        input_tokens=2_000_000,
        output_tokens=500_000,
    )
    # 2 * 1.0 input + 0.5 * 5.0 output
    assert cost == pytest.approx(2 + 2.5)


def test_compute_cost_usd_codex():
    """Short-context Codex requests use the current OpenAI list prices."""
    cost = compute_cost_usd(
        model="gpt-5.6-terra",
        input_tokens=200_000,
        output_tokens=50_000,
        cache_read_tokens=100_000,
    )
    # 0.2 * 2.0 input + 0.05 * 12.0 output + 0.1 * 0.2 cache_read
    assert cost == pytest.approx(0.4 + 0.6 + 0.02)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-sol", Rates(8.00, 10.00, 16.00, 0.80, 30.00)),
        ("gpt-5.6-terra", Rates(4.00, 5.00, 8.00, 0.40, 18.00)),
        ("gpt-5.6-luna", Rates(0.40, 0.50, 0.80, 0.04, 1.80)),
    ],
)
def test_rates_for_uses_long_context_codex_rates(model, expected):
    """Codex requests switch to the long-context table at the published boundary."""
    assert rates_for(model, input_tokens=LONG_CONTEXT_INPUT_TOKENS) != expected
    assert rates_for(model, input_tokens=LONG_CONTEXT_INPUT_TOKENS + 1) == expected


def test_compute_cost_usd_codex_long_context():
    """All Codex token classes use long-context rates above 272K input tokens."""
    cost = compute_cost_usd(
        model="gpt-5.6-sol",
        input_tokens=LONG_CONTEXT_INPUT_TOKENS + 1,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_5m=1_000_000,
        cache_creation_1h=1_000_000,
    )
    assert cost == pytest.approx(2.176008 + 30 + 0.80 + 10 + 16)


def test_compute_cost_usd_synthetic_is_zero():
    """Synthetic + unknown models always cost $0."""
    assert (
        compute_cost_usd(
            model="<synthetic>",
            input_tokens=10_000_000,
            output_tokens=10_000_000,
        )
        == 0
    )
    assert (
        compute_cost_usd(
            model="brand-new-model",
            input_tokens=10_000_000,
        )
        == 0
    )


# All five rate dimensions feed SESSION_COST_SUBQUERY; a typo in any one
# would silently mis-bill (cache_read in particular is the dominant cost for
# typical Claude Code sessions).
_SQL_RATE_PAIRS = [
    ("input", PRICING_INPUT_RATE_SQL),
    ("output", PRICING_OUTPUT_RATE_SQL),
    ("cache_read", PRICING_CACHE_READ_RATE_SQL),
    ("cache_write_5m", PRICING_CACHE_WRITE_5M_RATE_SQL),
    ("cache_write_1h", PRICING_CACHE_WRITE_1H_RATE_SQL),
]

_SQL_TEST_MODELS = [
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-1",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-haiku-3-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "<synthetic>",
    "unknown-model",
]


@pytest.mark.parametrize(
    "input_tokens", [0, LONG_CONTEXT_INPUT_TOKENS, LONG_CONTEXT_INPUT_TOKENS + 1]
)
@pytest.mark.parametrize("model", _SQL_TEST_MODELS)
@pytest.mark.parametrize(("attr", "sql"), _SQL_RATE_PAIRS)
def test_pricing_sql_matches_python(model, attr, sql, input_tokens):
    """Each SQL CASE rate must agree with the Python ``Rates`` attribute."""
    import duckdb  # noqa: PLC0415

    conn = duckdb.connect(":memory:")
    row = conn.execute(
        f"SELECT {sql} FROM (SELECT ? AS model, ? AS input_tokens)",
        [model, input_tokens],
    ).fetchone()
    assert row is not None, model
    # DuckDB returns Decimal for fractional CASE-derived numerics; coerce to
    # float so the equality check is value-based.
    assert float(row[0]) == pytest.approx(
        getattr(rates_for(model, input_tokens=input_tokens), attr)
    )
