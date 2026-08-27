"""The two cache-break detectors must agree on the real e2e transcripts.

Before ``cache_requests`` there were two rules with different thresholds
(300s with a cache_creation > cache_read condition for the session divider;
270s with a read-drop ratio and a 5k floor for the tokenscape event track),
so one session could be marked broken on one tab and clean on the other.
These tests pin the agreement to the shared view rather than to a snapshot
of either detector's output.
"""

from __future__ import annotations

import duckdb
import pytest

from introspect.api.handlers.tokenscape import _fetch_ttl_break_uuids
from introspect.cache_ttl import cache_miss_event_rows
from introspect.db import materialize_views

from .conftest import E2E_DATA_DIR


@pytest.fixture(scope="module")
def e2e_conn():
    """Materialized DB over the checked-in e2e transcripts."""
    conn = duckdb.connect(":memory:")
    materialize_views(
        conn,
        str(E2E_DATA_DIR / "projects" / "**" / "*.jsonl"),
        0,
        resolve_projects=False,
    )
    yield conn
    conn.close()


def _session_ids(conn) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT session_id FROM cache_requests ORDER BY 1"
        ).fetchall()
    ]


def test_e2e_data_exercises_the_view(e2e_conn):
    """Guard the guard: an empty view would make the next test vacuous."""
    assert _session_ids(e2e_conn)


def test_divider_and_tokenscape_agree_on_every_event(e2e_conn):
    """Both tracks resolve to the identical set of request uuids.

    Now that both read ``cache_requests`` this is close to arithmetic — that
    is the point. It fails the moment either surface reintroduces a private
    threshold or filter of its own, which is exactly the regression that
    made one session read as broken on one tab and clean on the other.
    """
    for session_id in _session_ids(e2e_conn):
        divider = {
            event["uuid"]
            for event in cache_miss_event_rows(e2e_conn, session_id=session_id)
        }
        tokenscape = _fetch_ttl_break_uuids(e2e_conn, session_id, sidechain=False)
        assert divider == tokenscape, f"detectors disagree on {session_id}"


def test_no_event_is_counted_as_both_waste_and_break(e2e_conn):
    """The recoverable/unrecoverable split is a partition, not two filters."""
    overlap = e2e_conn.execute(
        "SELECT COUNT(*) FROM cache_requests"
        " WHERE gap_recoverable AND gap_unrecoverable"
    ).fetchone()
    assert overlap[0] == 0
    orphan = e2e_conn.execute(
        "SELECT COUNT(*) FROM cache_requests"
        " WHERE cache_miss AND NOT gap_recoverable AND NOT gap_unrecoverable"
    ).fetchone()
    assert orphan[0] == 0


def test_prefix_invariant_holds_on_every_warm_request(e2e_conn):
    """A warm request reads exactly what the previous request left behind.

    ``prefix_total = cache_read + cache_creation`` is what makes the
    counterfactual honest — if it drifts, the simulated read/write split
    stops meaning anything.
    """
    bad = e2e_conn.execute(
        """
        SELECT COUNT(*) FROM cache_requests
        WHERE seq > 1 AND NOT cache_miss AND NOT structural_invalidation
          AND NOT prefix_shrank
          AND cache_read_tokens > prev_prefix_total
        """
    ).fetchone()
    assert bad[0] == 0


def test_split_sums_to_the_reported_cache_creation_total(e2e_conn):
    """Where the nested 5m/1h split exists it must equal the flat total."""
    mismatched = e2e_conn.execute(
        """
        SELECT COUNT(*) FROM assistant_message_costs
        WHERE cache_creation_5m + cache_creation_1h > 0
          AND cache_creation_5m + cache_creation_1h <> cache_creation_tokens
        """
    ).fetchone()
    assert mismatched[0] == 0
