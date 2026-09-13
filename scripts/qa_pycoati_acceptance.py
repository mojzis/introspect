"""Black-box functional QA for the project's locked Pycoati acceptance flow."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYCOATI_VERSION = "0.2.9"
SUBPROCESS_TIMEOUT_SECONDS = 120
ACCEPTED_NODEID = "tests/test_fixture.py::test_accept_me"
STOPPED_NODEID = "tests/test_fixture.py::test_signal_stopped"
ACTIVE_NODEID = "tests/test_fixture.py::test_active_signal"
SUBPROCESS_TEST_NAMES = (
    "test_checked_run",
    "test_check_call",
    "test_check_output",
    "test_check_returncode",
    "test_unchecked_run",
    "test_check_false",
    "test_replaced_subprocess_name",
    "test_swallowed_checked_run",
    "test_suppressed_checked_run",
)


class QaFailure(RuntimeError):
    """A failed black-box QA expectation."""


@dataclass(frozen=True)
class AcceptanceEvidence:
    """Inventories and shortlist observations from the three acceptance modes."""

    initial_inventory: dict[str, Any]
    default_inventory: dict[str, Any]
    raw_inventory: dict[str, Any]
    included_inventory: dict[str, Any]
    default_shortlist: list[str]
    raw_shortlist: list[str]
    included_shortlist: list[str]


def _fail(message: str, *, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise QaFailure(message)
    raise QaFailure(message) from cause


def _run_command(
    command: list[str], *, cwd: Path, phase: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        _fail(
            f"{phase} timed out after {SUBPROCESS_TIMEOUT_SECONDS} seconds",
            cause=error,
        )


def _scan(project: Path, *flags: str) -> dict[str, Any]:
    output = project / "inventory.json"
    command = [
        "uv",
        "run",
        "--locked",
        "pycoati",
        str(project),
        "--python",
        sys.executable,
        *flags,
        "--output",
        str(output),
    ]
    completed = _run_command(command, cwd=ROOT, phase="Pycoati scan")
    if completed.returncode != 0:
        _fail(f"Pycoati failed ({completed.returncode}): {completed.stderr.strip()}")
    return {
        "inventory": json.loads(output.read_text()),
        "stderr": completed.stderr,
    }


def _tests(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["nodeid"]: record for record in inventory["test_functions"]}


def _without_shortlist(inventory: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in inventory.items()
        if key not in {"top_suspicious", "accepted"}
    }
    result["suite"] = {
        key: value
        for key, value in result["suite"].items()
        if key not in {"runtime_seconds", "slowest_tests"}
    }
    result["test_functions"] = [
        {key: value for key, value in record.items() if key != "accepted_signals"}
        for record in inventory["test_functions"]
    ]
    return result


def _write_fixture(project: Path) -> None:
    (project / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "pycoati-qa-fixture"
            version = "0.1.0"
            requires-python = ">=3.11"
            """
        ).lstrip()
    )
    (project / "child.py").write_text("raise SystemExit(0)\n")
    package = project / "pycoati_qa_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("def touch():\n    return True\n")
    (project / "tests").mkdir()
    (project / "tests" / "test_fixture.py").write_text(
        dedent(
            """
            import contextlib
            import subprocess
            import sys
            from pathlib import Path
            from unittest.mock import Mock

            import pycoati_qa_fixture


            CHILD = str(Path(__file__).parents[1] / "child.py")


            def test_checked_run():
                assert pycoati_qa_fixture.touch()
                subprocess.run([sys.executable, CHILD], check=True)


            def test_check_call():
                subprocess.check_call([sys.executable, CHILD])


            def test_check_output():
                subprocess.check_output([sys.executable, CHILD])


            def test_check_returncode():
                completed = subprocess.run([sys.executable, CHILD])
                completed.check_returncode()


            def test_unchecked_run():
                subprocess.run([sys.executable, CHILD])


            def test_check_false():
                subprocess.run([sys.executable, CHILD], check=False)


            def test_replaced_subprocess_name():
                subprocess = Mock()
                subprocess.run([sys.executable, CHILD], check=True)


            def test_swallowed_checked_run():
                try:
                    subprocess.run([sys.executable, CHILD], check=True)
                except subprocess.CalledProcessError:
                    pass


            def test_suppressed_checked_run():
                with contextlib.suppress(subprocess.CalledProcessError):
                    subprocess.run([sys.executable, CHILD], check=True)


            def test_accept_me():
                value = 1
                value += 1


            def test_signal_stopped():
                assert 1 == 1


            def test_active_signal():
                worker = Mock()
                worker()
                assert worker.called
            """
        ).lstrip()
    )


def _assert_subprocess_contract(scan: dict[str, Any]) -> None:
    inventory = scan["inventory"]
    records = _tests(inventory)
    expected_external = {
        "test_checked_run": 1,
        "test_check_call": 1,
        "test_check_output": 1,
        "test_check_returncode": 1,
        "test_unchecked_run": 0,
        "test_check_false": 0,
        "test_replaced_subprocess_name": 0,
        "test_swallowed_checked_run": 0,
        "test_suppressed_checked_run": 0,
    }
    for name, expected in expected_external.items():
        nodeid = f"tests/test_fixture.py::{name}"
        actual = records[nodeid]["external_verification_count"]
        if actual != expected:
            _fail(f"{name}: expected {expected}, got {actual}")
    if not inventory["tool"]["ran_pytest"] or not inventory["tool"]["ran_coverage"]:
        _fail(
            "successful subprocess fixture did not run pytest+coverage: "
            f"{scan['stderr'].strip()}"
        )


def _verify_version(inventory: dict[str, Any]) -> None:
    actual_version = inventory["tool"]["version"]
    if actual_version != EXPECTED_PYCOATI_VERSION:
        _fail(f"expected Pycoati {EXPECTED_PYCOATI_VERSION}, got {actual_version}")


def _verify_failure_propagation(project: Path) -> None:
    child = project / "child.py"
    child.write_text("raise SystemExit(3)\n")
    try:
        failed_test = _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_fixture.py::test_checked_run",
                "-q",
            ],
            cwd=project,
            phase="focused pytest failure probe",
        )
        if (
            failed_test.returncode == 0
            or "CalledProcessError" not in failed_test.stdout
        ):
            _fail("pytest failure diagnostic was not observed")
        failed_inventory = _scan(project, "--no-accept")["inventory"]
        if not failed_inventory["tool"]["ran_pytest"]:
            _fail("a test failure was mistaken for a broken pytest run")
    finally:
        child.write_text("raise SystemExit(0)\n")


def _write_acceptance_baseline(project: Path, fingerprint: str) -> None:
    (project / ".pycoati-accept.toml").write_text(
        dedent(
            f"""
            schema_version = "1"

            [[accept]]
            test = "{ACCEPTED_NODEID}"
            signals = ["zero_asserts"]
            reason = "No-op fixture retained to verify acceptance filtering."
            reviewed = "2026-09-13"
            fingerprint = "{fingerprint}"
            """
        ).lstrip()
    )


def _collect_acceptance_evidence(
    project: Path, initial_inventory: dict[str, Any]
) -> AcceptanceEvidence:
    accepted_fingerprint = _tests(initial_inventory)[ACCEPTED_NODEID]["fingerprint"]
    _write_acceptance_baseline(project, accepted_fingerprint)
    default_inventory = _scan(project)["inventory"]
    raw_inventory = _scan(project, "--no-accept")["inventory"]
    included_inventory = _scan(project, "--include-accepted")["inventory"]
    return AcceptanceEvidence(
        initial_inventory=initial_inventory,
        default_inventory=default_inventory,
        raw_inventory=raw_inventory,
        included_inventory=included_inventory,
        default_shortlist=default_inventory["top_suspicious"]["test_functions"],
        raw_shortlist=raw_inventory["top_suspicious"]["test_functions"],
        included_shortlist=included_inventory["top_suspicious"]["test_functions"],
    )


def _verify_acceptance_filtering(evidence: AcceptanceEvidence) -> None:
    if ACCEPTED_NODEID in evidence.default_shortlist:
        _fail("accepted finding remained actionable")
    if (
        ACCEPTED_NODEID not in evidence.raw_shortlist
        or ACCEPTED_NODEID not in evidence.included_shortlist
    ):
        _fail("raw/include-accepted did not retain accepted finding")
    if ACTIVE_NODEID not in evidence.default_shortlist:
        _fail("active mock signal was incorrectly suppressed")
    reported = evidence.default_inventory["accepted"]["findings"]
    if not reported or reported[0]["test"] != ACCEPTED_NODEID:
        _fail("accepted finding was not reported")
    if evidence.included_inventory["accepted"]["included_in_shortlist"] is not True:
        _fail("include-accepted flag was not reported")
    if evidence.raw_inventory["accepted"]["path"] is not None:
        _fail("--no-accept read the baseline")


def _verify_audit_parity(evidence: AcceptanceEvidence) -> None:
    normalized_audits = {
        "default": _without_shortlist(evidence.default_inventory),
        "raw": _without_shortlist(evidence.raw_inventory),
        "include-accepted": _without_shortlist(evidence.included_inventory),
    }
    raw_audit = normalized_audits["raw"]
    for label, audit in normalized_audits.items():
        if audit != raw_audit:
            differing = [key for key in raw_audit if raw_audit[key] != audit[key]]
            _fail(f"raw/{label} audit records diverged: {differing}")
    for inventory in (
        evidence.default_inventory,
        evidence.raw_inventory,
        evidence.included_inventory,
    ):
        if not inventory["tool"]["ran_pytest"] or not inventory["tool"]["ran_coverage"]:
            _fail("acceptance scan did not run pytest+coverage")


def _write_stale_baseline(
    project: Path, *, accepted_fingerprint: str, stopped_fingerprint: str
) -> None:
    (project / ".pycoati-accept.toml").write_text(
        dedent(
            f"""
            schema_version = "1"

            [[accept]]
            test = "{ACCEPTED_NODEID}"
            signals = ["zero_asserts"]
            reason = "No-op fixture retained to verify acceptance filtering."
            reviewed = "2026-09-13"
            fingerprint = "{accepted_fingerprint}"

            [[accept]]
            test = "{STOPPED_NODEID}"
            signal = "zero_asserts"
            reason = "Stale-state fixture: the test now has an assertion."
            fingerprint = "{stopped_fingerprint}"

            [[accept]]
            test = "tests/test_fixture.py::test_missing"
            signal = "zero_asserts"
            reason = "Stale-state fixture: the nodeid is intentionally absent."
            """
        ).lstrip()
    )


def _verify_stale_states(
    project: Path, initial_inventory: dict[str, Any]
) -> dict[str, str]:
    initial_records = _tests(initial_inventory)
    accepted_fingerprint = initial_records[ACCEPTED_NODEID]["fingerprint"]
    fixture = project / "tests" / "test_fixture.py"
    fixture.write_text(
        fixture.read_text().replace(
            "value = 1\n    value += 1",
            "value = 2\n    for _ in range(2):\n"
            "        value += 2\n    value = str(value)",
        )
    )
    _write_stale_baseline(
        project,
        accepted_fingerprint=accepted_fingerprint,
        stopped_fingerprint=initial_records[STOPPED_NODEID]["fingerprint"],
    )
    stale_scan = _scan(project)
    stale_inventory = stale_scan["inventory"]
    if "stale" not in stale_scan["stderr"].lower():
        _fail("stale acceptance warning was not emitted")
    stale_statuses = {
        item["test"]: item["status"] for item in stale_inventory["accepted"]["stale"]
    }
    expected_stale = {
        ACCEPTED_NODEID: "content_changed",
        STOPPED_NODEID: "signal_not_active",
        "tests/test_fixture.py::test_missing": "unknown_test",
    }
    if stale_statuses != expected_stale:
        current = _tests(stale_inventory)[ACCEPTED_NODEID]["fingerprint"]
        _fail(
            f"unexpected stale statuses: {stale_statuses}; "
            f"fingerprint={current} was={accepted_fingerprint}"
        )
    if ACCEPTED_NODEID not in stale_inventory["top_suspicious"]["test_functions"]:
        _fail("content-changed finding was incorrectly suppressed")
    return stale_statuses


def _report(evidence: AcceptanceEvidence, stale_statuses: dict[str, str]) -> None:
    initial_inventory = evidence.initial_inventory
    initial_records = _tests(initial_inventory)
    qa_result = {
        "status": "pass",
        "pycoati_version": EXPECTED_PYCOATI_VERSION,
        "pytest": initial_inventory["tool"]["ran_pytest"],
        "coverage": initial_inventory["tool"]["ran_coverage"],
        "test_count": initial_inventory["suite"]["test_count"],
        "line_coverage_pct": initial_inventory["suite"]["line_coverage_pct"],
        "subprocess_external_verification": {
            name: initial_records[f"tests/test_fixture.py::{name}"][
                "external_verification_count"
            ]
            for name in SUBPROCESS_TEST_NAMES
        },
        "shortlist_counts": {
            "raw_no_accept": len(evidence.raw_shortlist),
            "default": len(evidence.default_shortlist),
            "include_accepted": len(evidence.included_shortlist),
        },
        "accepted_count": len(evidence.default_inventory["accepted"]["findings"]),
        "stale_counts": {
            status: list(stale_statuses.values()).count(status)
            for status in ("unknown_test", "signal_not_active", "content_changed")
        },
        "active_signal_actionable": ACTIVE_NODEID in evidence.default_shortlist,
        "warnings_observed": [
            "pytest failure diagnostic",
            "stale acceptance warning",
        ],
    }
    sys.stdout.write(json.dumps(qa_result, sort_keys=True) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pycoati-qa-") as temp_dir:
        project = Path(temp_dir)
        _write_fixture(project)
        initial_scan = _scan(project, "--no-accept")
        initial_inventory = initial_scan["inventory"]
        _verify_version(initial_inventory)
        _assert_subprocess_contract(initial_scan)
        _verify_failure_propagation(project)
        evidence = _collect_acceptance_evidence(project, initial_inventory)
        _verify_acceptance_filtering(evidence)
        _verify_audit_parity(evidence)
        stale_statuses = _verify_stale_states(project, initial_inventory)
        _report(evidence, stale_statuses)


if __name__ == "__main__":
    main()
