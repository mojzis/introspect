"""Black-box functional QA for the project's locked Pycoati acceptance flow."""

# This verifier reports precise expectation failures and keeps one linear,
# disposable scenario; the project-wide exception/complexity rules are noisy
# for that purpose.
# ruff: noqa: TRY003, PLR0912, PLR0915

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class QaFailure(RuntimeError):
    """A failed black-box QA expectation."""


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
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise QaFailure(
            f"Pycoati failed ({completed.returncode}): {completed.stderr.strip()}"
        )
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
    for record in result["test_functions"]:
        record.pop("accepted_signals", None)
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
            raise QaFailure(f"{name}: expected {expected}, got {actual}")
    if not inventory["tool"]["ran_pytest"] or not inventory["tool"]["ran_coverage"]:
        raise QaFailure(
            "successful subprocess fixture did not run pytest+coverage: "
            f"{scan['stderr'].strip()}"
        )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pycoati-qa-") as temp_dir:
        project = Path(temp_dir)
        _write_fixture(project)

        initial = _scan(project, "--no-accept")
        initial_inventory = initial["inventory"]
        _assert_subprocess_contract(initial)

        (project / "child.py").write_text("raise SystemExit(3)\n")
        failed_test = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_fixture.py::test_checked_run",
                "-q",
            ],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        if (
            failed_test.returncode == 0
            or "CalledProcessError" not in failed_test.stdout
        ):
            raise QaFailure("pytest failure diagnostic was not observed")
        failed_child = _scan(project, "--no-accept")
        failed_inventory = failed_child["inventory"]
        if not failed_inventory["tool"]["ran_pytest"]:
            raise QaFailure("a test failure was mistaken for a broken pytest run")
        (project / "child.py").write_text("raise SystemExit(0)\n")

        records = _tests(initial_inventory)
        accepted_nodeid = "tests/test_fixture.py::test_accept_me"
        stopped_nodeid = "tests/test_fixture.py::test_signal_stopped"
        active_nodeid = "tests/test_fixture.py::test_active_signal"
        accepted_fingerprint = records[accepted_nodeid]["fingerprint"]
        stopped_fingerprint = records[stopped_nodeid]["fingerprint"]
        (project / ".pycoati-accept.toml").write_text(
            dedent(
                f"""
                schema_version = "1"

                [[accept]]
                test = "{accepted_nodeid}"
                signals = ["zero_asserts"]
                reason = "No-op fixture retained to verify acceptance filtering."
                reviewed = "2026-09-13"
                fingerprint = "{accepted_fingerprint}"
                """
            ).lstrip()
        )

        default = _scan(project)
        raw = _scan(project, "--no-accept")
        included = _scan(project, "--include-accepted")
        default_inventory = default["inventory"]
        raw_inventory = raw["inventory"]
        included_inventory = included["inventory"]
        default_shortlist = default_inventory["top_suspicious"]["test_functions"]
        raw_shortlist = raw_inventory["top_suspicious"]["test_functions"]
        included_shortlist = included_inventory["top_suspicious"]["test_functions"]
        if accepted_nodeid in default_shortlist:
            raise QaFailure("accepted finding remained actionable")
        if (
            accepted_nodeid not in raw_shortlist
            or accepted_nodeid not in included_shortlist
        ):
            raise QaFailure("raw/include-accepted did not retain accepted finding")
        if active_nodeid not in default_shortlist:
            raise QaFailure("active mock signal was incorrectly suppressed")
        if default_inventory["accepted"]["findings"][0]["test"] != accepted_nodeid:
            raise QaFailure("accepted finding was not reported")
        if included_inventory["accepted"]["included_in_shortlist"] is not True:
            raise QaFailure("include-accepted flag was not reported")
        if raw_inventory["accepted"]["path"] is not None:
            raise QaFailure("--no-accept read the baseline")
        normalized_audits = {
            "default": _without_shortlist(default_inventory),
            "raw": _without_shortlist(raw_inventory),
            "include-accepted": _without_shortlist(included_inventory),
        }
        raw_audit = normalized_audits["raw"]
        for label, audit in normalized_audits.items():
            if audit != raw_audit:
                differing = [key for key in raw_audit if raw_audit[key] != audit[key]]
                raise QaFailure(f"raw/{label} audit records diverged: {differing}")
        for inventory in (default_inventory, raw_inventory, included_inventory):
            if (
                not inventory["tool"]["ran_pytest"]
                or not inventory["tool"]["ran_coverage"]
            ):
                raise QaFailure("acceptance scan did not run pytest+coverage")

        (project / "tests" / "test_fixture.py").write_text(
            (project / "tests" / "test_fixture.py")
            .read_text()
            .replace(
                "value = 1\n    value += 1",
                "value = 2\n    for _ in range(2):\n"
                "        value += 2\n    value = str(value)",
            )
        )
        (project / ".pycoati-accept.toml").write_text(
            dedent(
                f"""
                schema_version = "1"

                [[accept]]
                test = "{accepted_nodeid}"
                signals = ["zero_asserts"]
                reason = "No-op fixture retained to verify acceptance filtering."
                reviewed = "2026-09-13"
                fingerprint = "{accepted_fingerprint}"

                [[accept]]
                test = "{stopped_nodeid}"
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
        stale = _scan(project)
        stale_inventory = stale["inventory"]
        if "stale" not in stale["stderr"].lower():
            raise QaFailure("stale acceptance warning was not emitted")
        stale_statuses = {
            item["test"]: item["status"]
            for item in stale_inventory["accepted"]["stale"]
        }
        expected_stale = {
            accepted_nodeid: "content_changed",
            stopped_nodeid: "signal_not_active",
            "tests/test_fixture.py::test_missing": "unknown_test",
        }
        if stale_statuses != expected_stale:
            raise QaFailure(
                f"unexpected stale statuses: {stale_statuses}; "
                "fingerprint="
                f"{_tests(stale_inventory)[accepted_nodeid]['fingerprint']} "
                f"was={accepted_fingerprint}"
            )
        if accepted_nodeid not in stale_inventory["top_suspicious"]["test_functions"]:
            raise QaFailure("content-changed finding was incorrectly suppressed")

        result = {
            "status": "pass",
            "pycoati_version": initial_inventory["tool"]["version"],
            "pytest": initial_inventory["tool"]["ran_pytest"],
            "coverage": initial_inventory["tool"]["ran_coverage"],
            "test_count": initial_inventory["suite"]["test_count"],
            "line_coverage_pct": initial_inventory["suite"]["line_coverage_pct"],
            "subprocess_external_verification": {
                name: _tests(initial_inventory)[f"tests/test_fixture.py::{name}"][
                    "external_verification_count"
                ]
                for name in [
                    "test_checked_run",
                    "test_check_call",
                    "test_check_output",
                    "test_check_returncode",
                    "test_unchecked_run",
                    "test_check_false",
                ]
            },
            "shortlist_counts": {
                "raw_no_accept": len(raw_shortlist),
                "default": len(default_shortlist),
                "include_accepted": len(included_shortlist),
            },
            "accepted_count": len(default_inventory["accepted"]["findings"]),
            "stale_counts": {
                status: list(stale_statuses.values()).count(status)
                for status in ("unknown_test", "signal_not_active", "content_changed")
            },
            "active_signal_actionable": active_nodeid in default_shortlist,
            "warnings_observed": [
                "pytest failure diagnostic",
                "stale acceptance warning",
            ],
        }
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
