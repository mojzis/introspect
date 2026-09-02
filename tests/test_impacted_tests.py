"""Tests for the commit hook's test-impact selector (`scripts/impacted_tests.py`).

The selector shells out to `gerenuk` and `tyf`; these tests stub `_run` so the
selection logic is exercised without either binary. The behaviour that matters
is the failure direction: every inconclusive answer must raise `SelectionFailed`
so the hook widens to the full suite, never narrows to nothing.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path("/repo")


def _load_module():
    """Import the script by path — `scripts/` is not an installed package."""
    path = REPO_ROOT / "scripts" / "impacted_tests.py"
    spec = importlib.util.spec_from_file_location("impacted_tests", path)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


impacted_tests = _load_module()


class _Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _gerenuk_report(**overrides):
    report = {
        "base": "main",
        "changed_symbols": [],
        "module_level_changes": [],
        "non_python_changes": [],
        "test_files_changed": [],
        "errors": [],
    }
    report.update(overrides)
    return report


def _stub_run(monkeypatch, *, gerenuk=None, tyf=None, git_grep=""):
    """Route each tool in the chain to a canned result."""
    calls = []

    def fake_run(cmd, stdin=None):
        calls.append(cmd)
        if cmd[0] == impacted_tests.GERENUK:
            return (
                gerenuk if gerenuk is not None else _Proc(json.dumps(_gerenuk_report()))
            )
        if cmd[0] == impacted_tests.TYF:
            return tyf if tyf is not None else _Proc(json.dumps([]))
        return _Proc(git_grep)

    monkeypatch.setattr(impacted_tests, "_run", fake_run)
    return calls


def test_no_changes_selects_nothing(monkeypatch):
    _stub_run(monkeypatch)
    assert impacted_tests.select(ROOT, None) == []


def test_changed_test_file_is_selected(monkeypatch):
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(_gerenuk_report(test_files_changed=["tests/test_db.py"]))
        ),
    )
    assert impacted_tests.select(ROOT, None) == ["tests/test_db.py"]


def test_changed_symbol_maps_through_tyf(monkeypatch):
    tyf_payload = [
        {
            "symbol": "materialize_views",
            "test_references": [
                {"file": "/repo/tests/test_db.py"},
                {"file": "/repo/tests/test_db.py"},
                {"file": "tests/test_projects.py"},
            ],
        }
    ]
    calls = _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(
                _gerenuk_report(
                    changed_symbols=[
                        {
                            "symbol": "introspect.db:materialize_views",
                            "kind": "function",
                        }
                    ]
                )
            )
        ),
        tyf=_Proc(json.dumps(tyf_payload)),
    )

    selected = impacted_tests.select(ROOT, None)

    # Absolute and relative reference paths collapse to one repo-relative entry.
    assert selected == ["tests/test_db.py", "tests/test_projects.py"]
    # gerenuk's "module:name" form is reduced to the bare name tyf expects.
    assert any(cmd[0] == impacted_tests.TYF for cmd in calls)


def test_single_symbol_tyf_object_response(monkeypatch):
    """tyf returns a bare object for one query and a list for several."""
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(
                _gerenuk_report(changed_symbols=[{"symbol": "introspect.db:only_one"}])
            )
        ),
        tyf=_Proc(
            json.dumps({"test_references": [{"file": "/repo/tests/test_db.py"}]})
        ),
    )
    assert impacted_tests.select(ROOT, None) == ["tests/test_db.py"]


def test_module_level_change_falls_back_to_git_grep(monkeypatch):
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(_gerenuk_report(module_level_changes=["introspect.projects"]))
        ),
        git_grep="tests/test_projects.py\n",
    )
    assert impacted_tests.select(ROOT, None) == ["tests/test_projects.py"]


@pytest.mark.parametrize(
    "broad_file", ["tests/conftest.py", "pyproject.toml", "uv.lock"]
)
def test_broad_change_runs_everything(monkeypatch, broad_file):
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(json.dumps(_gerenuk_report(non_python_changes=[broad_file]))),
    )
    with pytest.raises(impacted_tests.SelectionFailed):
        impacted_tests.select(ROOT, None)


def test_unrelated_non_python_change_does_not_widen(monkeypatch):
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(json.dumps(_gerenuk_report(non_python_changes=["README.md"]))),
    )
    assert impacted_tests.select(ROOT, None) == []


def test_gerenuk_errors_run_everything(monkeypatch):
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(json.dumps(_gerenuk_report(errors=["could not resolve base"]))),
    )
    with pytest.raises(impacted_tests.SelectionFailed):
        impacted_tests.select(ROOT, None)


def test_gerenuk_crash_runs_everything(monkeypatch):
    _stub_run(monkeypatch, gerenuk=_Proc("", returncode=2, stderr="boom"))
    with pytest.raises(impacted_tests.SelectionFailed):
        impacted_tests.select(ROOT, None)


def test_gerenuk_findings_exit_code_is_not_a_crash(monkeypatch):
    """gerenuk exits 1 when it has findings — that is a result, not a failure."""
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(_gerenuk_report(test_files_changed=["tests/test_db.py"])),
            returncode=1,
        ),
    )
    assert impacted_tests.select(ROOT, None) == ["tests/test_db.py"]


def test_tyf_crash_runs_everything(monkeypatch):
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(_gerenuk_report(changed_symbols=[{"symbol": "mod:sym"}]))
        ),
        tyf=_Proc("", returncode=1, stderr="daemon down"),
    )
    with pytest.raises(impacted_tests.SelectionFailed):
        impacted_tests.select(ROOT, None)


def test_unmapped_symbol_runs_everything(monkeypatch):
    """A changed symbol that no test reaches means the mapping is untrusted."""
    _stub_run(
        monkeypatch,
        gerenuk=_Proc(
            json.dumps(_gerenuk_report(changed_symbols=[{"symbol": "mod:sym"}]))
        ),
        tyf=_Proc(json.dumps([{"test_references": []}])),
    )
    with pytest.raises(impacted_tests.SelectionFailed):
        impacted_tests.select(ROOT, None)


def test_main_returns_run_everything_on_failure(monkeypatch):
    _stub_run(monkeypatch, gerenuk=_Proc("not json"))
    assert impacted_tests.main() == impacted_tests.RUN_EVERYTHING
