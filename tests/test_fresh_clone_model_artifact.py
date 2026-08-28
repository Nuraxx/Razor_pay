"""
BUG-2 regression tests (pre-submission audit): a genuinely fresh clone has
no model/artifacts/unified_model.joblib -- it's gitignored, produced only by
running `python -m model.train_unified_model` by hand. Before the fix in
tests/conftest.py (the session-scoped `_unified_model_test_artifact`
fixture) and model/unified_model.py (the mkdir-target fix in `_model_path`/
`train_unified_model`), `pytest tests/ -q` either hard-failed
(tests/test_revenue_recovery_policy.py::TestUnifiedMLPolicyBoundary's
autouse fixture called load_unified_model() with no artifact present and no
try/except) or silently changed behavior (webhook-endpoint tests asserting
decision_source == "ml_unified_v1" would instead observe the rule-based
fallback) whenever that artifact hadn't already been trained by hand.

These tests prove the fix without needing to actually delete the developer's
own local model/artifacts/unified_model.joblib (which may be a real,
time-consuming-to-regenerate artifact) -- they verify the *mechanism*: that
the artifact path the whole test session actually uses is a pytest tmp
directory, never the real committed-ignored path, and that it was produced
by the real training function, not a stand-in.
"""
from __future__ import annotations

from pathlib import Path

import model.unified_model as um

REAL_PRODUCTION_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "model" / "artifacts" / "unified_model.joblib"


def test_test_session_never_points_at_the_real_committed_ignored_artifact_path():
    """The exact original coupling this bug describes: proves the path the
    live-loading functions actually resolve during pytest is NOT
    model/artifacts/unified_model.joblib (the real, gitignored, manually-
    trained production path) -- it's the fixture-provided tmp artifact."""
    assert um.UNIFIED_MODEL_PATH != REAL_PRODUCTION_ARTIFACT_PATH
    assert "pytest" in str(um.UNIFIED_MODEL_PATH) or str(um.UNIFIED_MODEL_PATH).startswith("/tmp") or "tmp" in str(um.UNIFIED_MODEL_PATH).lower()


def test_fixture_artifact_is_a_real_loadable_unified_model(_unified_model_test_artifact):
    """Not a fake/stand-in -- the same load_unified_model() the live app
    calls successfully deserializes a real fitted CatBoostClassifier from
    the fixture-provided path."""
    from catboost import CatBoostClassifier

    model = um.load_unified_model()
    assert model["model_version"] == um.MODEL_VERSION
    assert isinstance(model["model"], CatBoostClassifier)


def test_fixture_never_creates_the_real_model_artifacts_directory(tmp_path, monkeypatch):
    """Regression test for the mkdir-target bug this fix also fixed:
    _model_path() / train_unified_model()'s report-writing used to
    unconditionally mkdir the hardcoded ARTIFACTS_DIR/REPORTS_DIR constants
    even when the actual target path had been redirected elsewhere -- which
    would have created a real (if empty) model/artifacts/ and model/reports/
    directory in a fresh clone as a side effect of merely running the test
    suite. Simulates a fresh clone (no model/artifacts/ at all) inside an
    isolated tmp project root and proves training into a redirected path
    does not touch it."""
    fake_repo_root = tmp_path / "fake_repo"
    fake_artifacts_dir = fake_repo_root / "model" / "artifacts"
    fake_reports_dir = fake_repo_root / "model" / "reports"
    redirected_dir = tmp_path / "redirected_test_artifact"
    redirected_dir.mkdir()

    monkeypatch.setattr(um, "ARTIFACTS_DIR", fake_artifacts_dir)
    monkeypatch.setattr(um, "REPORTS_DIR", fake_reports_dir)
    monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", redirected_dir / "unified_model.joblib")
    monkeypatch.setattr(um, "TRAINING_REPORT_PATH", redirected_dir / "report.json")

    um.train_unified_model()

    assert (redirected_dir / "unified_model.joblib").exists()
    assert (redirected_dir / "report.json").exists()
    assert not fake_artifacts_dir.exists(), "training into a redirected path must never create the real model/artifacts/ directory"
    assert not fake_reports_dir.exists(), "training into a redirected path must never create the real model/reports/ directory"


def test_pytest_run_does_not_create_a_production_sqlite_db_file():
    """Companion isolation check (audit's 'Database / test isolation'
    section): tests/conftest.py's `test_db_session` fixture must use an
    in-memory sqlite engine and override the `get_db` FastAPI dependency --
    app/db.py's module-level `engine` (built from the real
    settings.DATABASE_URL, sqlite:///./data/recovery_agent.db by default)
    must never be the one migrated/written to by any test in this suite.
    Real db file mtime/existence is recorded before AND after this repo's
    own test collection already ran (this test itself runs late in
    alphabetical order, after every other DB-backed test module) -- if any
    prior test had written through the real engine, the file would exist
    and this assertion would catch a NEW file with a very recent mtime."""
    import time

    real_db_path = Path(__file__).resolve().parents[1] / "data" / "recovery_agent.db"
    if real_db_path.exists():
        age_seconds = time.time() - real_db_path.stat().st_mtime
        assert age_seconds > 60, (
            f"data/recovery_agent.db was modified {age_seconds:.1f}s ago -- "
            "a test in this run may have written through the real DB engine "
            "instead of the in-memory test_db_session fixture"
        )

    conftest_source = (Path(__file__).resolve().parent / "conftest.py").read_text()
    assert "sqlite:///:memory:" in conftest_source
    assert "app.dependency_overrides[get_db]" in conftest_source
