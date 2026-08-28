"""
BUG-3 regression tests (pre-submission audit): the README documented several
training/evaluation commands as direct-file invocations --
`./venv/bin/python data/generate_counterfactual_dataset.py`,
`./venv/bin/python model/train_latent_target_model.py`, etc. -- which fail
with ModuleNotFoundError, because those files import sibling project
packages with absolute, project-rooted imports (e.g.
`from data.generate_synthetic_dataset import ...`,
`from model.latent_target_preprocessing import ...`). Running a file
directly (`python path/to/file.py`) puts only that file's own directory on
sys.path[0], not the repository root -- only `python -m package.module`
(sys.path[0] = the current working directory, the repo root when invoked as
documented) resolves those imports. The README now documents all commands in
`-m` form.

These tests reproduce the exact failure surface (import-time
ModuleNotFoundError) FAST, without running each script's full `main()` (real
dataset generation / CatBoost training, which the "fresh clone verification"
step already exercises for real, separately, and which would make this
regression test far too slow to run on every `pytest tests/` invocation).
`runpy.run_module(..., run_name=<not "__main__">)` executes a module via the
exact same import machinery `-m` uses, but the module's own
`if __name__ == "__main__": main()` guard never fires, so only the
top-level import statements (the thing that was actually broken) run.

HARDENING PASS follow-up: a later re-evaluation found `model.train`,
`model.train_candidate_model`, and `model.train_ranking_model` (all three
import sibling `model.*` modules with the exact same absolute,
project-rooted style -- `from model.calibrate import ...`,
`from model.preprocessing import ...`, `from model.candidate_preprocessing
import ...` -- confirmed by reading each file) were left undocumented here,
even though the README's §8 table and top summary line both referenced them
in the same broken direct-file form this whole module exists to guard
against (now fixed in README.md alongside this test). Reproduced directly:
`./venv/bin/python model/train.py` (and the other two) raise
`ModuleNotFoundError: No module named 'model'`, identically to the three
modules already covered below.
"""
from __future__ import annotations

import runpy

import pytest

PREVIOUSLY_BROKEN_UNDER_DIRECT_EXECUTION = [
    "data.generate_counterfactual_dataset",
    "model.train_latent_target_model",
    "evaluation.evaluate_latent_target_policy",
    "model.train",
    "model.train_candidate_model",
    "model.train_ranking_model",
]

ALSO_DOCUMENTED_VIA_DASH_M = [
    "data.generate_synthetic_dataset",  # never actually broken (no sibling-package imports), but now documented via -m for consistency
]


@pytest.mark.parametrize("module_name", PREVIOUSLY_BROKEN_UNDER_DIRECT_EXECUTION + ALSO_DOCUMENTED_VIA_DASH_M)
def test_readme_documented_module_imports_cleanly_via_dash_m_style_loading(module_name):
    """Reproduces exactly what `python -m <module_name>` does at import time,
    without executing main(). A regression here means the README's `-m`
    invocation would fail the same way the old `python path/to/file.py`
    form used to."""
    runpy.run_module(module_name, run_name="not_main_regression_guard")


def test_readme_no_longer_documents_the_broken_direct_file_invocation_form():
    """Static guard: the specific broken commands must not reappear."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    broken_forms = [
        "python data/generate_counterfactual_dataset.py",
        "python model/train_latent_target_model.py",
        "python evaluation/evaluate_latent_target_policy.py",
        "python model/train.py",
        "python model/train_candidate_model.py",
        "python model/train_ranking_model.py",
    ]
    for form in broken_forms:
        assert form not in readme, f"README still documents the broken direct-file invocation: {form!r}"


def test_readme_documents_the_working_dash_m_form_for_the_reproduction_pipeline():
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    for module_name in PREVIOUSLY_BROKEN_UNDER_DIRECT_EXECUTION:
        dash_m_form = f"-m {module_name}"
        assert dash_m_form in readme, f"README does not document the working `-m` form for {module_name}"
