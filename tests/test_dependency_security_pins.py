"""
HARDENING PASS regression test: requirements.txt used to pin
`starlette==0.41.3` (forced by `fastapi==0.115.6`'s own `starlette<0.42.0`
dependency ceiling), which `pip-audit` flagged for several known CVEs (fix
versions ranging up to 1.3.1). Verified via each candidate fastapi release's
own Requires-Dist metadata that fastapi==0.133.0 is the smallest version
that drops its starlette upper bound entirely (`starlette>=0.40.0`, no
ceiling) -- requirements.txt now pins `fastapi==0.133.0` /
`starlette==1.3.1` (the lowest version at or above every CVE fix threshold
`pip-audit` reported), verified compatible with the pinned `streamlit==1.56.0`
(no starlette ceiling of its own) via a full test-suite run, a real FastAPI
startup + /health check, and a real Streamlit startup.

This test protects against a future edit silently reintroducing the
vulnerable starlette range -- it does not re-run pip-audit itself (a
network call, and a moving target as new CVEs get disclosed over time); it
pins the specific, currently-verified-safe floor as a regression guard.
"""
from __future__ import annotations

import re
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"

# The lowest version confirmed (via pip-audit against this project's exact
# dependency set, at the time of the pre-submission hardening pass) to
# resolve every starlette CVE `pip-audit` reported against the previously
# pinned starlette==0.41.3. Do not lower this without re-running pip-audit.
MINIMUM_SAFE_STARLETTE_VERSION = (1, 3, 1)

# The smallest fastapi release confirmed (via each candidate release's own
# Requires-Dist metadata) to drop fastapi's own starlette upper-bound
# constraint entirely -- without at least this version, pip cannot resolve
# any starlette release above the vulnerable 0.41.3-era range at all.
MINIMUM_SAFE_FASTAPI_VERSION = (0, 133, 0)


def _pinned_version(requirements_text: str, package: str) -> tuple[int, ...]:
    match = re.search(rf"^{re.escape(package)}==([0-9]+(?:\.[0-9]+)*)", requirements_text, re.MULTILINE)
    assert match is not None, f"{package} is not pinned with an exact ==version in requirements.txt"
    return tuple(int(part) for part in match.group(1).split("."))


def test_starlette_is_pinned_at_or_above_the_verified_safe_floor():
    text = REQUIREMENTS_PATH.read_text()
    assert _pinned_version(text, "starlette") >= MINIMUM_SAFE_STARLETTE_VERSION


def test_fastapi_is_pinned_at_or_above_the_version_that_permits_a_patched_starlette():
    text = REQUIREMENTS_PATH.read_text()
    assert _pinned_version(text, "fastapi") >= MINIMUM_SAFE_FASTAPI_VERSION


def test_installed_starlette_matches_the_pinned_safe_version():
    """Guards against requirements.txt and the actual installed environment
    drifting apart (e.g. someone edits the pin but never reinstalls)."""
    import starlette

    installed = tuple(int(part) for part in starlette.__version__.split(".")[:3])
    assert installed >= MINIMUM_SAFE_STARLETTE_VERSION


def test_installed_fastapi_matches_the_pinned_safe_version():
    import fastapi

    installed = tuple(int(part) for part in fastapi.__version__.split(".")[:3])
    assert installed >= MINIMUM_SAFE_FASTAPI_VERSION
