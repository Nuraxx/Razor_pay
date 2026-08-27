"""
Deterministic, timezone-aware contact-hours gate.

FINAL PRE-SUBMISSION CORRECTION: the README claimed the compliance gate
"can visibly refuse an action (contact-hours, attempt caps, opt-out)" but
policy/compliance.py and policy/compliance_v2.py never actually implemented
a time-of-day check -- only the max-retry-attempts and opt-out/cancellation
checks existed. This module adds the missing check; policy/compliance.py and
policy/compliance_v2.py wire it into their existing payment/communication
gates (see evaluate_compliance / _evaluate_new_domain there).

Design:
  - Checks the SCHEDULED action's own `candidate_datetime` (the time policy
    picked for the retry/outreach to happen), never the current process
    clock -- a decision made today about an action scheduled for 2am
    tomorrow must be judged against 2am tomorrow, not against right now.
  - Timezone-aware: this codebase's own established convention (see
    app/main.py's "_strip_tzinfo" note, reused across policy/compliance.py
    and recovery/orchestrator.py) is that every naive datetime flowing
    through policy/compliance already REPRESENTS UTC (tzinfo stripped after
    being computed from `datetime.now(timezone.utc)`). This gate honors that
    convention: a naive input is treated as UTC, then converted to the
    configured local timezone for the actual wall-clock check. A tz-aware
    input is also accepted and converted normally.
  - Configurable via app/config.py::settings (CONTACT_HOURS_ENABLED /
    CONTACT_HOURS_TIMEZONE / CONTACT_HOURS_START / CONTACT_HOURS_END),
    never hardcoded -- see settings for the defaults (09:00-21:00 Asia/Kolkata,
    TRAI's own commercial-communication window, since this project is
    Razorpay/India-specific and no other project-documented convention
    exists).
  - Pure and deterministic: same (candidate_datetime, config) always
    produces the same verdict; no hidden dependency on "now".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ContactHoursConfig:
    enabled: bool
    timezone_name: str
    start: dt_time
    end: dt_time


def parse_contact_hours_config(*, enabled: bool, timezone_name: str, start_str: str, end_str: str) -> ContactHoursConfig:
    """`start_str`/`end_str` are "HH:MM" 24-hour local-time strings (e.g.
    settings.CONTACT_HOURS_START). Raises ValueError on a malformed value or
    unknown IANA timezone name -- fails loudly at config-load time rather
    than silently misbehaving at decision time."""
    return ContactHoursConfig(
        enabled=enabled,
        timezone_name=timezone_name,
        start=dt_time.fromisoformat(start_str),
        end=dt_time.fromisoformat(end_str),
    )


def is_within_contact_hours(candidate_datetime: datetime, config: ContactHoursConfig) -> tuple[bool, str]:
    """Returns (within_window, reason). `candidate_datetime` is the
    SCHEDULED action's own timestamp being evaluated -- never `datetime.now()`."""
    if not config.enabled:
        return True, "contact_hours_gate_disabled"

    aware = candidate_datetime if candidate_datetime.tzinfo is not None else candidate_datetime.replace(tzinfo=timezone.utc)
    local_time = aware.astimezone(ZoneInfo(config.timezone_name)).time()

    if config.start <= config.end:
        within = config.start <= local_time < config.end
    else:
        # Window crosses local midnight (e.g. start=22:00, end=06:00).
        within = local_time >= config.start or local_time < config.end

    window_desc = f"[{config.start.isoformat(timespec='minutes')}, {config.end.isoformat(timespec='minutes')}) {config.timezone_name}"
    if within:
        return True, f"within_contact_hours: {local_time.isoformat(timespec='minutes')} {config.timezone_name} is within {window_desc}"
    return False, f"outside_contact_hours: {local_time.isoformat(timespec='minutes')} {config.timezone_name} is outside {window_desc}"


def default_contact_hours_config() -> ContactHoursConfig:
    """Reads app/config.py::settings -- imported lazily here (not at module
    top-level) so this module has zero import-time dependency on app.config,
    keeping it trivially unit-testable in isolation (brief: pure, deterministic)."""
    from app.config import settings

    return parse_contact_hours_config(
        enabled=settings.CONTACT_HOURS_ENABLED,
        timezone_name=settings.CONTACT_HOURS_TIMEZONE,
        start_str=settings.CONTACT_HOURS_START,
        end_str=settings.CONTACT_HOURS_END,
    )
