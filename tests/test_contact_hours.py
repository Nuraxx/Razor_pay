"""
Final pre-submission correction: tests for policy/contact_hours.py, the
timezone-aware contact-hours gate the README claimed existed but the code
never implemented (see policy/compliance.py's communication-gate wiring).

Pure-function tests -- (candidate_datetime, config) in, (bool, reason) out.
No DB, no policy call, no dependency on the real current time.
"""
from datetime import datetime, timezone

import pytest

from policy.contact_hours import ContactHoursConfig, is_within_contact_hours, next_contact_hours_start, parse_contact_hours_config

IST_CONFIG = ContactHoursConfig(enabled=True, timezone_name="Asia/Kolkata", start=datetime(2000, 1, 1, 9, 0).time(), end=datetime(2000, 1, 1, 21, 0).time())


class TestBasicWindow:
    def test_before_allowed_window_is_blocked(self):
        # 2026-06-15 02:00 UTC = 07:30 IST -- before the 09:00 IST start.
        dt = datetime(2026, 6, 15, 2, 0, 0)
        within, reason = is_within_contact_hours(dt, IST_CONFIG)
        assert within is False
        assert "outside_contact_hours" in reason

    def test_inside_allowed_window_is_allowed(self):
        # 2026-06-15 10:00 UTC = 15:30 IST -- comfortably inside [09:00, 21:00).
        dt = datetime(2026, 6, 15, 10, 0, 0)
        within, reason = is_within_contact_hours(dt, IST_CONFIG)
        assert within is True
        assert "within_contact_hours" in reason

    def test_after_allowed_window_is_blocked(self):
        # 2026-06-15 18:00 UTC = 23:30 IST -- after the 21:00 IST end.
        dt = datetime(2026, 6, 15, 18, 0, 0)
        within, reason = is_within_contact_hours(dt, IST_CONFIG)
        assert within is False
        assert "outside_contact_hours" in reason

    def test_exact_start_boundary_is_inclusive(self):
        # 2026-06-15 03:30 UTC = 09:00 IST exactly -- start is inclusive.
        dt = datetime(2026, 6, 15, 3, 30, 0)
        within, _ = is_within_contact_hours(dt, IST_CONFIG)
        assert within is True

    def test_exact_end_boundary_is_exclusive(self):
        # 2026-06-15 15:30 UTC = 21:00 IST exactly -- end is exclusive.
        dt = datetime(2026, 6, 15, 15, 30, 0)
        within, _ = is_within_contact_hours(dt, IST_CONFIG)
        assert within is False


class TestTimezoneAwareness:
    def test_naive_datetime_is_treated_as_utc_then_converted(self):
        # 08:00 naive: a naive-hour-only comparison against [09:00, 21:00)
        # would wrongly call this "too early". Converted UTC -> IST it is
        # 13:30, well inside the window -- proves real timezone conversion
        # happens, not a naive hour-string comparison.
        dt = datetime(2026, 3, 10, 8, 0, 0)
        within, reason = is_within_contact_hours(dt, IST_CONFIG)
        assert within is True
        assert "13:30" in reason

    def test_explicit_utc_tzinfo_gives_identical_result_to_naive(self):
        naive = datetime(2026, 3, 10, 8, 0, 0)
        aware = datetime(2026, 3, 10, 8, 0, 0, tzinfo=timezone.utc)
        assert is_within_contact_hours(naive, IST_CONFIG) == is_within_contact_hours(aware, IST_CONFIG)

    def test_different_configured_timezone_shifts_the_verdict(self):
        # Same instant, two different configured timezones -> different
        # local wall-clock hour -> can legitimately disagree on the verdict.
        # 2026-03-10 08:00 UTC = 13:30 IST (within) = 03:00 America/New_York (outside).
        dt = datetime(2026, 3, 10, 8, 0, 0)
        ny_config = ContactHoursConfig(enabled=True, timezone_name="America/New_York", start=IST_CONFIG.start, end=IST_CONFIG.end)
        within_ist, _ = is_within_contact_hours(dt, IST_CONFIG)
        within_ny, _ = is_within_contact_hours(dt, ny_config)
        assert within_ist is True
        assert within_ny is False

    def test_configured_via_parse_contact_hours_config(self):
        config = parse_contact_hours_config(enabled=True, timezone_name="Asia/Kolkata", start_str="09:00", end_str="21:00")
        assert config == IST_CONFIG


class TestScheduledCandidateCrossingBoundary:
    def test_candidate_scheduled_across_a_dst_transition_still_resolves_correctly(self):
        # US DST spring-forward 2026-03-08 (America/New_York clocks jump
        # 02:00 -> 03:00). A candidate scheduled right around that instant
        # must still resolve to a real, valid local wall-clock hour rather
        # than raising or silently misbehaving -- zoneinfo handles this
        # transition automatically; this test proves the gate doesn't choke
        # on it.
        ny_config = ContactHoursConfig(enabled=True, timezone_name="America/New_York", start=IST_CONFIG.start, end=IST_CONFIG.end)
        before_transition = datetime(2026, 3, 8, 6, 30, 0, tzinfo=timezone.utc)  # 01:30 EST
        after_transition = datetime(2026, 3, 8, 7, 30, 0, tzinfo=timezone.utc)  # 03:30 EDT
        within_before, _ = is_within_contact_hours(before_transition, ny_config)
        within_after, _ = is_within_contact_hours(after_transition, ny_config)
        assert within_before is False  # 01:30 local -- outside [09:00, 21:00)
        assert within_after is False  # 03:30 local -- also outside

    def test_candidate_just_before_and_just_after_the_end_boundary(self):
        # A scheduled candidate one minute before vs. one minute after the
        # window's own end boundary -- the exact "crossing the boundary"
        # case the gate must get right, using real minute-level precision,
        # not just whole-hour examples.
        one_minute_before = datetime(2026, 6, 15, 15, 29, 0)  # 20:59 IST
        one_minute_after = datetime(2026, 6, 15, 15, 31, 0)  # 21:01 IST
        assert is_within_contact_hours(one_minute_before, IST_CONFIG)[0] is True
        assert is_within_contact_hours(one_minute_after, IST_CONFIG)[0] is False


class TestGateDisabled:
    def test_disabled_config_always_allows(self):
        disabled = ContactHoursConfig(enabled=False, timezone_name="Asia/Kolkata", start=IST_CONFIG.start, end=IST_CONFIG.end)
        dt = datetime(2026, 6, 15, 18, 0, 0)  # would be blocked if enabled
        within, reason = is_within_contact_hours(dt, disabled)
        assert within is True
        assert reason == "contact_hours_gate_disabled"


class TestOvernightWindow:
    def test_window_crossing_midnight(self):
        overnight = ContactHoursConfig(enabled=True, timezone_name="UTC", start=datetime(2000, 1, 1, 22, 0).time(), end=datetime(2000, 1, 1, 6, 0).time())
        assert is_within_contact_hours(datetime(2026, 6, 15, 23, 0, 0), overnight)[0] is True
        assert is_within_contact_hours(datetime(2026, 6, 15, 3, 0, 0), overnight)[0] is True
        assert is_within_contact_hours(datetime(2026, 6, 15, 12, 0, 0), overnight)[0] is False


class TestConfigParsingErrors:
    def test_malformed_time_string_raises(self):
        with pytest.raises(ValueError):
            parse_contact_hours_config(enabled=True, timezone_name="Asia/Kolkata", start_str="not-a-time", end_str="21:00")


# ---------------------------------------------------------------------------
# DEFER, DON'T TERMINATE (final pre-submission audit): next_contact_hours_start
# ---------------------------------------------------------------------------

class TestNextContactHoursStart:
    def test_already_within_window_returns_unchanged(self):
        dt = datetime(2026, 6, 15, 10, 0, 0)  # 15:30 IST -- within [09:00, 21:00)
        assert next_contact_hours_start(dt, IST_CONFIG) == dt

    def test_before_window_defers_to_later_today(self):
        # 02:00 UTC = 07:30 IST -- before 09:00 IST start.
        dt = datetime(2026, 6, 15, 2, 0, 0)
        deferred = next_contact_hours_start(dt, IST_CONFIG)
        assert deferred == datetime(2026, 6, 15, 3, 30, 0)  # 2026-06-15 09:00 IST = 03:30 UTC
        assert is_within_contact_hours(deferred, IST_CONFIG)[0] is True

    def test_after_window_defers_to_tomorrow(self):
        # 18:00 UTC = 23:30 IST -- after 21:00 IST end.
        dt = datetime(2026, 6, 15, 18, 0, 0)
        deferred = next_contact_hours_start(dt, IST_CONFIG)
        assert deferred == datetime(2026, 6, 16, 3, 30, 0)  # 2026-06-16 09:00 IST = 03:30 UTC
        assert is_within_contact_hours(deferred, IST_CONFIG)[0] is True

    def test_preserves_naive_utc_convention_on_the_way_out(self):
        dt = datetime(2026, 6, 15, 18, 0, 0)
        deferred = next_contact_hours_start(dt, IST_CONFIG)
        assert deferred.tzinfo is None

    def test_aware_input_gives_aware_output(self):
        dt = datetime(2026, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
        deferred = next_contact_hours_start(dt, IST_CONFIG)
        assert deferred.tzinfo is not None
        assert deferred.astimezone(timezone.utc).replace(tzinfo=None) == datetime(2026, 6, 16, 3, 30, 0)

    def test_disabled_config_returns_unchanged(self):
        disabled = ContactHoursConfig(enabled=False, timezone_name="Asia/Kolkata", start=IST_CONFIG.start, end=IST_CONFIG.end)
        dt = datetime(2026, 6, 15, 18, 0, 0)  # would be deferred if enabled
        assert next_contact_hours_start(dt, disabled) == dt

    def test_overnight_window_defers_correctly(self):
        overnight = ContactHoursConfig(enabled=True, timezone_name="UTC", start=datetime(2000, 1, 1, 22, 0).time(), end=datetime(2000, 1, 1, 6, 0).time())
        dt = datetime(2026, 6, 15, 12, 0, 0)  # midday -- outside [22:00, 06:00)
        deferred = next_contact_hours_start(dt, overnight)
        assert deferred == datetime(2026, 6, 15, 22, 0, 0)  # opens later the same day
        assert is_within_contact_hours(deferred, overnight)[0] is True
