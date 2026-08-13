from __future__ import annotations

import unittest
from datetime import datetime, timezone

from voice_assistant.tools.date_time_parser import parse_datetime, parse_period


class DateTimeParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc)

    def test_relative_reminder_time_is_timezone_aware(self) -> None:
        parsed = parse_datetime("in 2 minutes", "America/New_York", self.now)
        self.assertEqual(parsed.isoformat(), "2026-08-13T12:02:00-04:00")

    def test_named_weekday_resolves_to_future_weekday(self) -> None:
        parsed = parse_datetime("Friday at 7 PM", "America/New_York", self.now)
        self.assertEqual(parsed.isoformat(), "2026-08-14T19:00:00-04:00")

    def test_time_only_rolls_to_next_day_after_time_passed(self) -> None:
        parsed = parse_datetime("9:30 AM", "America/New_York", self.now)
        self.assertEqual(parsed.isoformat(), "2026-08-14T09:30:00-04:00")

    def test_afternoon_period_is_noon_to_five(self) -> None:
        start, end = parse_period("this afternoon", "America/New_York", self.now)
        self.assertEqual(start.isoformat(), "2026-08-13T12:00:00-04:00")
        self.assertEqual(end.isoformat(), "2026-08-13T17:00:00-04:00")


if __name__ == "__main__":
    unittest.main()
