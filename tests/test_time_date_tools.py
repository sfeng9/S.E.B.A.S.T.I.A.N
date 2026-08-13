from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from voice_assistant.tools.time_date import (
    TimezoneError,
    get_current_date,
    get_current_local_time,
    get_day_of_week,
)


class TimeDateToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(
            2026,
            8,
            12,
            6,
            27,
            tzinfo=timezone(timedelta(hours=-4), "EDT"),
        )

    def test_current_local_time_is_structured(self) -> None:
        result = get_current_local_time(self.now)
        self.assertEqual(result.display_time, "6:27 AM")
        self.assertEqual(result.utc_offset, "-0400")
        self.assertIn("2026-08-12T06:27:00", result.iso_datetime)

    def test_current_date_is_structured(self) -> None:
        result = get_current_date(self.now)
        self.assertEqual(result.display_date, "August 12, 2026")
        self.assertEqual(result.iso_date, "2026-08-12")

    def test_day_of_week_is_structured(self) -> None:
        result = get_day_of_week(self.now)
        self.assertEqual(result.day_of_week, "Wednesday")
        self.assertEqual(result.iso_date, "2026-08-12")

    def test_converts_same_instant_to_tokyo_timezone(self) -> None:
        result = get_current_local_time(
            self.now,
            timezone_name="Asia/Tokyo",
            requested_location="Tokyo",
            resolved_location="Tokyo, Japan",
            is_home=False,
        )

        self.assertEqual(result.display_time, "7:27 PM")
        self.assertEqual(result.timezone, "Asia/Tokyo")
        self.assertEqual(result.utc_offset, "+0900")
        self.assertEqual(result.resolved_location, "Tokyo, Japan")
        self.assertFalse(result.is_home)

    def test_invalid_timezone_is_rejected(self) -> None:
        with self.assertRaises(TimezoneError):
            get_current_local_time(self.now, timezone_name="Invalid/Timezone")


if __name__ == "__main__":
    unittest.main()
