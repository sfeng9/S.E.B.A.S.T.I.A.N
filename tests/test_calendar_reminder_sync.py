from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voice_assistant.config import LocationConfig, load_assistant_config
from voice_assistant.integrations.calendar_reminder_sync import (
    CalendarReminderSynchronizer,
)
from voice_assistant.integrations.reminders import ReminderStore


NOW = datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc)


def timed_event(
    event_id: str = "event-1",
    start: str = "2026-08-13T15:00:00-04:00",
    title: str = "Project meeting",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "title": title,
        "start": start,
        "end": "2026-08-13T16:00:00-04:00",
        "timezone": "America/New_York",
        "all_day": False,
        "status": "confirmed",
    }


class FakeCalendar:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.error: Exception | None = None

    def get_events_between(
        self,
        start: datetime,
        end: datetime,
        max_results: int | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.error is not None:
            raise self.error
        return [dict(item) for item in self.events]


class CalendarReminderSynchronizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = load_assistant_config()
        self.config = replace(
            base,
            home_location=LocationConfig(
                name="Cary, NC",
                latitude=35.79,
                longitude=-78.78,
                timezone="America/New_York",
            ),
            reminders=replace(
                base.reminders,
                database_path=Path(self.temp_dir.name) / "reminders.sqlite3",
                calendar_reminder_minutes_before=30,
            ),
        )
        self.store = ReminderStore(self.config.reminders.database_path)
        self.calendar = FakeCalendar(
            [
                timed_event(),
                {
                    "id": "all-day",
                    "title": "Vacation",
                    "start": "2026-08-14",
                    "end": "2026-08-15",
                    "all_day": True,
                    "status": "confirmed",
                },
            ]
        )
        self.sync = CalendarReminderSynchronizer(
            self.config,
            self.store,
            calendar=self.calendar,
            now_provider=lambda: NOW,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sync_creates_one_reminder_and_skips_all_day_events(self) -> None:
        first = self.sync.sync_once()
        second = self.sync.sync_once()
        pending = self.store.list_pending()

        self.assertEqual(first.reminders_synchronized, 1)
        self.assertEqual(first.events_skipped, 1)
        self.assertEqual(second.reminders_synchronized, 1)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].external_id, "event-1")
        self.assertEqual(pending[0].due_at_utc.isoformat(), "2026-08-13T18:30:00+00:00")
        self.assertEqual(pending[0].text, "You have Project meeting at 3 PM")

    def test_rescheduled_fired_event_becomes_pending_again(self) -> None:
        self.sync.sync_once()
        reminder = self.store.list_pending()[0]
        self.store.mark_fired(reminder.id)
        self.calendar.events = [
            timed_event(start="2026-08-13T16:00:00-04:00")
        ]

        self.sync.sync_once()
        pending = self.store.list_pending()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].id, reminder.id)
        self.assertEqual(pending[0].due_at_utc.isoformat(), "2026-08-13T19:30:00+00:00")

    def test_missing_event_is_cancelled_and_restored_if_it_reappears(self) -> None:
        self.calendar.events = [timed_event()]
        self.sync.sync_once()
        self.calendar.events = []
        result = self.sync.sync_once()
        self.assertEqual(result.reminders_cancelled, 1)
        self.assertEqual(self.store.list_pending(), [])

        self.calendar.events = [timed_event()]
        self.sync.sync_once()
        self.assertEqual(len(self.store.list_pending()), 1)

    def test_calendar_failure_does_not_cancel_existing_reminders(self) -> None:
        self.calendar.events = [timed_event()]
        self.sync.sync_once()
        self.calendar.error = RuntimeError("network unavailable")

        with self.assertRaises(RuntimeError):
            self.sync.sync_once()

        self.assertEqual(len(self.store.list_pending()), 1)

    def test_spoken_title_is_tts_safe(self) -> None:
        self.calendar.events = [
            timed_event(title="*** \u738b\u5c0f\u660e <person@example.com> Study \u2705")
        ]
        self.sync.sync_once()
        text = self.store.list_pending()[0].text
        self.assertEqual(text, "You have Study at 3 PM")
        self.assertTrue(text.isascii())


if __name__ == "__main__":
    unittest.main()
