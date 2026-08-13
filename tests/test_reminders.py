from __future__ import annotations

import tempfile
import unittest
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from voice_assistant.integrations.reminders import ReminderStore


class ReminderStoreTests(unittest.TestCase):
    def test_reminder_persists_claims_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reminders.sqlite3"
            store = ReminderStore(path)
            due = datetime.now(timezone.utc) - timedelta(seconds=1)
            created = store.create("test Sebastian's reminder system", due, "America/New_York")

            reopened = ReminderStore(path)
            self.assertEqual(reopened.list_pending()[0].id, created.id)
            claimed = reopened.claim_due(datetime.now(timezone.utc))
            self.assertEqual([item.id for item in claimed], [created.id])
            self.assertEqual(reopened.claim_due(datetime.now(timezone.utc)), [])

            reopened.mark_fired(created.id)
            self.assertEqual(reopened.list_pending(), [])

    def test_interrupted_firing_is_recovered_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reminders.sqlite3"
            store = ReminderStore(path)
            due = datetime.now(timezone.utc) - timedelta(seconds=1)
            created = store.create("recover me", due, "America/New_York")
            store.claim_due(datetime.now(timezone.utc))

            recovered = ReminderStore(path)
            self.assertEqual(recovered.list_pending()[0].id, created.id)

    def test_existing_database_is_migrated_for_external_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reminders.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        due_at_utc TEXT NOT NULL,
                        timezone_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        fired_at_utc TEXT
                    )
                    """
                )

            store = ReminderStore(path)
            reminder = store.upsert_external(
                "google_calendar:primary",
                "event-1",
                "You have a meeting at 3 PM",
                datetime.now(timezone.utc) + timedelta(hours=1),
                "America/New_York",
            )

            self.assertEqual(reminder.external_id, "event-1")
            self.assertEqual(len(store.list_pending()), 1)


if __name__ == "__main__":
    unittest.main()
