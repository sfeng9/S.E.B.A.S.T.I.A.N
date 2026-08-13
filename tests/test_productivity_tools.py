from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voice_assistant.assistant.productivity_tools import ProductivityToolHandler
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import LocationConfig, load_assistant_config
from voice_assistant.integrations.google_auth import GoogleIntegrationError
from voice_assistant.integrations.reminders import ReminderStore


NOW = datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc)
EVENT = {
    "id": "event-1",
    "title": "Project meeting",
    "start": "2026-08-14T15:00:00-04:00",
    "end": "2026-08-14T16:00:00-04:00",
    "timezone": "America/New_York",
    "location": "",
}


class FakeGmail:
    def __init__(self) -> None:
        self.details: list[str] = []

    def search_emails(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        return self.get_important_emails(max_results)

    def get_recent_emails(self, max_results: int | None = None) -> list[dict[str, Any]]:
        return self.get_important_emails(max_results)

    def get_unread_emails(self, max_results: int | None = None) -> list[dict[str, Any]]:
        return self.get_important_emails(max_results)

    def get_important_emails(self, max_results: int | None = None) -> list[dict[str, Any]]:
        return [
            {
                "id": "mail-1",
                "thread_id": "thread-1",
                "sender": "Professor Smith <smith@example.edu>",
                "subject": "Project deadline",
                "received_at": "2026-08-13T14:00:00+00:00",
                "snippet": "The deadline is Friday.",
                "is_unread": True,
                "is_important": True,
            }
        ]

    def get_email_details(self, message_id: str) -> dict[str, Any]:
        self.details.append(message_id)
        return {
            "id": message_id,
            "sender": "Professor Smith <smith@example.edu>",
            "subject": "Project deadline",
            "body": "Submit by Friday.",
        }


class FakeCalendar:
    def __init__(self) -> None:
        self.created: list[tuple[Any, ...]] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.delete_error: Exception | None = None
        self.events = [dict(EVENT)]

    def get_events_between(self, start: datetime, end: datetime, max_results: int | None = None, query: str | None = None) -> list[dict[str, Any]]:
        return [dict(item) for item in self.events]

    def get_next_event(self, now: datetime) -> dict[str, Any] | None:
        return dict(self.events[0]) if self.events else None

    def find_events(self, query: str, start: datetime, end: datetime, max_results: int | None = None) -> list[dict[str, Any]]:
        return [dict(item) for item in self.events if query.casefold() in item["title"].casefold()]

    def create_event(self, title: str, start: datetime, end: datetime, timezone_name: str, location: str | None = None, description: str | None = None) -> dict[str, Any]:
        self.created.append((title, start, end, timezone_name, location, description))
        return {**EVENT, "id": "created-1", "title": title, "start": start.isoformat(), "end": end.isoformat()}

    def update_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        self.updated.append((event_id, changes))
        return {**EVENT, "id": event_id, "start": changes.get("start", {}).get("dateTime", EVENT["start"]), "end": changes.get("end", {}).get("dateTime", EVENT["end"])}

    def delete_event(self, event_id: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(event_id)


class ProductivityToolTests(unittest.TestCase):
    def setUp(self) -> None:
        base = load_assistant_config()
        self.config = replace(
            base,
            home_location=LocationConfig(
                name="Cary, NC",
                latitude=35.79,
                longitude=-78.78,
                timezone="America/New_York",
            ),
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gmail = FakeGmail()
        self.calendar = FakeCalendar()
        self.handler = ProductivityToolHandler(
            self.config,
            gmail=self.gmail,
            calendar=self.calendar,
            reminder_store=ReminderStore(Path(self.temp_dir.name) / "reminders.sqlite3"),
            now_provider=lambda: NOW,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_email_followup_fetches_only_selected_message(self) -> None:
        listed = self.handler.execute("get_important_emails", {"max_results": 3})
        detailed = self.handler.execute("get_email_details", {"ordinal": 1})
        self.assertEqual(listed["count"], 1)
        self.assertEqual(detailed["email"]["content"], "Submit by Friday")
        self.assertEqual(detailed["email"]["sender_name"], "Professor Smith")
        self.assertEqual(self.gmail.details, ["mail-1"])

        self.handler.reset_session_context()
        missing = self.handler.execute("get_email_details", {"ordinal": 1})
        self.assertEqual(missing["error"], "email_reference_missing")

    def test_email_list_spoken_format_removes_metadata_and_symbols(self) -> None:
        self.gmail.get_important_emails = lambda max_results=None: [  # type: ignore[method-assign]
            {
                "id": "mail-private-id",
                "thread_id": "thread-private-id",
                "sender": 'Professor Smith <smith@example.edu>',
                "subject": "Private subject",
                "received_at": "2026-08-13T14:00:00+00:00",
                "snippet": "*** Project update: read https://example.com now.",
                "is_unread": True,
                "is_important": True,
            },
            {
                "id": "mail-2",
                "thread_id": "thread-2",
                "sender": '★★ 王小明 <person@example.com>',
                "subject": "你好",
                "received_at": "2026-08-13T15:00:00+00:00",
                "snippet": "你好 ✅ English status is ready.",
                "is_unread": True,
                "is_important": True,
            },
        ]

        result = self.handler.execute("get_important_emails", {})
        serialized = str(result)
        spoken = self.handler.spoken_override_for(("get_important_emails",))

        self.assertNotIn("mail-private-id", serialized)
        self.assertNotIn("smith@example.edu", serialized)
        self.assertNotIn("Private subject", serialized)
        self.assertNotIn("2026-08-13", serialized)
        self.assertNotIn("*", spoken)
        self.assertNotIn("王", spoken)
        self.assertNotIn("http", spoken)
        self.assertEqual(
            spoken,
            "The sender is Professor Smith. The snippet is: Project update read now. "
            "The sender is an unnamed sender. The snippet is: English status is ready.",
        )

    def test_explicit_event_creation_uses_default_duration(self) -> None:
        result = self.handler.execute(
            "create_event",
            {"title": "Test meeting", "start_datetime": "tomorrow at 3 PM"},
        )
        self.assertTrue(result["created"])
        _, start, end, zone, _, _ = self.calendar.created[0]
        self.assertEqual(start.isoformat(), "2026-08-14T15:00:00-04:00")
        self.assertEqual((end - start).total_seconds(), 3600)
        self.assertEqual(zone, "America/New_York")

    def test_update_is_not_applied_until_confirmed(self) -> None:
        proposed = self.handler.execute(
            "update_event",
            {
                "query": "Project meeting",
                "period": "tomorrow",
                "new_start_datetime": "tomorrow at 4 PM",
            },
        )
        self.assertTrue(proposed["confirmation_required"])
        self.assertEqual(self.calendar.updated, [])

        same_turn = self.handler.execute("confirm_calendar_action", {"confirm": True})
        self.assertEqual(same_turn["error"], "confirmation_must_be_new_turn")
        self.handler.begin_turn()
        confirmed = self.handler.execute("confirm_calendar_action", {"confirm": True})
        self.assertTrue(confirmed["updated"])
        self.assertEqual(self.calendar.updated[0][0], "event-1")
        self.assertEqual(
            self.calendar.updated[0][1]["start"]["dateTime"],
            "2026-08-14T16:00:00-04:00",
        )

    def test_delete_is_rejected_without_side_effect(self) -> None:
        proposed = self.handler.execute(
            "delete_event", {"query": "Project meeting", "period": "tomorrow"}
        )
        self.assertTrue(proposed["confirmation_required"])
        self.handler.begin_turn()
        rejected = self.handler.execute("confirm_calendar_action", {"confirm": False})
        self.assertTrue(rejected["cancelled"])
        self.assertEqual(self.calendar.deleted, [])

    def test_delete_speech_only_claims_success_after_confirmation(self) -> None:
        router = AssistantToolRouter(self.config, productivity=self.handler)
        deletion_schemas = router.schemas_for(
            "Delete Project meeting tomorrow.",
            (),
        )
        deletion_names = {item["function"]["name"] for item in deletion_schemas}
        self.assertIn("delete_event", deletion_names)
        self.assertNotIn("confirm_calendar_action", deletion_names)

        proposed = self.handler.execute(
            "delete_event", {"query": "Project meeting", "period": "tomorrow"}
        )
        self.assertTrue(proposed["confirmation_required"])
        self.assertEqual(self.calendar.deleted, [])
        self.assertEqual(
            self.handler.spoken_override_for(("delete_event",)),
            "Do you want me to delete Project meeting from your calendar?",
        )
        requirement = self.handler.tool_requirement("Yes, please.", [])
        self.assertEqual(requirement["tools"], ("confirm_calendar_action",))
        verbose_requirement = self.handler.tool_requirement(
            "Confirm delete Test Meeting at 3 PM.", []
        )
        self.assertEqual(
            verbose_requirement["tools"],
            ("confirm_calendar_action",),
        )
        self.assertIsNone(self.handler.tool_requirement("Cancel my reminder", []))

        self.handler.begin_turn()
        confirmation_schemas = router.schemas_for("Yes, delete it.", ())
        confirmation_names = {
            item["function"]["name"] for item in confirmation_schemas
        }
        self.assertIn("confirm_calendar_action", confirmation_names)
        self.assertNotIn("delete_event", confirmation_names)

        confirmed = self.handler.execute("confirm_calendar_action", {"confirm": True})
        self.assertTrue(confirmed["deleted"])
        self.assertEqual(self.calendar.deleted, ["event-1"])
        self.assertEqual(
            self.handler.spoken_override_for(("confirm_calendar_action",)),
            "I've deleted Project meeting from your calendar.",
        )

    def test_delete_resolves_spoken_time_when_google_text_search_does_not(self) -> None:
        proposed = self.handler.execute(
            "delete_event", {"query": "3 PM", "period": "tomorrow"}
        )

        self.assertTrue(proposed["confirmation_required"])
        self.assertEqual(proposed["event"]["id"], "event-1")
        self.assertEqual(self.calendar.deleted, [])

    def test_spoken_time_fallback_refuses_multiple_matches(self) -> None:
        self.calendar.events.append(
            {
                **EVENT,
                "id": "event-2",
                "title": "Another meeting",
            }
        )

        result = self.handler.execute(
            "delete_event", {"query": "3 PM", "period": "tomorrow"}
        )

        self.assertEqual(result["error"], "ambiguous_event")
        self.assertEqual(self.calendar.deleted, [])

    def test_failed_delete_confirmation_remains_pending_for_retry(self) -> None:
        self.handler.execute(
            "delete_event", {"query": "Project meeting", "period": "tomorrow"}
        )
        self.handler.begin_turn()
        self.calendar.delete_error = GoogleIntegrationError("temporary failure")
        failed = self.handler.execute("confirm_calendar_action", {"confirm": True})
        self.assertFalse(failed["ok"])
        self.assertEqual(self.calendar.deleted, [])
        self.assertEqual(
            self.handler.spoken_override_for(("confirm_calendar_action",)),
            "I couldn't change your Google Calendar right now.",
        )

        self.calendar.delete_error = None
        self.handler.begin_turn()
        retried = self.handler.execute("confirm_calendar_action", {"confirm": True})
        self.assertTrue(retried["deleted"])
        self.assertEqual(self.calendar.deleted, ["event-1"])

    def test_time_only_update_keeps_selected_event_date(self) -> None:
        proposed = self.handler.execute(
            "update_event",
            {
                "query": "Project meeting",
                "period": "tomorrow",
                "new_start_datetime": "4 PM",
            },
        )
        self.assertEqual(
            proposed["changes"]["start"]["dateTime"],
            "2026-08-14T16:00:00-04:00",
        )

    def test_reminder_is_persisted_outside_session(self) -> None:
        result = self.handler.execute(
            "create_reminder",
            {"text": "test Sebastian", "due_datetime": "in 2 minutes"},
        )
        self.assertTrue(result["created"])
        self.handler.reset_session_context()
        pending = self.handler.execute("get_pending_reminders", {})
        self.assertEqual(pending["count"], 1)
        self.assertEqual(pending["reminders"][0]["text"], "test Sebastian")


if __name__ == "__main__":
    unittest.main()
