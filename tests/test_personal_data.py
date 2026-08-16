from __future__ import annotations

import tempfile
import unittest
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from voice_assistant.assistant.personal_data_tools import (
    PERSONAL_DATA_TOOL_PERMISSIONS,
    PersonalDataToolHandler,
)
from voice_assistant.assistant.tool_permissions import ToolPermission
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.personal_data import PersonalDataStore
from voice_assistant.integrations.reminders import ReminderStore


NOW = datetime(2026, 8, 15, 16, 0, tzinfo=timezone.utc)


class PersonalDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        base = load_assistant_config()
        self.config = replace(
            base,
            home_location=replace(base.home_location, timezone="America/New_York"),
            personal_data=replace(base.personal_data, database_path=root / "sebastian.db", search_result_limit=8, spoken_item_limit=3),
            reminders=replace(base.reminders, database_path=root / "reminders.sqlite3"),
        )
        self.store = PersonalDataStore(self.config.personal_data.database_path, now_provider=lambda: NOW)
        self.reminders = ReminderStore(self.config.reminders.database_path)
        self.handler = PersonalDataToolHandler(
            self.config,
            store=self.store,
            reminder_store=self.reminders,
            now_provider=lambda: NOW,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_note_lifecycle_search_and_restart_persistence(self) -> None:
        created = self.handler.execute("create_note", {"content": "My air filter size is 16 by 20.", "tags": ["home"]})
        note_id = created["note"]["id"]
        searched = self.handler.execute("search_notes", {"query": "AC filter"})
        self.assertEqual(searched["notes"][0]["id"], note_id)
        updated = self.handler.execute("update_note", {"query": "air filter", "content": "My air filter size is 16 by 25."})
        self.assertTrue(updated["updated"])
        restarted = PersonalDataStore(self.config.personal_data.database_path)
        self.assertIn("16 by 25", restarted.notes.get(note_id).content)
        self.assertTrue(self.handler.execute("delete_note", {"note_id": note_id})["deleted"])
        self.assertIsNone(self.store.notes.get(note_id))

    def test_tasks_due_dates_status_and_explicit_linked_reminder(self) -> None:
        task = self.handler.execute("create_task", {"title": "Test assignment", "due_datetime": "tomorrow at 5 PM"})
        task_id = task["task"]["id"]
        self.assertEqual(task["task"]["due_at_utc"], "2026-08-16T21:00:00+00:00")
        self.assertIsNone(task["task"]["reminder_id"])
        self.assertEqual(self.reminders.list_pending(), [])
        tomorrow = self.handler.execute("list_tasks", {"status": "pending", "due_period": "tomorrow"})
        self.assertEqual(tomorrow["count"], 1)
        self.assertEqual(self.handler.execute("complete_task", {"task_id": task_id})["task"]["status"], "completed")
        self.assertEqual(self.handler.execute("reopen_task", {"task_id": task_id})["task"]["status"], "pending")
        deleted = self.handler.execute("delete_task", {"task_id": task_id})
        self.assertTrue(deleted["deleted"])
        self.assertIsNone(self.store.tasks.get(task_id))
        linked = self.handler.execute(
            "create_task",
            {"title": "Call professor", "due_datetime": "tomorrow at 6 PM", "reminder_datetime": "tomorrow at 5:30 PM"},
        )
        self.assertIsNotNone(linked["task"]["reminder_id"])
        self.assertEqual(len(self.reminders.list_pending()), 1)

    def test_list_items_duplicates_checked_readding_and_removal(self) -> None:
        list_id = self.handler.execute("create_list", {"name": "Test grocery list"})["list"]["id"]
        added = self.handler.execute("add_list_items", {"list_id": list_id, "items": ["milk", "eggs", "chicken", "rice"]})
        self.assertEqual(len(added["added"]), 4)
        duplicate = self.handler.execute("add_list_items", {"list_id": list_id, "items": ["Milk"]})
        self.assertEqual(len(duplicate["duplicates"]), 1)
        crossed = self.handler.execute("complete_list_item", {"list_id": list_id, "item_text": "milk"})
        self.assertTrue(crossed["item"]["completed"])
        reopened = self.handler.execute("add_list_items", {"list_id": list_id, "items": ["milk"]})
        self.assertEqual(len(reopened["reopened"]), 1)
        self.assertTrue(self.handler.execute("remove_list_item", {"list_id": list_id, "item_text": "eggs"})["removed"])
        self.assertEqual(len(self.store.lists.get_items(list_id)), 3)

    def test_alias_follow_up_and_session_reset(self) -> None:
        created = self.handler.execute("create_list", {"name": "Grocery list", "aliases": ["shopping list"]})
        list_id = created["list"]["id"]
        self.handler.execute("add_list_items", {"list_name": "shopping list", "items": ["milk", "eggs"]})
        self.handler.execute("get_list", {"list_name": "grocery list"})
        self.assertTrue(self.handler.execute("remove_list_item", {"item_text": "eggs"})["removed"])
        renamed = self.handler.execute("rename_list", {"list_id": list_id, "new_name": "Costco list"})
        self.assertEqual(renamed["list"]["name"], "Costco list")
        self.assertEqual(self.store.lists.find_by_name("grocery list", exact=True)[0].id, list_id)
        self.handler.reset_session_context()
        missing = self.handler.execute("remove_list_item", {"item_text": "milk"})
        self.assertEqual(missing["error"], "list_reference_missing")
        self.assertEqual(len(self.store.lists.get_items(list_id)), 1)

    def test_ambiguous_task_refuses_mutation(self) -> None:
        first = self.handler.execute("create_task", {"title": "Finish CSC homework"})["task"]["id"]
        second = self.handler.execute("create_task", {"title": "Finish ECE homework"})["task"]["id"]
        result = self.handler.execute("delete_task", {"query": "homework"})
        self.assertEqual(result["error"], "ambiguous_task")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertIsNotNone(self.store.tasks.get(first))
        self.assertIsNotNone(self.store.tasks.get(second))

    def test_bulk_delete_requires_separate_confirmation(self) -> None:
        list_id = self.handler.execute("create_list", {"name": "Temporary list"})["list"]["id"]
        self.handler.execute("add_list_items", {"list_id": list_id, "items": ["one", "two"]})
        self.assertTrue(self.handler.execute("clear_list", {"list_id": list_id})["confirmation_required"])
        same_turn = self.handler.execute("confirm_personal_data_action", {"confirm": True})
        self.assertEqual(same_turn["error"], "confirmation_must_be_new_turn")
        self.handler.begin_turn()
        self.assertTrue(self.handler.execute("confirm_personal_data_action", {"confirm": True})["cleared"])
        self.assertEqual(self.store.lists.get_items(list_id), [])

    def test_undo_and_database_failure(self) -> None:
        task_id = self.handler.execute("create_task", {"title": "Test homework"})["task"]["id"]
        self.handler.execute("complete_task", {"task_id": task_id})
        self.assertTrue(self.handler.execute("undo_personal_data_action", {})["undone"])
        self.assertEqual(self.store.tasks.get(task_id).status, "pending")
        bad_path = Path(self.temp_dir.name) / "not-a-database"
        bad_path.mkdir()
        bad_handler = PersonalDataToolHandler(
            self.config,
            store=PersonalDataStore(bad_path),
            reminder_store=self.reminders,
            now_provider=lambda: NOW,
        )
        failed = bad_handler.execute("create_note", {"content": "Fail safely."})
        self.assertEqual(failed["error"], "personal_data_unavailable")

    def test_reminder_partial_failure_and_bulk_delete_retry(self) -> None:
        class FailingReminderStore:
            def create(self, *args, **kwargs):
                raise sqlite3.OperationalError("simulated reminder failure")

        handler = PersonalDataToolHandler(
            self.config,
            store=self.store,
            reminder_store=FailingReminderStore(),
            now_provider=lambda: NOW,
        )
        partial = handler.execute(
            "create_task",
            {"title": "Partial test", "reminder_datetime": "tomorrow at 5 PM"},
        )
        self.assertTrue(partial["ok"])
        self.assertTrue(partial["partial_success"])
        self.assertIsNotNone(self.store.tasks.get(partial["task"]["id"]))

        list_id = handler.execute("create_list", {"name": "Retry list"})["list"]["id"]
        handler.execute("add_list_items", {"list_id": list_id, "items": ["one"]})
        handler.execute("clear_list", {"list_id": list_id})
        handler.begin_turn()
        with patch.object(
            self.store.lists,
            "clear",
            side_effect=sqlite3.OperationalError("simulated lock"),
        ):
            failed_clear = handler.execute(
                "confirm_personal_data_action", {"confirm": True}
            )
        self.assertEqual(failed_clear["error"], "personal_data_unavailable")
        self.assertTrue(handler.has_pending_confirmation)
        handler.begin_turn()
        retried = handler.execute("confirm_personal_data_action", {"confirm": True})
        self.assertTrue(retried["cleared"])

    def test_permissions_and_router_selection(self) -> None:
        self.assertEqual(PERSONAL_DATA_TOOL_PERMISSIONS["search_notes"], ToolPermission.READ_ONLY)
        self.assertEqual(PERSONAL_DATA_TOOL_PERMISSIONS["create_task"], ToolPermission.SAFE_ACTION)
        self.assertEqual(PERSONAL_DATA_TOOL_PERMISSIONS["clear_list"], ToolPermission.CONFIRM_REQUIRED)
        router = AssistantToolRouter(self.config, personal_data=self.handler, now_provider=lambda: NOW)
        names = {schema["function"]["name"] for schema in router.schemas_for("Add milk to my grocery list", ())}
        self.assertIn("add_list_items", names)
        self.assertNotIn("web_search", names)
        note_names = {
            schema["function"]["name"]
            for schema in router.schemas_for(
                "What did I tell you about my test air filter?", ()
            )
        }
        self.assertEqual(
            note_names,
            {
                "get_current_local_time",
                "get_current_date",
                "get_day_of_week",
                "get_current_weather",
                "search_notes",
            },
        )


if __name__ == "__main__":
    unittest.main()
