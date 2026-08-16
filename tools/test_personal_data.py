from __future__ import annotations

import tempfile
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_assistant.assistant.personal_data_tools import PersonalDataToolHandler
from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.personal_data import PersonalDataStore
from voice_assistant.integrations.reminders import ReminderStore


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        base = load_assistant_config()
        config = replace(
            base,
            home_location=replace(
                base.home_location,
                timezone=base.home_location.timezone or "America/New_York",
            ),
            personal_data=replace(
                base.personal_data,
                database_path=root / "sebastian.db",
            ),
            reminders=replace(
                base.reminders,
                database_path=root / "reminders.sqlite3",
            ),
        )
        now = lambda: datetime.now(timezone.utc)
        handler = PersonalDataToolHandler(
            config,
            store=PersonalDataStore(config.personal_data.database_path, now),
            reminder_store=ReminderStore(config.reminders.database_path),
            now_provider=now,
        )

        note = handler.execute(
            "create_note", {"content": "My test air filter is 16 by 20."}
        )
        require(bool(note and note["ok"]), "Note creation failed.")
        require(
            handler.execute("search_notes", {"query": "air filter"})["count"] == 1,
            "Note search failed.",
        )

        task = handler.execute(
            "create_task",
            {"title": "Test assignment", "due_datetime": "tomorrow at 5 PM"},
        )
        require(bool(task and task["ok"] and task["task"]["due_at_utc"]), "Task creation failed.")
        require(handler.execute("complete_task", {"task_id": task["task"]["id"]})["ok"], "Task completion failed.")
        require(handler.execute("reopen_task", {"task_id": task["task"]["id"]})["ok"], "Task reopen failed.")

        personal_list = handler.execute("create_list", {"name": "Test grocery list"})
        list_id = personal_list["list"]["id"]
        added = handler.execute(
            "add_list_items",
            {"list_id": list_id, "items": ["milk", "eggs", "chicken", "rice"]},
        )
        require(len(added["added"]) == 4, "Multiple list-item extraction failed.")
        duplicate = handler.execute(
            "add_list_items", {"list_id": list_id, "items": ["Milk"]}
        )
        require(len(duplicate["duplicates"]) == 1, "Duplicate prevention failed.")
        require(
            handler.execute(
                "complete_list_item", {"list_id": list_id, "item_text": "milk"}
            )["ok"],
            "List-item completion failed.",
        )

        restarted = PersonalDataStore(config.personal_data.database_path)
        require(restarted.notes.get(note["note"]["id"]) is not None, "Note did not persist.")
        require(restarted.tasks.get(task["task"]["id"]) is not None, "Task did not persist.")
        require(len(restarted.lists.get_items(list_id)) == 4, "List items did not persist.")

    print("Result: persistent notes, tasks, and lists diagnostic succeeded.")
    print("The diagnostic used a temporary database and left your real data unchanged.")


if __name__ == "__main__":
    main()
