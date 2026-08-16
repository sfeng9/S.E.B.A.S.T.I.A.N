from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from voice_assistant.assistant.tool_permissions import ToolPermission
from voice_assistant.config import AssistantConfig
from voice_assistant.integrations.personal_data import (
    DuplicateRecordError,
    ListItem,
    Note,
    PersonalDataError,
    PersonalDataStore,
    PersonalList,
    Task,
)
from voice_assistant.integrations.reminders import ReminderStore
from voice_assistant.tools.date_time_parser import (
    DateTimeParseError,
    parse_datetime,
    parse_period,
)


logger = logging.getLogger(__name__)

NOTE_TOOLS = frozenset(
    {"create_note", "search_notes", "get_note", "update_note", "delete_note", "list_notes"}
)
TASK_TOOLS = frozenset(
    {
        "create_task",
        "list_tasks",
        "find_tasks",
        "complete_task",
        "reopen_task",
        "update_task",
        "delete_task",
        "delete_completed_tasks",
    }
)
LIST_TOOLS = frozenset(
    {
        "create_list",
        "list_lists",
        "add_list_items",
        "remove_list_item",
        "complete_list_item",
        "uncomplete_list_item",
        "get_list",
        "rename_list",
        "clear_list",
        "delete_list",
    }
)
PERSONAL_DATA_META_TOOLS = frozenset(
    {"confirm_personal_data_action", "undo_personal_data_action"}
)

PERSONAL_DATA_TOOL_PERMISSIONS: Mapping[str, ToolPermission] = {
    "create_note": ToolPermission.SAFE_ACTION,
    "search_notes": ToolPermission.READ_ONLY,
    "get_note": ToolPermission.READ_ONLY,
    "update_note": ToolPermission.SAFE_ACTION,
    "delete_note": ToolPermission.SAFE_ACTION,
    "list_notes": ToolPermission.READ_ONLY,
    "create_task": ToolPermission.SAFE_ACTION,
    "list_tasks": ToolPermission.READ_ONLY,
    "find_tasks": ToolPermission.READ_ONLY,
    "complete_task": ToolPermission.SAFE_ACTION,
    "reopen_task": ToolPermission.SAFE_ACTION,
    "update_task": ToolPermission.SAFE_ACTION,
    "delete_task": ToolPermission.SAFE_ACTION,
    "delete_completed_tasks": ToolPermission.CONFIRM_REQUIRED,
    "create_list": ToolPermission.SAFE_ACTION,
    "list_lists": ToolPermission.READ_ONLY,
    "add_list_items": ToolPermission.SAFE_ACTION,
    "remove_list_item": ToolPermission.SAFE_ACTION,
    "complete_list_item": ToolPermission.SAFE_ACTION,
    "uncomplete_list_item": ToolPermission.SAFE_ACTION,
    "get_list": ToolPermission.READ_ONLY,
    "rename_list": ToolPermission.SAFE_ACTION,
    "clear_list": ToolPermission.CONFIRM_REQUIRED,
    "delete_list": ToolPermission.CONFIRM_REQUIRED,
    "confirm_personal_data_action": ToolPermission.SAFE_ACTION,
    "undo_personal_data_action": ToolPermission.SAFE_ACTION,
}

NOTE_CUE = re.compile(
    r"\b(?:remember|note|notes|wrote|written|write down|tell you|told you|mentioned)\b",
    re.IGNORECASE,
)
TASK_CUE = re.compile(
    r"\b(?:task|tasks|to[ -]?do|homework|assignment|need to do)\b",
    re.IGNORECASE,
)
LIST_CUE = re.compile(
    r"\b(?:list|lists|grocery|groceries|shopping|packing|costco)\b",
    re.IGNORECASE,
)
LIST_FOLLOW_UP_CUE = re.compile(
    r"\b(?:add|remove|delete|cross|check|uncheck|what(?:'s| is) on|clear|rename)\b",
    re.IGNORECASE,
)
TASK_FOLLOW_UP_CUE = re.compile(
    r"\b(?:mark|complete|done|reopen|delete|remove|first|second|third)\b",
    re.IGNORECASE,
)
NOTE_FOLLOW_UP_CUE = re.compile(
    r"\b(?:update|change|delete|remove|first|second|third|what did it say)\b",
    re.IGNORECASE,
)
UNDO_CUE = re.compile(r"^\s*(?:undo|undo that|put it back|revert that)[.!]?\s*$", re.I)
CONFIRMATION_CUE = re.compile(
    r"^\s*(?:yes|yeah|yep|confirm|do it|go ahead|please do|no|nope|don't|do not|cancel)\b",
    re.IGNORECASE,
)


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


ID = {"type": "integer", "minimum": 1}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100}
QUERY = {"type": "string", "description": "Short identifying text from the user's request."}
ORDINAL = {"type": "integer", "minimum": 1, "maximum": 20}
DATETIME = {
    "type": ["string", "null"],
    "description": (
        "A local expression such as tomorrow at 5 PM, Friday, tonight, by noon, "
        "or a timezone-aware ISO datetime. Use null only when the user explicitly clears it."
    ),
}
PERIOD = {
    "type": "string",
    "description": "A local period such as today, tomorrow, Friday, next Monday, or YYYY-MM-DD.",
}
LIST_REFERENCE = {
    "list_id": ID,
    "list_name": {
        "type": "string",
        "description": "List name or configured alias. Omit for a clear follow-up to the last shown list.",
    },
}
RECORD_REFERENCE = {
    "id": ID,
    "query": QUERY,
    "ordinal": ORDINAL,
}


PERSONAL_DATA_TOOL_SCHEMAS = (
    _schema(
        "create_note",
        "Persist a note only when the user explicitly asks Sebastian to remember, save, store, or take a note. Derive a concise title if none was spoken.",
        {
            "title": {"type": ["string", "null"]},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        },
        ("content",),
    ),
    _schema("search_notes", "Search local note titles, contents, and tags. Use for what the user previously asked Sebastian to remember.", {"query": QUERY, "limit": LIMIT}, ("query",)),
    _schema("get_note", "Get one note by ID or a recent result ordinal.", {"note_id": ID, "ordinal": ORDINAL}),
    _schema("list_notes", "List recent local notes without dumping the entire database.", {"limit": LIMIT}),
    _schema(
        "update_note",
        "Update one clearly identified note. If several notes match, the tool will refuse and return candidates.",
        {"note_id": ID, "query": QUERY, "ordinal": ORDINAL, "title": {"type": "string"}, "content": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20}},
    ),
    _schema("delete_note", "Delete one clearly identified note. Never use for many notes at once.", {"note_id": ID, "query": QUERY, "ordinal": ORDINAL}),
    _schema(
        "create_task",
        "Create one actionable local task. Store due dates without creating a reminder unless the user explicitly asks to be reminded.",
        {
            "title": {"type": "string"},
            "due_datetime": DATETIME,
            "reminder_datetime": DATETIME,
            "priority": {"type": ["string", "null"]},
            "project": {"type": ["string", "null"]},
            "category": {"type": ["string", "null"]},
        },
        ("title",),
    ),
    _schema(
        "list_tasks",
        "List local tasks, optionally filtered by status or due period. Use due_period=today for 'what do I need to do today?'.",
        {"status": {"type": "string", "enum": ["pending", "completed", "all"]}, "due_period": PERIOD, "limit": LIMIT},
    ),
    _schema("find_tasks", "Find a small set of tasks by title text.", {"query": QUERY, "status": {"type": "string", "enum": ["pending", "completed", "all"]}, "limit": LIMIT}, ("query",)),
    _schema("complete_task", "Mark one clearly identified task completed.", {"task_id": ID, "query": QUERY, "ordinal": ORDINAL}),
    _schema("reopen_task", "Return one clearly identified completed task to pending.", {"task_id": ID, "query": QUERY, "ordinal": ORDINAL}),
    _schema(
        "update_task",
        "Update one clearly identified task's title, due date, priority, project, or category.",
        {"task_id": ID, "query": QUERY, "ordinal": ORDINAL, "title": {"type": "string"}, "due_datetime": DATETIME, "priority": {"type": ["string", "null"]}, "project": {"type": ["string", "null"]}, "category": {"type": ["string", "null"]}},
    ),
    _schema("delete_task", "Delete one clearly identified task. Refuse ambiguous matches.", {"task_id": ID, "query": QUERY, "ordinal": ORDINAL}),
    _schema("delete_completed_tasks", "Request confirmation before deleting all completed tasks.", {}),
    _schema("create_list", "Create a named local list, optionally with explicit aliases.", {"name": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 20}}, ("name",)),
    _schema("list_lists", "List the user's local named lists.", {"limit": LIMIT}),
    _schema(
        "add_list_items",
        "Add separately extracted items to one list. Always pass multiple spoken items as separate array elements.",
        {**LIST_REFERENCE, "items": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 50}},
        ("items",),
    ),
    _schema("get_list", "Get one list and its items. Omit the list name only for a clear follow-up to the last referenced list.", {**LIST_REFERENCE, "include_completed": {"type": "boolean"}, "limit": LIMIT}),
    _schema("remove_list_item", "Remove one clearly identified item from the referenced list.", {**LIST_REFERENCE, "item_id": ID, "item_text": QUERY, "ordinal": ORDINAL}),
    _schema("complete_list_item", "Cross off or check one clearly identified list item without deleting it.", {**LIST_REFERENCE, "item_id": ID, "item_text": QUERY, "ordinal": ORDINAL}),
    _schema("uncomplete_list_item", "Uncheck one clearly identified list item.", {**LIST_REFERENCE, "item_id": ID, "item_text": QUERY, "ordinal": ORDINAL}),
    _schema("rename_list", "Rename one list. The former name remains an alias.", {**LIST_REFERENCE, "new_name": {"type": "string"}}, ("new_name",)),
    _schema("clear_list", "Request confirmation before removing every item from one list.", LIST_REFERENCE),
    _schema("delete_list", "Request confirmation before deleting an entire list and its items.", LIST_REFERENCE),
    _schema("confirm_personal_data_action", "Confirm or reject a pending bulk task/list deletion from the immediately preceding turn.", {"confirm": {"type": "boolean"}}, ("confirm",)),
    _schema("undo_personal_data_action", "Undo the most recent supported note, task, or list mutation in this active session.", {}),
)


@dataclass
class PendingAction:
    operation: str
    target_id: int | None
    target_name: str


@dataclass
class UndoAction:
    description: str
    callback: Callable[[], None]


class PersonalDataToolHandler:
    def __init__(
        self,
        config: AssistantConfig,
        store: PersonalDataStore | None = None,
        reminder_store: ReminderStore | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._timezone = config.home_location.timezone
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self.store = store or PersonalDataStore(
            config.personal_data.database_path,
            now_provider=self._now_provider,
        )
        self._reminders = reminder_store or ReminderStore(config.reminders.database_path)
        self._schema_by_name = {
            str(schema["function"]["name"]): schema for schema in PERSONAL_DATA_TOOL_SCHEMAS
        }
        self._last_note_ids: list[int] = []
        self._last_task_ids: list[int] = []
        self._last_list_id: int | None = None
        self._last_item_ids: list[int] = []
        self._pending: PendingAction | None = None
        self._confirmation_created_this_turn = False
        self._undo: UndoAction | None = None
        self._spoken_override: str | None = None

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return PERSONAL_DATA_TOOL_SCHEMAS

    @property
    def permissions(self) -> Mapping[str, ToolPermission]:
        return PERSONAL_DATA_TOOL_PERMISSIONS

    @property
    def has_pending_confirmation(self) -> bool:
        return self._pending is not None

    def clear_pending_confirmation(self) -> None:
        if self._pending is not None:
            logger.info("Cancelled pending personal-data action in favor of a newer action.")
        self._pending = None

    def reset_session_context(self) -> None:
        self._last_note_ids.clear()
        self._last_task_ids.clear()
        self._last_list_id = None
        self._last_item_ids.clear()
        self._pending = None
        self._undo = None
        self._spoken_override = None
        logger.info("Cleared transient personal-data references, confirmation, and undo state.")

    def begin_turn(self) -> None:
        self._confirmation_created_this_turn = False
        self._spoken_override = None

    def schemas_for(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        del history
        names = set(self._expected_tool_names(prompt))
        return tuple(self._schema_by_name[name] for name in self._schema_by_name if name in names)

    def tool_requirement(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        schemas = self.schemas_for(prompt, history)
        names = tuple(str(schema["function"]["name"]) for schema in schemas)
        if not names:
            return None
        if self._pending is not None and CONFIRMATION_CUE.search(prompt):
            names = ("confirm_personal_data_action",)
            instruction = "A personal-data bulk deletion is awaiting confirmation. Call confirm_personal_data_action with the user's direct yes or no response."
        elif UNDO_CUE.search(prompt):
            names = ("undo_personal_data_action",)
            instruction = "The user asked to undo the most recent personal-data change. Call undo_personal_data_action now."
        else:
            instruction = (
                "This request reads or changes Sebastian's persistent local notes, tasks, or lists. "
                f"Call {', '.join(names)} now. Do not answer from conversation memory, "
                "and do not claim a mutation succeeded without a successful tool result."
            )
        return {
            "tools": names,
            "instruction": instruction,
            "fallback": "I couldn't access your saved notes, tasks, or lists right now.",
        }

    def _expected_tool_names(self, prompt: str) -> tuple[str, ...]:
        text = " ".join(prompt.casefold().split())
        if self._pending is not None and CONFIRMATION_CUE.search(prompt):
            return ("confirm_personal_data_action",)
        if UNDO_CUE.search(prompt):
            return ("undo_personal_data_action",)

        names: list[str] = []
        note_request = bool(
            NOTE_CUE.search(prompt)
            or (self._last_note_ids and NOTE_FOLLOW_UP_CUE.search(prompt))
        )
        if note_request:
            if re.search(r"\b(?:delete|remove)\b", text):
                names.append("delete_note")
            elif re.search(r"\b(?:update|change|edit)\b", text):
                names.append("update_note")
            elif re.search(r"\b(?:remember|save|store|take a note|make a note|write down|create a note)\b", text):
                names.append("create_note")
            elif re.search(r"\b(?:list|show)\b.*\bnotes?\b", text):
                names.append("list_notes")
            else:
                names.append("search_notes")

        task_request = bool(
            TASK_CUE.search(prompt)
            or (self._last_task_ids and TASK_FOLLOW_UP_CUE.search(prompt))
        )
        if task_request:
            if re.search(r"\bdelete\b.*\bcompleted tasks?\b", text):
                names.append("delete_completed_tasks")
            elif re.search(r"\b(?:reopen|uncomplete|mark pending)\b", text):
                names.append("reopen_task")
            elif re.search(r"\b(?:mark|complete|done|finish)\b", text) and not re.search(r"\badd\b", text):
                names.append("complete_task")
            elif re.search(r"\b(?:delete|remove)\b", text):
                names.append("delete_task")
            elif re.search(r"\b(?:update|change|edit|rename)\b", text):
                names.append("update_task")
            elif re.search(r"\b(?:add|create|new)\b", text):
                names.append("create_task")
            elif re.search(r"\b(?:find|search)\b", text):
                names.append("find_tasks")
            else:
                names.append("list_tasks")

        list_request = bool(
            LIST_CUE.search(prompt)
            or (self._last_list_id and LIST_FOLLOW_UP_CUE.search(prompt))
        )
        if list_request:
            if re.search(r"\bclear\b", text):
                names.append("clear_list")
            elif re.search(r"\brename\b", text):
                names.append("rename_list")
            elif re.search(r"\b(?:uncheck|uncross|put back)\b", text):
                names.append("uncomplete_list_item")
            elif re.search(r"\b(?:cross|check off|mark off)\b", text):
                names.append("complete_list_item")
            elif re.search(r"\b(?:remove|delete)\b", text):
                if re.search(r"\bdelete\b.*\blist\b", text) and not re.search(r"\bfrom\b.*\blist\b", text):
                    names.append("delete_list")
                else:
                    names.append("remove_list_item")
            else:
                if re.search(r"\bcreate\b.*\blist\b", text):
                    names.append("create_list")
                if re.search(r"\badd\b", text):
                    names.append("add_list_items")
                if not names or re.search(r"\b(?:what|show|read)\b", text):
                    names.append("get_list")
                if re.search(r"\b(?:list|show)\b.*\b(?:my|all)\s+lists\b", text):
                    names = [name for name in names if name != "get_list"]
                    names.append("list_lists")

        return tuple(dict.fromkeys(names))

    def spoken_override_for(self, called_tools: Sequence[str]) -> str | None:
        if called_tools and set(called_tools).issubset(PERSONAL_DATA_TOOL_PERMISSIONS):
            return self._spoken_override
        return None

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "create_note": self._create_note,
            "search_notes": self._search_notes,
            "get_note": self._get_note,
            "list_notes": self._list_notes,
            "update_note": self._update_note,
            "delete_note": self._delete_note,
            "create_task": self._create_task,
            "list_tasks": self._list_tasks,
            "find_tasks": self._find_tasks,
            "complete_task": lambda value: self._set_task_status(value, "completed"),
            "reopen_task": lambda value: self._set_task_status(value, "pending"),
            "update_task": self._update_task,
            "delete_task": self._delete_task,
            "delete_completed_tasks": self._request_delete_completed_tasks,
            "create_list": self._create_list,
            "list_lists": self._list_lists,
            "add_list_items": self._add_list_items,
            "get_list": self._get_list,
            "remove_list_item": self._remove_list_item,
            "complete_list_item": lambda value: self._set_list_item(value, True),
            "uncomplete_list_item": lambda value: self._set_list_item(value, False),
            "rename_list": self._rename_list,
            "clear_list": self._request_clear_list,
            "delete_list": self._request_delete_list,
            "confirm_personal_data_action": self._confirm_action,
            "undo_personal_data_action": self._undo_action,
        }
        handler = handlers.get(name)
        if handler is None:
            return None
        try:
            return handler(args)
        except DuplicateRecordError as exc:
            logger.info("Personal-data duplicate prevented: tool=%s", name)
            self._spoken_override = _spoken(str(exc))
            return {"ok": False, "error": "duplicate", "message": str(exc)}
        except (DateTimeParseError, ValueError, TypeError) as exc:
            logger.warning("Personal-data validation failed: tool=%s error=%s", name, type(exc).__name__)
            self._spoken_override = "I couldn't understand enough detail to save that."
            return {"ok": False, "error": "invalid_arguments", "message": str(exc)}
        except (sqlite3.Error, OSError, PersonalDataError) as exc:
            logger.error("Persistent-data error: tool=%s error=%s", name, type(exc).__name__)
            self._spoken_override = "I couldn't access your saved notes, tasks, or lists right now."
            return {
                "ok": False,
                "error": "personal_data_unavailable",
                "message": "The local personal-data store is unavailable. Do not guess or claim a change succeeded.",
            }

    def _create_note(self, args: dict[str, Any]) -> dict[str, Any]:
        content = str(args["content"]).strip()
        title = str(args.get("title") or _derive_note_title(content)).strip()
        note = self.store.notes.create(title, content, _string_list(args.get("tags")))
        self._last_note_ids = [note.id]
        self._remember_undo("save that note", lambda: self.store.notes.delete(note.id))
        self._spoken_override = f"Saved your note, {_spoken(note.title)}."
        return {"ok": True, "created": True, "note": note.to_dict()}

    def _search_notes(self, args: dict[str, Any]) -> dict[str, Any]:
        notes = self.store.notes.search(
            str(args["query"]),
            int(args.get("limit", self._config.personal_data.search_result_limit)),
        )
        self._last_note_ids = [note.id for note in notes]
        if not notes:
            self._spoken_override = "I couldn't find a matching note."
        return {"ok": True, "count": len(notes), "notes": [note.to_dict() for note in notes]}

    def _get_note(self, args: dict[str, Any]) -> dict[str, Any]:
        note = self._select_note(args)
        if isinstance(note, dict):
            return note
        self._last_note_ids = [note.id]
        return {"ok": True, "note": note.to_dict()}

    def _list_notes(self, args: dict[str, Any]) -> dict[str, Any]:
        notes = self.store.notes.list(int(args.get("limit", self._config.personal_data.search_result_limit)))
        self._last_note_ids = [note.id for note in notes]
        if not notes:
            self._spoken_override = "You don't have any saved notes."
        return {"ok": True, "count": len(notes), "notes": [note.to_dict() for note in notes]}

    def _update_note(self, args: dict[str, Any]) -> dict[str, Any]:
        note = self._select_note(args)
        if isinstance(note, dict):
            return note
        if not any(key in args for key in ("title", "content", "tags")):
            raise ValueError("Specify a title, content, or tags to update.")
        updated = self.store.notes.update(
            note.id,
            title=args.get("title"),
            content=args.get("content"),
            tags=_string_list(args.get("tags")) if "tags" in args else None,
        )
        if updated is None:
            return self._not_found("note")
        self._last_note_ids = [updated.id]
        self._remember_undo(
            "update that note",
            lambda: self.store.notes.update(note.id, title=note.title, content=note.content, tags=note.tags),
        )
        self._spoken_override = f"Updated {_spoken(updated.title)}."
        return {"ok": True, "updated": True, "note": updated.to_dict()}

    def _delete_note(self, args: dict[str, Any]) -> dict[str, Any]:
        note = self._select_note(args)
        if isinstance(note, dict):
            return note
        deleted = self.store.notes.delete(note.id)
        if deleted is None:
            return self._not_found("note")
        self._last_note_ids = []
        self._remember_undo(
            "delete that note",
            lambda: self.store.notes.create(deleted.title, deleted.content, deleted.tags),
        )
        self._spoken_override = f"Deleted {_spoken(deleted.title)}."
        return {"ok": True, "deleted": True, "note": deleted.to_dict()}

    def _create_task(self, args: dict[str, Any]) -> dict[str, Any]:
        zone = self._timezone
        due = None
        if args.get("due_datetime"):
            zone = self._require_timezone()
            due = self._parse_optional_datetime(args.get("due_datetime"), zone)
        task = self.store.tasks.create(
            str(args["title"]),
            due_at_utc=_utc_iso(due),
            timezone_name=zone if due else None,
            priority=args.get("priority"),
            project=args.get("project"),
            category=args.get("category"),
        )
        self._last_task_ids = [task.id]
        self._remember_undo("create that task", lambda: self.store.tasks.delete(task.id))
        reminder = None
        if args.get("reminder_datetime"):
            try:
                zone = self._require_timezone()
                reminder_due = parse_datetime(str(args["reminder_datetime"]), zone, self._now_provider())
                reminder = self._reminders.create(task.title, reminder_due, zone)
                task = self.store.tasks.set_reminder(task.id, reminder.id) or task
            except (sqlite3.Error, OSError, ValueError) as exc:
                logger.error(
                    "Task saved but linked reminder failed: task_id=%d error=%s",
                    task.id,
                    type(exc).__name__,
                )
                self._spoken_override = (
                    f"Added {_spoken(task.title)} to your tasks, but I couldn't set the reminder."
                )
                return {
                    "ok": True,
                    "created": True,
                    "partial_success": True,
                    "task": task.to_dict(),
                    "reminder_error": "reminder_unavailable",
                }
        self._spoken_override = f"Added {_spoken(task.title)} to your tasks."
        result: dict[str, Any] = {"ok": True, "created": True, "task": task.to_dict()}
        if reminder is not None:
            result["linked_reminder"] = reminder.to_dict()
        return result

    def _list_tasks(self, args: dict[str, Any]) -> dict[str, Any]:
        status = str(args.get("status", "pending"))
        due_after = due_before = None
        if args.get("due_period"):
            start, end = parse_period(str(args["due_period"]), self._require_timezone(), self._now_provider())
            due_after, due_before = _utc_iso(start), _utc_iso(end)
        tasks = self.store.tasks.list(
            status=status,
            due_after_utc=due_after,
            due_before_utc=due_before,
            limit=int(args.get("limit", 20)),
        )
        return self._task_results(tasks, status)

    def _find_tasks(self, args: dict[str, Any]) -> dict[str, Any]:
        tasks = self.store.tasks.find(
            str(args["query"]),
            status=str(args.get("status", "all")),
            limit=int(args.get("limit", self._config.personal_data.search_result_limit)),
        )
        return self._task_results(tasks, str(args.get("status", "all")))

    def _task_results(self, tasks: list[Task], status: str) -> dict[str, Any]:
        self._last_task_ids = [task.id for task in tasks]
        if not tasks:
            self._spoken_override = "You don't have any matching tasks."
        elif len(tasks) > self._config.personal_data.spoken_item_limit:
            titles = _join_spoken(task.title for task in tasks[: self._config.personal_data.spoken_item_limit])
            self._spoken_override = f"You have {len(tasks)} matching tasks. The first few are {titles}."
        return {"ok": True, "count": len(tasks), "status": status, "tasks": [task.to_dict() for task in tasks]}

    def _set_task_status(self, args: dict[str, Any], status: str) -> dict[str, Any]:
        task = self._select_task(args)
        if isinstance(task, dict):
            return task
        updated = self.store.tasks.set_status(task.id, status)
        if updated is None:
            return self._not_found("task")
        self._last_task_ids = [updated.id]
        self._remember_undo(
            "change that task",
            lambda: self.store.tasks.set_status(task.id, task.status),
        )
        self._spoken_override = (
            f"Marked {_spoken(updated.title)} done."
            if status == "completed"
            else f"Reopened {_spoken(updated.title)}."
        )
        return {"ok": True, "task": updated.to_dict(), "status_changed": True}

    def _update_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task = self._select_task(args)
        if isinstance(task, dict):
            return task
        updates: dict[str, Any] = {}
        for key in ("title", "priority", "project", "category"):
            if key in args:
                updates[key] = args[key]
        if "due_datetime" in args:
            due = self._parse_optional_datetime(args.get("due_datetime"), self._require_timezone())
            updates["due_at_utc"] = _utc_iso(due)
            updates["timezone_name"] = self._timezone if due else None
        if not updates:
            raise ValueError("Specify at least one task field to update.")
        updated = self.store.tasks.update(task.id, **updates)
        if updated is None:
            return self._not_found("task")
        self._last_task_ids = [updated.id]
        self._remember_undo(
            "update that task",
            lambda: self.store.tasks.update(
                task.id,
                title=task.title,
                due_at_utc=task.due_at_utc,
                timezone_name=task.timezone_name,
                priority=task.priority,
                project=task.project,
                category=task.category,
            ),
        )
        self._spoken_override = f"Updated {_spoken(updated.title)}."
        return {"ok": True, "updated": True, "task": updated.to_dict()}

    def _delete_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task = self._select_task(args)
        if isinstance(task, dict):
            return task
        deleted = self.store.tasks.delete(task.id)
        if deleted is None:
            return self._not_found("task")
        self._last_task_ids = []
        self._remember_undo("delete that task", lambda: self._restore_task(deleted))
        self._spoken_override = f"Deleted {_spoken(deleted.title)}."
        return {"ok": True, "deleted": True, "task": deleted.to_dict()}

    def _request_delete_completed_tasks(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        completed = self.store.tasks.list(status="completed", limit=100)
        if not completed:
            self._spoken_override = "You don't have any completed tasks to delete."
            return {"ok": True, "deleted": False, "count": 0}
        self._pending = PendingAction("delete_completed_tasks", None, "completed tasks")
        self._confirmation_created_this_turn = True
        self._spoken_override = f"Delete all {len(completed)} completed tasks?"
        return {"ok": True, "confirmation_required": True, "operation": "delete_completed_tasks", "count": len(completed)}

    def _create_list(self, args: dict[str, Any]) -> dict[str, Any]:
        item = self.store.lists.create(str(args["name"]), _string_list(args.get("aliases")))
        self._last_list_id = item.id
        self._last_item_ids = []
        self._remember_undo("create that list", lambda: self.store.lists.delete(item.id))
        self._spoken_override = f"Created your {_spoken_list_name(item.name)}."
        return {"ok": True, "created": True, "list": item.to_dict()}

    def _list_lists(self, args: dict[str, Any]) -> dict[str, Any]:
        lists = self.store.lists.list(int(args.get("limit", 20)))
        if not lists:
            self._spoken_override = "You don't have any saved lists."
        elif len(lists) <= self._config.personal_data.spoken_item_limit:
            self._spoken_override = f"Your lists are {_join_spoken(item.name for item in lists)}."
        return {"ok": True, "count": len(lists), "lists": [item.to_dict() for item in lists]}

    def _add_list_items(self, args: dict[str, Any]) -> dict[str, Any]:
        target = self._select_list(args)
        if isinstance(target, dict):
            return target
        result = self.store.lists.add_items(target.id, _string_list(args.get("items")))
        changed = [*result["added"], *result["reopened"]]
        self._last_list_id = target.id
        self._last_item_ids = [item.id for item in changed] or [item.id for item in result["duplicates"]]
        if changed:
            added_ids = [item.id for item in result["added"]]
            reopened_ids = [item.id for item in result["reopened"]]
            self._remember_undo(
                "add those list items",
                lambda: self._undo_added_items(added_ids, reopened_ids),
            )
            self._spoken_override = f"Added {_join_spoken(item.text for item in changed)}."
        else:
            duplicate_text = _join_spoken(item.text for item in result["duplicates"])
            verb = "is" if len(result["duplicates"]) == 1 else "are"
            self._spoken_override = f"{duplicate_text} {verb} already on your list."
        return {
            "ok": True,
            "list": target.to_dict(),
            "added": [item.to_dict() for item in result["added"]],
            "reopened": [item.to_dict() for item in result["reopened"]],
            "duplicates": [item.to_dict() for item in result["duplicates"]],
        }

    def _get_list(self, args: dict[str, Any]) -> dict[str, Any]:
        target = self._select_list(args)
        if isinstance(target, dict):
            return target
        items = self.store.lists.get_items(
            target.id,
            include_completed=bool(args.get("include_completed", True)),
            limit=int(args.get("limit", 100)),
        )
        self._last_list_id = target.id
        self._last_item_ids = [item.id for item in items]
        active = [item for item in items if not item.completed]
        if not items:
            self._spoken_override = f"Your {_spoken_list_name(target.name)} is empty."
        elif len(active) > self._config.personal_data.spoken_item_limit:
            first = _join_spoken(item.text for item in active[: self._config.personal_data.spoken_item_limit])
            self._spoken_override = f"Your {_spoken_list_name(target.name)} has {len(active)} unchecked items. The first few are {first}."
        elif active:
            self._spoken_override = f"Your {_spoken_list_name(target.name)} has {_join_spoken(item.text for item in active)}."
        else:
            self._spoken_override = f"Everything on your {_spoken_list_name(target.name)} is checked off."
        return {"ok": True, "list": target.to_dict(), "count": len(items), "items": [item.to_dict() for item in items]}

    def _remove_list_item(self, args: dict[str, Any]) -> dict[str, Any]:
        selected = self._select_list_item(args)
        if isinstance(selected, dict):
            return selected
        target, item = selected
        removed = self.store.lists.remove_item(item.id)
        if removed is None:
            return self._not_found("list item")
        self._last_list_id = target.id
        self._last_item_ids = []
        self._remember_undo("remove that list item", lambda: self._restore_list_item(removed))
        self._spoken_override = f"Removed {_spoken(removed.text)}."
        return {"ok": True, "removed": True, "item": removed.to_dict(), "list": target.to_dict()}

    def _set_list_item(self, args: dict[str, Any], completed: bool) -> dict[str, Any]:
        selected = self._select_list_item(args)
        if isinstance(selected, dict):
            return selected
        target, item = selected
        updated = self.store.lists.set_item_completed(item.id, completed)
        if updated is None:
            return self._not_found("list item")
        self._last_list_id = target.id
        self._last_item_ids = [updated.id]
        self._remember_undo(
            "change that list item",
            lambda: self.store.lists.set_item_completed(item.id, item.completed),
        )
        self._spoken_override = (
            f"Crossed off {_spoken(updated.text)}."
            if completed
            else f"Put {_spoken(updated.text)} back on the list."
        )
        return {"ok": True, "item": updated.to_dict(), "list": target.to_dict()}

    def _rename_list(self, args: dict[str, Any]) -> dict[str, Any]:
        target = self._select_list(args)
        if isinstance(target, dict):
            return target
        renamed = self.store.lists.rename(target.id, str(args["new_name"]))
        if renamed is None:
            return self._not_found("list")
        self._last_list_id = renamed.id
        self._remember_undo("rename that list", lambda: self.store.lists.rename(target.id, target.name))
        self._spoken_override = f"Renamed it to {_spoken(renamed.name)}."
        return {"ok": True, "renamed": True, "list": renamed.to_dict()}

    def _request_clear_list(self, args: dict[str, Any]) -> dict[str, Any]:
        target = self._select_list(args)
        if isinstance(target, dict):
            return target
        count = len(self.store.lists.get_items(target.id, include_completed=True, limit=500))
        if count == 0:
            self._spoken_override = f"Your {_spoken_list_name(target.name)} is already empty."
            return {"ok": True, "cleared": False, "count": 0}
        self._pending = PendingAction("clear_list", target.id, target.name)
        self._confirmation_created_this_turn = True
        self._spoken_override = f"Clear all {count} items from your {_spoken_list_name(target.name)}?"
        return {"ok": True, "confirmation_required": True, "operation": "clear_list", "list": target.to_dict(), "count": count}

    def _request_delete_list(self, args: dict[str, Any]) -> dict[str, Any]:
        target = self._select_list(args)
        if isinstance(target, dict):
            return target
        self._pending = PendingAction("delete_list", target.id, target.name)
        self._confirmation_created_this_turn = True
        self._spoken_override = f"Delete your {_spoken_list_name(target.name)} and all of its items?"
        return {"ok": True, "confirmation_required": True, "operation": "delete_list", "list": target.to_dict()}

    def _confirm_action(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._confirmation_created_this_turn:
            self._spoken_override = "Please confirm that deletion in a separate response."
            return {"ok": False, "error": "confirmation_must_be_new_turn"}
        if self._pending is None:
            self._spoken_override = "There isn't a pending list or task deletion to confirm."
            return {"ok": False, "error": "no_pending_action"}
        pending = self._pending
        if not bool(args["confirm"]):
            self._pending = None
            self._spoken_override = "Okay, I didn't delete anything."
            return {"ok": True, "cancelled": True, "operation": pending.operation}
        if pending.operation == "delete_completed_tasks":
            count = self.store.tasks.delete_completed()
            self._pending = None
            self._spoken_override = f"Deleted {count} completed task{'s' if count != 1 else ''}."
            return {"ok": True, "deleted": True, "count": count}
        if pending.target_id is None:
            raise PersonalDataError("The pending list action has no target.")
        if pending.operation == "clear_list":
            items = self.store.lists.clear(pending.target_id)
            self._pending = None
            self._last_list_id = pending.target_id
            self._last_item_ids = []
            self._spoken_override = f"Cleared your {_spoken_list_name(pending.target_name)}."
            return {"ok": True, "cleared": True, "count": len(items)}
        deleted = self.store.lists.delete(pending.target_id)
        self._pending = None
        self._last_list_id = None
        self._last_item_ids = []
        self._spoken_override = f"Deleted your {_spoken_list_name(pending.target_name)}."
        return {"ok": True, "deleted": deleted is not None, "operation": "delete_list"}

    def _undo_action(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        if self._undo is None:
            self._spoken_override = "There isn't a recent saved-data change I can undo."
            return {"ok": False, "error": "nothing_to_undo"}
        action = self._undo
        self._undo = None
        action.callback()
        logger.info("Personal-data change undone: action=%s", action.description)
        self._spoken_override = "Undone."
        return {"ok": True, "undone": True, "action": action.description}

    def _select_note(self, args: dict[str, Any]) -> Note | dict[str, Any]:
        note_id = _reference_id(args, "note_id", self._last_note_ids)
        if note_id is not None:
            note = self.store.notes.get(note_id)
            return note if note is not None else self._not_found("note")
        query = str(args.get("query", "")).strip()
        if query:
            matches = self.store.notes.search(query, self._config.personal_data.search_result_limit)
            if len(matches) == 1:
                return matches[0]
            return self._ambiguous_or_missing("note", matches)
        if len(self._last_note_ids) == 1:
            note = self.store.notes.get(self._last_note_ids[0])
            return note if note is not None else self._not_found("note")
        return self._reference_missing("note")

    def _select_task(self, args: dict[str, Any]) -> Task | dict[str, Any]:
        task_id = _reference_id(args, "task_id", self._last_task_ids)
        if task_id is not None:
            task = self.store.tasks.get(task_id)
            return task if task is not None else self._not_found("task")
        query = str(args.get("query", "")).strip()
        if query:
            matches = self.store.tasks.find(query, status="all", limit=self._config.personal_data.search_result_limit)
            if len(matches) == 1:
                return matches[0]
            return self._ambiguous_or_missing("task", matches)
        if len(self._last_task_ids) == 1:
            task = self.store.tasks.get(self._last_task_ids[0])
            return task if task is not None else self._not_found("task")
        return self._reference_missing("task")

    def _select_list(self, args: dict[str, Any]) -> PersonalList | dict[str, Any]:
        if args.get("list_id") is not None:
            target = self.store.lists.get(int(args["list_id"]))
            return target if target is not None else self._not_found("list")
        name = str(args.get("list_name", "")).strip()
        if name:
            exact = self.store.lists.find_by_name(name, exact=True)
            matches = exact or self.store.lists.find_by_name(name)
            if len(matches) == 1:
                return matches[0]
            return self._ambiguous_or_missing("list", matches)
        if self._last_list_id is not None:
            target = self.store.lists.get(self._last_list_id)
            return target if target is not None else self._not_found("list")
        return self._reference_missing("list")

    def _select_list_item(
        self, args: dict[str, Any]
    ) -> tuple[PersonalList, ListItem] | dict[str, Any]:
        target = self._select_list(args)
        if isinstance(target, dict):
            return target
        item_id = _reference_id(args, "item_id", self._last_item_ids)
        items: list[ListItem]
        if item_id is not None:
            items = [item for item in self.store.lists.get_items(target.id, limit=500) if item.id == item_id]
        else:
            text = str(args.get("item_text", "")).strip()
            items = self.store.lists.find_items(target.id, text) if text else []
        if len(items) == 1:
            return target, items[0]
        return self._ambiguous_or_missing("list item", items)

    def _ambiguous_or_missing(
        self,
        kind: str,
        matches: Sequence[Note | Task | PersonalList | ListItem],
    ) -> dict[str, Any]:
        if not matches:
            return self._not_found(kind)
        logger.info("Persistent-data ambiguity detected: kind=%s matches=%d", kind, len(matches))
        labels = [_record_label(item) for item in matches[:5]]
        self._spoken_override = f"I found more than one matching {kind}: {_join_spoken(labels)}. Which one?"
        return {
            "ok": False,
            "error": f"ambiguous_{kind.replace(' ', '_')}",
            "candidates": [_record_summary(item) for item in matches[:5]],
            "message": "Ask one concise clarification question. Do not modify any record.",
        }

    def _not_found(self, kind: str) -> dict[str, Any]:
        self._spoken_override = f"I couldn't find that {kind}."
        return {"ok": False, "error": f"{kind.replace(' ', '_')}_not_found"}

    def _reference_missing(self, kind: str) -> dict[str, Any]:
        self._spoken_override = f"Which {kind} do you mean?"
        return {"ok": False, "error": f"{kind.replace(' ', '_')}_reference_missing"}

    def _remember_undo(self, description: str, callback: Callable[[], Any]) -> None:
        self._undo = UndoAction(description, lambda: callback() and None)

    def _undo_added_items(self, added_ids: Sequence[int], reopened_ids: Sequence[int]) -> None:
        for item_id in added_ids:
            self.store.lists.remove_item(item_id)
        for item_id in reopened_ids:
            self.store.lists.set_item_completed(item_id, True)

    def _restore_list_item(self, item: ListItem) -> None:
        result = self.store.lists.add_items(item.list_id, [item.text])
        restored = [*result["added"], *result["reopened"]]
        if item.completed and restored:
            self.store.lists.set_item_completed(restored[0].id, True)

    def _restore_task(self, task: Task) -> None:
        restored = self.store.tasks.create(
            task.title,
            due_at_utc=task.due_at_utc,
            timezone_name=task.timezone_name,
            priority=task.priority,
            reminder_id=task.reminder_id,
            project=task.project,
            category=task.category,
            recurrence=task.recurrence,
        )
        if task.status == "completed":
            self.store.tasks.set_status(restored.id, "completed")

    def _require_timezone(self) -> str:
        if not self._timezone:
            raise ValueError("Configure home_location.timezone before using task due dates.")
        return self._timezone

    def _parse_optional_datetime(self, value: Any, timezone_name: str) -> datetime | None:
        if value is None or not str(value).strip():
            return None
        return parse_datetime(str(value), timezone_name, self._now_provider())


def _derive_note_title(content: str) -> str:
    clean = " ".join(content.split()).strip(" .")
    match = re.match(r"(?:my |the )?(.{2,60}?)\s+(?:is|are)\s+", clean, re.I)
    if match and len(match.group(1).split()) <= 8:
        title = match.group(1)
    else:
        title = " ".join(clean.split()[:7])
    return title[:1].upper() + title[1:] if title else "Saved note"


def _reference_id(args: dict[str, Any], key: str, recent_ids: Sequence[int]) -> int | None:
    if args.get(key) is not None:
        return int(args[key])
    if args.get("ordinal") is not None:
        ordinal = int(args["ordinal"])
        if not 0 < ordinal <= len(recent_ids):
            raise ValueError("That result number is no longer available in this session.")
        return recent_ids[ordinal - 1]
    return None


def _record_label(item: Note | Task | PersonalList | ListItem) -> str:
    if isinstance(item, Note):
        return item.title
    if isinstance(item, Task):
        return item.title
    if isinstance(item, PersonalList):
        return item.name
    return item.text


def _record_summary(item: Note | Task | PersonalList | ListItem) -> dict[str, Any]:
    data = item.to_dict()
    if isinstance(item, Note):
        return {"id": item.id, "title": item.title}
    if isinstance(item, Task):
        return {key: data[key] for key in ("id", "title", "status", "due_at_utc")}
    if isinstance(item, PersonalList):
        return {"id": item.id, "name": item.name}
    return {"id": item.id, "text": item.text, "completed": item.completed}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of text values.")
    return [str(item).strip() for item in value if str(item).strip()]


def _spoken(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9 .,!?'-]+", " ", text)
    return " ".join(text.split()).strip(" .,:;-_") or "that item"


def _spoken_list_name(value: str) -> str:
    name = _spoken(value)
    return name if name.casefold().endswith(" list") else f"{name} list"


def _join_spoken(values: Sequence[str] | Any) -> str:
    items = [_spoken(value) for value in values]
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()
