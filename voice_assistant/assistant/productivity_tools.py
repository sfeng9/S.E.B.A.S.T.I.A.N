from __future__ import annotations

import html
import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from typing import Any, Protocol

from voice_assistant.config import AssistantConfig
from voice_assistant.integrations.google_auth import GoogleIntegrationError
from voice_assistant.integrations.google_calendar import GoogleCalendarClient, datetime_patch
from voice_assistant.integrations.gmail import GmailClient
from voice_assistant.integrations.reminders import ReminderStore
from voice_assistant.tools.date_time_parser import (
    TIME_ONLY_PATTERN,
    DateTimeParseError,
    home_now,
    parse_datetime,
    parse_period,
)


logger = logging.getLogger(__name__)

EMAIL_LIST_TOOLS = frozenset(
    {
        "search_emails",
        "get_recent_emails",
        "get_unread_emails",
        "get_important_emails",
    }
)
EMAIL_DETAIL_CHARACTER_LIMIT = 1200
CALENDAR_ACTION_TOOLS = frozenset(
    {"create_event", "update_event", "delete_event", "confirm_calendar_action"}
)
CALENDAR_OBJECT_CUE = re.compile(
    r"\b(?:calendar|schedule|meeting|event|appointment|dinner|lunch|dentist)\b",
    re.IGNORECASE,
)
CALENDAR_REFERENCE_CUE = re.compile(
    r"\b(?:it|that|this|one)\b",
    re.IGNORECASE,
)
CALENDAR_CREATE_CUE = re.compile(
    r"\b(?:add|book|create|put|schedule|set up)\b",
    re.IGNORECASE,
)
CALENDAR_UPDATE_CUE = re.compile(
    r"\b(?:change|move|rename|reschedule|shift|update)\b",
    re.IGNORECASE,
)
CALENDAR_DELETE_CUE = re.compile(
    r"\b(?:cancel|delete|remove)\b",
    re.IGNORECASE,
)
CONFIRMATION_RESPONSE_CUE = re.compile(
    r"(?:^\s*(?:yes|yeah|yep|confirm|do it|go ahead|please do|"
    r"no|nope|don't|do not)\b)|"
    r"(?:^\s*cancel(?:\s*,?\s*(?:it|that|please|thanks))*[.!]?\s*$)",
    re.IGNORECASE,
)
SPOKEN_TIME_CUE = re.compile(
    r"\b(?P<hour>1[0-2]|0?[1-9])(?:\s*:\s*(?P<minute>[0-5]\d))?\s*"
    r"(?P<period>a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)


class GmailProtocol(Protocol):
    def search_emails(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]: ...
    def get_recent_emails(self, max_results: int | None = None) -> list[dict[str, Any]]: ...
    def get_unread_emails(self, max_results: int | None = None) -> list[dict[str, Any]]: ...
    def get_important_emails(self, max_results: int | None = None) -> list[dict[str, Any]]: ...
    def get_email_details(self, message_id: str) -> dict[str, Any]: ...


class CalendarProtocol(Protocol):
    def get_events_between(self, start: datetime, end: datetime, max_results: int | None = None, query: str | None = None) -> list[dict[str, Any]]: ...
    def get_next_event(self, now: datetime) -> dict[str, Any] | None: ...
    def find_events(self, query: str, start: datetime, end: datetime, max_results: int | None = None) -> list[dict[str, Any]]: ...
    def create_event(self, title: str, start: datetime, end: datetime, timezone_name: str, location: str | None = None, description: str | None = None) -> dict[str, Any]: ...
    def update_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]: ...
    def delete_event(self, event_id: str) -> None: ...


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


MAX_RESULTS = {"type": "integer", "minimum": 1, "maximum": 20}
PERIOD = {
    "type": "string",
    "description": "Local date/period such as today, tomorrow, Friday, next Tuesday, this afternoon, this evening, or YYYY-MM-DD.",
}
DATETIME = {
    "type": "string",
    "description": "Timezone-aware ISO datetime or clear local expression such as tomorrow at 3 PM. Preserve AM/PM and date details from the user.",
}


PRODUCTIVITY_TOOL_SCHEMAS = (
    _schema("search_emails", "Search a small relevant Gmail result set. Use Gmail query syntax, e.g. from:(name) newer_than:30d or subject:(project).", {"query": {"type": "string"}, "max_results": MAX_RESULTS}, ["query"]),
    _schema("get_recent_emails", "Get a limited list of email received in the last seven days.", {"max_results": MAX_RESULTS}),
    _schema("get_unread_emails", "Get a limited list of unread recent email.", {"max_results": MAX_RESULTS}),
    _schema("get_important_emails", "Get and rank a small candidate set using Gmail important/unread/recent signals.", {"max_results": MAX_RESULTS}),
    _schema("get_email_details", "Fetch full content for one selected email only. Use after a result-list follow-up such as 'what did the first one say?'.", {"message_id": {"type": "string"}, "ordinal": {"type": "integer", "minimum": 1, "maximum": 20}}),
    _schema("get_events_for_date", "Get Calendar events for a local date or part of a day. Use for today's plan, tomorrow's schedule, afternoons, evenings, and named weekdays.", {"period": PERIOD}, ["period"]),
    _schema("get_events_between", "Get Calendar events in an explicit local date/time range.", {"start_datetime": DATETIME, "end_datetime": DATETIME}, ["start_datetime", "end_datetime"]),
    _schema("get_next_event", "Get the next upcoming Calendar event.", {}),
    _schema("find_events", "Find Calendar events by title/text in a period before modifying or deleting one.", {"query": {"type": "string"}, "period": PERIOD}, ["query"]),
    _schema("create_event", "Create an explicit, unambiguous Calendar event. If the user did not provide a required date or time, ask a clarification instead of calling. End defaults to the configured duration.", {"title": {"type": "string"}, "start_datetime": DATETIME, "end_datetime": DATETIME, "duration_minutes": {"type": "integer", "minimum": 1}, "location": {"type": ["string", "null"]}, "description": {"type": ["string", "null"]}}, ["title", "start_datetime"]),
    _schema("update_event", "Resolve one Calendar event and propose changes. This normally returns a confirmation requirement; do not claim it changed before confirmation.", {"event_id": {"type": "string"}, "query": {"type": "string"}, "period": PERIOD, "new_title": {"type": "string"}, "new_start_datetime": DATETIME, "new_end_datetime": DATETIME, "shift_minutes": {"type": "integer"}, "location": {"type": ["string", "null"]}, "description": {"type": ["string", "null"]}}),
    _schema("delete_event", "Resolve one Calendar event and request confirmation before deletion.", {"event_id": {"type": "string"}, "query": {"type": "string"}, "period": PERIOD}),
    _schema("confirm_calendar_action", "Confirm or reject the one pending Calendar update/delete from the immediately preceding conversation. Never use without a pending action.", {"confirm": {"type": "boolean"}}, ["confirm"]),
    _schema("create_reminder", "Create a persistent local spoken reminder. It survives session expiry and restart.", {"text": {"type": "string"}, "due_datetime": DATETIME}, ["text", "due_datetime"]),
    _schema("get_pending_reminders", "List upcoming local reminders.", {"max_results": MAX_RESULTS}),
    _schema("cancel_reminder", "Cancel one pending local reminder by ID.", {"reminder_id": {"type": "integer"}}, ["reminder_id"]),
)


@dataclass
class PendingCalendarAction:
    operation: str
    event: dict[str, Any]
    changes: dict[str, Any] | None = None


class ProductivityToolHandler:
    def __init__(
        self,
        config: AssistantConfig,
        gmail: GmailProtocol | None = None,
        calendar: CalendarProtocol | None = None,
        reminder_store: ReminderStore | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._timezone = config.home_location.timezone
        self._gmail = gmail or GmailClient(config.google, config.gmail)
        self._calendar = calendar or GoogleCalendarClient(config.google, config.calendar)
        self.reminder_store = reminder_store or ReminderStore(config.reminders.database_path)
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._last_email_ids: list[str] = []
        self._last_event_ids: list[str] = []
        self._pending: PendingCalendarAction | None = None
        self._confirmation_created_this_turn = False
        self._email_spoken_override: str | None = None
        self._calendar_spoken_override: str | None = None

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return PRODUCTIVITY_TOOL_SCHEMAS

    def reset_session_context(self) -> None:
        self._last_email_ids.clear()
        self._last_event_ids.clear()
        self._pending = None
        logger.info("Cleared transient Gmail/Calendar references and pending action.")

    def begin_turn(self) -> None:
        self._confirmation_created_this_turn = False
        self._email_spoken_override = None
        self._calendar_spoken_override = None

    def tool_requirement(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        del history
        if self._pending is not None and CONFIRMATION_RESPONSE_CUE.search(prompt):
            return {
                "tools": ("confirm_calendar_action",),
                "instruction": (
                    "A Calendar update or deletion is awaiting confirmation. Call "
                    "confirm_calendar_action now using the user's yes or no response. "
                    "Do not claim success without its tool result."
                ),
                "fallback": (
                    "I couldn't send that confirmation to Google Calendar, so I "
                    "didn't complete the change."
                ),
            }

        has_event_context = bool(
            CALENDAR_OBJECT_CUE.search(prompt)
            or (self._last_event_ids and CALENDAR_REFERENCE_CUE.search(prompt))
        )
        if not has_event_context:
            return None
        if CALENDAR_DELETE_CUE.search(prompt):
            expected_tool = "delete_event"
            action = "delete"
        elif CALENDAR_UPDATE_CUE.search(prompt):
            expected_tool = "update_event"
            action = "update"
        elif CALENDAR_CREATE_CUE.search(prompt):
            expected_tool = "create_event"
            action = "create"
        else:
            return None
        return {
            "tools": (expected_tool,),
            "instruction": (
                f"The user asked you to {action} a Calendar event. You must call "
                f"{expected_tool} now. Do not say the event was changed unless a "
                "Calendar tool result explicitly confirms it."
            ),
            "fallback": (
                "I couldn't use Google Calendar for that request, so I didn't "
                "make the change."
            ),
        }

    def spoken_override_for(self, called_tools: Sequence[str]) -> str | None:
        if called_tools and set(called_tools).issubset(EMAIL_LIST_TOOLS):
            return self._email_spoken_override
        if called_tools and CALENDAR_ACTION_TOOLS.intersection(called_tools):
            return self._calendar_spoken_override
        return None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        handlers = {
            "search_emails": self._search_emails,
            "get_recent_emails": self._recent_emails,
            "get_unread_emails": self._unread_emails,
            "get_important_emails": self._important_emails,
            "get_email_details": self._email_details,
            "get_events_for_date": self._events_for_date,
            "get_events_between": self._events_between,
            "get_next_event": self._next_event,
            "find_events": self._find_events,
            "create_event": self._create_event,
            "update_event": self._update_event,
            "delete_event": self._delete_event,
            "confirm_calendar_action": self._confirm_calendar_action,
            "create_reminder": self._create_reminder,
            "get_pending_reminders": self._pending_reminders,
            "cancel_reminder": self._cancel_reminder,
        }
        handler = handlers.get(name)
        if handler is None:
            return None
        try:
            return handler(arguments)
        except GoogleIntegrationError as exc:
            service = "email" if "email" in name or "gmail" in name else "Google Calendar"
            logger.warning("%s tool failed (%s): %s", name, type(exc).__name__, exc)
            if name in CALENDAR_ACTION_TOOLS:
                self._calendar_spoken_override = (
                    "I couldn't change your Google Calendar right now."
                )
            return {
                "ok": False,
                "error": "google_unavailable",
                "message": (
                    f"Respond with exactly this one sentence: I couldn't check your {service} right now. "
                    "Do not guess, expose setup details, ask a question, or offer more help."
                ),
            }
        except (DateTimeParseError, ValueError, TypeError) as exc:
            logger.warning("%s validation failed: %s", name, exc)
            return {"ok": False, "error": "invalid_arguments", "message": str(exc)}

    def _search_emails(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._email_list(self._gmail.search_emails(str(args["query"]), args.get("max_results")))

    def _recent_emails(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._email_list(self._gmail.get_recent_emails(args.get("max_results")))

    def _unread_emails(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._email_list(self._gmail.get_unread_emails(args.get("max_results")))

    def _important_emails(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._email_list(self._gmail.get_important_emails(args.get("max_results")))

    def _email_list(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self._last_email_ids = [str(item["id"]) for item in messages]
        spoken_messages = [
            _spoken_email_metadata(
                item,
                self._config.gmail.list_snippet_character_limit,
                index + 1,
            )
            for index, item in enumerate(
                messages[: self._config.gmail.spoken_result_limit]
            )
        ]
        if spoken_messages:
            self._email_spoken_override = " ".join(
                str(item["spoken_summary"]) for item in spoken_messages
            )
        else:
            self._email_spoken_override = "I didn't find any matching emails."
        return {
            "ok": True,
            "count": len(messages),
            "emails": spoken_messages,
            "message": (
                "For this email-only response, use only each spoken_summary exactly. "
                "Do not add an introduction, count, subject, address, date, label, "
                "identifier, symbol, or closing sentence."
            ),
        }

    def _email_details(self, args: dict[str, Any]) -> dict[str, Any]:
        message_id = str(args.get("message_id", "")).strip()
        if not message_id:
            ordinal = int(args.get("ordinal", 1))
            if ordinal < 1 or ordinal > len(self._last_email_ids):
                return {"ok": False, "error": "email_reference_missing", "message": "I no longer know which email you mean. Ask for it by sender or subject."}
            message_id = self._last_email_ids[ordinal - 1]
        result = self._gmail.get_email_details(message_id)
        self._last_email_ids = [message_id]
        content = str(result.get("body") or result.get("snippet") or "")
        return {
            "ok": True,
            "email": {
                "sender_name": _spoken_sender_name(str(result.get("sender", ""))),
                "content": _spoken_text(content, EMAIL_DETAIL_CHARACTER_LIMIT),
            },
            "message": (
                "Summarize this email in one or two short spoken sentences. Do not "
                "read addresses, metadata, symbols, or the content verbatim."
            ),
        }

    def _events_for_date(self, args: dict[str, Any]) -> dict[str, Any]:
        start, end = parse_period(str(args["period"]), self._require_timezone(), self._now_provider())
        return self._event_list(self._calendar.get_events_between(start, end), start, end)

    def _events_between(self, args: dict[str, Any]) -> dict[str, Any]:
        zone = self._require_timezone()
        start = parse_datetime(str(args["start_datetime"]), zone, self._now_provider(), prefer_future=False)
        end = parse_datetime(str(args["end_datetime"]), zone, self._now_provider(), prefer_future=False)
        return self._event_list(self._calendar.get_events_between(start, end), start, end)

    def _next_event(self, args: dict[str, Any]) -> dict[str, Any]:
        now = home_now(self._require_timezone(), self._now_provider())
        event = self._calendar.get_next_event(now)
        self._last_event_ids = [str(event["id"])] if event else []
        return {"ok": True, "event": event}

    def _find_events(self, args: dict[str, Any]) -> dict[str, Any]:
        start, end = self._search_window(args.get("period"))
        events = self._calendar.find_events(str(args["query"]), start, end)
        return self._event_list(events, start, end)

    def _event_list(self, events: list[dict[str, Any]], start: datetime, end: datetime) -> dict[str, Any]:
        self._last_event_ids = [str(item["id"]) for item in events]
        return {"ok": True, "count": len(events), "start": start.isoformat(), "end": end.isoformat(), "events": events}

    def _create_event(self, args: dict[str, Any]) -> dict[str, Any]:
        zone = self._require_timezone()
        title = str(args["title"]).strip()
        if not title:
            raise ValueError("An event title is required.")
        start = parse_datetime(str(args["start_datetime"]), zone, self._now_provider())
        if args.get("end_datetime"):
            end = parse_datetime(str(args["end_datetime"]), zone, self._now_provider())
        else:
            duration = int(args.get("duration_minutes", self._config.calendar.default_event_duration_minutes))
            end = start + timedelta(minutes=duration)
        event = self._calendar.create_event(title, start, end, zone, args.get("location"), args.get("description"))
        self._last_event_ids = [str(event["id"])]
        self._calendar_spoken_override = (
            f"I've added {_spoken_event_title(event)} to your calendar."
        )
        return {"ok": True, "created": True, "event": event}

    def _update_event(self, args: dict[str, Any]) -> dict[str, Any]:
        selected = self._select_event(args)
        if isinstance(selected, dict) and selected.get("selection_error"):
            self._calendar_spoken_override = _selection_error_speech(selected)
            return selected
        event = selected
        if not isinstance(event, dict):
            raise RuntimeError("Calendar event selection returned an invalid result.")
        changes = self._build_changes(event, args)
        if not changes:
            self._calendar_spoken_override = "I didn't find any changes to make."
            return {"ok": False, "error": "no_changes", "message": "No event changes were specified."}
        if self._config.calendar.confirm_updates:
            self._pending = PendingCalendarAction("update", event, changes)
            self._confirmation_created_this_turn = True
            self._calendar_spoken_override = (
                f"Do you want me to update {_spoken_event_title(event)}?"
            )
            return {"ok": True, "confirmation_required": True, "operation": "update", "event": _event_confirmation(event), "changes": changes, "message": "Ask one concise confirmation question. Do not say the event has changed yet."}
        updated = self._calendar.update_event(str(event["id"]), changes)
        self._calendar_spoken_override = (
            f"I've updated {_spoken_event_title(updated)}."
        )
        return {"ok": True, "updated": True, "event": updated}

    def _delete_event(self, args: dict[str, Any]) -> dict[str, Any]:
        selected = self._select_event(args)
        if isinstance(selected, dict) and selected.get("selection_error"):
            self._calendar_spoken_override = _selection_error_speech(selected)
            return selected
        event = selected
        if not isinstance(event, dict):
            raise RuntimeError("Calendar event selection returned an invalid result.")
        if self._config.calendar.confirm_deletes:
            self._pending = PendingCalendarAction("delete", event)
            self._confirmation_created_this_turn = True
            self._calendar_spoken_override = (
                f"Do you want me to delete {_spoken_event_title(event)} from your calendar?"  # nosec
            )
            return {"ok": True, "confirmation_required": True, "operation": "delete", "event": _event_confirmation(event), "message": "Ask one concise confirmation question. Do not say the event is deleted yet."}
        self._calendar.delete_event(str(event["id"]))
        self._calendar_spoken_override = (
            f"I've deleted {_spoken_event_title(event)} from your calendar."
        )
        return {"ok": True, "deleted": True, "event": _event_confirmation(event)}

    def _confirm_calendar_action(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._confirmation_created_this_turn:
            self._calendar_spoken_override = (
                "Please confirm that Calendar change in a separate response."
            )
            return {"ok": False, "error": "confirmation_must_be_new_turn", "message": "Wait for the user's next response before confirming this Calendar action."}
        if self._pending is None:
            self._calendar_spoken_override = (
                "There isn't a pending Calendar change to confirm."
            )
            return {"ok": False, "error": "no_pending_action", "message": "There is no pending Calendar change to confirm."}
        pending = self._pending
        if not bool(args["confirm"]):
            self._pending = None
            logger.info("Pending Calendar %s rejected for event_id=%s", pending.operation, pending.event["id"])
            self._calendar_spoken_override = (
                f"Okay, I didn't {pending.operation} {_spoken_event_title(pending.event)}."
            )
            return {"ok": True, "cancelled": True, "operation": pending.operation}
        event_id = str(pending.event["id"])
        if pending.operation == "update":
            event = self._calendar.update_event(event_id, pending.changes or {})
            self._pending = None
            self._last_event_ids = [event_id]
            self._calendar_spoken_override = (
                f"I've updated {_spoken_event_title(event)}."
            )
            return {"ok": True, "updated": True, "event": event}
        self._calendar.delete_event(event_id)
        self._pending = None
        self._last_event_ids = []
        self._calendar_spoken_override = (
            f"I've deleted {_spoken_event_title(pending.event)} from your calendar."
        )
        return {"ok": True, "deleted": True, "event": _event_confirmation(pending.event)}

    def _select_event(self, args: dict[str, Any]) -> dict[str, Any]:
        event_id = str(args.get("event_id", "")).strip()
        query = str(args.get("query", "")).strip()
        start, end = self._search_window(args.get("period"))
        if query:
            events = self._calendar.find_events(query, start, end)
            if not events:
                candidates = self._calendar.get_events_between(start, end)
                events = _events_matching_spoken_time(candidates, query, self._require_timezone())
                if events:
                    logger.info(
                        "Calendar event resolved by spoken start time after text search returned no results: matches=%d",
                        len(events),
                    )
        elif event_id:
            events = self._calendar.get_events_between(start, end)
            events = [item for item in events if str(item.get("id")) == event_id]
        elif len(self._last_event_ids) == 1:
            target = self._last_event_ids[0]
            events = self._calendar.get_events_between(start, end)
            events = [item for item in events if str(item.get("id")) == target]
        else:
            return {"selection_error": True, "ok": False, "error": "event_reference_missing", "message": "I need the event title or date to identify it."}
        if not events:
            return {"selection_error": True, "ok": False, "error": "event_not_found", "message": "I couldn't find that event."}
        if len(events) > 1:
            self._last_event_ids = [str(item["id"]) for item in events]
            return {"selection_error": True, "ok": False, "error": "ambiguous_event", "candidates": [_event_confirmation(item) for item in events[:5]], "message": "Multiple events match. Ask one concise clarification question before changing anything."}
        return events[0]

    def _build_changes(self, event: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        zone = self._require_timezone()
        changes: dict[str, Any] = {}
        if args.get("new_title"):
            changes["summary"] = str(args["new_title"])
        old_start = _event_datetime(event.get("start"), zone)
        old_end = _event_datetime(event.get("end"), zone)
        new_start = None
        if args.get("new_start_datetime"):
            new_start = _parse_update_datetime(str(args["new_start_datetime"]), old_start, zone, self._now_provider())
        elif args.get("shift_minutes") is not None:
            new_start = old_start + timedelta(minutes=int(args["shift_minutes"]))
        if new_start is not None:
            changes["start"] = datetime_patch(new_start, zone)
            if args.get("new_end_datetime"):
                new_end = _parse_update_datetime(str(args["new_end_datetime"]), new_start, zone, self._now_provider())
            else:
                new_end = new_start + (old_end - old_start)
            changes["end"] = datetime_patch(new_end, zone)
        elif args.get("new_end_datetime"):
            changes["end"] = datetime_patch(_parse_update_datetime(str(args["new_end_datetime"]), old_start, zone, self._now_provider()), zone)
        for source, target in (("location", "location"), ("description", "description")):
            if source in args:
                changes[target] = args[source] or ""
        return changes

    def _create_reminder(self, args: dict[str, Any]) -> dict[str, Any]:
        zone = self._require_timezone()
        due = parse_datetime(str(args["due_datetime"]), zone, self._now_provider())
        reminder = self.reminder_store.create(str(args["text"]), due, zone)
        return {"ok": True, "created": True, "reminder": reminder.to_dict()}

    def _pending_reminders(self, args: dict[str, Any]) -> dict[str, Any]:
        reminders = self.reminder_store.list_pending(int(args.get("max_results", 10)))
        return {"ok": True, "count": len(reminders), "reminders": [item.to_dict() for item in reminders]}

    def _cancel_reminder(self, args: dict[str, Any]) -> dict[str, Any]:
        reminder_id = int(args["reminder_id"])
        cancelled = self.reminder_store.cancel(reminder_id)
        return {"ok": cancelled, "cancelled": cancelled, "reminder_id": reminder_id, "message": "Reminder cancelled." if cancelled else "I couldn't find that pending reminder."}

    def _search_window(self, period: Any) -> tuple[datetime, datetime]:
        zone = self._require_timezone()
        if period:
            return parse_period(str(period), zone, self._now_provider())
        start = home_now(zone, self._now_provider())
        return start, start + timedelta(days=30)

    def _require_timezone(self) -> str:
        if not self._timezone:
            raise ValueError("Configure home_location.timezone before using Calendar or reminders.")
        return self._timezone


def _event_datetime(value: Any, timezone_name: str) -> datetime:
    if not value:
        raise ValueError("The selected event has no usable start/end time.")
    text = str(value)
    if len(text) == 10:
        text += "T00:00:00"
    return parse_datetime(text, timezone_name, prefer_future=False)


def _parse_update_datetime(
    value: str,
    event_reference: datetime,
    timezone_name: str,
    now: datetime,
) -> datetime:
    if TIME_ONLY_PATTERN.match(value.strip()):
        previous_day_end = event_reference.replace(
            hour=23, minute=59, second=59, microsecond=0
        ) - timedelta(days=1)
        return parse_datetime(value, timezone_name, previous_day_end)
    return parse_datetime(value, timezone_name, now)


def _event_confirmation(event: dict[str, Any]) -> dict[str, Any]:
    return {key: event.get(key) for key in ("id", "title", "start", "end", "timezone", "location")}


def _spoken_event_title(event: dict[str, Any]) -> str:
    return _spoken_text(
        str(event.get("title", "")),
        100,
        empty_fallback="that event",
    )


def _selection_error_speech(result: dict[str, Any]) -> str:
    if result.get("error") == "ambiguous_event":
        return "I found multiple matching events. Which one do you mean?"
    if result.get("error") == "event_not_found":
        return "I couldn't find that event."
    return "I need the event title or date before I can change it."


def _events_matching_spoken_time(
    events: list[dict[str, Any]],
    query: str,
    timezone_name: str,
) -> list[dict[str, Any]]:
    match = SPOKEN_TIME_CUE.search(query)
    if match is None:
        return []
    hour = int(match.group("hour")) % 12
    if match.group("period").casefold().startswith("p"):
        hour += 12
    minute = int(match.group("minute") or 0)
    matches: list[dict[str, Any]] = []
    for event in events:
        if event.get("all_day"):
            continue
        try:
            start = _event_datetime(event.get("start"), timezone_name)
        except (DateTimeParseError, ValueError, TypeError):
            continue
        if start.hour == hour and start.minute == minute:
            matches.append(event)
    return matches


def _spoken_email_metadata(
    message: dict[str, Any],
    snippet_limit: int,
    reference: int,
) -> dict[str, Any]:
    sender_name = _spoken_sender_name(str(message.get("sender", "")))
    snippet = _spoken_text(str(message.get("snippet", "")), snippet_limit)
    return {
        "reference": reference,
        "sender_name": sender_name,
        "snippet": snippet,
        "spoken_summary": f"The sender is {sender_name}. The snippet is: {snippet}.",
    }


def _spoken_sender_name(raw_sender: str) -> str:
    display_name, _ = parseaddr(raw_sender)
    cleaned = _spoken_text(display_name, 80, empty_fallback="an unnamed sender")
    return cleaned.rstrip(".?!") or "an unnamed sender"


def _spoken_text(
    value: str,
    max_characters: int,
    empty_fallback: str = "No readable English preview is available",
) -> str:
    text = html.unescape(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S+@\S+\.\S+\b", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9 .,!?'-]+", " ", text)
    text = re.sub(r"([*#_=~|<>\[\]{}])+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-_")
    if not text:
        return empty_fallback
    if len(text) > max_characters:
        shortened = text[: max_characters + 1].rsplit(" ", 1)[0].strip()
        text = shortened or text[:max_characters].strip()
    return text.rstrip(".?!")
