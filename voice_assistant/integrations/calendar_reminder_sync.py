from __future__ import annotations

import html
import logging
import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from voice_assistant.config import AssistantConfig
from voice_assistant.integrations.google_calendar import GoogleCalendarClient
from voice_assistant.integrations.reminders import ReminderStore


logger = logging.getLogger(__name__)


class CalendarReader(Protocol):
    def get_events_between(
        self,
        start: datetime,
        end: datetime,
        max_results: int | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class CalendarReminderSyncResult:
    events_seen: int
    reminders_synchronized: int
    events_skipped: int
    reminders_cancelled: int


class CalendarReminderSynchronizer:
    def __init__(
        self,
        config: AssistantConfig,
        reminder_store: ReminderStore,
        calendar: CalendarReader | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._store = reminder_store
        self._calendar = calendar or GoogleCalendarClient(
            config.google,
            config.calendar,
        )
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._source = f"google_calendar:{config.calendar.calendar_id}"

    def sync_once(self) -> CalendarReminderSyncResult:
        timezone_name = self._config.home_location.timezone
        if not timezone_name:
            raise ValueError(
                "Configure home_location.timezone before synchronizing Calendar reminders."
            )
        try:
            home_zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown home timezone: {timezone_name}") from exc

        now = self._now_provider()
        if now.tzinfo is None:
            raise ValueError("Calendar reminder synchronization requires an aware clock.")
        now = now.astimezone(timezone.utc)
        window_end = now + timedelta(
            hours=self._config.reminders.calendar_sync_lookahead_hours
        )
        logger.info(
            "Calendar reminder sync started: calendar=%s lookahead_hours=%d lead_minutes=%d",
            self._config.calendar.calendar_id,
            self._config.reminders.calendar_sync_lookahead_hours,
            self._config.reminders.calendar_reminder_minutes_before,
        )
        events = self._calendar.get_events_between(
            now,
            window_end,
            max_results=self._config.reminders.calendar_sync_max_results,
        )

        active_ids: set[str] = set()
        synchronized = 0
        skipped = 0
        for event in events:
            external_id = str(event.get("id", "")).strip()
            if (
                not external_id
                or bool(event.get("all_day"))
                or str(event.get("status", "")).casefold() == "cancelled"
            ):
                skipped += 1
                continue
            try:
                start = _event_start(event, home_zone)
            except (TypeError, ValueError, ZoneInfoNotFoundError):
                logger.warning(
                    "Skipping Calendar reminder with malformed start: event_id=%s",
                    external_id,
                )
                skipped += 1
                continue
            if start <= now:
                skipped += 1
                continue

            active_ids.add(external_id)
            due = start - timedelta(
                minutes=self._config.reminders.calendar_reminder_minutes_before
            )
            title = _spoken_event_title(str(event.get("title", "")))
            local_start = start.astimezone(home_zone)
            text = f"You have {title} at {_spoken_time(local_start)}"
            self._store.upsert_external(
                source=self._source,
                external_id=external_id,
                text=text,
                due_at=due,
                timezone_name=timezone_name,
            )
            synchronized += 1

        cancelled = self._store.cancel_missing_external(self._source, active_ids)
        result = CalendarReminderSyncResult(
            events_seen=len(events),
            reminders_synchronized=synchronized,
            events_skipped=skipped,
            reminders_cancelled=cancelled,
        )
        logger.info(
            "Calendar reminder sync completed: events_seen=%d synchronized=%d skipped=%d cancelled=%d",
            result.events_seen,
            result.reminders_synchronized,
            result.events_skipped,
            result.reminders_cancelled,
        )
        return result


class CalendarReminderSyncWorker:
    def __init__(
        self,
        synchronizer: CalendarReminderSynchronizer,
        interval_seconds: float,
    ) -> None:
        self._synchronizer = synchronizer
        self._interval_seconds = max(1.0, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="calendar-reminder-sync",
            daemon=True,
        )
        self._thread.start()
        logger.info("Calendar reminder background sync started.")

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout_seconds))
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._synchronizer.sync_once()
            except Exception as exc:
                logger.warning(
                    "Calendar reminder sync failed (%s): %s",
                    type(exc).__name__,
                    exc,
                )
            if self._stop_event.wait(self._interval_seconds):
                break


def _event_start(event: dict[str, Any], home_zone: ZoneInfo) -> datetime:
    value = str(event.get("start", "")).strip()
    if not value or len(value) == 10:
        raise ValueError("Timed Calendar event has no date-time start.")
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        event_timezone = str(event.get("timezone", "")).strip()
        parsed = parsed.replace(tzinfo=ZoneInfo(event_timezone) if event_timezone else home_zone)
    return parsed.astimezone(timezone.utc)


def _spoken_event_title(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"https?://\S+|www\.\S+|\b\S+@\S+\b", " ", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    if not text:
        return "an event"
    if len(text) <= 100:
        return text
    shortened = text[:100].rsplit(" ", 1)[0].strip()
    return shortened or "an event"


def _spoken_time(value: datetime) -> str:
    hour = int(value.strftime("%I"))
    suffix = value.strftime("%p")
    if value.minute == 0:
        return f"{hour} {suffix}"
    return f"{hour}:{value.minute:02d} {suffix}"
