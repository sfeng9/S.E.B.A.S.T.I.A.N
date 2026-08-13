from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from voice_assistant.config import CalendarConfig, GoogleConfig
from voice_assistant.integrations.google_auth import (
    CALENDAR_EVENTS_SCOPE,
    GoogleIntegrationError,
    GoogleOAuthProvider,
    build_google_service,
)


logger = logging.getLogger(__name__)


class GoogleCalendarClient:
    def __init__(
        self,
        google_config: GoogleConfig,
        calendar_config: CalendarConfig,
        service: Any | None = None,
    ) -> None:
        self._config = calendar_config
        self._service = service
        self._auth = GoogleOAuthProvider(
            google_config.credentials_path,
            google_config.calendar_token_path,
            [CALENDAR_EVENTS_SCOPE],
        )

    def get_events_between(
        self,
        start: datetime,
        end: datetime,
        max_results: int | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValueError("Calendar range must use aware datetimes with end after start.")
        limit = max(1, min(2500, max_results or self._config.max_results))
        logger.info(
            "Calendar query started: start=%s end=%s query=%r max_results=%d",
            start.isoformat(),
            end.isoformat(),
            query,
            limit,
        )
        kwargs: dict[str, Any] = {
            "calendarId": self._config.calendar_id,
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": min(2500, limit),
        }
        if query:
            kwargs["q"] = query
        try:
            events: list[dict[str, Any]] = []
            page_token: str | None = None
            while len(events) < limit:
                request_kwargs = dict(kwargs)
                request_kwargs["maxResults"] = min(2500, limit - len(events))
                if page_token:
                    request_kwargs["pageToken"] = page_token
                response = self._get_service().events().list(**request_kwargs).execute()
                items = response.get("items", []) if isinstance(response, dict) else []
                events.extend(
                    _parse_event(item) for item in items if isinstance(item, dict)
                )
                page_token = (
                    str(response.get("nextPageToken", "")).strip()
                    if isinstance(response, dict)
                    else ""
                )
                if not page_token:
                    break
            events = events[:limit]
        except GoogleIntegrationError:
            raise
        except Exception as exc:
            logger.warning("Calendar query failed: %s", type(exc).__name__)
            raise GoogleIntegrationError("Calendar query failed.") from exc
        logger.info("Calendar query completed: result_count=%d", len(events))
        return events

    def get_events_for_date(
        self, start: datetime, end: datetime, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        return self.get_events_between(start, end, max_results=max_results)

    def get_next_event(self, now: datetime) -> dict[str, Any] | None:
        from datetime import timedelta

        events = self.get_events_between(now, now + timedelta(days=365), max_results=1)
        return events[0] if events else None

    def find_events(
        self,
        query: str,
        start: datetime,
        end: datetime,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.get_events_between(start, end, max_results=max_results, query=query)

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        timezone_name: str,
        location: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        body = _event_body(title, start, end, timezone_name, location, description)
        logger.info("Calendar event creation started: title=%r start=%s", title, start.isoformat())
        try:
            raw = (
                self._get_service()
                .events()
                .insert(calendarId=self._config.calendar_id, body=body)
                .execute()
            )
            result = _parse_event(raw)
        except GoogleIntegrationError:
            raise
        except Exception as exc:
            logger.warning("Calendar event creation failed: %s", type(exc).__name__)
            raise GoogleIntegrationError("Calendar event creation failed.") from exc
        logger.info("Calendar event created: event_id=%s title=%r", result["id"], result["title"])
        return result

    def update_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        safe_fields = {key: value for key, value in changes.items() if key in {
            "summary", "start", "end", "location", "description"
        }}
        logger.info("Calendar event update started: event_id=%s fields=%s", event_id, sorted(safe_fields))
        try:
            raw = (
                self._get_service()
                .events()
                .patch(
                    calendarId=self._config.calendar_id,
                    eventId=event_id,
                    body=safe_fields,
                )
                .execute()
            )
            result = _parse_event(raw)
        except GoogleIntegrationError:
            raise
        except Exception as exc:
            logger.warning("Calendar event update failed: event_id=%s error=%s", event_id, type(exc).__name__)
            raise GoogleIntegrationError("Calendar event update failed.") from exc
        logger.info("Calendar event updated: event_id=%s", event_id)
        return result

    def delete_event(self, event_id: str) -> None:
        logger.info("Calendar event deletion started: event_id=%s", event_id)
        try:
            (
                self._get_service()
                .events()
                .delete(calendarId=self._config.calendar_id, eventId=event_id)
                .execute()
            )
        except GoogleIntegrationError:
            raise
        except Exception as exc:
            logger.warning("Calendar event deletion failed: event_id=%s error=%s", event_id, type(exc).__name__)
            raise GoogleIntegrationError("Calendar event deletion failed.") from exc
        logger.info("Calendar event deleted: event_id=%s", event_id)

    def _get_service(self) -> Any:
        if self._service is None:
            self._service = build_google_service("calendar", "v3", self._auth)
        return self._service


def _event_body(
    title: str,
    start: datetime,
    end: datetime,
    timezone_name: str,
    location: str | None,
    description: str | None,
) -> dict[str, Any]:
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Event start/end must be timezone-aware and end after start.")
    body: dict[str, Any] = {
        "summary": title,
        "start": {"dateTime": start.isoformat(), "timeZone": timezone_name},
        "end": {"dateTime": end.isoformat(), "timeZone": timezone_name},
    }
    if location:
        body["location"] = location
    if description:
        body["description"] = description
    return body


def datetime_patch(value: datetime, timezone_name: str) -> dict[str, str]:
    return {"dateTime": value.isoformat(), "timeZone": timezone_name}


def _parse_event(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw.get("id"):
        raise GoogleIntegrationError("Google Calendar returned malformed event data.")
    start = raw.get("start") if isinstance(raw.get("start"), dict) else {}
    end = raw.get("end") if isinstance(raw.get("end"), dict) else {}
    return {
        "id": str(raw["id"]),
        "title": str(raw.get("summary", "(untitled event)")),
        "start": str(start.get("dateTime") or start.get("date") or ""),
        "end": str(end.get("dateTime") or end.get("date") or ""),
        "timezone": str(start.get("timeZone") or end.get("timeZone") or ""),
        "all_day": "date" in start,
        "location": str(raw.get("location", "")),
        "description": str(raw.get("description", ""))[:1000],
        "status": str(raw.get("status", "")),
        "html_link": str(raw.get("htmlLink", "")),
    }
