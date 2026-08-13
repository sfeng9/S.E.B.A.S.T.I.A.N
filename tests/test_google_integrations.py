from __future__ import annotations

import base64
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from voice_assistant.config import GoogleConfig, load_assistant_config
from voice_assistant.integrations.gmail import GmailClient
from voice_assistant.integrations.google_auth import (
    GMAIL_READONLY_SCOPE,
    GoogleCredentialsMissingError,
    GoogleOAuthProvider,
)
from voice_assistant.integrations.google_calendar import GoogleCalendarClient


class RequestResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def execute(self) -> Any:
        return self.value


class FakeGmailService:
    def users(self) -> "FakeGmailService":
        return self

    def messages(self) -> "FakeGmailService":
        return self

    def list(self, **kwargs: Any) -> RequestResult:
        return RequestResult({"messages": [{"id": "normal"}, {"id": "important"}]})

    def get(self, **kwargs: Any) -> RequestResult:
        message_id = kwargs["id"]
        body = base64.urlsafe_b64encode(b"Submit the project by Friday.").decode()
        return RequestResult(
            {
                "id": message_id,
                "threadId": f"thread-{message_id}",
                "internalDate": "1786636800000",
                "labelIds": ["UNREAD"] + (["IMPORTANT"] if message_id == "important" else []),
                "snippet": "Project deadline update",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "Professor Smith <smith@example.edu>"},
                        {"name": "Subject", "value": f"Subject {message_id}"},
                    ],
                    "body": {"data": body},
                },
            }
        )


class FakeCalendarService:
    def __init__(self) -> None:
        self.inserted: dict[str, Any] | None = None

    def events(self) -> "FakeCalendarService":
        return self

    def insert(self, **kwargs: Any) -> RequestResult:
        self.inserted = kwargs["body"]
        return RequestResult({"id": "new-event", **kwargs["body"]})


class FakePagedCalendarService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def events(self) -> "FakePagedCalendarService":
        return self

    def list(self, **kwargs: Any) -> RequestResult:
        self.calls.append(kwargs)
        event_number = 2 if kwargs.get("pageToken") else 1
        response: dict[str, Any] = {
            "items": [
                {
                    "id": f"event-{event_number}",
                    "summary": f"Event {event_number}",
                    "start": {"dateTime": f"2026-08-14T{event_number + 12}:00:00-04:00"},
                    "end": {"dateTime": f"2026-08-14T{event_number + 13}:00:00-04:00"},
                    "status": "confirmed",
                }
            ]
        }
        if event_number == 1:
            response["nextPageToken"] = "page-2"
        return RequestResult(response)


class GoogleIntegrationTests(unittest.TestCase):
    def test_missing_oauth_client_fails_without_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = GoogleOAuthProvider(
                Path(temp_dir) / "missing.json",
                Path(temp_dir) / "token.json",
                [GMAIL_READONLY_SCOPE],
            )
            with self.assertRaises(GoogleCredentialsMissingError):
                provider.get_credentials()

    def test_gmail_important_results_are_structured_and_ranked(self) -> None:
        base = load_assistant_config()
        client = GmailClient(base.google, base.gmail, service=FakeGmailService())
        results = client.get_important_emails(max_results=2)
        self.assertEqual(results[0]["id"], "important")
        self.assertTrue(results[0]["is_important"])
        self.assertNotIn("body", results[0])

        details = client.get_email_details("important")
        self.assertEqual(details["body"], "Submit the project by Friday.")

    def test_calendar_creation_sends_timezone_aware_payload(self) -> None:
        base = load_assistant_config()
        service = FakeCalendarService()
        client = GoogleCalendarClient(base.google, base.calendar, service=service)
        start = datetime.fromisoformat("2026-08-14T19:00:00-04:00")
        end = datetime.fromisoformat("2026-08-14T20:00:00-04:00")
        result = client.create_event(
            "Test dinner", start, end, "America/New_York", location="Cary"
        )
        self.assertEqual(result["id"], "new-event")
        self.assertEqual(service.inserted["start"]["timeZone"], "America/New_York")
        self.assertEqual(service.inserted["start"]["dateTime"], start.isoformat())

    def test_calendar_query_follows_pagination(self) -> None:
        base = load_assistant_config()
        service = FakePagedCalendarService()
        client = GoogleCalendarClient(base.google, base.calendar, service=service)
        start = datetime.fromisoformat("2026-08-14T00:00:00-04:00")
        end = datetime.fromisoformat("2026-08-15T00:00:00-04:00")

        events = client.get_events_between(start, end, max_results=10)

        self.assertEqual([item["id"] for item in events], ["event-1", "event-2"])
        self.assertEqual(len(service.calls), 2)
        self.assertEqual(service.calls[1]["pageToken"], "page-2")


if __name__ == "__main__":
    unittest.main()
