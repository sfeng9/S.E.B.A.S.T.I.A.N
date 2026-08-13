from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from voice_assistant.config import GmailConfig, GoogleConfig
from voice_assistant.integrations.google_auth import (
    GMAIL_READONLY_SCOPE,
    GoogleIntegrationError,
    GoogleOAuthProvider,
    build_google_service,
)


logger = logging.getLogger(__name__)


class GmailClient:
    def __init__(
        self,
        google_config: GoogleConfig,
        gmail_config: GmailConfig,
        service: Any | None = None,
    ) -> None:
        self._config = gmail_config
        self._service = service
        self._auth = GoogleOAuthProvider(
            google_config.credentials_path,
            google_config.gmail_token_path,
            [GMAIL_READONLY_SCOPE],
        )

    def search_emails(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        limit = self._limit(max_results)
        logger.info("Gmail search started: query=%r max_results=%d", query, limit)
        try:
            response = (
                self._get_service()
                .users()
                .messages()
                .list(userId="me", q=query, maxResults=limit)
                .execute()
            )
            stubs = response.get("messages", []) if isinstance(response, dict) else []
            results = [self._get_metadata(str(item["id"])) for item in stubs if item.get("id")]
        except GoogleIntegrationError:
            raise
        except Exception as exc:
            logger.warning("Gmail search failed: %s", type(exc).__name__)
            raise GoogleIntegrationError("Gmail search failed.") from exc
        logger.info("Gmail search completed: result_count=%d", len(results))
        return results

    def get_recent_emails(self, max_results: int | None = None) -> list[dict[str, Any]]:
        return self.search_emails("newer_than:7d", max_results)

    def get_unread_emails(self, max_results: int | None = None) -> list[dict[str, Any]]:
        return self.search_emails("is:unread newer_than:30d", max_results)

    def get_important_emails(self, max_results: int | None = None) -> list[dict[str, Any]]:
        requested = self._limit(max_results)
        candidates = self.search_emails(
            "newer_than:14d {is:important is:unread}",
            self._config.important_candidate_limit,
        )
        ranked = sorted(candidates, key=_importance_score, reverse=True)
        results = ranked[:requested]
        logger.info(
            "Gmail important ranking completed: candidates=%d returned=%d",
            len(candidates),
            len(results),
        )
        return results

    def get_email_details(self, message_id: str) -> dict[str, Any]:
        logger.info("Gmail detail fetch started: message_id=%s", message_id)
        try:
            raw = (
                self._get_service()
                .users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
            result = self._parse_message(raw, include_body=True)
        except GoogleIntegrationError:
            raise
        except Exception as exc:
            logger.warning(
                "Gmail detail fetch failed: message_id=%s error=%s",
                message_id,
                type(exc).__name__,
            )
            raise GoogleIntegrationError("Email detail fetch failed.") from exc
        logger.info(
            "Gmail detail fetch completed: message_id=%s subject=%r",
            message_id,
            result.get("subject", ""),
        )
        return result

    def _get_metadata(self, message_id: str) -> dict[str, Any]:
        raw = (
            self._get_service()
            .users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        return self._parse_message(raw, include_body=False)

    def _parse_message(self, raw: Any, include_body: bool) -> dict[str, Any]:
        if not isinstance(raw, dict) or not raw.get("id"):
            raise GoogleIntegrationError("Gmail returned malformed message data.")
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        headers = {
            str(item.get("name", "")).casefold(): str(item.get("value", ""))
            for item in payload.get("headers", [])
            if isinstance(item, dict)
        }
        labels = {str(label) for label in raw.get("labelIds", [])}
        received_at = _received_at(raw, headers.get("date"))
        result: dict[str, Any] = {
            "id": str(raw["id"]),
            "thread_id": str(raw.get("threadId", "")),
            "sender": headers.get("from", "Unknown sender"),
            "subject": headers.get("subject", "(no subject)"),
            "received_at": received_at,
            "snippet": str(raw.get("snippet", ""))[
                : self._config.list_snippet_character_limit
            ],
            "is_unread": "UNREAD" in labels,
            "is_important": "IMPORTANT" in labels,
        }
        if include_body:
            body = _extract_body(payload)
            result["body"] = body[: self._config.detail_body_character_limit]
            result["body_truncated"] = len(body) > self._config.detail_body_character_limit
        return result

    def _get_service(self) -> Any:
        if self._service is None:
            self._service = build_google_service("gmail", "v1", self._auth)
        return self._service

    def _limit(self, requested: int | None) -> int:
        if requested is None:
            return self._config.max_results
        return max(1, min(self._config.important_candidate_limit, int(requested)))


def _received_at(raw: dict[str, Any], date_header: str | None) -> str:
    internal = raw.get("internalDate")
    if internal is not None:
        try:
            return datetime.fromtimestamp(int(internal) / 1000, timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            pass
    if date_header:
        try:
            parsed = parsedate_to_datetime(date_header)
            return parsed.isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    return ""


def _importance_score(message: dict[str, Any]) -> tuple[int, str]:
    score = int(bool(message.get("is_important"))) * 4
    score += int(bool(message.get("is_unread"))) * 2
    text = f"{message.get('sender', '')} {message.get('subject', '')}".casefold()
    if any(word in text for word in ("urgent", "deadline", "professor", "action required")):
        score += 1
    return score, str(message.get("received_at", ""))


def _extract_body(payload: dict[str, Any]) -> str:
    plain: list[str] = []
    html: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = str(part.get("mimeType", ""))
        body = part.get("body") if isinstance(part.get("body"), dict) else {}
        data = body.get("data")
        if data and mime in {"text/plain", "text/html"}:
            decoded = _decode_base64url(str(data))
            (plain if mime == "text/plain" else html).append(decoded)
        for child in part.get("parts", []):
            if isinstance(child, dict):
                walk(child)

    walk(payload)
    if plain:
        return "\n".join(plain).strip()
    if html:
        import re

        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", "\n".join(html))).strip()
    return ""


def _decode_base64url(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""
