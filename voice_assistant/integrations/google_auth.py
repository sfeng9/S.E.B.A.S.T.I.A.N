from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"


class GoogleIntegrationError(RuntimeError):
    """A safe, user-facing boundary for Google setup and API failures."""


class GoogleCredentialsMissingError(GoogleIntegrationError):
    pass


class GoogleOAuthProvider:
    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        scopes: Sequence[str],
    ) -> None:
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = tuple(scopes)

    def get_credentials(self, interactive: bool = False) -> Any:
        if not self.credentials_path.exists():
            raise GoogleCredentialsMissingError(
                f"Google OAuth client file is missing: {self.credentials_path}"
            )

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise GoogleIntegrationError(
                "Google dependencies are not installed. Run pip install -r requirements.txt."
            ) from exc

        credentials = None
        if self.token_path.exists():
            try:
                credentials = Credentials.from_authorized_user_file(
                    str(self.token_path), list(self.scopes)
                )
            except (OSError, ValueError) as exc:
                logger.warning("Google token file could not be read: %s", self.token_path)
                if not interactive:
                    raise GoogleIntegrationError(
                        "The saved Google authorization is invalid. Re-authenticate Google."
                    ) from exc

        if credentials is not None and credentials.valid:
            return credentials

        if credentials is not None and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                self._save(credentials)
                logger.info("Google OAuth token refreshed for %s.", self.token_path.name)
                return credentials
            except Exception as exc:
                logger.warning(
                    "Google OAuth refresh failed for %s: %s",
                    self.token_path.name,
                    type(exc).__name__,
                )
                if not interactive:
                    raise GoogleIntegrationError(
                        "Google authorization expired and could not be refreshed."
                    ) from exc

        if not interactive:
            raise GoogleIntegrationError(
                f"Google is not authorized. Run tools/authenticate_google.py for {self.token_path.stem}."
            )

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path), list(self.scopes)
            )
            credentials = flow.run_local_server(port=0)
        except Exception as exc:
            logger.warning("Interactive Google OAuth failed: %s", type(exc).__name__)
            raise GoogleIntegrationError("Google authorization did not complete.") from exc
        self._save(credentials)
        logger.info("Google OAuth authorization saved for %s.", self.token_path.name)
        return credentials

    def _save(self, credentials: Any) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(credentials.to_json(), encoding="utf-8")


def build_google_service(
    api_name: str,
    api_version: str,
    auth: GoogleOAuthProvider,
) -> Any:
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise GoogleIntegrationError(
            "Google dependencies are not installed. Run pip install -r requirements.txt."
        ) from exc
    try:
        return build(
            api_name,
            api_version,
            credentials=auth.get_credentials(interactive=False),
            cache_discovery=False,
        )
    except GoogleIntegrationError:
        raise
    except Exception as exc:
        logger.warning("Google %s client creation failed: %s", api_name, type(exc).__name__)
        raise GoogleIntegrationError(f"Could not connect to Google {api_name}.") from exc
