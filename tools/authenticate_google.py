from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.google_auth import (
    CALENDAR_EVENTS_SCOPE,
    GMAIL_READONLY_SCOPE,
    GoogleIntegrationError,
    GoogleOAuthProvider,
)
from voice_assistant.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authorize Sebastian with Google OAuth.")
    parser.add_argument(
        "--service",
        choices=("gmail", "calendar", "all"),
        default="all",
        help="Authorize one service or both using separate least-privilege tokens.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    config = load_assistant_config()
    requested = ("gmail", "calendar") if args.service == "all" else (args.service,)
    providers = {
        "gmail": GoogleOAuthProvider(
            config.google.credentials_path,
            config.google.gmail_token_path,
            [GMAIL_READONLY_SCOPE],
        ),
        "calendar": GoogleOAuthProvider(
            config.google.credentials_path,
            config.google.calendar_token_path,
            [CALENDAR_EVENTS_SCOPE],
        ),
    }
    if not config.google.credentials_path.exists():
        print("Google OAuth client credentials are not configured.")
        print(f"Place the downloaded Desktop OAuth JSON at:\n{config.google.credentials_path}")
        return 2

    for service in requested:
        print(f"Authorizing {service}. Your browser will open for Google consent...")
        try:
            providers[service].get_credentials(interactive=True)
        except GoogleIntegrationError as exc:
            print(f"Authorization failed for {service}: {exc}")
            return 1
        print(f"Authorized {service}; token saved to {providers[service].token_path}")
    print("Google authorization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
