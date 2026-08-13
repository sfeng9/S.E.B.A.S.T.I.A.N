from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.calendar_reminder_sync import (
    CalendarReminderSynchronizer,
)
from voice_assistant.integrations.reminders import ReminderStore
from voice_assistant.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize upcoming Google Calendar events into local reminders."
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    config = load_assistant_config()
    if not config.reminders.calendar_sync_enabled:
        print("Result: automatic Calendar reminders are disabled in configuration.")
        return 2
    if not config.home_location.timezone:
        print("Result: configure home_location.timezone before synchronizing reminders.")
        return 2
    if not config.google.calendar_token_path.exists():
        print("Result: authorize Google Calendar before synchronizing reminders.")
        return 2

    store = ReminderStore(config.reminders.database_path)
    try:
        result = CalendarReminderSynchronizer(config, store).sync_once()
    except Exception as exc:
        print(f"Result: Calendar reminder synchronization failed: {exc}")
        return 1

    print(f"Events checked: {result.events_seen}")
    print(f"Timed reminders synchronized: {result.reminders_synchronized}")
    print(f"All-day or unusable events skipped: {result.events_skipped}")
    print(f"Stale reminders cancelled: {result.reminders_cancelled}")
    print("Result: Calendar reminder synchronization succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
