from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import find_output_device
from voice_assistant.audio.speaker_test import play_wav_file
from voice_assistant.audio.text_to_speech import PiperSynthesizer
from voice_assistant.config import PROJECT_ROOT, load_assistant_config, load_device_config
from voice_assistant.integrations.reminders import ReminderStore
from voice_assistant.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test persistent reminder firing and speech.")
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--text", default="test Sebastian's reminder system")
    parser.add_argument("--play", action="store_true", help="Play the spoken reminder.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    config = load_assistant_config()
    if not config.home_location.timezone:
        print("Configure home_location.timezone before testing reminders.")
        return 2
    due = datetime.now(timezone.utc) + timedelta(seconds=max(0.1, args.delay_seconds))
    store = ReminderStore(config.reminders.database_path)
    reminder = store.create(args.text, due, config.home_location.timezone)
    print(f"Created reminder {reminder.id}; reopening the database to verify persistence.")

    store = ReminderStore(config.reminders.database_path)
    wait_seconds = max(0.0, (due - datetime.now(timezone.utc)).total_seconds()) + 0.1
    time.sleep(wait_seconds)
    claimed = [item for item in store.claim_due(datetime.now(timezone.utc)) if item.id == reminder.id]
    if not claimed:
        print("Result: reminder was not claimed when due.")
        return 1

    wav_path = PROJECT_ROOT / "outputs" / f"reminder_test_{reminder.id}.wav"
    try:
        PiperSynthesizer(config.text_to_speech).synthesize(
            f"Reminder: {reminder.text}.", wav_path
        )
        if args.play:
            devices = load_device_config()
            speaker = find_output_device(
                devices.speaker.device_id, devices.speaker.name_query
            )
            play_wav_file(
                speaker,
                wav_path,
                devices.speaker.sample_rate,
                devices.speaker.channels,
            )
        store.mark_fired(reminder.id)
    except Exception as exc:
        store.release(reminder.id)
        print(f"Result: reminder speech failed and was returned to pending: {exc}")
        return 1

    print(f"WAV saved: {wav_path}")
    print("Result: reminder persisted, became due, synthesized, and was marked fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
