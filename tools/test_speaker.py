from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import find_output_device
from voice_assistant.audio.speaker_test import play_test_tone
from voice_assistant.config import load_device_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a short tone on the configured assistant speaker.")
    parser.add_argument("--speaker", help="Output device name query.")
    parser.add_argument("--device-id", type=int, help="Output device id from tools/list_audio_devices.py.")
    parser.add_argument("--seconds", type=float, default=1.0, help="Tone duration.")
    parser.add_argument("--sample-rate", type=int, help="Output sample rate.")
    parser.add_argument("--channels", type=int, help="Output channels.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the speaker device without playing audio.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_device_config()
    speaker_config = config.speaker

    try:
        device = find_output_device(
            device_id=args.device_id if args.device_id is not None else speaker_config.device_id,
            name_query=args.speaker if args.speaker else speaker_config.name_query,
        )
    except RuntimeError as exc:
        print(exc)
        return 1

    sample_rate = args.sample_rate or speaker_config.sample_rate
    channels = args.channels or speaker_config.channels

    print(f"Testing speaker [{device.id}] {device.name} ({device.host_api})")
    if args.dry_run:
        print(f"Dry run: selected output at {sample_rate} Hz, {channels} channel(s).")
        return 0

    play_test_tone(device=device, seconds=args.seconds, sample_rate=sample_rate, channels=channels)
    print("Result: speaker test tone completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
