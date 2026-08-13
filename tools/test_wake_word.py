from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import find_input_device
from voice_assistant.audio.wake_word import WakeWordDetector, listen_for_wake_word
from voice_assistant.config import load_assistant_config, load_device_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for the configured wake word.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to listen; use 0 to listen until Ctrl+C.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assistant_config = load_assistant_config()
    device_config = load_device_config()
    wake_config = assistant_config.wake_word

    try:
        microphone = find_input_device(
            device_config.microphone.device_id,
            device_config.microphone.name_query,
        )
        detector = WakeWordDetector(wake_config)
    except (RuntimeError, ValueError) as exc:
        print(f"Result: wake-word startup failed: {exc}")
        return 1

    print(f"Microphone: [{microphone.id}] {microphone.name} ({microphone.host_api})")
    print(
        f"Listening for '{wake_config.phrase}' at threshold "
        f"{wake_config.threshold:.2f}. Press Ctrl+C to stop."
    )

    try:
        result = listen_for_wake_word(
            detector=detector,
            device_id=microphone.id,
            source_rate=device_config.microphone.sample_rate,
            timeout_seconds=args.timeout,
        )
    except KeyboardInterrupt:
        print("\nWake-word test stopped.")
        return 0
    except Exception as exc:
        print(f"Result: microphone stream failed: {exc}")
        return 1

    if result.overflow_count:
        print(f"Warning: microphone input overflowed {result.overflow_count} time(s).")
    if result.prediction is not None:
        print(
            f"Detected: {wake_config.phrase} "
            f"(score {result.prediction.score:.3f})"
        )
        return 0

    print(f"Result: no detection; highest score was {result.highest_score:.3f}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
