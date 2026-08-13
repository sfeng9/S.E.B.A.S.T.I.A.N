from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import find_input_device
from voice_assistant.audio.mic_test import record_mic_test
from voice_assistant.config import PROJECT_ROOT, load_device_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and test the configured assistant microphone.")
    parser.add_argument("--mic", help="Input device name query, for example BlackShark or TONOR.")
    parser.add_argument("--device-id", type=int, help="Input device id from tools/list_audio_devices.py.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration.")
    parser.add_argument("--sample-rate", type=int, help="Recording sample rate.")
    parser.add_argument("--channels", type=int, help="Recording channels.")
    parser.add_argument("--wav", type=Path, default=PROJECT_ROOT / "outputs" / "mic_test.wav")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_device_config()
    mic_config = config.microphone

    try:
        device = find_input_device(
            device_id=args.device_id if args.device_id is not None else mic_config.device_id,
            name_query=args.mic if args.mic else mic_config.name_query,
        )
    except RuntimeError as exc:
        print(exc)
        return 1

    sample_rate = args.sample_rate or mic_config.sample_rate
    channels = args.channels or mic_config.channels

    print(f"Testing microphone [{device.id}] {device.name} ({device.host_api})")
    print(f"Recording {args.seconds:.1f}s at {sample_rate} Hz, {channels} channel(s). Speak now.")

    result = record_mic_test(
        device=device,
        seconds=args.seconds,
        sample_rate=sample_rate,
        channels=channels,
        wav_path=args.wav,
    )

    print(f"RMS level:  {result.rms:.6f}")
    print(f"Peak level: {result.peak:.6f}")
    print(f"WAV saved:  {result.wav_path}")

    if result.likely_receiving_audio:
        print("Result: microphone is receiving audio.")
        return 0

    print("Result: microphone was detected, but the sample level was very low.")
    print("Check Windows microphone privacy, input volume, mute controls, and the selected device id.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
