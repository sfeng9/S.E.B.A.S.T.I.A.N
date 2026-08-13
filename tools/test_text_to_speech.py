from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import find_output_device
from voice_assistant.audio.speaker_test import play_wav_file
from voice_assistant.audio.text_to_speech import PiperSynthesizer
from voice_assistant.config import PROJECT_ROOT, load_assistant_config, load_device_config


DEFAULT_TEXT = "Good morning. Local speech synthesis is working through the configured speaker."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test local text-to-speech and speaker routing.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize.")
    parser.add_argument("--wav", type=Path, default=PROJECT_ROOT / "outputs" / "tts_test.wav")
    parser.add_argument("--no-play", action="store_true", help="Generate the WAV without playing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assistant_config = load_assistant_config()
    print(f"Loading Piper voice {assistant_config.text_to_speech.voice}...")
    try:
        synthesizer = PiperSynthesizer(assistant_config.text_to_speech)
        result = synthesizer.synthesize(args.text, args.wav)
    except RuntimeError as exc:
        print(f"Result: synthesis failed: {exc}")
        return 1

    print(f"WAV saved: {result.wav_path}")
    print(f"Timing: {result.audio_seconds:.2f}s audio generated in {result.elapsed_seconds:.2f}s")
    if args.no_play:
        print("Result: local text-to-speech synthesis succeeded.")
        return 0

    device_config = load_device_config().speaker
    try:
        speaker = find_output_device(device_config.device_id, device_config.name_query)
        print(f"Playing through [{speaker.id}] {speaker.name} ({speaker.host_api})")
        play_wav_file(
            device=speaker,
            wav_path=result.wav_path,
            sample_rate=device_config.sample_rate,
            channels=device_config.channels,
        )
    except Exception as exc:
        print(f"Result: playback failed: {exc}")
        return 1

    print("Result: local text-to-speech and configured playback succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
