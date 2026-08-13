from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import find_input_device
from voice_assistant.audio.mic_test import record_mic_test
from voice_assistant.audio.speech_to_text import FasterWhisperTranscriber
from voice_assistant.config import PROJECT_ROOT, load_assistant_config, load_device_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test local microphone-to-text transcription.")
    parser.add_argument("--audio", type=Path, help="Transcribe an existing audio file instead of recording.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Microphone recording duration.")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Override inference device.")
    parser.add_argument("--model", help="Override the configured Whisper model.")
    parser.add_argument("--compute-type", help="Override inference precision, for example int8_float16.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assistant_config = load_assistant_config()
    stt_config = assistant_config.speech_to_text
    if args.device:
        compute_type = "float16" if args.device == "cuda" else "int8"
        stt_config = replace(stt_config, device=args.device, compute_type=compute_type)
    if args.model:
        stt_config = replace(stt_config, model=args.model)
    if args.compute_type:
        stt_config = replace(stt_config, compute_type=args.compute_type)

    audio_path = args.audio
    if audio_path is None:
        audio_path = PROJECT_ROOT / "outputs" / "stt_test.wav"
        device_config = load_device_config().microphone
        try:
            microphone = find_input_device(device_config.device_id, device_config.name_query)
            print(f"Recording from [{microphone.id}] {microphone.name} ({microphone.host_api})")
            print(f"Speak now for {args.seconds:.1f} seconds...")
            recording = record_mic_test(
                device=microphone,
                seconds=args.seconds,
                sample_rate=device_config.sample_rate,
                channels=device_config.channels,
                wav_path=audio_path,
            )
        except Exception as exc:
            print(f"Result: recording failed: {exc}")
            return 1

        if not recording.likely_receiving_audio:
            print("Result: recording level was too low to transcribe reliably.")
            return 2

    print(
        f"Loading Faster Whisper {stt_config.model} on "
        f"{stt_config.device}/{stt_config.compute_type}..."
    )
    try:
        transcriber = FasterWhisperTranscriber(stt_config)
        result = transcriber.transcribe(audio_path)
    except RuntimeError as exc:
        print(f"Result: transcription failed: {exc}")
        return 1

    print(f"Transcript: {result.text or '(no speech detected)'}")
    print(f"Language: {result.language} ({result.language_probability:.1%})")
    print(f"Timing: {result.audio_seconds:.2f}s audio in {result.elapsed_seconds:.2f}s")
    print("Result: local speech-to-text succeeded.")
    return 0 if result.text else 2


if __name__ == "__main__":
    raise SystemExit(main())
