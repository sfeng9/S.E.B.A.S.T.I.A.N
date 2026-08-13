from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.assistant.voice_assistant import VoiceAssistant
from voice_assistant.audio.devices import find_input_device, find_output_device
from voice_assistant.audio.mic_test import record_mic_test
from voice_assistant.audio.speaker_test import play_wav_file
from voice_assistant.config import PROJECT_ROOT, load_assistant_config, load_device_config
from voice_assistant.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local push-to-talk assistant turn.")
    parser.add_argument("--audio", type=Path, help="Use an existing audio file instead of recording.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Microphone recording duration.")
    parser.add_argument("--no-play", action="store_true", help="Generate the response without playing it.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    assistant_config = load_assistant_config()
    device_config = load_device_config()

    print("Preparing Faster Whisper and Piper...")
    try:
        assistant = VoiceAssistant(assistant_config)
    except (RuntimeError, ValueError) as exc:
        print(f"Result: startup failed: {exc}")
        return 1

    input_wav = args.audio
    if input_wav is None:
        input_wav = PROJECT_ROOT / "outputs" / "assistant_input.wav"
        try:
            microphone = find_input_device(
                device_config.microphone.device_id,
                device_config.microphone.name_query,
            )
            print(f"Microphone: [{microphone.id}] {microphone.name} ({microphone.host_api})")
            input("Press Enter when you are ready to speak...")
            print(f"Listening for {args.seconds:.1f} seconds...")
            recording = record_mic_test(
                device=microphone,
                seconds=args.seconds,
                sample_rate=device_config.microphone.sample_rate,
                channels=device_config.microphone.channels,
                wav_path=input_wav,
            )
        except (EOFError, KeyboardInterrupt):
            print("Result: cancelled.")
            return 130
        except Exception as exc:
            print(f"Result: recording failed: {exc}")
            return 1

        if not recording.likely_receiving_audio:
            print("Result: microphone level was too low for a reliable request.")
            return 2

    response_wav = PROJECT_ROOT / "outputs" / "assistant_response.wav"
    print("Transcribing and thinking locally...")
    try:
        turn = assistant.process_audio(input_wav, response_wav)
    except RuntimeError as exc:
        print(f"Result: assistant turn failed: {exc}")
        return 1

    print(f"You: {turn.transcription.text}")
    print(f"Assistant: {turn.reply.text}")
    print(
        f"Timing: STT {turn.transcription.elapsed_seconds:.2f}s, "
        f"LLM {turn.reply.total_seconds:.2f}s, "
        f"TTS {turn.synthesis.elapsed_seconds:.2f}s, "
        f"turn {turn.total_seconds:.2f}s"
    )

    if not args.no_play:
        try:
            speaker = find_output_device(
                device_config.speaker.device_id,
                device_config.speaker.name_query,
            )
            print(f"Speaker: [{speaker.id}] {speaker.name} ({speaker.host_api})")
            play_wav_file(
                device=speaker,
                wav_path=response_wav,
                sample_rate=device_config.speaker.sample_rate,
                channels=device_config.speaker.channels,
            )
        except Exception as exc:
            print(f"Result: response generated, but playback failed: {exc}")
            return 1

    print("Result: complete local voice turn succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
