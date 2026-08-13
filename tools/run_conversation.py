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
    parser = argparse.ArgumentParser(description="Run a local push-to-talk conversation.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Recording duration per turn.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    assistant_config = load_assistant_config()
    device_config = load_device_config()

    print("Preparing local conversation models...")
    try:
        assistant = VoiceAssistant(assistant_config)
        microphone = find_input_device(
            device_config.microphone.device_id,
            device_config.microphone.name_query,
        )
        speaker = find_output_device(
            device_config.speaker.device_id,
            device_config.speaker.name_query,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Result: startup failed: {exc}")
        return 1

    print(f"Microphone: [{microphone.id}] {microphone.name} ({microphone.host_api})")
    print(f"Speaker: [{speaker.id}] {speaker.name} ({speaker.host_api})")
    print("Press Enter to speak, type 'clear' to reset context, or type 'q' to quit.")

    input_wav = PROJECT_ROOT / "outputs" / "conversation_input.wav"
    response_wav = PROJECT_ROOT / "outputs" / "conversation_response.wav"

    while True:
        try:
            command = input("\nReady> ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation ended.")
            return 0

        if command in {"q", "quit", "exit"}:
            print("Conversation ended.")
            return 0
        if command == "clear":
            assistant.clear_conversation()
            print("Conversation context cleared.")
            continue
        if command:
            print("Type only Enter, 'clear', or 'q'.")
            continue

        print(f"Listening for {args.seconds:.1f} seconds...")
        try:
            recording = record_mic_test(
                device=microphone,
                seconds=args.seconds,
                sample_rate=device_config.microphone.sample_rate,
                channels=device_config.microphone.channels,
                wav_path=input_wav,
            )
        except Exception as exc:
            print(f"Recording failed: {exc}")
            continue

        if not recording.likely_receiving_audio:
            print("Microphone level was too low. Try that turn again.")
            continue

        print("Processing locally...")
        try:
            turn = assistant.process_audio(input_wav, response_wav)
        except RuntimeError as exc:
            print(f"Assistant turn failed: {exc}")
            continue

        print(f"You: {turn.transcription.text}")
        print(f"Assistant: {turn.reply.text}")
        print(
            f"Timing: STT {turn.transcription.elapsed_seconds:.2f}s, "
            f"LLM {turn.reply.total_seconds:.2f}s, "
            f"TTS {turn.synthesis.elapsed_seconds:.2f}s, "
            f"turn {turn.total_seconds:.2f}s; "
            f"context {assistant.conversation_turns} turns, "
            f"~{assistant.context_tokens}/{assistant.max_context_tokens} tokens"
        )
        try:
            play_wav_file(
                device=speaker,
                wav_path=response_wav,
                sample_rate=device_config.speaker.sample_rate,
                channels=device_config.speaker.channels,
            )
        except Exception as exc:
            print(f"Response generated, but playback failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
