from __future__ import annotations

import argparse
import atexit
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.assistant.voice_assistant import VoiceAssistant
from voice_assistant.audio.command_recorder import record_command_until_silence
from voice_assistant.audio.devices import find_input_device, find_output_device
from voice_assistant.audio.speaker_test import play_test_tone, play_wav_file
from voice_assistant.audio.wake_word import WakeWordDetector, listen_for_wake_word
from voice_assistant.config import PROJECT_ROOT, load_assistant_config, load_device_config
from voice_assistant.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local assistant with continuous wake-word activation."
    )
    parser.add_argument(
        "--max-seconds",
        "--seconds",
        dest="max_seconds",
        type=float,
        help="Maximum request duration; defaults to command_recording.max_seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one completed assistant turn.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    assistant_config = load_assistant_config()
    device_config = load_device_config()
    wake_config = assistant_config.wake_word
    recording_config = assistant_config.command_recording
    max_seconds = (
        recording_config.max_seconds
        if args.max_seconds is None
        else args.max_seconds
    )

    print("Preparing lightweight assistant services and wake-word model...")
    try:
        assistant = VoiceAssistant(assistant_config)
        detector = WakeWordDetector(wake_config)
        microphone = find_input_device(
            device_config.microphone.device_id,
            device_config.microphone.name_query,
        )
        speaker = find_output_device(
            device_config.speaker.device_id,
            device_config.speaker.name_query,
        )
        assistant.start_background_services()
        atexit.register(assistant.close)
    except (RuntimeError, ValueError) as exc:
        print(f"Result: startup failed: {exc}")
        return 1

    input_wav = PROJECT_ROOT / "outputs" / "wake_assistant_input.wav"
    response_wav = PROJECT_ROOT / "outputs" / "wake_assistant_response.wav"

    print(f"Microphone: [{microphone.id}] {microphone.name} ({microphone.host_api})")
    print(f"Speaker: [{speaker.id}] {speaker.name} ({speaker.host_api})")
    print(f"Ready. Say '{wake_config.phrase}' or press Ctrl+C to quit.")

    while True:
        try:
            wake_result = listen_for_wake_word(
                detector=detector,
                device_id=microphone.id,
                source_rate=device_config.microphone.sample_rate,
                maintenance_callback=assistant.maintenance,
                maintenance_interval_seconds=(
                    assistant_config.resources.maintenance_interval_seconds
                ),
            )
        except KeyboardInterrupt:
            print("\nAssistant stopped.")
            return 0
        except Exception as exc:
            print(f"Wake-word listening failed: {exc}")
            return 1

        if wake_result.maintenance_interrupt:
            for reminder in assistant.take_due_reminders():
                reminder_wav = PROJECT_ROOT / "outputs" / f"reminder_{reminder.id}.wav"
                try:
                    assistant.synthesize_reminder(reminder, reminder_wav)
                    play_wav_file(
                        device=speaker,
                        wav_path=reminder_wav,
                        sample_rate=device_config.speaker.sample_rate,
                        channels=device_config.speaker.channels,
                    )
                    assistant.complete_reminder(reminder.id)
                except Exception as exc:
                    assistant.release_reminder(reminder.id)
                    print(f"Reminder {reminder.id} could not be played: {exc}")
            print(f"Ready. Say '{wake_config.phrase}'.")
            continue
        if wake_result.prediction is None:
            continue
        print(f"Wake word detected (score {wake_result.prediction.score:.3f}).")

        try:
            play_test_tone(
                device=speaker,
                seconds=0.12,
                sample_rate=device_config.speaker.sample_rate,
                channels=device_config.speaker.channels,
                frequency=880.0,
                volume=0.12,
            )
            print(f"Listening until you stop speaking (max {max_seconds:.1f}s)...")
            recording = record_command_until_silence(
                device=microphone,
                sample_rate=device_config.microphone.sample_rate,
                channels=device_config.microphone.channels,
                config=recording_config,
                wav_path=input_wav,
                max_seconds=max_seconds,
            )
        except KeyboardInterrupt:
            print("\nAssistant stopped.")
            return 0
        except Exception as exc:
            print(f"Recording failed: {exc}")
            time.sleep(0.25)
            continue

        if not recording.speech_detected:
            print("No request detected. Returning to wake-word listening.")
            continue

        print(
            f"Request captured in {recording.seconds:.2f}s "
            f"(stopped on {recording.stop_reason})."
        )
        if recording.overflow_count:
            print(
                f"Warning: microphone input overflowed "
                f"{recording.overflow_count} time(s)."
            )

        print("Processing locally...")
        try:
            turn = assistant.process_audio(input_wav, response_wav)
        except Exception as exc:
            print(f"Assistant turn failed: {exc}")
            continue

        print(f"You: {turn.transcription.text}")
        print(f"Assistant: {turn.reply.text}")
        if turn.reply.tool_calls:
            logging.getLogger(__name__).info(
                "Tools used this turn: %s",
                ", ".join(turn.reply.tool_calls),
            )
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

        if args.once:
            print("Result: wake-activated local voice turn succeeded.")
            return 0

        time.sleep(0.25)
        print(f"Ready. Say '{wake_config.phrase}'.")


if __name__ == "__main__":
    raise SystemExit(main())
