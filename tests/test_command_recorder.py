from __future__ import annotations

import tempfile
import unittest
import wave
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voice_assistant.audio.command_recorder import record_command_until_silence
from voice_assistant.audio.devices import AudioDevice
from voice_assistant.config import CommandRecordingConfig


SAMPLE_RATE = 48000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class FakeInputStream:
    def __init__(self, frames: Iterator[np.ndarray]) -> None:
        self.frames = frames

    def __enter__(self) -> FakeInputStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        if frames != FRAME_SAMPLES:
            raise AssertionError(f"Unexpected frame size: {frames}")
        return next(self.frames), False


class FakeVad:
    def __init__(self, decisions: Iterator[bool]) -> None:
        self.decisions = decisions

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        if len(frame) != FRAME_SAMPLES * 2 or sample_rate != SAMPLE_RATE:
            raise AssertionError("Unexpected VAD frame format")
        return next(self.decisions)


class CommandRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = AudioDevice(
            id=15,
            name="Test microphone",
            host_api="Windows WASAPI",
            max_input_channels=1,
            max_output_channels=0,
            default_sample_rate=SAMPLE_RATE,
        )
        self.config = CommandRecordingConfig(
            max_seconds=1.0,
            speech_start_timeout=0.3,
            silence_seconds=0.09,
            pre_roll_ms=60,
            frame_ms=FRAME_MS,
            vad_mode=2,
        )

    def test_stops_after_trailing_silence_and_keeps_pre_roll(self) -> None:
        decisions = iter([False, True, True, False, False, False])
        frames = iter(
            [np.full((FRAME_SAMPLES, 1), index * 100, dtype=np.int16) for index in range(6)]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "request.wav"
            with (
                patch(
                    "voice_assistant.audio.command_recorder.sd.InputStream",
                    return_value=FakeInputStream(frames),
                ),
                patch(
                    "voice_assistant.audio.command_recorder.webrtcvad.Vad",
                    return_value=FakeVad(decisions),
                ),
            ):
                result = record_command_until_silence(
                    device=self.device,
                    sample_rate=SAMPLE_RATE,
                    channels=1,
                    config=self.config,
                    wav_path=wav_path,
                )

            self.assertTrue(result.speech_detected)
            self.assertEqual(result.stop_reason, "silence")
            self.assertAlmostEqual(result.seconds, 0.18)
            self.assertEqual(result.wav_path, wav_path)
            with wave.open(str(wav_path), "rb") as wav:
                self.assertEqual(wav.getframerate(), SAMPLE_RATE)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getnframes(), FRAME_SAMPLES * 6)

    def test_returns_without_file_when_speech_never_starts(self) -> None:
        config = replace(self.config, speech_start_timeout=0.09)
        decisions = iter([False, False, False])
        frames = iter([np.zeros((FRAME_SAMPLES, 1), dtype=np.int16) for _ in range(3)])

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "request.wav"
            with (
                patch(
                    "voice_assistant.audio.command_recorder.sd.InputStream",
                    return_value=FakeInputStream(frames),
                ),
                patch(
                    "voice_assistant.audio.command_recorder.webrtcvad.Vad",
                    return_value=FakeVad(decisions),
                ),
            ):
                result = record_command_until_silence(
                    device=self.device,
                    sample_rate=SAMPLE_RATE,
                    channels=1,
                    config=config,
                    wav_path=wav_path,
                )

            self.assertFalse(result.speech_detected)
            self.assertEqual(result.stop_reason, "speech start timeout")
            self.assertIsNone(result.wav_path)
            self.assertFalse(wav_path.exists())


if __name__ == "__main__":
    unittest.main()
