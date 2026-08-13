from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from voice_assistant.audio.wake_word import (
    WakeWordPrediction,
    listen_for_wake_word,
)


class FakeInputStream:
    def __init__(self, frames: int) -> None:
        self.frames = frames

    def __enter__(self) -> FakeInputStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        if frames != self.frames:
            raise AssertionError("Unexpected wake-word frame size.")
        return np.zeros((frames, 1), dtype=np.int16), False


class FakeDetector:
    def __init__(self) -> None:
        self.config = SimpleNamespace(frame_ms=80)

    def reset(self) -> None:
        return None

    def predict(self, samples: np.ndarray) -> WakeWordPrediction:
        return WakeWordPrediction(detected=True, score=0.9, scores={"test": 0.9})


class QuietDetector(FakeDetector):
    def predict(self, samples: np.ndarray) -> WakeWordPrediction:
        return WakeWordPrediction(detected=False, score=0.01, scores={"test": 0.01})


class WakeWordMaintenanceTests(unittest.TestCase):
    def test_idle_maintenance_runs_without_interrupting_detection(self) -> None:
        detector = FakeDetector()
        callback = Mock()
        source_rate = 16_000
        frames = source_rate * detector.config.frame_ms // 1000

        with (
            patch(
                "voice_assistant.audio.wake_word.sd.InputStream",
                return_value=FakeInputStream(frames),
            ),
            patch(
                "voice_assistant.audio.wake_word.time.monotonic",
                side_effect=[0.0, 1.0, 1.0],
            ),
        ):
            result = listen_for_wake_word(
                detector=detector,  # type: ignore[arg-type]
                device_id=1,
                source_rate=source_rate,
                maintenance_callback=callback,
                maintenance_interval_seconds=0.25,
            )

        callback.assert_called_once_with()
        self.assertIsNotNone(result.prediction)
        self.assertEqual(result.prediction.score, 0.9)

    def test_due_reminder_interrupts_idle_listening(self) -> None:
        detector = QuietDetector()
        source_rate = 16_000
        frames = source_rate * detector.config.frame_ms // 1000
        with (
            patch(
                "voice_assistant.audio.wake_word.sd.InputStream",
                return_value=FakeInputStream(frames),
            ),
            patch(
                "voice_assistant.audio.wake_word.time.monotonic",
                side_effect=[0.0, 1.0, 1.0],
            ),
        ):
            result = listen_for_wake_word(
                detector=detector,  # type: ignore[arg-type]
                device_id=1,
                source_rate=source_rate,
                maintenance_callback=lambda: True,
                maintenance_interval_seconds=0.25,
            )

        self.assertTrue(result.maintenance_interrupt)
        self.assertIsNone(result.prediction)


if __name__ == "__main__":
    unittest.main()
