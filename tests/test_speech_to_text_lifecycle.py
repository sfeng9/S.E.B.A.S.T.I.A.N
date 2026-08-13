from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voice_assistant.audio.speech_to_text import FasterWhisperTranscriber
from voice_assistant.config import SpeechToTextConfig


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeCTranslateModel:
    def __init__(self) -> None:
        self.model_is_loaded = True
        self.load_calls = 0
        self.unload_calls = 0

    def load_model(self) -> None:
        self.model_is_loaded = True
        self.load_calls += 1

    def unload_model(self) -> None:
        self.model_is_loaded = False
        self.unload_calls += 1


class FakeWhisperModel:
    def __init__(self) -> None:
        self.model = FakeCTranslateModel()

    def transcribe(self, path: str, **kwargs: object) -> tuple[list[object], object]:
        segments = [SimpleNamespace(text="Test transcript")]
        info = SimpleNamespace(language="en", language_probability=1.0, duration=1.0)
        return segments, info


class SpeechToTextLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SpeechToTextConfig(
            model="small.en",
            device="cuda",
            compute_type="int8_float16",
            language="en",
            beam_size=1,
        )
        self.clock = FakeClock()

    def test_lazy_load_idle_unload_and_reload(self) -> None:
        backend = FakeWhisperModel()
        with patch.object(
            FasterWhisperTranscriber,
            "_create_model",
            return_value=backend,
        ) as create_model:
            transcriber = FasterWhisperTranscriber(
                self.config,
                lazy=True,
                clock=self.clock,
            )
            self.assertFalse(transcriber.is_loaded)
            create_model.assert_not_called()

            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = Path(temp_dir) / "input.wav"
                audio_path.touch()
                result = transcriber.transcribe(audio_path)
                self.assertEqual(result.text, "Test transcript")
                self.assertTrue(transcriber.is_loaded)
                create_model.assert_called_once()

                self.assertFalse(transcriber.unload_if_idle(10.0, now=9.9))
                self.assertTrue(transcriber.unload_if_idle(10.0, now=10.0))
                self.assertFalse(transcriber.is_loaded)
                self.assertEqual(backend.model.unload_calls, 1)

                transcriber.transcribe(audio_path)
                self.assertTrue(transcriber.is_loaded)
                self.assertEqual(backend.model.load_calls, 1)
                create_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
