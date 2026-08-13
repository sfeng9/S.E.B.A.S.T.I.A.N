from __future__ import annotations

import math
import time
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from voice_assistant.config import WakeWordConfig


WAKE_WORD_SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeWordPrediction:
    detected: bool
    score: float
    scores: dict[str, float]


@dataclass(frozen=True)
class WakeWordListenResult:
    prediction: WakeWordPrediction | None
    highest_score: float
    overflow_count: int
    maintenance_interrupt: bool = False


def resample_for_wake_word(samples: np.ndarray, source_rate: int) -> np.ndarray:
    mono = np.asarray(samples).reshape(-1)
    if mono.dtype != np.int16:
        mono = np.clip(mono, -32768, 32767).astype(np.int16)
    if source_rate == WAKE_WORD_SAMPLE_RATE:
        return mono

    divisor = math.gcd(source_rate, WAKE_WORD_SAMPLE_RATE)
    converted = resample_poly(
        mono.astype(np.float32),
        WAKE_WORD_SAMPLE_RATE // divisor,
        source_rate // divisor,
    )
    return np.clip(np.rint(converted), -32768, 32767).astype(np.int16)


class WakeWordDetector:
    def __init__(self, config: WakeWordConfig) -> None:
        self.config = config
        self._validate_config()

        import openwakeword
        from openwakeword.model import Model

        resource_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
        required_runtime_models = [
            resource_dir / "melspectrogram.onnx",
            resource_dir / "embedding_model.onnx",
        ]
        if config.vad_threshold > 0:
            required_runtime_models.append(resource_dir / "silero_vad.onnx")
        missing = [path for path in required_runtime_models if not path.exists()]
        if missing:
            raise RuntimeError(
                "openWakeWord runtime models are missing. Run "
                ".\\.venv\\Scripts\\python.exe .\\tools\\setup_wake_word.py"
            )

        self._model_name = config.model_path.stem
        self._model = Model(
            wakeword_models=[str(config.model_path)],
            vad_threshold=config.vad_threshold,
            inference_framework="onnx",
        )
        logger.info(
            "Wake-word model %s loaded with ONNX Runtime on CPU.",
            config.model_path.name,
        )

    def predict(self, samples_16khz: np.ndarray) -> WakeWordPrediction:
        samples = np.asarray(samples_16khz, dtype=np.int16).reshape(-1)
        scores = {
            name: float(score)
            for name, score in self._model.predict(samples).items()
        }
        score = scores.get(self._model_name, max(scores.values(), default=0.0))
        return WakeWordPrediction(
            detected=score >= self.config.threshold,
            score=score,
            scores=scores,
        )

    def reset(self) -> None:
        self._model.reset()

    def _validate_config(self) -> None:
        if not self.config.model_path.exists():
            raise RuntimeError(
                f"Wake-word model not found: {self.config.model_path}\n"
                "Train Sebastian using docs/train-sebastian.md, then place the ONNX "
                "file at that path."
            )
        if self.config.model_path.suffix.casefold() != ".onnx":
            raise ValueError("Windows wake-word models must use the .onnx format.")
        if not 0 < self.config.threshold <= 1:
            raise ValueError("Wake-word threshold must be greater than 0 and at most 1.")
        if self.config.frame_ms <= 0 or self.config.frame_ms % 80 != 0:
            raise ValueError("Wake-word frame_ms must be a positive multiple of 80.")


def listen_for_wake_word(
    detector: WakeWordDetector,
    device_id: int,
    source_rate: int,
    timeout_seconds: float | None = None,
    maintenance_callback: Callable[[], bool | None] | None = None,
    maintenance_interval_seconds: float = 5.0,
) -> WakeWordListenResult:
    source_frames = int(source_rate * detector.config.frame_ms / 1000)
    deadline = (
        None
        if timeout_seconds is None or timeout_seconds <= 0
        else time.monotonic() + timeout_seconds
    )
    highest_score = 0.0
    overflow_count = 0
    detector.reset()
    next_maintenance = time.monotonic() + max(0.25, maintenance_interval_seconds)

    with sd.InputStream(
        samplerate=source_rate,
        blocksize=source_frames,
        device=device_id,
        channels=1,
        dtype="int16",
    ) as stream:
        while deadline is None or time.monotonic() < deadline:
            audio, overflowed = stream.read(source_frames)
            overflow_count += int(overflowed)
            samples = resample_for_wake_word(audio[:, 0], source_rate)
            prediction = detector.predict(samples)
            highest_score = max(highest_score, prediction.score)
            interrupted = False
            if maintenance_callback is not None and time.monotonic() >= next_maintenance:
                interrupted = bool(maintenance_callback())
                next_maintenance = time.monotonic() + max(
                    0.25, maintenance_interval_seconds
                )
            if prediction.detected:
                return WakeWordListenResult(
                    prediction=prediction,
                    highest_score=highest_score,
                    overflow_count=overflow_count,
                )
            if interrupted:
                return WakeWordListenResult(
                    prediction=None,
                    highest_score=highest_score,
                    overflow_count=overflow_count,
                    maintenance_interrupt=True,
                )

    return WakeWordListenResult(
        prediction=None,
        highest_score=highest_score,
        overflow_count=overflow_count,
    )
