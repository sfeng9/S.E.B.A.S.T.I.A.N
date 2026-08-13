from __future__ import annotations

import os
import sys
import time
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_assistant.config import SpeechToTextConfig


_DLL_DIRECTORY_HANDLES: list[Any] = []
logger = logging.getLogger(__name__)


def _configure_windows_cuda_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    dll_directories: list[Path] = []
    for relative_path in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        dll_directory = site_packages / relative_path
        if dll_directory.is_dir():
            dll_directories.append(dll_directory)
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_directory)))

    if dll_directories:
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in dll_directories), existing_path]
        )


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    audio_seconds: float
    elapsed_seconds: float


class FasterWhisperTranscriber:
    def __init__(
        self,
        config: SpeechToTextConfig,
        *,
        lazy: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _configure_windows_cuda_dlls()
        self.config = config
        self._clock = clock
        self._model: Any | None = None
        self._last_used_at: float | None = None
        if not lazy:
            self.load()

    @property
    def is_loaded(self) -> bool:
        return bool(
            self._model is not None and self._model.model.model_is_loaded
        )

    def load(self) -> None:
        if self.is_loaded:
            return

        started = time.perf_counter()
        logger.info(
            "Loading Faster Whisper model %s on %s/%s.",
            self.config.model,
            self.config.device,
            self.config.compute_type,
        )
        if self._model is None:
            self._model = self._create_model()
        else:
            try:
                self._model.model.load_model()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not reload Faster Whisper model {self.config.model!r}: {exc}"
                ) from exc
        logger.info(
            "Faster Whisper model loaded in %.2fs.",
            time.perf_counter() - started,
        )

    def unload(self) -> bool:
        if not self.is_loaded:
            return False
        try:
            self._model.model.unload_model()
        except Exception as exc:
            logger.warning("Could not unload Faster Whisper model: %s", exc)
            return False
        logger.info(
            "Faster Whisper model unloaded from %s; wake listening remains active.",
            self.config.device,
        )
        return True

    def unload_if_idle(
        self,
        idle_seconds: float,
        now: float | None = None,
    ) -> bool:
        if not self.is_loaded or self._last_used_at is None:
            return False
        current = self._clock() if now is None else now
        if current - self._last_used_at < idle_seconds:
            return False
        return self.unload()

    def _create_model(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install the project requirements first."
            ) from exc

        try:
            return WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Faster Whisper model {self.config.model!r} on "
                f"{self.config.device}/{self.config.compute_type}: {exc}"
            ) from exc

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not audio_path.is_file():
            raise RuntimeError(f"Audio file does not exist: {audio_path}")

        self.load()
        started = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                str(audio_path),
                language=self.config.language,
                beam_size=self.config.beam_size,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise RuntimeError(f"Faster Whisper transcription failed: {exc}") from exc
        finally:
            self._last_used_at = self._clock()

        elapsed = time.perf_counter() - started
        return TranscriptionResult(
            text=text,
            language=str(info.language),
            language_probability=float(info.language_probability),
            audio_seconds=float(info.duration),
            elapsed_seconds=elapsed,
        )
