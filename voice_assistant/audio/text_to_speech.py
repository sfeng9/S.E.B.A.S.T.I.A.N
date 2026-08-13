from __future__ import annotations

import time
import wave
import logging
from dataclasses import dataclass
from pathlib import Path

from voice_assistant.config import TextToSpeechConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisResult:
    wav_path: Path
    voice: str
    audio_seconds: float
    elapsed_seconds: float


class PiperSynthesizer:
    def __init__(self, config: TextToSpeechConfig) -> None:
        if config.provider.casefold() != "piper":
            raise ValueError(f"Unsupported TTS provider: {config.provider!r}")
        if not config.model_path.is_file():
            raise RuntimeError(
                f"Piper voice model is missing: {config.model_path}. "
                f"Download the {config.voice} voice first."
            )

        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts is not installed. Install the project requirements first."
            ) from exc

        self.config = config
        self._synthesis_config = SynthesisConfig(
            volume=config.volume,
            length_scale=config.length_scale,
        )
        try:
            self._voice = PiperVoice.load(config.model_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load Piper voice {config.voice!r}: {exc}") from exc
        logger.info("Piper voice %s loaded on CPU.", config.voice)

    def synthesize(self, text: str, wav_path: Path) -> SynthesisResult:
        text = text.strip()
        if not text:
            raise RuntimeError("Cannot synthesize empty text.")

        wav_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                self._voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=self._synthesis_config,
                )
        except Exception as exc:
            raise RuntimeError(f"Piper speech synthesis failed: {exc}") from exc
        elapsed = time.perf_counter() - started

        with wave.open(str(wav_path), "rb") as wav_file:
            audio_seconds = wav_file.getnframes() / wav_file.getframerate()

        return SynthesisResult(
            wav_path=wav_path,
            voice=self.config.voice,
            audio_seconds=audio_seconds,
            elapsed_seconds=elapsed,
        )
