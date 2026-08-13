from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

from voice_assistant.audio.devices import AudioDevice


@dataclass(frozen=True)
class MicTestResult:
    device: AudioDevice
    seconds: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    likely_receiving_audio: bool
    wav_path: Path | None


def record_mic_test(
    device: AudioDevice,
    seconds: float,
    sample_rate: int,
    channels: int,
    wav_path: Path | None = None,
) -> MicTestResult:
    frames = int(seconds * sample_rate)
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device.id,
    )
    sd.wait()

    rms = float(np.sqrt(np.mean(np.square(recording))))
    peak = float(np.max(np.abs(recording)))
    likely_receiving_audio = peak > 0.02 or rms > 0.005

    if wav_path is not None:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(wav_path, recording, sample_rate)

    return MicTestResult(
        device=device,
        seconds=seconds,
        sample_rate=sample_rate,
        channels=channels,
        rms=rms,
        peak=peak,
        likely_receiving_audio=likely_receiving_audio,
        wav_path=wav_path,
    )


def _write_wav(path: Path, recording: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(recording, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1 if pcm16.ndim == 1 else pcm16.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
