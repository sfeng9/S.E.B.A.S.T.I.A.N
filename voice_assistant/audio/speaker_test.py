from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from voice_assistant.audio.devices import AudioDevice


def play_test_tone(
    device: AudioDevice,
    seconds: float,
    sample_rate: int,
    channels: int,
    frequency: float = 440.0,
    volume: float = 0.15,
) -> None:
    frames = int(seconds * sample_rate)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    wave = np.sin(2 * math.pi * frequency * t, dtype=np.float32) * volume

    if channels > 1:
        wave = np.repeat(wave[:, np.newaxis], channels, axis=1)

    sd.play(wave, samplerate=sample_rate, device=device.id)
    sd.wait()


def play_wav_file(
    device: AudioDevice,
    wav_path: Path,
    sample_rate: int,
    channels: int,
) -> None:
    with wave.open(str(wav_path), "rb") as wav_file:
        source_channels = wav_file.getnchannels()
        source_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, found {sample_width * 8}-bit audio.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio.reshape(-1, source_channels)
    if source_rate != sample_rate:
        audio = _resample(audio, source_rate, sample_rate)

    if audio.shape[1] == 1 and channels > 1:
        audio = np.repeat(audio, channels, axis=1)
    elif audio.shape[1] > 1 and channels == 1:
        audio = np.mean(audio, axis=1, keepdims=True)
    elif audio.shape[1] != channels:
        raise RuntimeError(
            f"Cannot map {audio.shape[1]} WAV channels to {channels} output channels."
        )

    sd.play(audio, samplerate=sample_rate, device=device.id)
    sd.wait()


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_frames = max(1, round(len(audio) * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, target_frames, endpoint=False)
    channels = [
        np.interp(target_positions, source_positions, audio[:, channel])
        for channel in range(audio.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)
