from __future__ import annotations

import math
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad

from voice_assistant.audio.devices import AudioDevice
from voice_assistant.config import CommandRecordingConfig


WEBRTC_SAMPLE_RATES = {8000, 16000, 32000, 48000}
WEBRTC_FRAME_MS = {10, 20, 30}


@dataclass(frozen=True)
class CommandRecordingResult:
    device: AudioDevice
    seconds: float
    speech_detected: bool
    stop_reason: str
    rms: float
    peak: float
    overflow_count: int
    wav_path: Path | None


def record_command_until_silence(
    device: AudioDevice,
    sample_rate: int,
    channels: int,
    config: CommandRecordingConfig,
    wav_path: Path,
    max_seconds: float | None = None,
) -> CommandRecordingResult:
    _validate_settings(sample_rate, channels, config)
    duration_limit = config.max_seconds if max_seconds is None else max_seconds
    if duration_limit <= 0:
        raise ValueError("Command recording max_seconds must be greater than zero.")

    frame_samples = sample_rate * config.frame_ms // 1000
    pre_roll_frames = max(1, math.ceil(config.pre_roll_ms / config.frame_ms))
    start_timeout_frames = max(
        1,
        math.ceil(config.speech_start_timeout * 1000 / config.frame_ms),
    )
    silence_frames = max(
        1,
        math.ceil(config.silence_seconds * 1000 / config.frame_ms),
    )
    maximum_frames = max(1, math.ceil(duration_limit * 1000 / config.frame_ms))

    vad = webrtcvad.Vad(config.vad_mode)
    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
    captured: list[np.ndarray] = []
    speech_started = False
    trailing_silence = 0
    overflow_count = 0
    stop_reason = "maximum duration"

    with sd.InputStream(
        samplerate=sample_rate,
        blocksize=frame_samples,
        device=device.id,
        channels=channels,
        dtype="int16",
    ) as stream:
        for frame_number in range(maximum_frames):
            audio, overflowed = stream.read(frame_samples)
            overflow_count += int(overflowed)
            mono = np.ascontiguousarray(audio[:, 0])
            is_speech = vad.is_speech(mono.tobytes(), sample_rate)

            if not speech_started:
                pre_roll.append(mono.copy())
                if is_speech:
                    speech_started = True
                    captured.extend(pre_roll)
                    pre_roll.clear()
                elif frame_number + 1 >= start_timeout_frames:
                    stop_reason = "speech start timeout"
                    break
                continue

            captured.append(mono.copy())
            if is_speech:
                trailing_silence = 0
            else:
                trailing_silence += 1
                if trailing_silence >= silence_frames:
                    stop_reason = "silence"
                    break

    if not speech_started or not captured:
        return CommandRecordingResult(
            device=device,
            seconds=0.0,
            speech_detected=False,
            stop_reason=stop_reason,
            rms=0.0,
            peak=0.0,
            overflow_count=overflow_count,
            wav_path=None,
        )

    pcm16 = np.concatenate(captured)
    normalized = pcm16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    peak = float(np.max(np.abs(normalized)))
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm16_wav(wav_path, pcm16, sample_rate)

    return CommandRecordingResult(
        device=device,
        seconds=len(pcm16) / sample_rate,
        speech_detected=True,
        stop_reason=stop_reason,
        rms=rms,
        peak=peak,
        overflow_count=overflow_count,
        wav_path=wav_path,
    )


def _validate_settings(
    sample_rate: int,
    channels: int,
    config: CommandRecordingConfig,
) -> None:
    if sample_rate not in WEBRTC_SAMPLE_RATES:
        supported = ", ".join(str(rate) for rate in sorted(WEBRTC_SAMPLE_RATES))
        raise ValueError(f"WebRTC VAD sample rate must be one of: {supported}.")
    if channels != 1:
        raise ValueError("Command recording currently requires a mono microphone.")
    if config.frame_ms not in WEBRTC_FRAME_MS:
        raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30.")
    if config.vad_mode not in {0, 1, 2, 3}:
        raise ValueError("WebRTC VAD mode must be between 0 and 3.")
    if config.speech_start_timeout <= 0 or config.silence_seconds <= 0:
        raise ValueError("Speech and silence timeouts must be greater than zero.")
    if config.pre_roll_ms < 0:
        raise ValueError("Command recording pre_roll_ms cannot be negative.")


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype(np.int16, copy=False).tobytes())
