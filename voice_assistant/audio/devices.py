from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sounddevice as sd


@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


def list_devices() -> list[AudioDevice]:
    host_apis = sd.query_hostapis()
    devices = sd.query_devices()

    results: list[AudioDevice] = []
    for idx, device in enumerate(devices):
        host_api = host_apis[device["hostapi"]]["name"]
        results.append(
            AudioDevice(
                id=idx,
                name=str(device["name"]),
                host_api=str(host_api),
                max_input_channels=int(device["max_input_channels"]),
                max_output_channels=int(device["max_output_channels"]),
                default_sample_rate=float(device["default_samplerate"]),
            )
        )

    return results


def find_input_device(device_id: int | None, name_query: str | None) -> AudioDevice:
    return _find_device(device_id, name_query, want_input=True)


def find_output_device(device_id: int | None, name_query: str | None) -> AudioDevice:
    return _find_device(device_id, name_query, want_input=False)


def _find_device(device_id: int | None, name_query: str | None, want_input: bool) -> AudioDevice:
    devices = list_devices()
    candidates = [d for d in devices if d.is_input] if want_input else [d for d in devices if d.is_output]
    kind = "input" if want_input else "output"

    if device_id is not None:
        for device in candidates:
            if device.id == device_id:
                return device
        raise RuntimeError(f"No {kind} audio device found with id {device_id}.")

    if name_query:
        query = name_query.casefold()
        matches = [device for device in candidates if query in device.name.casefold()]
        if matches:
            return sorted(matches, key=lambda d: (d.host_api != "Windows WASAPI", d.id))[0]
        available = _format_devices(candidates)
        raise RuntimeError(f"No {kind} audio device matched {name_query!r}.\n\nAvailable {kind} devices:\n{available}")

    default_input, default_output = sd.default.device
    default_id = default_input if want_input else default_output
    for device in candidates:
        if device.id == default_id:
            return device

    raise RuntimeError(f"No default {kind} audio device is available.")


def _format_devices(devices: Iterable[AudioDevice]) -> str:
    lines = []
    for device in devices:
        lines.append(
            f"  [{device.id}] {device.name} | {device.host_api} | "
            f"in={device.max_input_channels} out={device.max_output_channels} "
            f"rate={device.default_sample_rate:.0f}"
        )
    return "\n".join(lines) if lines else "  (none)"
