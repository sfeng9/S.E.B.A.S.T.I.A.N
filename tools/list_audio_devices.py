from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.audio.devices import list_devices


def main() -> int:
    devices = list_devices()
    if not devices:
        print("No audio devices found.")
        return 1

    print("Audio devices:")
    for device in devices:
        direction = []
        if device.is_input:
            direction.append("input")
        if device.is_output:
            direction.append("output")
        direction_text = "/".join(direction) if direction else "unavailable"
        print(
            f"[{device.id:>2}] {direction_text:<12} {device.name} "
            f"| {device.host_api} | in={device.max_input_channels} "
            f"out={device.max_output_channels} | default_rate={device.default_sample_rate:.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
