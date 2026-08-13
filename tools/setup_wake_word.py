from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwakeword.utils import download_models

from voice_assistant.config import load_assistant_config


def main() -> int:
    config = load_assistant_config().wake_word
    print("Downloading openWakeWord feature models...")
    download_models(model_names=["__runtime_only__"])
    print("Result: openWakeWord runtime is ready.")

    if config.model_path.exists():
        print(f"Sebastian model: {config.model_path}")
    else:
        print(f"Sebastian model still needed: {config.model_path}")
        print("Next: follow docs/train-sebastian.md and place sebastian.onnx there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
