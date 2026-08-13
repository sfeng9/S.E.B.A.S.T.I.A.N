from __future__ import annotations

import argparse
import json
import shutil
# This diagnostic runs nvidia-smi with fixed arguments and no shell.
import subprocess  # nosec B404
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.assistant.llm import OllamaClient
from voice_assistant.audio.speech_to_text import FasterWhisperTranscriber
from voice_assistant.config import load_assistant_config
from voice_assistant.http_utils import validated_http_url
from voice_assistant.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Ollama and Faster Whisper GPU resource lifetimes."
    )
    parser.add_argument(
        "--keep-alive",
        default="5s",
        help="Temporary Ollama keep-alive used by this test.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=8.0,
        help="How long to wait for the temporary Ollama keep-alive to expire.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional WAV to test Faster Whisper load, unload, and reload state.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def gpu_memory() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "unavailable (nvidia-smi was not found)"
    try:
        result = subprocess.run(  # nosec B603
            [
                executable,
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"
    values = result.stdout.strip().split(",")
    if len(values) != 3:
        return result.stdout.strip()
    used, total, utilization = (value.strip() for value in values)
    return f"{used} MiB / {total} MiB, GPU utilization {utilization}%"


def running_models(base_url: str) -> list[dict[str, Any]]:
    url = f"{validated_http_url(base_url)}/api/ps"
    # The destination was validated as HTTP(S) immediately above.
    with request.urlopen(url, timeout=10.0) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models", [])
    return models if isinstance(models, list) else []


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    config = load_assistant_config()
    temporary_llm = replace(config.llm, keep_alive=args.keep_alive)

    print(f"GPU before Ollama request: {gpu_memory()}")
    client = OllamaClient(temporary_llm)
    reply = client.chat("Reply with only the word ready.")
    print(f"Ollama reply: {reply.text}")
    active = running_models(config.llm.base_url)
    print(f"Ollama active models: {', '.join(model.get('name', '?') for model in active) or '(none)'}")
    print(f"GPU with Ollama active: {gpu_memory()}")

    print(
        f"Waiting {args.wait_seconds:.1f}s for keep_alive={args.keep_alive} to expire..."
    )
    time.sleep(args.wait_seconds)
    after_wait = running_models(config.llm.base_url)
    print(
        "Ollama models after wait: "
        f"{', '.join(model.get('name', '?') for model in after_wait) or '(none)'}"
    )
    print(f"GPU after Ollama idle unload: {gpu_memory()}")

    if args.audio is not None:
        transcriber = FasterWhisperTranscriber(config.speech_to_text, lazy=True)
        print(f"Faster Whisper initially loaded: {transcriber.is_loaded}")
        result = transcriber.transcribe(args.audio)
        print(f"Transcript: {result.text}")
        print(f"GPU with Faster Whisper active: {gpu_memory()}")
        unloaded = transcriber.unload()
        time.sleep(1.0)
        print(f"Faster Whisper unloaded: {unloaded and not transcriber.is_loaded}")
        print(f"GPU after Faster Whisper unload: {gpu_memory()}")

    configured_model = config.llm.model.casefold()
    still_loaded = any(
        str(model.get("name", "")).casefold() == configured_model
        for model in after_wait
    )
    if still_loaded:
        print(
            "Result: configured Ollama model is still active. Another client may "
            "have refreshed it, or the wait was too short."
        )
        return 1
    print("Result: resource lifetime checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
