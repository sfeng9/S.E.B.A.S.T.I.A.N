from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.assistant.llm import OllamaClient
from voice_assistant.config import load_assistant_config


DEFAULT_PROMPT = (
    "I have a meeting at 10:15 and rain is expected this afternoon. "
    "What should I remember? Reply in one short sentence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the configured local Ollama model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to Ollama.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_assistant_config()
    client = OllamaClient(config.llm)

    print(f"Testing {config.llm.provider} model {config.llm.model}...")
    try:
        reply = client.chat(args.prompt)
    except RuntimeError as exc:
        print(f"Result: failed: {exc}")
        return 1

    print(f"Response: {reply.text}")
    print(f"Total time: {reply.total_seconds:.2f}s (load: {reply.load_seconds:.2f}s)")
    if reply.tokens_per_second is not None:
        print(f"Generation speed: {reply.tokens_per_second:.1f} tokens/s")
    print(f"Tokens: {reply.prompt_tokens} prompt, {reply.response_tokens} response")
    print("Result: local Ollama inference succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
