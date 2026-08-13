from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.assistant.llm import OllamaClient
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import load_assistant_config
from voice_assistant.logging_config import configure_logging


def main() -> int:
    configure_logging(debug=True)
    config = load_assistant_config()
    if config.google.gmail_token_path.exists() or config.google.calendar_token_path.exists():
        print("Refusing failure simulation because a real Google token exists.")
        return 2
    llm = OllamaClient(config.llm)
    router = AssistantToolRouter(config)
    prompts = (
        "What's my plan today?",
        "Do I have any important emails?",
        "What time is it?",
    )
    for prompt in prompts:
        reply = llm.chat(prompt, tool_executor=router)
        print(f"Prompt: {prompt}")
        print(f"Tools: {', '.join(reply.tool_calls) or '(none)'}")
        print(f"Reply: {reply.text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
