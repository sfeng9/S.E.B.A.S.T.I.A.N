from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_assistant.assistant.llm import OllamaClient
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import load_assistant_config


def main() -> int:
    config = load_assistant_config()
    tools = AssistantToolRouter(config)
    llm = OllamaClient(config.llm)

    stable_question = "What is photosynthesis?"
    stable = llm.chat(stable_question, tool_executor=tools)
    _print_result(stable_question, stable.text, stable.tool_calls)

    current_question = "What's the latest news about Jungkook?"
    current = llm.chat(
        current_question,
        history=[
            {"role": "user", "content": stable_question},
            {"role": "assistant", "content": stable.text},
        ],
        tool_executor=tools,
    )
    _print_result(current_question, current.text, current.tool_calls)

    source_question = "Which source said that?"
    sources = llm.chat(
        source_question,
        history=[
            {"role": "user", "content": current_question},
            {"role": "assistant", "content": current.text},
        ],
        tool_executor=tools,
    )
    _print_result(source_question, sources.text, sources.tool_calls)

    checks = (
        "web_search" not in stable.tool_calls,
        "web_search" in current.tool_calls,
        "get_last_web_sources" in sources.tool_calls,
    )
    if not all(checks):
        print("Result: live web routing diagnostic failed.")
        return 1
    print("Result: local knowledge, live search, and source follow-up routing succeeded.")
    return 0


def _print_result(question: str, answer: str, tool_calls: tuple[str, ...]) -> None:
    print(f"\nUser: {question}")
    print(f"Tools: {', '.join(tool_calls) if tool_calls else 'none'}")
    print(f"Sebastian: {answer}")


if __name__ == "__main__":
    raise SystemExit(main())
