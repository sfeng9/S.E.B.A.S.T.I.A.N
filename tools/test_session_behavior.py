from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_assistant.assistant.llm import OllamaClient
from voice_assistant.assistant.session_memory import SessionMemory
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import load_assistant_config
from voice_assistant.logging_config import configure_logging


PROMPTS = (
    "What's the weather in Tokyo?",
    "What about tomorrow?",
    "And Friday?",
    "What time is it there?",
    "Name the city we're discussing in one word.",
    "Remember the code word cedar for this session.",
    "What code word did I give you?",
    "Add the color blue to that code word.",
    "What color did I add?",
    "What city were we discussing?",
    "What was the code word?",
    "Summarize the city, code word, and color.",
)


class RecordingRouter:
    def __init__(self, router: AssistantToolRouter) -> None:
        self.router = router
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return self.router.schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return self.router.execute(name, arguments)


def main() -> int:
    configure_logging(debug=False)
    config = load_assistant_config()
    client = OllamaClient(config.llm)
    router = RecordingRouter(AssistantToolRouter(config))
    memory = SessionMemory(config.conversation)
    failures = 0

    for turn_number, prompt in enumerate(PROMPTS, start=1):
        history = memory.begin_interaction()
        call_count = len(router.calls)
        try:
            reply = client.chat(prompt, history=history, tool_executor=router)
        except RuntimeError as exc:
            print(f"Turn {turn_number} failed: {exc}")
            return 1
        memory.record_turn(prompt, reply.text)
        new_calls = router.calls[call_count:]
        print(f"\nTurn {turn_number}: {prompt}")
        print(f"Tools: {', '.join(name for name, _ in new_calls) or '(none)'}")
        print(f"Sebastian: {reply.text}")
        print(
            f"Context: {memory.turn_count} turns, "
            f"~{memory.estimated_tokens}/{memory.max_context_tokens} tokens"
        )

        if turn_number in {2, 3, 4}:
            expected_tool = (
                "get_current_local_time" if turn_number == 4 else "get_current_weather"
            )
            matching = [args for name, args in new_calls if name == expected_tool]
            if not matching or not any(
                "tokyo" in str(args.get("location", "")).casefold()
                for args in matching
            ):
                print("Result: Tokyo context was not preserved in the tool call.")
                failures += 1
        if turn_number == 3 and not any(
            args.get("forecast_day", "").casefold() == "friday"
            for name, args in new_calls
            if name == "get_current_weather"
        ):
            print("Result: Friday was not passed to the weather tool.")
            failures += 1

    final_text = memory.history[-1]["content"].casefold()
    for expected in ("tokyo", "cedar", "blue"):
        if expected not in final_text:
            print(f"Result: final summary forgot {expected!r}.")
            failures += 1

    if memory.turn_count != 12:
        print(f"Result: expected 12 retained turns, found {memory.turn_count}.")
        failures += 1

    print("\nResult: session behavior passed." if not failures else "\nResult: session behavior failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
