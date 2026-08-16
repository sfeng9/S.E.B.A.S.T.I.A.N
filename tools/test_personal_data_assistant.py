from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_assistant.assistant.llm import ChatMessage, OllamaClient
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import load_assistant_config


class RecordingRouter:
    def __init__(self, router: AssistantToolRouter) -> None:
        self.router = router
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return self.router.execute(name, arguments)


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        base = load_assistant_config()
        config = replace(
            base,
            home_location=replace(
                base.home_location,
                timezone=base.home_location.timezone or "America/New_York",
            ),
            personal_data=replace(base.personal_data, database_path=root / "sebastian.db"),
            reminders=replace(base.reminders, database_path=root / "reminders.sqlite3"),
        )
        client = OllamaClient(config.llm)
        router = RecordingRouter(AssistantToolRouter(config))
        history: list[ChatMessage] = []
        checks = (
            ("Remember that my test air filter is 16 by 20.", {"create_note"}),
            ("What did I tell you about my test air filter?", {"search_notes"}),
            ("Add test assignment due tomorrow at 5 PM to my tasks.", {"create_task"}),
            ("What tasks are due tomorrow?", {"list_tasks"}),
            ("Create a test grocery list.", {"create_list"}),
            ("Add milk, eggs, chicken, and rice to my test grocery list.", {"add_list_items"}),
            ("What's on my test grocery list?", {"get_list"}),
            ("Remove the chicken.", {"remove_list_item"}),
        )
        failures = 0
        for prompt, expected in checks:
            call_start = len(router.calls)
            print(f"\nPrompt: {prompt}")
            try:
                reply = client.chat(prompt, history=history, tool_executor=router)
            except RuntimeError as exc:
                print(f"Result: failed: {exc}")
                failures += 1
                continue
            called = {name for name, _ in router.calls[call_start:]}
            print(f"Tools: {', '.join(name for name, _ in router.calls[call_start:]) or '(none)'}")
            print(f"Sebastian: {reply.text}")
            if expected.issubset(called):
                print("Result: passed")
            else:
                print(f"Result: missing {', '.join(sorted(expected - called))}")
                failures += 1
            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply.text},
                ]
            )
            history = history[-8:]

    print("\nThe diagnostic used a temporary database and left your real data unchanged.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
