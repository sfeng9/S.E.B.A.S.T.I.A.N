from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any

from voice_assistant.assistant.llm import ChatMessage, OllamaClient
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.location import ResolvedLocation
from voice_assistant.integrations.open_meteo import WeatherUnavailableError
from voice_assistant.logging_config import configure_logging


PROMPTS = (
    ("What's the weather?", {"get_current_weather"}, None),
    ("What's it like outside?", {"get_current_weather"}, None),
    ("What's the weather in Cary?", {"get_current_weather"}, "Cary"),
    ("What's the weather in Raleigh?", {"get_current_weather"}, "Raleigh"),
    ("What's the weather in Tokyo?", {"get_current_weather"}, "Tokyo"),
    ("What's the weather in London?", {"get_current_weather"}, "London"),
    ("How hot is it in Miami?", {"get_current_weather"}, "Miami"),
    (
        "What's the forecast in Paris tomorrow?",
        {"get_current_weather"},
        "Paris",
    ),
    ("What time is it?", {"get_current_local_time"}, None),
    ("What time is it in Tokyo?", {"get_current_local_time"}, "Tokyo"),
    ("What time is it in London?", {"get_current_local_time"}, "London"),
    (
        "It's the time in Taipei right now.",
        {"get_current_local_time"},
        "Taipei",
    ),
    (
        "Give me the time and weather.",
        {"get_current_local_time", "get_current_weather"},
        None,
    ),
    (
        "Give me the time and weather in Tokyo.",
        {"get_current_local_time", "get_current_weather"},
        "Tokyo",
    ),
    (
        "What's the weather in FakeCityThatDoesNotExist?",
        {"get_current_weather"},
        None,
    ),
    ("What's the weather in Springfield?", {"get_current_weather"}, "Springfield"),
)


class FailingWeatherClient:
    def get_current_weather(
        self,
        location: ResolvedLocation,
        forecast_day: str = "today",
    ) -> None:
        raise WeatherUnavailableError("Simulated network failure.")


class RecordingToolRouter:
    def __init__(self, router: AssistantToolRouter) -> None:
        self.router = router
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return self.router.schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return self.router.execute(name, arguments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Sebastian's location-aware time and weather tools."
    )
    parser.add_argument(
        "--simulate-weather-failure",
        action="store_true",
        help="Return a controlled weather failure to the model.",
    )
    parser.add_argument(
        "--follow-ups-only",
        action="store_true",
        help="Run only the conversational location follow-up checks.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(debug=args.debug)
    config = load_assistant_config()
    client = OllamaClient(config.llm)
    weather_client = FailingWeatherClient() if args.simulate_weather_failure else None
    router = RecordingToolRouter(
        AssistantToolRouter(config, weather_client=weather_client)
    )
    prompts = (
        (
            ("What's the weather?", {"get_current_weather"}, None),
            (
                "Give me the time and weather in Tokyo.",
                {"get_current_local_time", "get_current_weather"},
                "Tokyo",
            ),
        )
        if args.simulate_weather_failure
        else (() if args.follow_ups_only else PROMPTS)
    )

    failures = 0
    for prompt, expected_tools, expected_location in prompts:
        print(f"\nPrompt: {prompt}")
        try:
            reply = client.chat(prompt, tool_executor=router)
        except RuntimeError as exc:
            print(f"Result: failed: {exc}")
            failures += 1
            continue
        used_tools = set(reply.tool_calls)
        print(f"Tools: {', '.join(reply.tool_calls) or '(none)'}")
        print(f"Sebastian: {reply.text}")
        missing = expected_tools - used_tools
        if missing:
            print(f"Result: missing expected tools: {', '.join(sorted(missing))}")
            failures += 1
            continue
        if expected_location and expected_location.casefold() not in reply.text.casefold():
            print(f"Result: response did not identify {expected_location}.")
            failures += 1
            continue
        print("Result: passed")

    if not args.simulate_weather_failure:
        follow_ups = (
            (
                "What's the weather in Tokyo?",
                "What about tomorrow?",
                "get_current_weather",
                "Tokyo",
                "tomorrow",
            ),
            (
                "What time is it in London?",
                "And Tokyo?",
                "get_current_local_time",
                "Tokyo",
                None,
            ),
        )
        for first_prompt, follow_prompt, tool_name, place, forecast_day in follow_ups:
            print(f"\nFollow-up: {first_prompt} / {follow_prompt}")
            try:
                first = client.chat(first_prompt, tool_executor=router)
                history: list[ChatMessage] = [
                    {"role": "user", "content": first_prompt},
                    {"role": "assistant", "content": first.text},
                ]
                call_count = len(router.calls)
                follow = client.chat(
                    follow_prompt,
                    history=history,
                    tool_executor=router,
                )
            except RuntimeError as exc:
                print(f"Result: failed: {exc}")
                failures += 1
                continue

            new_calls = router.calls[call_count:]
            print(f"Tools: {', '.join(name for name, _ in new_calls) or '(none)'}")
            print(f"Sebastian: {follow.text}")
            matching = [arguments for name, arguments in new_calls if name == tool_name]
            valid = bool(matching) and any(
                place.casefold() in str(arguments.get("location", "")).casefold()
                and (
                    forecast_day is None
                    or arguments.get("forecast_day") == forecast_day
                )
                for arguments in matching
            )
            if not valid:
                print("Result: follow-up did not preserve the expected tool context.")
                failures += 1
            else:
                print("Result: passed")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
