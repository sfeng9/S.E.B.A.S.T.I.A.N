from __future__ import annotations

import unittest
import json
from typing import Any
from urllib import error
from unittest.mock import patch

from voice_assistant.assistant.llm import OllamaClient
from voice_assistant.config import load_assistant_config


class FakeToolRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_local_time",
                    "description": "Get time",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "get_current_local_time":
            return {"ok": True, "display_time": "6:27 AM"}
        return {"ok": True, "temperature_f": 68, "condition": "partly cloudy"}


class RequiredCalendarRouter(FakeToolRouter):
    def __init__(self) -> None:
        super().__init__()
        self.schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "delete_event",
                    "description": "Delete event",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )

    def tool_requirement(self, prompt: str, history: list[dict[str, Any]]) -> dict[str, object]:
        return {
            "tools": ("delete_event",),
            "instruction": "Call delete_event. Do not claim the event was deleted.",
            "fallback": "I couldn't use Google Calendar, so I didn't delete it.",
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"ok": True, "confirmation_required": True}

    def spoken_override_for(self, tools: tuple[str, ...]) -> str | None:
        if "delete_event" in tools:
            return "Do you want me to delete Test Meeting from your calendar?"
        return None


class LlmToolLoopTests(unittest.TestCase):
    def test_calendar_action_cannot_succeed_without_required_tool(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = RequiredCalendarRouter()
        false_success = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "I've removed it."},
        }

        with patch.object(
            client,
            "_post_json",
            side_effect=[false_success, false_success, false_success],
        ) as post:
            reply = client.chat(
                "Remove that test meeting, please for me.",
                tool_executor=router,
            )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(router.calls, [])
        self.assertEqual(
            reply.text,
            "I couldn't use Google Calendar, so I didn't delete it.",
        )

    def test_calendar_action_uses_tool_and_controlled_confirmation_speech(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = RequiredCalendarRouter()
        false_success = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "I've removed it."},
        }
        tool_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "delete_event", "arguments": {}}}
                ],
            },
        }

        with patch.object(
            client,
            "_post_json",
            side_effect=[false_success, tool_response, false_success],
        ):
            reply = client.chat(
                "Remove that test meeting, please for me.",
                tool_executor=router,
            )

        self.assertEqual(router.calls, [("delete_event", {})])
        self.assertEqual(reply.tool_calls, ("delete_event",))
        self.assertEqual(
            reply.text,
            "Do you want me to delete Test Meeting from your calendar?",
        )

    def test_retries_transient_ollama_connection_failure(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"message": {"role": "assistant", "content": "Recovered."}}
        ).encode("utf-8")

        with (
            patch(
                "voice_assistant.assistant.llm.request.urlopen",
                side_effect=[error.URLError("connection reset"), response],
            ) as urlopen,
            patch("voice_assistant.assistant.llm.time.sleep") as sleep,
        ):
            result = client._post_json("/api/chat", {"model": "qwen3:8b"})

        self.assertEqual(result["message"]["content"], "Recovered.")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_tool_results_preserve_unicode_without_escape_expansion(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = FakeToolRouter()
        tool_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_current_weather", "arguments": {}}}
                ],
            },
        }
        final_response = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Done."},
        }
        router.execute = lambda name, arguments: {  # type: ignore[method-assign]
            "ok": True,
            "subject": "研究项目",
        }

        with patch.object(
            client, "_post_json", side_effect=[tool_response, final_response]
        ) as post:
            client.chat("What's the weather?", tool_executor=router)

        tool_content = next(
            message["content"]
            for message in post.call_args_list[1].args[1]["messages"]
            if message["role"] == "tool"
        )
        self.assertIn("研究项目", tool_content)
        self.assertNotIn("\\u", tool_content)

    def test_controlled_spoken_override_replaces_model_wording(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = FakeToolRouter()
        router.spoken_override_for = lambda tools: (  # type: ignore[attr-defined]
            "The sender is Professor Smith. The snippet is: The deadline is Friday."
        )
        tool_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_current_weather", "arguments": {}}}
                ],
            },
        }
        verbose_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "A very long response with metadata that should not be spoken.",
            },
        }

        with patch.object(
            client, "_post_json", side_effect=[tool_response, verbose_response]
        ):
            reply = client.chat("Any important emails?", tool_executor=router)

        self.assertEqual(
            reply.text,
            "The sender is Professor Smith. The snippet is: The deadline is Friday.",
        )

    def test_executes_parallel_tools_before_final_answer(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = FakeToolRouter()
        first_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_local_time",
                            "arguments": {"location": "Tokyo"},
                        }
                    },
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "Tokyo"},
                        }
                    },
                ],
            },
            "total_duration": 100,
            "eval_count": 2,
            "eval_duration": 50,
        }
        final_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "It's 6:27 AM and 68 degrees with partly cloudy skies.",
            },
            "total_duration": 200,
            "eval_count": 10,
            "eval_duration": 100,
        }

        with patch.object(
            client,
            "_post_json",
            side_effect=[first_response, final_response],
        ) as post:
            reply = client.chat(
                "Give me the time and weather in Tokyo.",
                tool_executor=router,
            )

        self.assertEqual(
            router.calls,
            [
                ("get_current_local_time", {"location": "Tokyo"}),
                ("get_current_weather", {"location": "Tokyo"}),
            ],
        )
        self.assertEqual(
            reply.tool_calls,
            ("get_current_local_time", "get_current_weather"),
        )
        self.assertIn("6:27 AM", reply.text)
        second_payload = post.call_args_list[1].args[1]
        tool_messages = [
            message for message in second_payload["messages"] if message["role"] == "tool"
        ]
        self.assertEqual(len(tool_messages), 2)

    def test_retries_one_empty_response_before_tool_call(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = FakeToolRouter()
        empty_response = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": ""},
        }
        tool_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_local_time",
                            "arguments": {"location": "Tokyo"},
                        }
                    }
                ],
            },
        }
        final_response = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "It's 7:30 PM in Tokyo."},
        }

        with patch.object(
            client,
            "_post_json",
            side_effect=[empty_response, tool_response, final_response],
        ) as post:
            reply = client.chat(
                "And Tokyo?",
                history=[
                    {"role": "user", "content": "What time is it in London?"},
                    {"role": "assistant", "content": "It's 11:30 AM in London."},
                ],
                tool_executor=router,
            )

        self.assertEqual(reply.tool_calls, ("get_current_local_time",))
        recovery_messages = post.call_args_list[1].args[1]["messages"]
        self.assertTrue(
            any(
                "previous response was empty" in message.get("content", "")
                for message in recovery_messages
            )
        )

    def test_retries_current_information_answer_that_skipped_tools(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = FakeToolRouter()
        guessed_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "The current time in Taipei is 3:45 PM.",
            },
        }
        tool_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_local_time",
                            "arguments": {"location": "Taipei"},
                        }
                    }
                ],
            },
        }
        final_response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "The current time in Taipei is 7:20 PM.",
            },
        }

        with patch.object(
            client,
            "_post_json",
            side_effect=[guessed_response, tool_response, final_response],
        ) as post:
            reply = client.chat(
                "It's the time in Taipei right now.",
                tool_executor=router,
            )

        self.assertEqual(reply.tool_calls, ("get_current_local_time",))
        self.assertIn("7:20 PM", reply.text)
        retry_messages = post.call_args_list[1].args[1]["messages"]
        self.assertTrue(
            any(
                "answered a possible current time" in message.get("content", "")
                for message in retry_messages
            )
        )

    def test_ordinary_answer_does_not_trigger_missing_tool_retry(self) -> None:
        client = OllamaClient(load_assistant_config().llm)
        router = FakeToolRouter()
        response = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "Melatonin helps regulate the sleep-wake cycle.",
            },
        }

        with patch.object(client, "_post_json", return_value=response) as post:
            reply = client.chat("What's melatonin?", tool_executor=router)

        self.assertEqual(reply.tool_calls, ())
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
