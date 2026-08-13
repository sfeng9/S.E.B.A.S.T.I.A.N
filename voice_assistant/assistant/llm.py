from __future__ import annotations

import json
import re
import logging
import time
from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import Any, Protocol
from urllib import error, request

from voice_assistant.config import LlmConfig
from voice_assistant.http_utils import validated_http_url


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are Sebastian, a concise personal voice assistant. Give direct, natural "
    "answers that sound good when spoken aloud. Use plain spoken prose rather than "
    "Markdown. Always use the available tools for current time, date, day, weather, "
    "email, Calendar, schedules, reminders, outdoor conditions, or clothing advice "
    "based on weather. Tool use is mandatory "
    "even when speech recognition produces an awkward statement or fragment such as "
    "'It's the time in Taipei right now.' Interpret that as a current-time request. "
    "Never guess current "
    "information. Pass an explicitly named place to each relevant tool; omit location "
    "for here, home, outside, or when no place is named. Never invent coordinates, "
    "timezones, or weather. For combined requests, call every tool needed and pass "
    "the same explicit location to each one before answering. "
    "If weather fails, say you couldn't get it right now, but still answer any other "
    "parts whose tools succeeded. Keep ordinary time and weather answers to one or "
    "two short sentences, normally under 35 words. For a general weather question, "
    "mention the temperature, condition, feels-like temperature when useful, and "
    "today's high. Do not mention today's low unless the user explicitly asks for it. "
    "For a future day, mention the forecast condition and high, plus rain chance when useful. "
    "Omit humidity, precipitation, and wind unless asked, notable, or relevant. Give "
    "jacket or umbrella advice only when the user asks for it. Mention an explicit "
    "location in the response, but do not repeatedly name the home location. If a "
    "tool reports an ambiguous location, ask one short clarification question. Use "
    "conversation history to preserve locations in follow-up questions. A short "
    "follow-up such as 'What about tomorrow?' or 'And Tokyo?' means repeat the "
    "previous weather or time request with the changed day or place, and must call "
    "the corresponding tool again. Always return a spoken answer after tool results. Do not add "
    "generic offers to help at the end. Except for a required location clarification, "
    "end after the requested information and never ask a follow-up question. For "
    "Gmail, fetch only a small relevant list and fetch full content only when the user "
    "asks what a selected message says. For email lists, use only the tool's cleaned "
    "spoken_summary fields. Never read email addresses, subjects, dates, IDs, labels, "
    "asterisks, symbols, or raw metadata aloud. Never claim to have checked Gmail without a "
    "tool result. Use Calendar tools for plans, availability, and events in the home "
    "timezone. Create an event only when title, date, and start time are clear; ask one "
    "concise clarification when required information is missing. Updates and deletions "
    "may return confirmation_required: ask the indicated question and do not claim "
    "success. On the user's next yes/no response, call confirm_calendar_action. Never "
    "modify an ambiguous event. Use create_reminder for reminders; they are persistent, "
    "not conversation memory. Do not expose private reasoning."
)

CURRENT_INFORMATION_CUE = re.compile(
    r"\b(?:time|date|day|weather|forecast|temperature|outside|rain(?:ing)?|"
    r"jacket|coat|umbrella|email|mail|inbox|calendar|schedule|appointment|"
    r"meeting|event|plan|afternoon|evening|remind(?:er)?)\b",
    re.IGNORECASE,
)

ChatMessage = dict[str, Any]


class ToolExecutor(Protocol):
    @property
    def schemas(self) -> Sequence[dict[str, Any]]: ...

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LlmReply:
    text: str
    model: str
    prompt_tokens: int
    response_tokens: int
    total_seconds: float
    load_seconds: float
    tokens_per_second: float | None
    tool_calls: tuple[str, ...]


class OllamaClient:
    def __init__(self, config: LlmConfig, timeout_seconds: float = 120.0) -> None:
        if config.provider.casefold() != "ollama":
            raise ValueError(f"Unsupported LLM provider: {config.provider!r}")
        self.config = config
        self._base_url = validated_http_url(config.base_url)
        self.timeout_seconds = timeout_seconds
        logger.info(
            "Ollama client ready: model=%s context=%d keep_alive=%s. "
            "The Ollama server may unload the model after this idle duration.",
            config.model,
            config.context_window,
            config.keep_alive,
        )

    def chat(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history: Sequence[ChatMessage] = (),
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 4,
    ) -> LlmReply:
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": prompt},
        ]
        tool_schemas: list[dict[str, Any]] = []
        if tool_executor is not None:
            begin_turn = getattr(tool_executor, "begin_turn", None)
            if callable(begin_turn):
                begin_turn()
            schema_selector = getattr(tool_executor, "schemas_for", None)
            tool_schemas = list(
                schema_selector(prompt, history)
                if callable(schema_selector)
                else tool_executor.schemas
            )
        responses: list[dict[str, Any]] = []
        called_tools: list[str] = []
        empty_response_retried = False
        missing_tool_retries = 0
        requirement_provider = (
            getattr(tool_executor, "tool_requirement", None)
            if tool_executor is not None
            else None
        )
        tool_requirement = (
            requirement_provider(prompt, history)
            if callable(requirement_provider)
            else None
        )

        for round_number in range(max_tool_rounds + 1):
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": self.config.keep_alive,
                "options": {
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.context_window,
                },
            }
            if tool_schemas:
                payload["tools"] = tool_schemas

            if round_number == 0:
                logger.info(
                    "Ollama request started: model=%s keep_alive=%s history_messages=%d.",
                    self.config.model,
                    self.config.keep_alive,
                    len(history),
                )
            else:
                logger.debug("Ollama tool round %d started.", round_number + 1)
            response = self._post_json("/api/chat", payload)
            responses.append(response)
            load_seconds = int(response.get("load_duration", 0)) / 1_000_000_000
            if load_seconds >= 1.0:
                logger.info(
                    "Ollama model load/reload took %.2fs.",
                    load_seconds,
                )
            elif load_seconds > 0:
                logger.debug("Ollama warm model setup took %.2fs.", load_seconds)
            message = response.get("message")
            if not isinstance(message, dict):
                raise RuntimeError("Ollama response did not contain a valid message.")
            raw_tool_calls = message.get("tool_calls") or []
            if not isinstance(raw_tool_calls, list):
                raise RuntimeError("Ollama returned malformed tool calls.")

            if not raw_tool_calls:
                text = str(message.get("content", "")).strip()
                if not text:
                    if empty_response_retried:
                        raise RuntimeError("Ollama returned an empty response twice.")
                    empty_response_retried = True
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your previous response was empty. Respond to the "
                                "latest user request now. Infer elliptical follow-ups "
                                "from conversation history, call any required tools, "
                                "and return a concise spoken answer."
                            ),
                        }
                    )
                    continue
                expected_tools = {
                    str(name)
                    for name in (
                        tool_requirement.get("tools", ())
                        if isinstance(tool_requirement, dict)
                        else ()
                    )
                }
                explicit_tool_missing = bool(expected_tools) and not expected_tools.intersection(
                    called_tools
                )
                current_tool_missing = (
                    not expected_tools
                    and not called_tools
                    and CURRENT_INFORMATION_CUE.search(prompt) is not None
                )
                if tool_executor is not None and (
                    explicit_tool_missing or current_tool_missing
                ):
                    if missing_tool_retries >= 2:
                        fallback = (
                            str(tool_requirement.get("fallback", "")).strip()
                            if isinstance(tool_requirement, dict)
                            else ""
                        )
                        if not fallback:
                            fallback = (
                                "I couldn't verify that request with the required tool, "
                                "so I didn't complete it."
                            )
                        logger.warning(
                            "Required tool was not called after retries: expected=%s called=%s",
                            sorted(expected_tools) or ["any current-information tool"],
                            called_tools,
                        )
                        return self._build_reply(fallback, responses, called_tools)
                    missing_tool_retries += 1
                    instruction = (
                        str(tool_requirement.get("instruction", "")).strip()
                        if isinstance(tool_requirement, dict)
                        else ""
                    )
                    if not instruction:
                        instruction = (
                            "You answered a possible current time, date, weather, "
                            "email, Calendar, or reminder request without a tool. "
                            "Re-evaluate the latest user message, including imperfect "
                            "speech transcription. If it requests current information, "
                            "you must call the appropriate tool now and must not reuse "
                            "or guess the previous answer."
                        )
                    messages.extend(
                        [
                            {"role": "assistant", "content": text},
                            {
                                "role": "system",
                                "content": instruction,
                            },
                        ]
                    )
                    continue
                reply = self._build_reply(text, responses, called_tools)
                override_provider = getattr(
                    tool_executor, "spoken_override_for", None
                )
                if callable(override_provider):
                    spoken_override = override_provider(tuple(called_tools))
                    if spoken_override:
                        logger.info(
                            "Applying controlled spoken response for tools: %s",
                            ", ".join(called_tools),
                        )
                        reply = replace(reply, text=str(spoken_override))
                return reply

            if tool_executor is None:
                raise RuntimeError("Ollama requested a tool, but no tool executor is available.")
            if round_number >= max_tool_rounds:
                raise RuntimeError("Ollama exceeded the maximum number of tool rounds.")

            messages.append(
                {
                    "role": "assistant",
                    "content": str(message.get("content", "")),
                    "tool_calls": raw_tool_calls,
                }
            )
            for raw_call in raw_tool_calls:
                name, arguments = self._parse_tool_call(raw_call)
                called_tools.append(name)
                result = tool_executor.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(
                            result,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    }
                )

        raise RuntimeError("Ollama tool loop ended without a response.")

    def _build_reply(
        self,
        text: str,
        responses: Sequence[dict[str, Any]],
        called_tools: Sequence[str],
    ) -> LlmReply:
        eval_count = sum(int(response.get("eval_count", 0)) for response in responses)
        eval_duration = sum(
            int(response.get("eval_duration", 0)) for response in responses
        )
        tokens_per_second = None
        if eval_count > 0 and eval_duration > 0:
            tokens_per_second = eval_count / (eval_duration / 1_000_000_000)
        final_response = responses[-1]
        reply = LlmReply(
            text=text,
            model=str(final_response.get("model", self.config.model)),
            prompt_tokens=sum(
                int(response.get("prompt_eval_count", 0)) for response in responses
            ),
            response_tokens=eval_count,
            total_seconds=sum(
                int(response.get("total_duration", 0)) for response in responses
            )
            / 1_000_000_000,
            load_seconds=sum(
                int(response.get("load_duration", 0)) for response in responses
            )
            / 1_000_000_000,
            tokens_per_second=tokens_per_second,
            tool_calls=tuple(called_tools),
        )
        logger.info(
            "Ollama response complete: %.2fs, tools=%d, keep_alive=%s.",
            reply.total_seconds,
            len(reply.tool_calls),
            self.config.keep_alive,
        )
        return reply

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw_call, dict):
            return "unknown_tool", {}
        function = raw_call.get("function")
        if not isinstance(function, dict):
            return "unknown_tool", {}
        name = str(function.get("name", "unknown_tool"))
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, dict):
            return name, raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return name, {}
            return name, parsed if isinstance(parsed, dict) else {}
        return name, {}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        connection_attempts = 3
        for attempt in range(1, connection_attempts + 1):
            try:
                # The destination was validated as HTTP(S) during initialization.
                with request.urlopen(  # nosec B310
                    http_request, timeout=self.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Ollama request failed ({exc.code}): {detail}"
                ) from exc
            except (error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt >= connection_attempts:
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self.config.base_url}. "
                        "Start Ollama and try again."
                    ) from exc
                delay = 0.5 * attempt
                logger.warning(
                    "Ollama connection failed during request; retrying in %.1fs "
                    "(attempt %d/%d).",
                    delay,
                    attempt + 1,
                    connection_attempts,
                )
                time.sleep(delay)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError("Ollama returned a malformed JSON response.") from exc

        raise RuntimeError("Ollama request failed without a response.")
