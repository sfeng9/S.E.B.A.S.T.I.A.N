from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from voice_assistant.assistant.llm import ChatMessage
from voice_assistant.config import ConversationConfig


logger = logging.getLogger(__name__)


class SessionMemory:
    """Bounded transient conversation state with inactivity expiration."""

    def __init__(
        self,
        config: ConversationConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = config.session_ttl_minutes * 60.0
        self._max_context_tokens = config.max_context_tokens
        self._clock = clock
        self._history: list[ChatMessage] = []
        self._last_activity: float | None = None

    def begin_interaction(self) -> tuple[ChatMessage, ...]:
        now = self._clock()
        self.expire_if_idle(now)
        if self._last_activity is None:
            logger.info(
                "Conversation session started: ttl=%.1f minutes context_budget=%d tokens",
                self._ttl_seconds / 60.0,
                self._max_context_tokens,
            )
        else:
            logger.debug("Conversation session activity; inactivity timer reset.")
        self._last_activity = now
        return tuple(self._history)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self._history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        self._last_activity = self._clock()
        self._trim_to_budget()

    def expire_if_idle(self, now: float | None = None) -> bool:
        if self._last_activity is None:
            return False
        current = self._clock() if now is None else now
        if current - self._last_activity < self._ttl_seconds:
            return False

        expired_turns = self.turn_count
        self._history.clear()
        self._last_activity = None
        logger.info(
            "Conversation session expired after %.1f minutes idle; cleared %d turns.",
            self._ttl_seconds / 60.0,
            expired_turns,
        )
        return True

    def reset_session(self) -> None:
        cleared_turns = self.turn_count
        self._history.clear()
        self._last_activity = None
        logger.info("Conversation session reset; cleared %d turns.", cleared_turns)

    @property
    def history(self) -> Sequence[ChatMessage]:
        return tuple(self._history)

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2

    @property
    def estimated_tokens(self) -> int:
        return sum(estimate_message_tokens(message) for message in self._history)

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    @property
    def is_active(self) -> bool:
        return self._last_activity is not None

    def _trim_to_budget(self) -> None:
        removed_turns = 0
        while len(self._history) >= 4 and self.estimated_tokens > self._max_context_tokens:
            del self._history[:2]
            removed_turns += 1

        if removed_turns:
            logger.info(
                "Conversation context trimmed: removed %d oldest turns; "
                "remaining=%d turns estimated_tokens=%d/%d.",
                removed_turns,
                self.turn_count,
                self.estimated_tokens,
                self._max_context_tokens,
            )


def estimate_message_tokens(message: dict[str, Any]) -> int:
    content = str(message.get("content", ""))
    # Conservative approximation for English prose plus per-message framing.
    return 4 + max(1, math.ceil(len(content) / 4))
