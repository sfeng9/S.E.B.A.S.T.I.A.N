from __future__ import annotations

import unittest

from voice_assistant.assistant.session_memory import SessionMemory
from voice_assistant.config import ConversationConfig


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SessionMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.memory = SessionMemory(
            ConversationConfig(
                session_ttl_minutes=0.05,
                max_context_tokens=1000,
            ),
            clock=self.clock,
        )

    def test_inactivity_ttl_resets_after_each_interaction(self) -> None:
        self.memory.begin_interaction()
        self.memory.record_turn("Remember Tokyo.", "Tokyo is the active topic.")
        self.clock.advance(2.0)

        history = self.memory.begin_interaction()
        self.assertEqual(len(history), 2)
        self.memory.record_turn("What about tomorrow?", "Tomorrow in Tokyo...")
        self.clock.advance(2.0)

        self.assertFalse(self.memory.expire_if_idle())
        self.assertEqual(self.memory.turn_count, 2)
        self.clock.advance(1.1)
        self.assertTrue(self.memory.expire_if_idle())
        self.assertEqual(self.memory.turn_count, 0)
        self.assertFalse(self.memory.is_active)

    def test_more_than_six_turns_survive_when_under_budget(self) -> None:
        for turn in range(12):
            self.memory.begin_interaction()
            self.memory.record_turn(f"User turn {turn}", f"Assistant turn {turn}")

        self.assertEqual(self.memory.turn_count, 12)
        self.assertIn("User turn 0", self.memory.history[0]["content"])
        self.assertIn("Assistant turn 11", self.memory.history[-1]["content"])

    def test_context_trimming_removes_old_complete_turns(self) -> None:
        memory = SessionMemory(
            ConversationConfig(
                session_ttl_minutes=30,
                max_context_tokens=128,
            ),
            clock=self.clock,
        )
        for turn in range(8):
            memory.begin_interaction()
            memory.record_turn(
                f"User {turn}: " + ("word " * 20),
                f"Assistant {turn}: " + ("answer " * 20),
            )

        self.assertLess(memory.turn_count, 8)
        self.assertEqual(len(memory.history) % 2, 0)
        self.assertEqual(memory.history[0]["role"], "user")
        self.assertEqual(memory.history[-1]["role"], "assistant")
        self.assertIn("Assistant 7", memory.history[-1]["content"])

    def test_manual_reset_clears_only_session_state(self) -> None:
        self.memory.begin_interaction()
        self.memory.record_turn("Temporary topic", "Temporary response")

        self.memory.reset_session()

        self.assertEqual(self.memory.turn_count, 0)
        self.assertEqual(tuple(self.memory.history), ())
        self.assertFalse(self.memory.is_active)


if __name__ == "__main__":
    unittest.main()
