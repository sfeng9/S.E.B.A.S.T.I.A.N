from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from voice_assistant.assistant.llm import LlmReply, OllamaClient
from voice_assistant.assistant.session_memory import SessionMemory
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.audio.speech_to_text import (
    FasterWhisperTranscriber,
    TranscriptionResult,
)
from voice_assistant.audio.text_to_speech import PiperSynthesizer, SynthesisResult
from voice_assistant.config import AssistantConfig
from voice_assistant.integrations.calendar_reminder_sync import (
    CalendarReminderSynchronizer,
    CalendarReminderSyncWorker,
)
from voice_assistant.integrations.reminders import Reminder, ReminderScheduler


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceTurnResult:
    transcription: TranscriptionResult
    reply: LlmReply
    synthesis: SynthesisResult
    total_seconds: float


class VoiceAssistant:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.transcriber = FasterWhisperTranscriber(
            config.speech_to_text,
            lazy=True,
        )
        self.llm = OllamaClient(config.llm)
        self.tools = AssistantToolRouter(config)
        self.synthesizer = PiperSynthesizer(config.text_to_speech)
        self.session = SessionMemory(config.conversation)
        self.reminder_scheduler = ReminderScheduler(
            self.tools.reminder_store,
            config.reminders.poll_interval_seconds,
        )
        self.calendar_reminder_worker = CalendarReminderSyncWorker(
            CalendarReminderSynchronizer(config, self.tools.reminder_store),
            config.reminders.calendar_sync_interval_seconds,
        )
        self._background_services_started = False
        self._due_reminders: list[Reminder] = []
        self._stt_idle_seconds = config.resources.stt_idle_unload_minutes * 60.0
        logger.info(
            "Heavy model policy: Faster Whisper lazy load, unload after %.1f minutes "
            "idle; Ollama keep_alive=%s. Piper and wake-word inference are CPU-backed.",
            config.resources.stt_idle_unload_minutes,
            config.llm.keep_alive,
        )

    def start_background_services(self) -> None:
        if self._background_services_started:
            return
        self._background_services_started = True
        if not self.config.reminders.calendar_sync_enabled:
            logger.info("Automatic Calendar reminders are disabled.")
            return
        if not self.config.home_location.timezone:
            logger.warning(
                "Automatic Calendar reminders require home_location.timezone."
            )
            return
        if not self.config.google.calendar_token_path.exists():
            logger.warning(
                "Automatic Calendar reminders are waiting for Calendar authorization."
            )
            return
        self.calendar_reminder_worker.start()

    def close(self) -> None:
        self.calendar_reminder_worker.stop()

    def process_audio(self, audio_path: Path, response_wav_path: Path) -> VoiceTurnResult:
        started = time.perf_counter()
        transcription = self.transcriber.transcribe(audio_path)
        if not transcription.text:
            raise RuntimeError("No speech was detected in the recording.")

        if self.session.expire_if_idle():
            self.tools.reset_session_context()
        history = self.session.begin_interaction()
        reply = self.llm.chat(
            transcription.text,
            history=history,
            tool_executor=self.tools,
        )
        synthesis = self.synthesizer.synthesize(reply.text, response_wav_path)
        self.session.record_turn(transcription.text, reply.text)
        return VoiceTurnResult(
            transcription=transcription,
            reply=reply,
            synthesis=synthesis,
            total_seconds=time.perf_counter() - started,
        )

    def maintenance(self) -> bool:
        if self.session.expire_if_idle():
            self.tools.reset_session_context()
        self.transcriber.unload_if_idle(self._stt_idle_seconds)
        claimed = self.reminder_scheduler.poll()
        if claimed:
            self._due_reminders.extend(claimed)
        return bool(self._due_reminders)

    def take_due_reminders(self) -> tuple[Reminder, ...]:
        reminders = tuple(self._due_reminders)
        self._due_reminders.clear()
        return reminders

    def synthesize_reminder(self, reminder: Reminder, wav_path: Path) -> SynthesisResult:
        return self.synthesizer.synthesize(f"Reminder: {reminder.text}.", wav_path)

    def complete_reminder(self, reminder_id: int) -> None:
        self.tools.reminder_store.mark_fired(reminder_id)

    def release_reminder(self, reminder_id: int) -> None:
        self.tools.reminder_store.release(reminder_id)

    def reset_session(self) -> None:
        self.session.reset_session()
        self.tools.reset_session_context()

    def clear_conversation(self) -> None:
        self.reset_session()

    @property
    def conversation_turns(self) -> int:
        return self.session.turn_count

    @property
    def context_tokens(self) -> int:
        return self.session.estimated_tokens

    @property
    def max_context_tokens(self) -> int:
        return self.session.max_context_tokens
