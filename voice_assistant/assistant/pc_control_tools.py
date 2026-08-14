from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from voice_assistant.config import ApplicationConfig, AssistantConfig
from voice_assistant.integrations.pc_control import (
    POWER_ACTIONS,
    ApplicationNotFoundError,
    FeatureUnavailableError,
    PcControlError,
    ProtectedProcessError,
    WindowsPcController,
)


logger = logging.getLogger(__name__)


class ToolPermission(str, Enum):
    READ_ONLY = "read_only"
    SAFE_ACTION = "safe_action"
    CONFIRM_REQUIRED = "confirm_required"
    DISABLED = "disabled"


TOOL_PERMISSIONS: Mapping[str, ToolPermission] = {
    "open_application": ToolPermission.SAFE_ACTION,
    "close_application": ToolPermission.SAFE_ACTION,
    "get_application_status": ToolPermission.READ_ONLY,
    "get_system_volume": ToolPermission.READ_ONLY,
    "set_system_volume": ToolPermission.SAFE_ACTION,
    "change_system_volume": ToolPermission.SAFE_ACTION,
    "mute_system_volume": ToolPermission.SAFE_ACTION,
    "unmute_system_volume": ToolPermission.SAFE_ACTION,
    "media_play_pause": ToolPermission.SAFE_ACTION,
    "media_next_track": ToolPermission.SAFE_ACTION,
    "media_previous_track": ToolPermission.SAFE_ACTION,
    "media_stop": ToolPermission.SAFE_ACTION,
    "lock_computer": ToolPermission.SAFE_ACTION,
    "take_screenshot": ToolPermission.SAFE_ACTION,
    "get_system_status": ToolPermission.READ_ONLY,
    "get_gpu_status": ToolPermission.READ_ONLY,
    "get_top_processes": ToolPermission.READ_ONLY,
    "request_power_action": ToolPermission.CONFIRM_REQUIRED,
    "confirm_pc_action": ToolPermission.CONFIRM_REQUIRED,
}

CONFIRMATION_RESPONSE_CUE = re.compile(
    r"^\s*(?:yes|yeah|yep|confirm|do it|go ahead|please do|no|nope|don't|do not|cancel)\b",
    re.IGNORECASE,
)
APP_OPEN_CUE = re.compile(r"\b(?:open|launch|start)\b", re.IGNORECASE)
APP_CLOSE_CUE = re.compile(r"\b(?:close|quit|exit)\b", re.IGNORECASE)
APP_STATUS_CUE = re.compile(
    r"\b(?:is|are)\b.{0,30}\b(?:open|running)|\b(?:open|running)\b.{0,30}\b(?:is|are)\b",
    re.IGNORECASE,
)
VOLUME_CUE = re.compile(
    r"\b(?:volume|mute|unmute|louder|quieter|turn (?:it|this|that) (?:up|down)|"
    r"(?:increase|decrease|raise|lower) (?:it|this|that)(?: by)?\b)",
    re.IGNORECASE,
)
MEDIA_CUE = re.compile(
    r"\b(?:pause|resume|play|skip|next (?:song|track)|previous (?:song|track)|stop (?:music|media|playback))\b",
    re.IGNORECASE,
)
SCREENSHOT_CUE = re.compile(r"\b(?:screen ?shot|capture (?:my |the )?screen)\b", re.IGNORECASE)
LOCK_CUE = re.compile(r"\block\b.{0,20}\b(?:computer|pc|workstation|screen)\b", re.IGNORECASE)
POWER_CUE = re.compile(
    r"\b(?:shut ?down|restart|reboot|log ?out|sign ?out|hibernate|"
    r"(?:sleep|suspend) (?:my |the )?(?:computer|pc|system)|"
    r"put (?:my |the )?(?:computer|pc|system) to sleep)\b",
    re.IGNORECASE,
)
GPU_CUE = re.compile(r"\b(?:gpu|graphics card|vram)\b", re.IGNORECASE)
TOP_PROCESS_CUE = re.compile(
    r"\b(?:what(?:'s| is) using|top processes?|programs? (?:are )?running|"
    r"most (?:cpu|memory|ram))\b",
    re.IGNORECASE,
)
SYSTEM_STATUS_CUE = re.compile(
    r"\b(?:cpu usage|ram|memory usage|disk space|storage space|uptime|"
    r"how long has (?:my |the )?(?:pc|computer))\b",
    re.IGNORECASE,
)
EXTERNAL_INSTRUCTION_CONTEXT_CUE = re.compile(
    r"\b(?:search (?:the )?(?:web|internet)|web(?:page|site| result)|"
    r"email|message|calendar description|document)\b.{0,40}\b(?:says?|contains?|"
    r"mentions?|for|about)\b",
    re.IGNORECASE,
)
NON_APPLICATION_OPEN_CUE = re.compile(
    r"\b(?:email|mail|inbox|calendar|event|weather|search|web|website|url|file|folder|"
    r"document|settings?)\b",
    re.IGNORECASE,
)


class PcControllerProtocol(Protocol):
    def open_application(self, application: ApplicationConfig) -> dict[str, Any]: ...
    def close_application(self, application: ApplicationConfig) -> dict[str, Any]: ...
    def force_close_application(self, application: ApplicationConfig) -> dict[str, Any]: ...
    def is_application_running(self, application: ApplicationConfig) -> dict[str, Any]: ...
    def get_system_volume(self) -> dict[str, Any]: ...
    def set_system_volume(self, percent: int) -> dict[str, Any]: ...
    def change_system_volume(self, delta_percent: int) -> dict[str, Any]: ...
    def set_system_mute(self, muted: bool) -> dict[str, Any]: ...
    def send_media_key(self, action: str) -> dict[str, Any]: ...
    def lock_computer(self) -> dict[str, Any]: ...
    def perform_power_action(self, action: str) -> dict[str, Any]: ...
    def take_screenshot(self) -> dict[str, Any]: ...
    def get_system_status(self) -> dict[str, Any]: ...
    def get_top_processes(self, sort_by: str, limit: int) -> dict[str, Any]: ...
    def get_gpu_status(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PendingPcAction:
    action: str
    arguments: dict[str, Any]
    description: str
    expires_at: datetime


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def pc_tool_schemas(config: AssistantConfig) -> tuple[dict[str, Any], ...]:
    app_ids = [application.identifier for application in config.pc_control.applications]
    application = {
        "type": "string",
        "enum": app_ids,
        "description": "Trusted configured application identifier. Map spoken aliases to one of these IDs.",
    }
    return (
        _schema("open_application", "Open one application from the configured allowlist. Never pass a path or command.", {"app_name": application}, ["app_name"]),
        _schema("close_application", "Gracefully close one allowlisted noncritical application. A force close, if needed, requires later confirmation.", {"app_name": application}, ["app_name"]),
        _schema("get_application_status", "Check whether one allowlisted application is running.", {"app_name": application}, ["app_name"]),
        _schema("get_system_volume", "Get the Windows default media-output volume and mute state."),
        _schema("set_system_volume", "Set Windows default media-output volume from 0 to 100 percent. This is separate from Sebastian's configured TTS device.", {"percent": {"type": "integer", "minimum": 0, "maximum": 100}}, ["percent"]),
        _schema("change_system_volume", f"Raise or lower Windows default media-output volume. Pass a signed percentage-point amount. When the user gives no amount, omit delta_percent and the configured {config.pc_control.volume_step_percent}-point step will be applied in the requested direction.", {"delta_percent": {"type": "integer", "minimum": -100, "maximum": 100}}),
        _schema("mute_system_volume", "Mute the Windows default media output."),
        _schema("unmute_system_volume", "Unmute the Windows default media output."),
        _schema("media_play_pause", "Send the global Windows play/pause media key for pause, play, or resume requests."),
        _schema("media_next_track", "Send the global Windows next-track media key."),
        _schema("media_previous_track", "Send the global Windows previous-track media key."),
        _schema("media_stop", "Send the global Windows stop-media key."),
        _schema("lock_computer", "Lock this Windows computer after an explicit direct user request. This cannot unlock or enter a password."),
        _schema("take_screenshot", "Capture all screens locally to the configured screenshot directory. Do not upload or inspect the image."),
        _schema("get_system_status", "Read current CPU, RAM, system-disk, and uptime statistics on demand."),
        _schema("get_gpu_status", "Read NVIDIA GPU utilization, temperature, and VRAM statistics on demand."),
        _schema("get_top_processes", "Read a short list of processes using the most CPU or memory. This never terminates them.", {"sort_by": {"type": "string", "enum": ["cpu", "memory"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["sort_by"]),
        _schema("request_power_action", "Request shutdown, restart, logout, sleep, or hibernate. This only creates an expiring confirmation and never executes immediately.", {"action": {"type": "string", "enum": sorted(POWER_ACTIONS)}}, ["action"]),
        _schema("confirm_pc_action", "Confirm or reject the one recent pending PC action. Call only for the user's separate direct yes/no response.", {"confirm": {"type": "boolean"}}, ["confirm"]),
    )


class PcControlToolHandler:
    def __init__(
        self,
        config: AssistantConfig,
        controller: PcControllerProtocol | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._controller = controller or WindowsPcController(config.pc_control)
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._schemas = pc_tool_schemas(config) if config.pc_control.enabled else ()
        self._applications = {
            application.identifier: application
            for application in config.pc_control.applications
        }
        self._application_names: dict[str, ApplicationConfig] = {}
        for application in config.pc_control.applications:
            for name in (application.identifier, *application.aliases):
                self._application_names[_normalized_name(name)] = application
        self._pending: PendingPcAction | None = None
        self._confirmation_created_this_turn = False
        self._spoken_override: str | None = None
        self._current_prompt = ""

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return self._schemas

    @property
    def permissions(self) -> Mapping[str, ToolPermission]:
        return TOOL_PERMISSIONS

    def begin_turn(self) -> None:
        self._confirmation_created_this_turn = False
        self._spoken_override = None
        self._current_prompt = ""

    def reset_session_context(self) -> None:
        self._pending = None
        self._spoken_override = None
        logger.info("Cleared pending PC-control confirmation.")

    def clear_pending_confirmation(self) -> None:
        if self._pending is not None:
            logger.info("Cancelled pending PC action in favor of a newer action.")
        self._pending = None

    @property
    def has_pending_confirmation(self) -> bool:
        return self._pending is not None

    def matches_prompt(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]] = (),
    ) -> bool:
        if not self._config.pc_control.enabled:
            return False
        self._current_prompt = prompt
        if self._pending is not None and CONFIRMATION_RESPONSE_CUE.search(prompt):
            return True
        context = " ".join(
            [*(str(item.get("content", "")) for item in history[-2:]), prompt]
        )
        return self._expected_tool(context) is not None

    def tool_requirement(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        del history
        self._current_prompt = prompt
        if not self._config.pc_control.enabled:
            return None
        if self._pending is not None and CONFIRMATION_RESPONSE_CUE.search(prompt):
            return {
                "tools": ("confirm_pc_action",),
                "instruction": (
                    "A PC action is awaiting confirmation. Call confirm_pc_action "
                    "now using this separate direct yes or no response. Do not "
                    "claim the action happened without the tool result."
                ),
                "fallback": "I couldn't confirm that PC action, so I didn't do it.",
            }
        expected = self._expected_tool(prompt)
        if expected is None:
            return None
        fallback = (
            "I couldn't find that application on this computer."
            if expected in {"open_application", "close_application", "get_application_status"}
            else "I couldn't complete that PC request."
        )
        return {
            "tools": (expected,),
            "instruction": (
                f"The direct user request requires the {expected} PC tool. Call it "
                "now and use only its result. Never simulate or claim a PC action."
            ),
            "fallback": fallback,
        }

    def spoken_override_for(self, called_tools: Sequence[str]) -> str | None:
        if any(name in TOOL_PERMISSIONS for name in called_tools):
            return self._spoken_override
        return None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        handlers = {
            "open_application": self._open_application,
            "close_application": self._close_application,
            "get_application_status": self._application_status,
            "get_system_volume": self._get_volume,
            "set_system_volume": self._set_volume,
            "change_system_volume": self._change_volume,
            "mute_system_volume": lambda _: self._set_mute(True),
            "unmute_system_volume": lambda _: self._set_mute(False),
            "media_play_pause": lambda _: self._media("play_pause"),
            "media_next_track": lambda _: self._media("next"),
            "media_previous_track": lambda _: self._media("previous"),
            "media_stop": lambda _: self._media("stop"),
            "lock_computer": self._lock,
            "take_screenshot": self._screenshot,
            "get_system_status": self._system_status,
            "get_gpu_status": self._gpu_status,
            "get_top_processes": self._top_processes,
            "request_power_action": self._request_power_action,
            "confirm_pc_action": self._confirm_action,
        }
        handler = handlers.get(name)
        if handler is None:
            return None
        permission = TOOL_PERMISSIONS[name]
        logger.info("PC action requested: tool=%s permission=%s", name, permission.value)
        try:
            result = handler(arguments)
            logger.info("PC action succeeded: tool=%s", name)
            return result
        except ApplicationNotFoundError:
            app_name = self._safe_app_label(arguments.get("app_name"))
            self._spoken_override = f"I couldn't find {app_name} on this computer."
            logger.warning("Application resolution failed: app=%s", app_name)
            return {"ok": False, "error": "application_not_found", "message": self._spoken_override}
        except ProtectedProcessError:
            self._spoken_override = "I can't terminate that Windows system process."
            logger.warning("Protected process termination was blocked.")
            return {"ok": False, "error": "protected_process", "message": self._spoken_override}
        except FeatureUnavailableError as exc:
            self._spoken_override = self._failure_message(name)
            logger.warning("PC feature unavailable: tool=%s error=%s", name, type(exc).__name__)
            return {"ok": False, "error": "feature_unavailable", "message": self._spoken_override}
        except (PcControlError, OSError, ValueError, TypeError) as exc:
            self._spoken_override = self._failure_message(name)
            logger.warning("PC action failed: tool=%s error=%s", name, type(exc).__name__)
            return {"ok": False, "error": "pc_action_failed", "message": self._spoken_override}

    def _open_application(self, arguments: dict[str, Any]) -> dict[str, Any]:
        application = self._resolve_application(arguments)
        result = self._controller.open_application(application)
        display = _display_name(application)
        self._spoken_override = (
            f"{display} is already open."
            if result.get("already_running")
            else f"Opening {display}."
        )
        logger.info("Application open resolved: app=%s", application.identifier)
        return {"ok": True, "application": application.identifier, **result}

    def _close_application(self, arguments: dict[str, Any]) -> dict[str, Any]:
        application = self._resolve_application(arguments)
        result = self._controller.close_application(application)
        display = _display_name(application)
        if result.get("already_closed"):
            self._spoken_override = f"{display} isn't running."
        elif result.get("still_running"):
            self._set_pending(
                "force_close_application",
                {"app_name": application.identifier},
                f"force close {display}",
            )
            self._spoken_override = f"{display} didn't close normally. Force close it?"
            result = {**result, "confirmation_required": True}
        else:
            self._spoken_override = f"Closed {display}."
        logger.info(
            "Application close result: app=%s still_running=%s",
            application.identifier,
            bool(result.get("still_running")),
        )
        return {"ok": True, "application": application.identifier, **result}

    def _application_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        application = self._resolve_application(arguments)
        result = self._controller.is_application_running(application)
        display = _display_name(application)
        self._spoken_override = (
            f"{display} is running." if result.get("running") else f"{display} isn't running."
        )
        return {"ok": True, "application": application.identifier, **result}

    def _get_volume(self, _: dict[str, Any]) -> dict[str, Any]:
        result = self._controller.get_system_volume()
        self._spoken_override = _volume_spoken(result)
        return {"ok": True, **result}

    def _set_volume(self, arguments: dict[str, Any]) -> dict[str, Any]:
        percent = _bounded_int(arguments.get("percent"), 0, 100, "percent")
        result = self._controller.set_system_volume(percent)
        self._spoken_override = _volume_spoken(result)
        return {"ok": True, **result}

    def _change_volume(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_delta = arguments.get("delta_percent")
        if raw_delta is None:
            step = self._config.pc_control.volume_step_percent
            delta = -step if re.search(
                r"\b(?:down|decrease|lower|quieter)\b",
                self._current_prompt,
                re.IGNORECASE,
            ) else step
        else:
            delta = _bounded_int(raw_delta, -100, 100, "delta_percent")
        result = self._controller.change_system_volume(delta)
        self._spoken_override = _volume_spoken(result)
        return {"ok": True, **result}

    def _set_mute(self, muted: bool) -> dict[str, Any]:
        result = self._controller.set_system_mute(muted)
        self._spoken_override = "Muted." if muted else "Unmuted."
        return {"ok": True, **result}

    def _media(self, action: str) -> dict[str, Any]:
        result = self._controller.send_media_key(action)
        self._spoken_override = "Done."
        return {"ok": True, **result}

    def _lock(self, _: dict[str, Any]) -> dict[str, Any]:
        result = self._controller.lock_computer()
        self._spoken_override = "Locking your computer."
        return {"ok": True, **result}

    def _screenshot(self, _: dict[str, Any]) -> dict[str, Any]:
        result = self._controller.take_screenshot()
        self._spoken_override = "Screenshot saved."
        return {"ok": True, **result}

    def _system_status(self, _: dict[str, Any]) -> dict[str, Any]:
        logger.info("System-status query started.")
        result = self._controller.get_system_status()
        prompt = self._current_prompt.casefold()
        if re.search(r"\b(?:ram|memory)\b", prompt):
            self._spoken_override = (
                f"You're using about {result['memory_used_gb']:g} of "
                f"{result['memory_total_gb']:g} gigabytes of RAM."
            )
        elif re.search(r"\b(?:disk|storage)\b", prompt):
            self._spoken_override = (
                f"Your system drive has about {result['disk_free_gb']:g} gigabytes "
                f"free out of {result['disk_total_gb']:g}."
            )
        elif "uptime" in prompt or "how long" in prompt:
            self._spoken_override = _uptime_spoken(int(result["uptime_seconds"]))
        elif "cpu" in prompt:
            self._spoken_override = f"CPU usage is around {result['cpu_percent']:g} percent."
        else:
            self._spoken_override = (
                f"CPU usage is around {result['cpu_percent']:g} percent, and you're "
                f"using about {result['memory_used_gb']:g} of "
                f"{result['memory_total_gb']:g} gigabytes of RAM."
            )
        return {"ok": True, **result}

    def _gpu_status(self, _: dict[str, Any]) -> dict[str, Any]:
        logger.info("GPU-status query started.")
        result = self._controller.get_gpu_status()
        prompt = self._current_prompt.casefold()
        if "temperature" in prompt or "hot" in prompt:
            self._spoken_override = (
                f"Your GPU temperature is {result['temperature_c']:g} degrees Celsius."
            )
        elif "vram" in prompt or "memory" in prompt:
            self._spoken_override = (
                f"You're using about {result['vram_used_mb']:g} of "
                f"{result['vram_total_mb']:g} megabytes of VRAM."
            )
        else:
            self._spoken_override = (
                f"GPU usage is {result['gpu_utilization_percent']:g} percent at "
                f"{result['temperature_c']:g} degrees Celsius."
            )
        return {"ok": True, **result}

    def _top_processes(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sort_by = str(arguments.get("sort_by", "memory")).casefold()
        if sort_by not in {"cpu", "memory"}:
            raise ValueError("Invalid process sort order.")
        limit = _bounded_int(arguments.get("limit", 5), 1, 10, "limit")
        logger.info("Process-status query started: sort_by=%s limit=%d", sort_by, limit)
        result = self._controller.get_top_processes(sort_by, limit)
        self._spoken_override = _top_processes_spoken(result)
        return {"ok": True, **result}

    def _request_power_action(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action", "")).casefold()
        if action not in POWER_ACTIONS:
            raise ValueError("Invalid power action.")
        description = _power_description(action)
        self._set_pending("power_action", {"action": action}, description)
        self._spoken_override = f"{description.capitalize()} now?"
        logger.info("PC confirmation requested: action=%s", action)
        return {
            "ok": True,
            "confirmation_required": True,
            "action": action,
            "expires_at": self._pending.expires_at.isoformat() if self._pending else None,
        }

    def _confirm_action(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._confirmation_created_this_turn:
            self._spoken_override = "Please confirm that PC action in a separate response."
            return {"ok": False, "error": "confirmation_must_be_new_turn", "message": self._spoken_override}
        pending = self._pending
        if pending is None:
            self._spoken_override = "There isn't a pending PC action to confirm."
            return {"ok": False, "error": "no_pending_action", "message": self._spoken_override}
        if self._now() > pending.expires_at:
            self._pending = None
            self._spoken_override = "That confirmation expired, so I didn't do it."
            logger.info("PC confirmation expired: action=%s", pending.action)
            return {"ok": False, "error": "confirmation_expired", "message": self._spoken_override}
        if not isinstance(arguments.get("confirm"), bool):
            raise ValueError("confirm must be true or false.")
        if not arguments["confirm"]:
            self._pending = None
            self._spoken_override = f"Okay, I won't {pending.description}."
            logger.info("PC confirmation rejected: action=%s", pending.action)
            return {"ok": True, "cancelled": True, "action": pending.action}

        self._pending = None
        logger.info("PC confirmation accepted: action=%s", pending.action)
        if pending.action == "power_action":
            action = str(pending.arguments["action"])
            result = self._controller.perform_power_action(action)
            self._spoken_override = _power_started_spoken(action)
            return {"ok": True, "confirmed": True, **result}
        if pending.action == "force_close_application":
            application = self._resolve_application(pending.arguments)
            result = self._controller.force_close_application(application)
            self._spoken_override = f"Force closed {_display_name(application)}."
            return {"ok": True, "confirmed": True, "application": application.identifier, **result}
        raise ValueError("Unknown pending action.")

    def _set_pending(self, action: str, arguments: dict[str, Any], description: str) -> None:
        self._pending = PendingPcAction(
            action=action,
            arguments=dict(arguments),
            description=description,
            expires_at=self._now()
            + timedelta(seconds=self._config.pc_control.confirmation_timeout_seconds),
        )
        self._confirmation_created_this_turn = True

    def _resolve_application(self, arguments: dict[str, Any]) -> ApplicationConfig:
        raw_name = arguments.get("app_name")
        if not isinstance(raw_name, str):
            raise ApplicationNotFoundError("missing application")
        application = self._application_names.get(_normalized_name(raw_name))
        if application is None:
            raise ApplicationNotFoundError(raw_name)
        return application

    def _expected_tool(self, prompt: str) -> str | None:
        if EXTERNAL_INSTRUCTION_CONTEXT_CUE.search(prompt):
            return None
        application_named = any(
            re.search(rf"\b{re.escape(name)}\b", prompt, re.IGNORECASE)
            for application in self._applications.values()
            for name in (application.identifier, *application.aliases)
        )
        if application_named and APP_OPEN_CUE.search(prompt):
            return "open_application"
        if application_named and APP_CLOSE_CUE.search(prompt):
            return "close_application"
        if application_named and APP_STATUS_CUE.search(prompt):
            return "get_application_status"
        if APP_OPEN_CUE.search(prompt) and not NON_APPLICATION_OPEN_CUE.search(prompt):
            return "open_application"
        if APP_CLOSE_CUE.search(prompt) and not NON_APPLICATION_OPEN_CUE.search(prompt):
            return "close_application"
        if SCREENSHOT_CUE.search(prompt):
            return "take_screenshot"
        if LOCK_CUE.search(prompt):
            return "lock_computer"
        if POWER_CUE.search(prompt):
            return "request_power_action"
        if GPU_CUE.search(prompt):
            return "get_gpu_status"
        if TOP_PROCESS_CUE.search(prompt):
            return "get_top_processes"
        if SYSTEM_STATUS_CUE.search(prompt):
            return "get_system_status"
        if VOLUME_CUE.search(prompt):
            lowered = prompt.casefold()
            if "unmute" in lowered:
                return "unmute_system_volume"
            if "mute" in lowered:
                return "mute_system_volume"
            if re.search(r"\b(?:what|where|level|at)\b", lowered):
                return "get_system_volume"
            if re.search(r"\b(?:up|down|increase|decrease|raise|lower|louder|quieter)\b", lowered):
                return "change_system_volume"
            return "set_system_volume"
        if MEDIA_CUE.search(prompt):
            lowered = prompt.casefold()
            if "next" in lowered or "skip" in lowered:
                return "media_next_track"
            if "previous" in lowered:
                return "media_previous_track"
            if "stop" in lowered:
                return "media_stop"
            return "media_play_pause"
        return None

    def _failure_message(self, tool_name: str) -> str:
        if tool_name in {"get_gpu_status"}:
            return "I couldn't read the GPU status right now."
        if tool_name in {"get_system_status", "get_top_processes"}:
            return "I couldn't read the system status right now."
        if "volume" in tool_name:
            return "I couldn't change or read the volume right now."
        if tool_name == "take_screenshot":
            return "I couldn't take a screenshot."
        if tool_name in {"open_application", "close_application", "get_application_status"}:
            return "I couldn't control that application."
        return "I couldn't complete that PC action."

    def _safe_app_label(self, value: Any) -> str:
        if isinstance(value, str):
            application = self._application_names.get(_normalized_name(value))
            if application is not None:
                return _display_name(application)
        return "that application"

    def _now(self) -> datetime:
        value = self._now_provider()
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _bounded_int(value: Any, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    converted = int(value)
    if converted < minimum or converted > maximum:
        raise ValueError(f"{name} is outside the allowed range.")
    return converted


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _display_name(application: ApplicationConfig) -> str:
    return application.identifier.replace("_", " ").title()


def _volume_spoken(result: dict[str, Any]) -> str:
    percent = int(result.get("percent", 0))
    return f"The volume is muted at {percent} percent." if result.get("muted") else f"The volume is at {percent} percent."


def _power_description(action: str) -> str:
    return {
        "shutdown": "shut down the computer",
        "restart": "restart the computer",
        "logout": "log out",
        "sleep": "put the computer to sleep",
        "hibernate": "hibernate the computer",
    }[action]


def _power_started_spoken(action: str) -> str:
    return {
        "shutdown": "Shutting down in ten seconds.",
        "restart": "Restarting in ten seconds.",
        "logout": "Logging out.",
        "sleep": "Putting the computer to sleep.",
        "hibernate": "Hibernating the computer.",
    }[action]


def _top_processes_spoken(result: dict[str, Any]) -> str:
    processes = list(result.get("processes", ()))
    if not processes:
        return "I couldn't find any process usage to report."
    leader = processes[0]
    name = _spoken_process_name(str(leader.get("name", "Unknown")))
    if result.get("sort_by") == "cpu":
        usage = f"{float(leader.get('cpu_percent', 0)):g} percent"
        resource = "CPU"
    else:
        memory_gb = float(leader.get("memory_mb", 0)) / 1024
        usage = f"{memory_gb:.1f} gigabytes"
        resource = "memory"
    followers = [
        _spoken_process_name(str(item.get("name", "Unknown")))
        for item in processes[1:3]
    ]
    suffix = f", followed by {' and '.join(followers)}" if followers else ""
    return f"{name} is using the most {resource} at about {usage}{suffix}."


def _spoken_process_name(value: str) -> str:
    name = re.sub(r"\.exe$", "", value, flags=re.IGNORECASE)
    return name.replace("_", " ")


def _uptime_spoken(seconds: int) -> str:
    days, remainder = divmod(max(0, seconds), 86400)
    hours = remainder // 3600
    if days:
        return f"Your computer has been running for about {days} days and {hours} hours."
    return f"Your computer has been running for about {hours} hours."
