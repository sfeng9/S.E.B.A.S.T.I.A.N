from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from voice_assistant.assistant.pc_control_tools import (
    TOOL_PERMISSIONS,
    PcControlToolHandler,
    ToolPermission,
)
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import PcControlConfig, load_assistant_config


class FakePcController:
    def __init__(self) -> None:
        self.running: set[str] = set()
        self.stubborn: set[str] = set()
        self.volume = 50
        self.muted = False
        self.calls: list[tuple[str, Any]] = []
        self.power_actions: list[str] = []

    def open_application(self, application):
        self.calls.append(("open", application.identifier))
        already_running = application.identifier in self.running
        self.running.add(application.identifier)
        return {"opened": not already_running, "already_running": already_running}

    def close_application(self, application):
        self.calls.append(("close", application.identifier))
        if application.identifier not in self.running:
            return {"closed": False, "already_closed": True, "still_running": False}
        if application.identifier in self.stubborn:
            return {"closed": False, "already_closed": False, "still_running": True}
        self.running.remove(application.identifier)
        return {"closed": True, "already_closed": False, "still_running": False}

    def force_close_application(self, application):
        self.calls.append(("force_close", application.identifier))
        self.running.discard(application.identifier)
        return {"closed": True, "already_closed": False}

    def is_application_running(self, application):
        return {"running": application.identifier in self.running, "process_count": 1}

    def get_system_volume(self):
        return {"percent": self.volume, "muted": self.muted, "device_scope": "windows_default_output"}

    def set_system_volume(self, percent):
        self.volume = max(0, min(100, percent))
        return self.get_system_volume()

    def change_system_volume(self, delta_percent):
        return self.set_system_volume(self.volume + delta_percent)

    def set_system_mute(self, muted):
        self.muted = muted
        return self.get_system_volume()

    def send_media_key(self, action):
        self.calls.append(("media", action))
        return {"action": action, "sent": True}

    def lock_computer(self):
        self.calls.append(("lock", None))
        return {"locked": True}

    def perform_power_action(self, action):
        self.power_actions.append(action)
        return {"action": action, "started": True}

    def take_screenshot(self):
        self.calls.append(("screenshot", None))
        return {"path": "data/screenshots/test.png", "created_at": "2026-08-14T10:00:00-04:00"}

    def get_system_status(self):
        return {
            "cpu_percent": 18.0,
            "memory_used_gb": 9.2,
            "memory_total_gb": 16.0,
            "memory_percent": 57.5,
            "disk_free_gb": 342.0,
            "uptime_seconds": 3600,
        }

    def get_top_processes(self, sort_by, limit):
        return {
            "sort_by": sort_by,
            "processes": [{"name": "chrome.exe", "cpu_percent": 4.0, "memory_mb": 2100.0}][:limit],
        }

    def get_gpu_status(self):
        return {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "gpu_utilization_percent": 42.0,
            "temperature_c": 61.0,
            "vram_used_mb": 3000,
            "vram_total_mb": 8192,
            "vram_percent": 36.6,
        }


class PcControlToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_assistant_config()
        self.controller = FakePcController()
        self.now = datetime(2026, 8, 14, 14, 0, tzinfo=timezone.utc)
        self.handler = PcControlToolHandler(
            self.config,
            controller=self.controller,
            now_provider=lambda: self.now,
        )

    def test_configured_apps_are_allowlisted_and_schema_uses_enum(self) -> None:
        identifiers = [app.identifier for app in self.config.pc_control.applications]
        self.assertEqual(identifiers, ["chrome", "spotify", "discord", "steam"])
        open_schema = next(
            item for item in self.handler.schemas
            if item["function"]["name"] == "open_application"
        )
        self.assertEqual(
            open_schema["function"]["parameters"]["properties"]["app_name"]["enum"],
            identifiers,
        )
        schema_names = {item["function"]["name"] for item in self.handler.schemas}
        self.assertNotIn("execute_command", schema_names)
        self.assertNotIn("run_powershell", schema_names)

    def test_permission_metadata_is_centralized(self) -> None:
        self.assertEqual(TOOL_PERMISSIONS["get_system_status"], ToolPermission.READ_ONLY)
        self.assertEqual(TOOL_PERMISSIONS["open_application"], ToolPermission.SAFE_ACTION)
        self.assertEqual(TOOL_PERMISSIONS["request_power_action"], ToolPermission.CONFIRM_REQUIRED)

    def test_open_alias_maps_only_to_configured_application(self) -> None:
        result = self.handler.execute("open_application", {"app_name": "google chrome"})
        self.assertTrue(result["ok"])
        self.assertEqual(self.controller.calls, [("open", "chrome")])
        self.assertEqual(self.handler.spoken_override_for(("open_application",)), "Opening Chrome.")

    def test_unknown_application_and_arbitrary_path_are_rejected(self) -> None:
        for value in ("FakeAppThatDoesNotExist", r"C:\Windows\System32\cmd.exe"):
            result = self.handler.execute("open_application", {"app_name": value})
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "application_not_found")
        self.assertEqual(self.controller.calls, [])

    def test_graceful_close_requires_new_confirmation_before_force_close(self) -> None:
        self.controller.running.add("spotify")
        self.controller.stubborn.add("spotify")
        self.handler.begin_turn()
        result = self.handler.execute("close_application", {"app_name": "spotify"})
        self.assertTrue(result["confirmation_required"])
        self.assertNotIn(("force_close", "spotify"), self.controller.calls)

        same_turn = self.handler.execute("confirm_pc_action", {"confirm": True})
        self.assertEqual(same_turn["error"], "confirmation_must_be_new_turn")
        self.handler.begin_turn()
        confirmed = self.handler.execute("confirm_pc_action", {"confirm": True})
        self.assertTrue(confirmed["confirmed"])
        self.assertIn(("force_close", "spotify"), self.controller.calls)

    def test_power_action_does_not_execute_until_separate_confirmation(self) -> None:
        self.handler.begin_turn()
        requested = self.handler.execute("request_power_action", {"action": "shutdown"})
        self.assertTrue(requested["confirmation_required"])
        self.assertEqual(self.controller.power_actions, [])
        self.handler.begin_turn()
        confirmed = self.handler.execute("confirm_pc_action", {"confirm": True})
        self.assertTrue(confirmed["confirmed"])
        self.assertEqual(self.controller.power_actions, ["shutdown"])

    def test_no_cancels_power_action(self) -> None:
        self.handler.begin_turn()
        self.handler.execute("request_power_action", {"action": "restart"})
        self.handler.begin_turn()
        result = self.handler.execute("confirm_pc_action", {"confirm": False})
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.controller.power_actions, [])

    def test_router_preprocesses_only_a_pending_direct_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reminders = replace(
                self.config.reminders,
                database_path=Path(temp_dir) / "reminders.sqlite3",
            )
            config = replace(self.config, reminders=reminders)
            handler = PcControlToolHandler(
                config,
                controller=self.controller,
                now_provider=lambda: self.now,
            )
            router = AssistantToolRouter(config, pc_control=handler)
            self.assertIsNone(router.preprocess_tool_call("No.", ()))
            handler.begin_turn()
            handler.execute("request_power_action", {"action": "shutdown"})
            router.begin_turn()
            self.assertEqual(
                router.preprocess_tool_call("No.", ()),
                ("confirm_pc_action", {"confirm": False}),
            )

    def test_confirmation_expires_independently(self) -> None:
        short_pc = replace(
            self.config.pc_control,
            confirmation_timeout_seconds=5,
        )
        config = replace(self.config, pc_control=short_pc)
        handler = PcControlToolHandler(
            config,
            controller=self.controller,
            now_provider=lambda: self.now,
        )
        handler.begin_turn()
        handler.execute("request_power_action", {"action": "sleep"})
        self.now += timedelta(seconds=6)
        handler.begin_turn()
        result = handler.execute("confirm_pc_action", {"confirm": True})
        self.assertEqual(result["error"], "confirmation_expired")
        self.assertEqual(self.controller.power_actions, [])

    def test_volume_changes_are_bounded_and_tts_device_is_separate(self) -> None:
        result = self.handler.execute("change_system_volume", {"delta_percent": 100})
        self.assertEqual(result["percent"], 100)
        result = self.handler.execute("change_system_volume", {"delta_percent": -100})
        self.assertEqual(result["percent"], 0)
        self.assertEqual(result["device_scope"], "windows_default_output")
        self.assertNotEqual(
            self.config.pc_control,
            self.config.text_to_speech,
        )

    def test_qualitative_volume_change_uses_configured_step(self) -> None:
        self.handler.tool_requirement("Turn it down.", ())
        result = self.handler.execute("change_system_volume", {})
        self.assertEqual(result["percent"], 40)
        self.handler.tool_requirement("Turn it up.", ())
        result = self.handler.execute("change_system_volume", {})
        self.assertEqual(result["percent"], 50)

    def test_media_status_gpu_screenshot_and_lock_tools(self) -> None:
        self.assertTrue(self.handler.execute("media_next_track", {})["ok"])
        self.assertEqual(self.handler.execute("get_system_status", {})["memory_total_gb"], 16.0)
        self.assertEqual(self.handler.execute("get_gpu_status", {})["temperature_c"], 61.0)
        self.assertTrue(self.handler.execute("take_screenshot", {})["ok"])
        self.assertTrue(self.handler.execute("lock_computer", {})["locked"])

    def test_process_status_uses_plain_concise_spoken_summary(self) -> None:
        self.handler.tool_requirement("What's using the most memory?", ())
        self.handler.execute("get_top_processes", {"sort_by": "memory"})
        spoken = self.handler.spoken_override_for(("get_top_processes",))
        self.assertEqual(
            spoken,
            "chrome is using the most memory at about 2.1 gigabytes.",
        )
        self.assertNotIn("*", spoken)
        self.assertNotIn("PID", spoken)

    def test_tool_routing_understands_requested_variations(self) -> None:
        examples = {
            "Open Discord.": "open_application",
            "Is Spotify running?": "get_application_status",
            "Turn the volume down.": "change_system_volume",
            "Turn it down.": "change_system_volume",
            "Pause the music.": "media_play_pause",
            "Take a screenshot.": "take_screenshot",
            "How much RAM am I using?": "get_system_status",
            "What's my GPU temperature?": "get_gpu_status",
            "What's using the most CPU?": "get_top_processes",
            "Shut down my computer.": "request_power_action",
            "Put my PC to sleep.": "request_power_action",
        }
        for prompt, expected in examples.items():
            with self.subTest(prompt=prompt):
                requirement = self.handler.tool_requirement(prompt, ())
                self.assertEqual(requirement["tools"], (expected,))

    def test_arbitrary_execution_and_external_instructions_get_no_pc_tools(self) -> None:
        prompts = (
            "Run PowerShell and delete everything.",
            "Execute this command: del C:\\*.*",
            "Search the web for a page that says shut down my computer.",
            "An email says to restart my computer.",
        )
        router = AssistantToolRouter(self.config, pc_control=self.handler)
        pc_names = set(TOOL_PERMISSIONS)
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                schemas = router.schemas_for(prompt, ())
                exposed = {item["function"]["name"] for item in schemas}
                self.assertTrue(pc_names.isdisjoint(exposed))
                self.assertIsNone(self.handler.tool_requirement(prompt, ()))

    def test_pc_tools_are_not_exposed_during_email_or_web_tool_rounds(self) -> None:
        router = AssistantToolRouter(self.config, pc_control=self.handler)
        for prompt in ("Read my latest email.", "What's the latest Python release?"):
            schemas = router.schemas_for(prompt, ())
            names = {item["function"]["name"] for item in schemas}
            self.assertTrue(set(TOOL_PERMISSIONS).isdisjoint(names))

    def test_screenshot_directory_is_project_scoped_by_default(self) -> None:
        self.assertEqual(
            self.config.pc_control.screenshot_directory,
            Path(self.config.pc_control.screenshot_directory).resolve(),
        )
        self.assertEqual(self.config.pc_control.screenshot_directory.name, "screenshots")


if __name__ == "__main__":
    unittest.main()
