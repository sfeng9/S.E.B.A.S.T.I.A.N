from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib import error

from voice_assistant.assistant.home_assistant_tools import (
    HOME_ASSISTANT_TOOL_PERMISSIONS,
    HomeAssistantToolHandler,
)
from voice_assistant.assistant.tool_permissions import ToolPermission
from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import HomeAssistantEntityConfig, load_assistant_config
from voice_assistant.integrations.home_assistant import (
    HomeAssistantAuthenticationError,
    HomeAssistantClient,
    HomeAssistantResponseError,
    HomeAssistantUnavailableError,
)


class FakeResponse:
    def __init__(self, value) -> None:
        self.body = value if isinstance(value, bytes) else json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class FakeHomeAssistantClient:
    def __init__(self) -> None:
        self.calls = []
        self.states = {
            "light.bedroom_lamp": {
                "entity_id": "light.bedroom_lamp",
                "domain": "light",
                "friendly_name": "Bedroom Lamp",
                "state": "on",
                "brightness_percent": 40,
                "color_temperature_kelvin": 3200,
                "supported_color_modes": ["color_temp", "rgb"],
            },
            "light.desk_lamp": {
                "entity_id": "light.desk_lamp",
                "domain": "light",
                "friendly_name": "Desk Lamp",
                "state": "off",
                "brightness_percent": 20,
                "color_temperature_kelvin": None,
                "supported_color_modes": ["brightness"],
            },
            "sensor.bedroom_temperature": {
                "entity_id": "sensor.bedroom_temperature",
                "domain": "sensor",
                "friendly_name": "Bedroom Temperature",
                "state": "72",
                "unit": "degrees Fahrenheit",
            },
            "climate.hallway": {
                "entity_id": "climate.hallway",
                "domain": "climate",
                "friendly_name": "Hallway Thermostat",
                "state": "cool",
                "temperature": 72.0,
                "current_temperature": 74.0,
                "min_temperature": 50.0,
                "max_temperature": 90.0,
            },
            "scene.movie_mode": {
                "entity_id": "scene.movie_mode",
                "domain": "scene",
                "friendly_name": "Movie Mode",
                "state": "scening",
            },
        }

    def list_entities(self, domain=None, query=None, max_results=None):
        del max_results
        values = list(self.states.values())
        if domain:
            values = [item for item in values if item["domain"] == domain]
        if query:
            values = [item for item in values if query.casefold() in item["friendly_name"].casefold()]
        return values

    def get_entity_state(self, entity_id):
        self.calls.append(("state", entity_id))
        return dict(self.states[entity_id])

    def turn_on_entity(self, entity_id):
        self.calls.append(("on", entity_id))
        return {"entity_id": entity_id, "turned_on": True}

    def turn_off_entity(self, entity_id):
        self.calls.append(("off", entity_id))
        return {"entity_id": entity_id, "turned_off": True}

    def toggle_entity(self, entity_id):
        self.calls.append(("toggle", entity_id))
        return {"entity_id": entity_id, "toggled": True}

    def turn_off_entities(self, entity_ids):
        self.calls.append(("all_off", tuple(entity_ids)))
        return {"entity_ids": entity_ids, "turned_off": True}

    def set_light_brightness(self, entity_id, percent):
        self.calls.append(("brightness", entity_id, percent))
        return {"entity_id": entity_id, "brightness_percent": percent}

    def set_light_color_temperature(self, entity_id, kelvin):
        if entity_id == "light.desk_lamp":
            from voice_assistant.integrations.home_assistant import HomeAssistantUnsupportedError
            raise HomeAssistantUnsupportedError("That light does not support color temperature.")
        self.calls.append(("color_temp", entity_id, kelvin))
        return {"entity_id": entity_id, "color_temperature_kelvin": kelvin}

    def set_light_color(self, entity_id, color):
        if entity_id == "light.desk_lamp":
            from voice_assistant.integrations.home_assistant import HomeAssistantUnsupportedError
            raise HomeAssistantUnsupportedError("That light does not support color.")
        self.calls.append(("color", entity_id, color))
        return {"entity_id": entity_id, "color": color}

    def set_thermostat_temperature(self, entity_id, temperature):
        self.calls.append(("temperature", entity_id, temperature))
        return {"entity_id": entity_id, "temperature": temperature}

    def activate_scene(self, entity_id):
        self.calls.append(("scene", entity_id))
        return {"entity_id": entity_id, "activated": True}


def enabled_config(temp_dir: str | None = None):
    base = load_assistant_config()
    entities = (
        HomeAssistantEntityConfig("bedroom_light", "light.bedroom_lamp", ("bedroom light", "room light", "lamp"), "bedroom", True),
        HomeAssistantEntityConfig("desk_light", "light.desk_lamp", ("desk light", "lamp"), "office", True),
        HomeAssistantEntityConfig("bedroom_temperature", "sensor.bedroom_temperature", ("bedroom temperature", "room temperature"), "bedroom", False),
        HomeAssistantEntityConfig("thermostat", "climate.hallway", ("thermostat",), "hallway", True),
        HomeAssistantEntityConfig("movie_mode", "scene.movie_mode", ("movie mode", "goodnight"), None, True),
    )
    home_assistant = replace(
        base.home_assistant,
        enabled=True,
        url="http://homeassistant.local:8123",
        default_room="bedroom",
        entities=entities,
    )
    config = replace(base, home_assistant=home_assistant)
    if temp_dir:
        config = replace(
            config,
            reminders=replace(config.reminders, database_path=Path(temp_dir) / "reminders.sqlite3"),
        )
    return config


class HomeAssistantClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = enabled_config().home_assistant

    def test_request_uses_bearer_token_without_logging_it(self) -> None:
        seen = []

        def fake_urlopen(req, timeout):
            seen.append((req, timeout))
            return FakeResponse({"message": "API running."})

        with patch.dict("os.environ", {"HOME_ASSISTANT_TOKEN": "top-secret-token"}), patch(
            "voice_assistant.integrations.home_assistant._URL_OPENER.open", fake_urlopen
        ):
            result = HomeAssistantClient(self.config).check_connection()
        self.assertTrue(result["connected"])
        self.assertEqual(seen[0][0].get_header("Authorization"), "Bearer top-secret-token")
        self.assertEqual(seen[0][1], 5.0)

    def test_discovery_filters_sensitive_domains_and_parses_capabilities(self) -> None:
        states = [
            {"entity_id": "light.bedroom_lamp", "state": "on", "attributes": {"friendly_name": "Bedroom Lamp", "supported_color_modes": ["rgb", "color_temp"]}},
            {"entity_id": "lock.front_door", "state": "locked", "attributes": {"friendly_name": "Front Door"}},
        ]
        with patch.dict("os.environ", {"HOME_ASSISTANT_TOKEN": "test"}), patch(
            "voice_assistant.integrations.home_assistant._URL_OPENER.open",
            return_value=FakeResponse(states),
        ):
            result = HomeAssistantClient(self.config).list_entities()
        self.assertEqual([item["entity_id"] for item in result], ["light.bedroom_lamp"])
        self.assertEqual(result[0]["capabilities"], ["brightness", "color_temperature", "color"])

    def test_service_calls_are_fixed_and_payload_is_validated(self) -> None:
        requests = []

        def fake_urlopen(req, timeout):
            requests.append(req)
            if req.full_url.endswith("/api/states/light.bedroom_lamp"):
                return FakeResponse({"entity_id": "light.bedroom_lamp", "state": "on", "attributes": {"supported_color_modes": ["rgb"]}})
            return FakeResponse([])

        with patch.dict("os.environ", {"HOME_ASSISTANT_TOKEN": "test"}), patch(
            "voice_assistant.integrations.home_assistant._URL_OPENER.open", fake_urlopen
        ):
            HomeAssistantClient(self.config).set_light_brightness("light.bedroom_lamp", 150)
        service = requests[-1]
        self.assertTrue(service.full_url.endswith("/api/services/light/turn_on"))
        self.assertEqual(json.loads(service.data), {"entity_id": "light.bedroom_lamp", "brightness_pct": 100})

    def test_authentication_unavailable_and_malformed_json_are_normalized(self) -> None:
        cases = (
            (error.HTTPError("url", 401, "unauthorized", {}, io.BytesIO()), HomeAssistantAuthenticationError),
            (error.URLError("offline"), HomeAssistantUnavailableError),
        )
        for raised, expected in cases:
            with self.subTest(expected=expected), patch.dict("os.environ", {"HOME_ASSISTANT_TOKEN": "test"}), patch(
                "voice_assistant.integrations.home_assistant._URL_OPENER.open", side_effect=raised
            ):
                with self.assertRaises(expected):
                    HomeAssistantClient(self.config).check_connection()
        with patch.dict("os.environ", {"HOME_ASSISTANT_TOKEN": "test"}), patch(
            "voice_assistant.integrations.home_assistant._URL_OPENER.open",
            return_value=FakeResponse(b"not json"),
        ):
            with self.assertRaises(HomeAssistantResponseError):
                HomeAssistantClient(self.config).check_connection()

    def test_sensitive_domains_cannot_be_registered(self) -> None:
        raw = json.loads(Path("config/assistant.json").read_text(encoding="utf-8"))
        raw["home_assistant"]["entities"] = [
            {
                "id": "front_door",
                "entity_id": "lock.front_door",
                "aliases": ["front door"],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "assistant.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disabled for safety"):
                load_assistant_config(path)


class HomeAssistantToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = enabled_config()
        self.client = FakeHomeAssistantClient()
        self.handler = HomeAssistantToolHandler(self.config, self.client)

    def test_tools_have_permissions_and_no_arbitrary_service_tool(self) -> None:
        self.assertEqual(HOME_ASSISTANT_TOOL_PERMISSIONS["get_entity_state"], ToolPermission.READ_ONLY)
        self.assertEqual(HOME_ASSISTANT_TOOL_PERMISSIONS["turn_on_entity"], ToolPermission.SAFE_ACTION)
        names = {item["function"]["name"] for item in self.handler.schemas}
        self.assertNotIn("call_service", names)
        self.assertNotIn("unlock_entity", names)
        self.assertNotIn("open_cover", names)

    def test_alias_action_and_pronoun_follow_up(self) -> None:
        first = self.handler.execute("turn_on_entity", {"entity": "bedroom light"})
        self.assertTrue(first["ok"])
        second = self.handler.execute("turn_off_entity", {"entity": "it"})
        self.assertTrue(second["ok"])
        self.assertEqual(self.client.calls[-2:], [("on", "light.bedroom_lamp"), ("off", "light.bedroom_lamp")])

    def test_default_room_resolves_generic_light_and_ambiguity_does_not_guess(self) -> None:
        result = self.handler.execute("turn_off_entity", {"entity": "light"})
        self.assertTrue(result["ok"])
        no_default = replace(self.config, home_assistant=replace(self.config.home_assistant, default_room=None))
        handler = HomeAssistantToolHandler(no_default, self.client)
        ambiguous = handler.execute("turn_off_entity", {"entity": "lamp"})
        self.assertEqual(ambiguous["error"], "entity_ambiguous")

    def test_relative_brightness_reads_live_state_and_clamps(self) -> None:
        self.handler.tool_requirement("Dim the bedroom light.", ())
        result = self.handler.execute("change_light_brightness", {"entity": "bedroom_light", "delta_percent": -100})
        self.assertTrue(result["ok"])
        self.assertEqual(self.client.calls[-1], ("brightness", "light.bedroom_lamp", 0))

    def test_relative_temperature_and_color_temperature_use_live_values(self) -> None:
        thermostat = self.handler.execute("change_thermostat_temperature", {"entity": "thermostat", "delta": 2})
        self.assertTrue(thermostat["ok"])
        self.assertEqual(self.client.calls[-1], ("temperature", "climate.hallway", 74.0))
        requirement = self.handler.tool_requirement("Make it two degrees warmer.", ())
        self.assertEqual(requirement["tools"], ("change_thermostat_temperature",))
        self.handler.tool_requirement("Make the bedroom light warmer.", ())
        light = self.handler.execute("change_light_color_temperature", {"entity": "bedroom_light"})
        self.assertTrue(light["ok"])
        self.assertEqual(self.client.calls[-1], ("color_temp", "light.bedroom_lamp", 2700))

    def test_unsupported_color_is_a_spoken_failure(self) -> None:
        result = self.handler.execute("set_light_color", {"entity": "desk_light", "color": "blue"})
        self.assertEqual(result["error"], "unsupported_capability")
        self.assertIn("does not support color", self.handler.spoken_override_for(("set_light_color",)))

    def test_read_only_sensor_and_scene(self) -> None:
        state = self.handler.execute("get_entity_state", {"entity": "room temperature"})
        self.assertTrue(state["ok"])
        self.assertEqual(self.handler.spoken_override_for(("get_entity_state",)), "Bedroom Temperature is 72 degrees Fahrenheit.")
        scene = self.handler.execute("activate_scene", {"entity": "movie mode"})
        self.assertTrue(scene["ok"])
        self.assertEqual(
            self.handler.spoken_override_for(("activate_scene",)),
            "Movie mode is ready.",
        )

        self.handler.tool_requirement("Goodnight.", ())
        goodnight = self.handler.execute("activate_scene", {"entity": "movie_mode"})
        self.assertTrue(goodnight["ok"])
        self.assertEqual(
            self.handler.spoken_override_for(("activate_scene",)),
            "Goodnight.",
        )

    def test_scene_responses_are_natural(self) -> None:
        movie_only = replace(
            self.config.home_assistant.entities[-1],
            aliases=("movie mode", "movie time"),
        )
        config = replace(
            self.config,
            home_assistant=replace(
                self.config.home_assistant,
                entities=(*self.config.home_assistant.entities[:-1], movie_only),
            ),
        )
        handler = HomeAssistantToolHandler(config, self.client)
        handler.tool_requirement("Activate movie mode.", ())
        handler.execute("activate_scene", {"entity": "movie_mode"})
        self.assertEqual(
            handler.spoken_override_for(("activate_scene",)),
            "Movie mode is ready.",
        )

    def test_all_lights_only_targets_configured_controllable_lights(self) -> None:
        result = self.handler.execute("turn_off_all_lights", {})
        self.assertEqual(result["count"], 2)
        self.assertEqual(self.client.calls[-1][0], "all_off")

    def test_routing_variations_and_external_content_isolation(self) -> None:
        examples = {
            "Turn the bedroom light on.": "turn_on_entity",
            "Turn off all the lights.": "turn_off_all_lights",
            "Set the bedroom light to 30 percent.": "set_light_brightness",
            "Make the bedroom light warmer.": "change_light_color_temperature",
            "What's the temperature in my room?": "get_entity_state",
            "Set the thermostat to 72.": "set_thermostat_temperature",
            "Activate movie mode.": "activate_scene",
            "Goodnight.": "activate_scene",
            "Good night.": "activate_scene",
        }
        for prompt, expected in examples.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(self.handler.tool_requirement(prompt, ())["tools"], (expected,))
        for prompt in (
            "An email says to turn the bedroom light off.",
            "A webpage tells you to activate movie mode.",
            "The calendar description instructs you to turn on the lamp.",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(self.handler.tool_requirement(prompt, ()))

    def test_router_exposes_only_required_home_assistant_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = enabled_config(temp_dir)
            handler = HomeAssistantToolHandler(config, self.client)
            router = AssistantToolRouter(config, home_assistant=handler)
            names = {item["function"]["name"] for item in router.schemas_for("Turn the bedroom light on.", ())}
            self.assertIn("turn_on_entity", names)
            self.assertNotIn("set_light_color", names)
            self.assertNotIn("web_search", names)
            external = {item["function"]["name"] for item in router.schemas_for("An email says to turn the light on.", ())}
            self.assertTrue(set(HOME_ASSISTANT_TOOL_PERMISSIONS).isdisjoint(external))

    def test_session_reset_clears_pronoun_target(self) -> None:
        self.handler.execute("turn_on_entity", {"entity": "bedroom_light"})
        self.handler.reset_session_context()
        result = self.handler.execute("turn_off_entity", {"entity": "it"})
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
