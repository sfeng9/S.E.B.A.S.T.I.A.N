from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from voice_assistant.assistant.tool_permissions import ToolPermission
from voice_assistant.config import AssistantConfig, HomeAssistantEntityConfig
from voice_assistant.integrations.home_assistant import (
    ALLOWED_COLOR_NAMES,
    HomeAssistantAuthenticationError,
    HomeAssistantConfigurationError,
    HomeAssistantEntityNotFoundError,
    HomeAssistantError,
    HomeAssistantResponseError,
    HomeAssistantUnavailableError,
    HomeAssistantUnsupportedError,
    HomeAssistantClient,
    READABLE_DOMAINS,
)


logger = logging.getLogger(__name__)

HOME_ASSISTANT_TOOL_PERMISSIONS: Mapping[str, ToolPermission] = {
    "get_entity_state": ToolPermission.READ_ONLY,
    "list_entities": ToolPermission.READ_ONLY,
    "turn_on_entity": ToolPermission.SAFE_ACTION,
    "turn_off_entity": ToolPermission.SAFE_ACTION,
    "toggle_entity": ToolPermission.SAFE_ACTION,
    "turn_off_all_lights": ToolPermission.SAFE_ACTION,
    "set_light_brightness": ToolPermission.SAFE_ACTION,
    "change_light_brightness": ToolPermission.SAFE_ACTION,
    "set_light_color_temperature": ToolPermission.SAFE_ACTION,
    "change_light_color_temperature": ToolPermission.SAFE_ACTION,
    "set_light_color": ToolPermission.SAFE_ACTION,
    "set_thermostat_temperature": ToolPermission.SAFE_ACTION,
    "change_thermostat_temperature": ToolPermission.SAFE_ACTION,
    "activate_scene": ToolPermission.SAFE_ACTION,
}

EXTERNAL_INSTRUCTION_CUE = re.compile(
    r"\b(?:email|message|web(?:page|site| result)|calendar description|document)\b",
    re.IGNORECASE,
)
HA_DEVICE_CUE = re.compile(
    r"\b(?:home assistant|lights?|lamps?|bulbs?|thermostat|climate|temperature|humidity|"
    r"sensors?|fans?|plugs?|switches|scenes?|mode)\b",
    re.IGNORECASE,
)
HA_ACTION_CUE = re.compile(
    r"\b(?:turn|switch|toggle|set|make|dim|brighten|activate)\b",
    re.IGNORECASE,
)
PRONOUN_CUE = re.compile(r"^(?:it|that|this|the light|the lamp|the device)$", re.IGNORECASE)


class HomeAssistantClientProtocol(Protocol):
    def list_entities(self, domain=None, query=None, max_results=None): ...
    def get_entity_state(self, entity_id): ...
    def turn_on_entity(self, entity_id): ...
    def turn_off_entity(self, entity_id): ...
    def toggle_entity(self, entity_id): ...
    def turn_off_entities(self, entity_ids): ...
    def set_light_brightness(self, entity_id, percent): ...
    def set_light_color_temperature(self, entity_id, kelvin): ...
    def set_light_color(self, entity_id, color): ...
    def set_thermostat_temperature(self, entity_id, temperature): ...
    def activate_scene(self, entity_id): ...


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


def home_assistant_tool_schemas(config: AssistantConfig) -> tuple[dict[str, Any], ...]:
    identifiers = [entity.identifier for entity in config.home_assistant.entities]
    target = {
        "type": "string",
        "enum": identifiers,
        "description": (
            "Configured entity identifier. Resolve the user's natural name to one of "
            "these IDs. For an 'it' follow-up, reuse the preceding Home Assistant ID."
        ),
    }
    discovery = _schema(
            "list_entities",
            "Discover a short read-only list of safe Home Assistant entities. Do not use this for routine control when a configured entity already matches.",
            {
                "domain": {"type": "string", "enum": sorted(READABLE_DOMAINS)},
                "query": {"type": "string", "maxLength": 100},
            },
        )
    if not identifiers:
        return (discovery,)
    return (
        _schema("get_entity_state", "Read the live state of one configured Home Assistant entity.", {"entity": target}, ["entity"]),
        discovery,
        _schema("turn_on_entity", "Turn on one configured light, switch, or fan.", {"entity": target}, ["entity"]),
        _schema("turn_off_entity", "Turn off one configured light, switch, or fan.", {"entity": target}, ["entity"]),
        _schema("toggle_entity", "Toggle one configured light, switch, or fan.", {"entity": target}, ["entity"]),
        _schema("turn_off_all_lights", "Turn off all configured controllable lights."),
        _schema("set_light_brightness", "Set a configured light to a brightness percentage from 0 to 100.", {"entity": target, "percent": {"type": "integer", "minimum": 0, "maximum": 100}}, ["entity", "percent"]),
        _schema("change_light_brightness", f"Raise or lower a configured light's live brightness by percentage points. Use a signed amount; omit it for the configured {config.home_assistant.brightness_step_percent}-point step.", {"entity": target, "delta_percent": {"type": "integer", "minimum": -100, "maximum": 100}}, ["entity"]),
        _schema("set_light_color_temperature", "Set a configured light's color temperature in Kelvin. Use 2700 for warm white and 5000 for cool white.", {"entity": target, "kelvin": {"type": "integer", "minimum": 1500, "maximum": 10000}}, ["entity", "kelvin"]),
        _schema("change_light_color_temperature", f"Make a configured light warmer or cooler from its live value. Use a signed Kelvin delta; negative is warmer and positive is cooler. Omit it for the configured {config.home_assistant.color_temperature_step_kelvin}-Kelvin step.", {"entity": target, "delta_kelvin": {"type": "integer", "minimum": -8500, "maximum": 8500}}, ["entity"]),
        _schema("set_light_color", "Set a configured RGB-capable light to a supported named color.", {"entity": target, "color": {"type": "string", "enum": sorted(ALLOWED_COLOR_NAMES)}}, ["entity", "color"]),
        _schema("set_thermostat_temperature", "Set one configured climate entity's target temperature.", {"entity": target, "temperature": {"type": "number", "minimum": -50, "maximum": 150}}, ["entity", "temperature"]),
        _schema("change_thermostat_temperature", "Raise or lower a configured climate entity from its live target temperature. Use a signed degree delta.", {"entity": target, "delta": {"type": "number", "minimum": -20, "maximum": 20}}, ["entity", "delta"]),
        _schema("activate_scene", "Activate one configured ordinary Home Assistant scene.", {"entity": target}, ["entity"]),
    )


class HomeAssistantToolHandler:
    def __init__(
        self,
        config: AssistantConfig,
        client: HomeAssistantClientProtocol | None = None,
    ) -> None:
        self._config = config.home_assistant
        self._client = client or HomeAssistantClient(self._config)
        self._entities = {item.identifier: item for item in self._config.entities}
        self._names: dict[str, list[HomeAssistantEntityConfig]] = {}
        for entity in self._config.entities:
            names = (entity.identifier, entity.entity_id, *entity.aliases)
            for name in names:
                self._names.setdefault(_normalize(name), []).append(entity)
        self._schemas = home_assistant_tool_schemas(config) if self._config.enabled else ()
        self._last_entity: HomeAssistantEntityConfig | None = None
        self._spoken_override: str | None = None
        self._current_prompt = ""

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return self._schemas

    @property
    def permissions(self) -> Mapping[str, ToolPermission]:
        return HOME_ASSISTANT_TOOL_PERMISSIONS

    def begin_turn(self) -> None:
        self._spoken_override = None

    def reset_session_context(self) -> None:
        self._last_entity = None
        self._spoken_override = None
        self._current_prompt = ""
        logger.info("Cleared transient Home Assistant entity context.")

    def matches_prompt(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]] = (),
    ) -> bool:
        del history
        if not self._config.enabled or EXTERNAL_INSTRUCTION_CUE.search(prompt):
            return False
        return self._expected_tool(prompt) is not None

    def tool_requirement(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        del history
        self._current_prompt = prompt
        if not self._config.enabled or EXTERNAL_INSTRUCTION_CUE.search(prompt):
            return None
        expected = self._expected_tool(prompt)
        if expected is None:
            return None
        return {
            "tools": (expected,),
            "instruction": (
                f"This is a direct smart-home request. Call {expected} now using only "
                "a configured entity identifier. Never claim success without its result."
            ),
            "fallback": "I couldn't complete that Home Assistant request.",
        }

    def spoken_override_for(self, called_tools: Sequence[str]) -> str | None:
        if set(called_tools).intersection(HOME_ASSISTANT_TOOL_PERMISSIONS):
            return self._spoken_override
        return None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if name not in HOME_ASSISTANT_TOOL_PERMISSIONS:
            return None
        self._spoken_override = None
        try:
            if name == "list_entities":
                return self._list_entities(arguments)
            if name == "turn_off_all_lights":
                return self._turn_off_all_lights()

            domains = _domains_for_tool(name)
            entity = self._resolve_entity(str(arguments.get("entity", "")), domains)
            if entity is None:
                return self._resolution_failure(domains)
            if name not in {"get_entity_state"} and not entity.allow_control:
                return self._failure("control_disabled", "Control is disabled for that device.")
            self._last_entity = entity
            logger.info(
                "Home Assistant entity resolved: registry_id=%s entity_id=%s",
                entity.identifier,
                entity.entity_id,
            )
            if name == "get_entity_state":
                result = self._client.get_entity_state(entity.entity_id)
                self._spoken_override = _state_spoken(result)
            elif name == "turn_on_entity":
                result = self._client.turn_on_entity(entity.entity_id)
                self._spoken_override = f"Turned on {_display_name(entity)}."
            elif name == "turn_off_entity":
                result = self._client.turn_off_entity(entity.entity_id)
                self._spoken_override = f"Turned off {_display_name(entity)}."
            elif name == "toggle_entity":
                result = self._client.toggle_entity(entity.entity_id)
                self._spoken_override = f"Toggled {_display_name(entity)}."
            elif name == "set_light_brightness":
                percent = max(0, min(100, int(arguments.get("percent"))))
                result = self._client.set_light_brightness(entity.entity_id, percent)
                self._spoken_override = f"Set {_display_name(entity)} to {percent} percent."
            elif name == "change_light_brightness":
                state = self._client.get_entity_state(entity.entity_id)
                current = state.get("brightness_percent")
                if current is None:
                    raise HomeAssistantUnsupportedError("That light doesn't report brightness control.")
                default_delta = self._config.brightness_step_percent
                delta = int(arguments.get("delta_percent", _signed_default(self._current_prompt, default_delta)))
                percent = max(0, min(100, int(current) + delta))
                result = self._client.set_light_brightness(entity.entity_id, percent)
                self._spoken_override = f"Set {_display_name(entity)} to {percent} percent."
            elif name == "set_light_color_temperature":
                kelvin = max(1500, min(10000, int(arguments.get("kelvin"))))
                result = self._client.set_light_color_temperature(entity.entity_id, kelvin)
                self._spoken_override = f"Set {_display_name(entity)} to {kelvin} Kelvin."
            elif name == "change_light_color_temperature":
                state = self._client.get_entity_state(entity.entity_id)
                current = state.get("color_temperature_kelvin")
                if current is None:
                    raise HomeAssistantUnsupportedError("That light doesn't support color temperature.")
                step = self._config.color_temperature_step_kelvin
                default_delta = -step if re.search(r"\bwarmer\b", self._current_prompt, re.IGNORECASE) else step
                delta = int(arguments.get("delta_kelvin", default_delta))
                kelvin = max(1500, min(10000, round(float(current) + delta)))
                result = self._client.set_light_color_temperature(entity.entity_id, kelvin)
                self._spoken_override = f"Adjusted {_display_name(entity)} to {kelvin} Kelvin."
            elif name == "set_light_color":
                color = str(arguments.get("color", "")).casefold().strip()
                result = self._client.set_light_color(entity.entity_id, color)
                self._spoken_override = f"Set {_display_name(entity)} to {color}."
            elif name == "set_thermostat_temperature":
                temperature = float(arguments.get("temperature"))
                result = self._client.set_thermostat_temperature(entity.entity_id, temperature)
                self._spoken_override = f"Set {_display_name(entity)} to {_number(temperature)} degrees."
            elif name == "change_thermostat_temperature":
                state = self._client.get_entity_state(entity.entity_id)
                current = state.get("temperature")
                if current is None:
                    raise HomeAssistantUnsupportedError("That thermostat doesn't report a target temperature.")
                delta = float(arguments.get("delta"))
                temperature = float(current) + delta
                result = self._client.set_thermostat_temperature(entity.entity_id, temperature)
                self._spoken_override = f"Set {_display_name(entity)} to {_number(temperature)} degrees."
            else:
                result = self._client.activate_scene(entity.entity_id)
                self._spoken_override = _scene_spoken(
                    entity,
                    self._current_prompt,
                    str(arguments.get("entity", "")),
                )
            logger.info("Home Assistant action succeeded: tool=%s entity=%s", name, entity.entity_id)
            return {"ok": True, "registry_id": entity.identifier, **result}
        except HomeAssistantUnsupportedError as exc:
            logger.info("Home Assistant capability unsupported: tool=%s error=%s", name, exc)
            return self._failure("unsupported_capability", str(exc))
        except HomeAssistantConfigurationError as exc:
            logger.warning("Home Assistant is unconfigured: %s", exc)
            return self._failure("home_assistant_unconfigured", "Home Assistant isn't configured yet.")
        except HomeAssistantAuthenticationError:
            logger.warning("Home Assistant authentication failed.")
            return self._failure("home_assistant_authentication_failed", "I couldn't authenticate with Home Assistant.")
        except HomeAssistantEntityNotFoundError:
            logger.warning("Home Assistant entity was not found: tool=%s", name)
            return self._failure("entity_not_found", "I couldn't find that device in Home Assistant.")
        except HomeAssistantUnavailableError:
            logger.warning("Home Assistant request failed because the server was unavailable.")
            return self._failure("home_assistant_unavailable", "I couldn't reach Home Assistant right now.")
        except HomeAssistantResponseError:
            logger.warning("Home Assistant returned an invalid response.")
            return self._failure("home_assistant_bad_response", "Home Assistant returned an invalid response.")
        except (TypeError, ValueError) as exc:
            logger.info("Rejected invalid Home Assistant tool arguments: tool=%s error=%s", name, exc)
            return self._failure("invalid_arguments", "That smart-home request had an invalid value.")
        except HomeAssistantError as exc:
            logger.warning("Home Assistant action failed: %s", exc)
            return self._failure("home_assistant_error", "I couldn't complete that Home Assistant request.")

    def _list_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        domain = arguments.get("domain")
        query = arguments.get("query")
        results = self._client.list_entities(domain=domain, query=query)
        if not results:
            self._spoken_override = "I couldn't find any matching Home Assistant devices."
        else:
            names = [str(item.get("friendly_name") or item.get("entity_id")) for item in results[:5]]
            self._spoken_override = "I found " + ", ".join(names) + "."
        return {"ok": True, "entities": results, "count": len(results)}

    def _turn_off_all_lights(self) -> dict[str, Any]:
        lights = [
            item for item in self._config.entities
            if item.entity_id.startswith("light.") and item.allow_control
        ]
        if not lights:
            return self._failure("entity_not_configured", "No controllable lights are configured.")
        result = self._client.turn_off_entities([item.entity_id for item in lights])
        self._spoken_override = "Turned off all configured lights."
        logger.info("Home Assistant all-lights action succeeded: count=%d", len(lights))
        return {"ok": True, "count": len(lights), **result}

    def _resolve_entity(
        self,
        value: str,
        domains: set[str] | None,
    ) -> HomeAssistantEntityConfig | None:
        self._ambiguous: list[HomeAssistantEntityConfig] = []
        normalized = _normalize(value)
        candidates = [
            item for item in self._config.entities
            if domains is None or item.entity_id.split(".", 1)[0] in domains
        ]
        if PRONOUN_CUE.fullmatch(normalized):
            return self._last_entity if self._last_entity in candidates else None
        exact = [item for item in self._names.get(normalized, ()) if item in candidates]
        if len(exact) == 1:
            return exact[0]
        matches = [
            item for item in candidates
            if any(_normalize(alias) in normalized or normalized in _normalize(alias)
                   for alias in (item.identifier, *item.aliases))
        ]
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            return matches[0]
        room = _normalize(self._config.default_room or "")
        if room:
            room_matches = [item for item in (matches or candidates) if _normalize(item.room or "") == room]
            if len(room_matches) == 1:
                return room_matches[0]
        if not normalized and self._last_entity in candidates:
            return self._last_entity
        if len(candidates) == 1 and normalized in {"", "light", "lamp", "thermostat", "fan", "switch", "sensor", "scene"}:
            return candidates[0]
        self._ambiguous = matches if len(matches) > 1 else []
        return None

    def _resolution_failure(self, domains: set[str] | None) -> dict[str, Any]:
        ambiguous = getattr(self, "_ambiguous", [])
        if ambiguous:
            logger.info("Home Assistant entity resolution was ambiguous: count=%d", len(ambiguous))
            choices = [_display_name(item) for item in ambiguous[:5]]
            return self._failure("entity_ambiguous", "Which device do you mean: " + " or ".join(choices) + "?")
        domain_name = next(iter(domains)) if domains and len(domains) == 1 else "device"
        return self._failure("entity_not_configured", f"I couldn't find a configured {domain_name} matching that name.")

    def _failure(self, error: str, message: str) -> dict[str, Any]:
        self._spoken_override = message
        return {"ok": False, "error": error, "message": message}

    def _expected_tool(self, prompt: str) -> str | None:
        text = prompt.casefold()
        normalized_text = _normalize(prompt)
        if not HA_DEVICE_CUE.search(text) and not self._configured_alias_in_prompt(text, {"scene"}):
            if self._last_entity is None or not re.search(r"\b(?:it|that|this)\b", text):
                return None
        if re.search(r"\b(?:all|every)\b.{0,15}\blights?\b", text) and re.search(r"\b(?:off|out)\b", text):
            return "turn_off_all_lights"
        if re.search(r"\b(?:scene|mode|movie time|goodnight)\b", text) and HA_ACTION_CUE.search(text):
            return "activate_scene"
        if "goodnight" in normalized_text and self._configured_alias_in_prompt(text, {"scene"}):
            return "activate_scene"
        last_domain = (
            self._last_entity.entity_id.split(".", 1)[0]
            if self._last_entity is not None
            else None
        )
        if re.search(r"\b(?:warmer|cooler)\b", text) and (
            re.search(r"\b(?:thermostat|climate)\b", text)
            or (
                last_domain == "climate"
                and not re.search(r"\b(?:lights?|lamps?|bulbs?)\b", text)
            )
        ):
            return "change_thermostat_temperature"
        if re.search(r"\b(?:thermostat|climate)\b", text) and re.search(r"\b(?:set|make)\b", text):
            return "set_thermostat_temperature"
        if re.search(r"\b(?:dim|dimmer|brighter|brighten)\b", text):
            return "change_light_brightness"
        if re.search(r"\b(?:brightness|percent)\b", text):
            return "set_light_brightness"
        if re.search(r"\b(?:warmer|cooler)\b", text):
            return "change_light_color_temperature"
        if re.search(r"\b(?:warm white|cool white|kelvin|color temperature)\b", text):
            return "set_light_color_temperature"
        if re.search(r"\b(?:red|orange|yellow|green|cyan|blue|purple|pink|white)\b", text) and HA_ACTION_CUE.search(text):
            return "set_light_color"
        if re.search(r"\b(?:turn|switch)\b.{0,40}\bon\b", text):
            return "turn_on_entity"
        if re.search(r"\b(?:turn|switch)\b.{0,40}\boff\b", text):
            return "turn_off_entity"
        if re.search(r"\btoggle\b", text):
            return "toggle_entity"
        if re.search(r"\b(?:activate|turn on)\b", text) and re.search(r"\b(?:scene|mode)\b", text):
            return "activate_scene"
        if re.search(r"\b(?:list|find|show|discover)\b", text) and "home assistant" in text:
            return "list_entities"
        if re.search(r"\b(?:is|what|status|state|temperature|humidity|battery|brightness)\b", text):
            return "get_entity_state"
        return None

    def _configured_alias_in_prompt(self, prompt: str, domains: set[str]) -> bool:
        return any(
            item.entity_id.split(".", 1)[0] in domains
            and any(_normalize(alias) in _normalize(prompt) for alias in item.aliases)
            for item in self._config.entities
        )


def _domains_for_tool(name: str) -> set[str] | None:
    if name in {"set_light_brightness", "change_light_brightness", "set_light_color_temperature", "change_light_color_temperature", "set_light_color"}:
        return {"light"}
    if name in {"set_thermostat_temperature", "change_thermostat_temperature"}:
        return {"climate"}
    if name == "activate_scene":
        return {"scene"}
    if name in {"turn_on_entity", "turn_off_entity", "toggle_entity"}:
        return {"light", "switch", "fan"}
    return None


def _normalize(value: str) -> str:
    normalized = " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    )
    return re.sub(r"\bgood night\b", "goodnight", normalized)


def _display_name(entity: HomeAssistantEntityConfig) -> str:
    return entity.aliases[0] if entity.aliases else entity.identifier.replace("_", " ")


def _scene_spoken(
    entity: HomeAssistantEntityConfig,
    prompt: str,
    requested_name: str,
) -> str:
    if "goodnight" in _normalize(prompt) or _normalize(requested_name) == "goodnight":
        return "Goodnight."
    return f"{_display_name(entity).capitalize()} is ready."


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _signed_default(prompt: str, magnitude: int) -> int:
    if re.search(r"\b(?:dim|dimmer|lower|decrease|down)\b", prompt, re.IGNORECASE):
        return -magnitude
    return magnitude


def _state_spoken(state: dict[str, Any]) -> str:
    name = str(state.get("friendly_name") or state.get("entity_id") or "That device")
    value = str(state.get("state", "unknown"))
    unit = _spoken_unit(state.get("unit"))
    if state.get("domain") == "light":
        brightness = state.get("brightness_percent")
        suffix = f" at {brightness} percent" if brightness is not None and value == "on" else ""
        return f"{name} is {value}{suffix}."
    if state.get("domain") == "climate":
        current = state.get("current_temperature")
        target = state.get("temperature")
        if current is not None and target is not None:
            return f"{name} is {_number(float(current))} degrees, set to {_number(float(target))}."
    spoken_value = f"{value} {unit}".strip()
    return f"{name} is {spoken_value}."


def _spoken_unit(value: Any) -> str:
    unit = str(value or "").strip()
    return {
        "°F": "degrees Fahrenheit",
        "°C": "degrees Celsius",
        "%": "percent",
    }.get(unit, unit)
