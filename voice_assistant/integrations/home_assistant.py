from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib import error, parse, request

from voice_assistant.config import HomeAssistantConfig
from voice_assistant.http_utils import validated_http_url


logger = logging.getLogger(__name__)


class _NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


_URL_OPENER = request.build_opener(_NoRedirectHandler())

ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z0-9_]+$")
READABLE_DOMAINS = frozenset(
    {"light", "switch", "fan", "sensor", "binary_sensor", "climate", "scene"}
)
ON_OFF_DOMAINS = frozenset({"light", "switch", "fan"})
ALLOWED_COLOR_NAMES = frozenset(
    {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "white"}
)
COLOR_MODES = frozenset({"hs", "xy", "rgb", "rgbw", "rgbww"})
BRIGHTNESS_MODES = COLOR_MODES | {"brightness", "color_temp", "white"}


class HomeAssistantError(RuntimeError):
    pass


class HomeAssistantConfigurationError(HomeAssistantError):
    pass


class HomeAssistantAuthenticationError(HomeAssistantError):
    pass


class HomeAssistantUnavailableError(HomeAssistantError):
    pass


class HomeAssistantResponseError(HomeAssistantError):
    pass


class HomeAssistantEntityNotFoundError(HomeAssistantError):
    pass


class HomeAssistantUnsupportedError(HomeAssistantError):
    pass


class HomeAssistantClient:
    """Small REST client exposing only predefined safe Home Assistant operations."""

    def __init__(self, config: HomeAssistantConfig) -> None:
        self._config = config
        self._base_url: str | None = None
        self._url_error: str | None = None
        if config.url:
            try:
                self._base_url = validated_http_url(config.url)
            except ValueError as exc:
                self._url_error = str(exc)

    def check_connection(self) -> dict[str, Any]:
        response = self._request("GET", "/api/")
        return {
            "connected": isinstance(response, dict),
            "message": str(response.get("message", "")) if isinstance(response, dict) else "",
        }

    def list_entities(
        self,
        domain: str | None = None,
        query: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized_domain = domain.casefold().strip() if domain else None
        if normalized_domain is not None and normalized_domain not in READABLE_DOMAINS:
            raise ValueError("Unsupported Home Assistant discovery domain.")
        normalized_query = str(query or "").strip().casefold()
        if len(normalized_query) > 100:
            raise ValueError("Home Assistant entity query is too long.")
        limit = max(1, min(self._config.max_results, int(max_results or self._config.max_results)))
        response = self._request("GET", "/api/states")
        if not isinstance(response, list):
            raise HomeAssistantResponseError("Home Assistant returned malformed states.")

        results: list[dict[str, Any]] = []
        for item in response:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id", "")).casefold()
            if not ENTITY_ID_PATTERN.fullmatch(entity_id):
                continue
            entity_domain = entity_id.split(".", 1)[0]
            if entity_domain not in READABLE_DOMAINS:
                continue
            attributes = item.get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            friendly_name = str(attributes.get("friendly_name", entity_id))
            if normalized_domain and entity_domain != normalized_domain:
                continue
            if normalized_query and normalized_query not in (
                f"{entity_id} {friendly_name}".casefold()
            ):
                continue
            results.append(
                {
                    "entity_id": entity_id,
                    "friendly_name": friendly_name,
                    "domain": entity_domain,
                    "state": str(item.get("state", "unknown")),
                    "unit": _optional_text(attributes.get("unit_of_measurement")),
                    "capabilities": _capabilities(entity_domain, attributes),
                }
            )
            if len(results) >= limit:
                break
        logger.info(
            "Home Assistant entity discovery complete: domain=%s query=%s count=%d",
            normalized_domain or "any-safe",
            normalized_query or "none",
            len(results),
        )
        return results

    def get_entity_state(self, entity_id: str) -> dict[str, Any]:
        normalized = _validated_entity_id(entity_id)
        domain = normalized.split(".", 1)[0]
        if domain not in READABLE_DOMAINS:
            raise HomeAssistantUnsupportedError(f"Unsupported entity domain: {domain}")
        try:
            response = self._request(
                "GET",
                f"/api/states/{parse.quote(normalized, safe='.')}",
            )
        except HomeAssistantEntityNotFoundError:
            raise
        if not isinstance(response, dict):
            raise HomeAssistantResponseError("Home Assistant returned malformed state data.")
        attributes = response.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        result = {
            "entity_id": normalized,
            "domain": domain,
            "friendly_name": str(attributes.get("friendly_name", normalized)),
            "state": str(response.get("state", "unknown")),
            "unit": _optional_text(attributes.get("unit_of_measurement")),
            "device_class": _optional_text(attributes.get("device_class")),
            "brightness_percent": _brightness_percent(attributes.get("brightness")),
            "color_temperature_kelvin": _optional_number(
                attributes.get("color_temp_kelvin")
            ),
            "supported_color_modes": _string_list(
                attributes.get("supported_color_modes")
            ),
            "temperature": _optional_number(attributes.get("temperature")),
            "current_temperature": _optional_number(
                attributes.get("current_temperature")
            ),
            "min_temperature": _optional_number(attributes.get("min_temp")),
            "max_temperature": _optional_number(attributes.get("max_temp")),
        }
        logger.info("Home Assistant state query succeeded: entity=%s", normalized)
        return result

    def turn_on_entity(self, entity_id: str) -> dict[str, Any]:
        state = self._require_domain(entity_id, ON_OFF_DOMAINS)
        self._call_service(state["domain"], "turn_on", [state["entity_id"]])
        return {"entity_id": state["entity_id"], "turned_on": True}

    def turn_off_entity(self, entity_id: str) -> dict[str, Any]:
        state = self._require_domain(entity_id, ON_OFF_DOMAINS)
        self._call_service(state["domain"], "turn_off", [state["entity_id"]])
        return {"entity_id": state["entity_id"], "turned_off": True}

    def toggle_entity(self, entity_id: str) -> dict[str, Any]:
        state = self._require_domain(entity_id, ON_OFF_DOMAINS)
        self._call_service(state["domain"], "toggle", [state["entity_id"]])
        return {"entity_id": state["entity_id"], "toggled": True}

    def turn_off_entities(self, entity_ids: list[str]) -> dict[str, Any]:
        normalized = [_validated_entity_id(value) for value in entity_ids]
        if not normalized or len(normalized) > 50:
            raise ValueError("The Home Assistant entity target list is invalid.")
        states = [self._require_domain(value, {"light"}) for value in normalized]
        self._call_service("light", "turn_off", [item["entity_id"] for item in states])
        return {"entity_ids": normalized, "turned_off": True}

    def set_light_brightness(self, entity_id: str, percent: int) -> dict[str, Any]:
        state = self._require_domain(entity_id, {"light"})
        if not _supports_brightness(state):
            raise HomeAssistantUnsupportedError("That light does not support brightness.")
        bounded = max(0, min(100, int(percent)))
        self._call_service(
            "light",
            "turn_on",
            [state["entity_id"]],
            {"brightness_pct": bounded},
        )
        return {"entity_id": state["entity_id"], "brightness_percent": bounded}

    def set_light_color_temperature(
        self,
        entity_id: str,
        kelvin: int,
    ) -> dict[str, Any]:
        state = self._require_domain(entity_id, {"light"})
        if "color_temp" not in set(state["supported_color_modes"]):
            raise HomeAssistantUnsupportedError(
                "That light does not support color temperature."
            )
        bounded = max(1500, min(10000, int(kelvin)))
        self._call_service(
            "light",
            "turn_on",
            [state["entity_id"]],
            {"color_temp_kelvin": bounded},
        )
        return {"entity_id": state["entity_id"], "color_temperature_kelvin": bounded}

    def set_light_color(self, entity_id: str, color: str) -> dict[str, Any]:
        state = self._require_domain(entity_id, {"light"})
        normalized_color = str(color).casefold().strip()
        if normalized_color not in ALLOWED_COLOR_NAMES:
            raise ValueError("Unsupported light color.")
        if not COLOR_MODES.intersection(state["supported_color_modes"]):
            raise HomeAssistantUnsupportedError("That light does not support color.")
        self._call_service(
            "light",
            "turn_on",
            [state["entity_id"]],
            {"color_name": normalized_color},
        )
        return {"entity_id": state["entity_id"], "color": normalized_color}

    def set_thermostat_temperature(
        self,
        entity_id: str,
        temperature: float,
    ) -> dict[str, Any]:
        state = self._require_domain(entity_id, {"climate"})
        minimum = state.get("min_temperature")
        maximum = state.get("max_temperature")
        target = float(temperature)
        if minimum is not None and target < float(minimum):
            target = float(minimum)
        if maximum is not None and target > float(maximum):
            target = float(maximum)
        if not -50 <= target <= 150:
            raise ValueError("Thermostat temperature is outside the safe range.")
        self._call_service(
            "climate",
            "set_temperature",
            [state["entity_id"]],
            {"temperature": target},
        )
        return {"entity_id": state["entity_id"], "temperature": target}

    def activate_scene(self, entity_id: str) -> dict[str, Any]:
        state = self._require_domain(entity_id, {"scene"})
        self._call_service("scene", "turn_on", [state["entity_id"]])
        return {"entity_id": state["entity_id"], "activated": True}

    def _require_domain(
        self,
        entity_id: str,
        allowed_domains: set[str] | frozenset[str],
    ) -> dict[str, Any]:
        state = self.get_entity_state(entity_id)
        if state["domain"] not in allowed_domains:
            raise HomeAssistantUnsupportedError(
                f"The {state['domain']} domain does not support this action."
            )
        return state

    def _call_service(
        self,
        domain: str,
        service: str,
        entity_ids: list[str],
        data: dict[str, Any] | None = None,
    ) -> None:
        allowed_services = {
            "light": {"turn_on", "turn_off", "toggle"},
            "switch": {"turn_on", "turn_off", "toggle"},
            "fan": {"turn_on", "turn_off", "toggle"},
            "climate": {"set_temperature"},
            "scene": {"turn_on"},
        }
        if service not in allowed_services.get(domain, set()):
            raise ValueError("The Home Assistant service is not allowlisted.")
        targets = [_validated_entity_id(value) for value in entity_ids]
        if any(value.split(".", 1)[0] != domain for value in targets):
            raise ValueError("Home Assistant target domain mismatch.")
        payload: dict[str, Any] = {"entity_id": targets if len(targets) > 1 else targets[0]}
        if data:
            payload.update(data)
        logger.info(
            "Home Assistant service call: domain=%s service=%s targets=%d",
            domain,
            service,
            len(targets),
        )
        response = self._request(
            "POST",
            f"/api/services/{domain}/{service}",
            payload,
        )
        if not isinstance(response, list):
            raise HomeAssistantResponseError(
                "Home Assistant returned malformed service data."
            )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        base_url = self._configured_base_url()
        token = self._load_token()
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        logger.debug("Home Assistant request: method=%s path=%s", method, path)
        api_request = request.Request(
            f"{base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with _URL_OPENER.open(
                api_request,
                timeout=self._config.timeout_seconds,
            ) as response:
                raw = response.read()
        except error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise HomeAssistantAuthenticationError(
                    "Home Assistant rejected the access token."
                ) from exc
            if exc.code == 404:
                raise HomeAssistantEntityNotFoundError(
                    "Home Assistant entity was not found."
                ) from exc
            raise HomeAssistantResponseError(
                f"Home Assistant returned HTTP {exc.code}."
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise HomeAssistantUnavailableError(
                "Home Assistant could not be reached."
            ) from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HomeAssistantResponseError(
                "Home Assistant returned malformed JSON."
            ) from exc

    def _configured_base_url(self) -> str:
        if not self._config.enabled:
            raise HomeAssistantConfigurationError(
                "Home Assistant is disabled in configuration."
            )
        if self._base_url is None:
            detail = self._url_error or "Home Assistant URL is not configured."
            raise HomeAssistantConfigurationError(detail)
        return self._base_url

    def _load_token(self) -> str:
        token: str | None = os.environ.get(self._config.token_env_var, "").strip() or None
        if token:
            return token
        try:
            token = self._config.token_path.read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            token = None
        except OSError as exc:
            raise HomeAssistantConfigurationError(
                "Home Assistant token file could not be read."
            ) from exc
        if not token:
            raise HomeAssistantConfigurationError(
                "Home Assistant access token is not configured."
            )
        return token


def _validated_entity_id(value: str) -> str:
    normalized = str(value).casefold().strip()
    if not ENTITY_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid Home Assistant entity ID.")
    return normalized


def _supports_brightness(state: dict[str, Any]) -> bool:
    modes = set(state.get("supported_color_modes") or ())
    return bool(BRIGHTNESS_MODES.intersection(modes)) or state.get(
        "brightness_percent"
    ) is not None


def _capabilities(domain: str, attributes: dict[str, Any]) -> list[str]:
    capabilities: list[str] = []
    modes = set(_string_list(attributes.get("supported_color_modes")))
    if domain == "light":
        if BRIGHTNESS_MODES.intersection(modes) or "brightness" in attributes:
            capabilities.append("brightness")
        if "color_temp" in modes:
            capabilities.append("color_temperature")
        if COLOR_MODES.intersection(modes):
            capabilities.append("color")
    if domain == "climate":
        capabilities.append("temperature")
    return capabilities


def _brightness_percent(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return max(0, min(100, round((float(value) / 255) * 100)))
    except (TypeError, ValueError):
        return None


def _optional_number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).casefold() for item in value]
