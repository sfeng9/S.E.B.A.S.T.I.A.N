from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

from voice_assistant.assistant.productivity_tools import ProductivityToolHandler
from voice_assistant.config import AssistantConfig
from voice_assistant.integrations.location import (
    LocationAmbiguousError,
    LocationConfigurationError,
    LocationError,
    LocationNotFoundError,
    LocationResolver,
    LocationResponseError,
    LocationUnavailableError,
    ResolvedLocation,
)
from voice_assistant.integrations.open_meteo import OpenMeteoClient, WeatherError
from voice_assistant.tools.time_date import (
    TimezoneError,
    get_current_date,
    get_current_local_time,
    get_day_of_week,
)


logger = logging.getLogger(__name__)


class WeatherClient(Protocol):
    def get_current_weather(
        self,
        location: ResolvedLocation,
        forecast_day: str = "today",
    ) -> Any: ...


class Resolver(Protocol):
    def resolve(self, location: str | None = None) -> ResolvedLocation: ...


LOCATION_PROPERTY = {
    "type": ["string", "null"],
    "description": (
        "City, region, country, postal code, or place named by the user. Omit or "
        "use null when no place is named, or when the user says here, home, back "
        "home, or outside; those mean the configured home location."
    ),
}


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = (
    _tool_schema(
        "get_current_local_time",
        "Get the current time in a named place or the configured home location. This tool is mandatory for any current-time request, including awkward speech transcripts, statements, or fragments such as 'It's the time in Taipei right now.' Never calculate or recall a timezone offset yourself.",
        {"location": LOCATION_PROPERTY},
    ),
    _tool_schema(
        "get_current_date",
        "Get today's calendar date in a named place or the configured home location.",
        {"location": LOCATION_PROPERTY},
    ),
    _tool_schema(
        "get_day_of_week",
        "Get the current day of the week in a named place or the configured home location.",
        {"location": LOCATION_PROPERTY},
    ),
    _tool_schema(
        "get_current_weather",
        "Get weather for a named place or the configured home location. Use for weather, forecasts for today or tomorrow, temperature, outdoor conditions, rain, wind, humidity, or clothing questions.",
        {
            "location": LOCATION_PROPERTY,
            "forecast_day": {
                "type": "string",
                "description": (
                    "Requested forecast period: today, tomorrow, a weekday such "
                    "as Friday, or an ISO date within the next seven days. Omit "
                    "for current conditions today."
                ),
            },
        },
    ),
)


class AssistantToolRouter:
    def __init__(
        self,
        config: AssistantConfig,
        weather_client: WeatherClient | None = None,
        location_resolver: Resolver | None = None,
        now_provider: Callable[[], datetime] | None = None,
        productivity: ProductivityToolHandler | None = None,
    ) -> None:
        self._locations = location_resolver or LocationResolver(
            config.home_location,
            timeout_seconds=config.weather.timeout_seconds,
        )
        self._weather = weather_client or OpenMeteoClient(config.weather)
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._productivity = productivity or ProductivityToolHandler(
            config,
            now_provider=self._now_provider,
        )

    @property
    def schemas(self) -> Sequence[dict[str, Any]]:
        return (*TOOL_SCHEMAS, *self._productivity.schemas)

    def schemas_for(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        context = " ".join(
            [*(str(item.get("content", "")) for item in history[-4:]), prompt]
        ).casefold()
        selected = list(TOOL_SCHEMAS)
        productivity = list(self._productivity.schemas)
        requirement = self._productivity.tool_requirement(prompt, history)
        required_names = {
            str(name)
            for name in (
                requirement.get("tools", ())
                if isinstance(requirement, dict)
                else ()
            )
        }
        if required_names:
            selected.extend(
                item
                for item in productivity
                if item["function"]["name"] in required_names
            )
            logger.debug(
                "Restricting productivity tools to required action: %s",
                ", ".join(sorted(required_names)),
            )
            return tuple(selected)
        include_all = "briefing" in context
        groups = (
            (
                ("email", "mail", "inbox", "sender", "professor"),
                {"search_emails", "get_recent_emails", "get_unread_emails", "get_important_emails", "get_email_details"},
            ),
            (
                ("calendar", "schedule", "plan", "event", "appointment", "meeting", "dinner", "lunch", "dentist", "free at", "move it", "cancel it", "what do i have", "do i have anything", "afternoon", "evening"),
                {"get_events_for_date", "get_events_between", "get_next_event", "find_events", "create_event", "update_event", "delete_event", "confirm_calendar_action"},
            ),
            (
                ("remind", "reminder"),
                {"create_reminder", "get_pending_reminders", "cancel_reminder"},
            ),
        )
        for cues, names in groups:
            if include_all or any(cue in context for cue in cues):
                selected.extend(
                    item for item in productivity
                    if item["function"]["name"] in names
                )
        logger.debug("Exposing %d tools for current request.", len(selected))
        return tuple(selected)

    @property
    def reminder_store(self):
        return self._productivity.reminder_store

    def reset_session_context(self) -> None:
        self._productivity.reset_session_context()

    def begin_turn(self) -> None:
        self._productivity.begin_turn()

    def tool_requirement(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        return self._productivity.tool_requirement(prompt, history)

    def spoken_override_for(self, called_tools: Sequence[str]) -> str | None:
        return self._productivity.spoken_override_for(called_tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        logger.info("Tool selected: %s", name)
        try:
            if name == "get_current_local_time":
                return self._get_time(arguments)
            if name == "get_current_date":
                return self._get_date(arguments)
            if name == "get_day_of_week":
                return self._get_day(arguments)
            if name == "get_current_weather":
                return self._get_weather(arguments)
            productivity_result = self._productivity.execute(name, arguments)
            if productivity_result is not None:
                return productivity_result
        except LocationError as exc:
            return self._location_error(exc)
        except TimezoneError as exc:
            logger.warning("Time tool failed (%s): %s", type(exc).__name__, exc)
            return {
                "ok": False,
                "error": "timezone_unavailable",
                "message": (
                    "The timezone for that location is unavailable. Do not guess "
                    "the time; tell the user you couldn't get it right now."
                ),
            }

        logger.warning("Unknown tool requested: %s", name)
        return {
            "ok": False,
            "error": "unknown_tool",
            "message": "The requested tool is unavailable. Do not invent a result.",
        }

    def _get_time(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location = self._resolve(arguments)
        result = get_current_local_time(
            self._now_provider(),
            timezone_name=location.timezone,
            requested_location=location.requested_location,
            resolved_location=location.resolved_location,
            is_home=location.is_home,
        ).to_dict()
        logger.info(
            "Time tool succeeded for %s: %s",
            _log_location(location),
            result["display_time"],
        )
        return {"ok": True, **result}

    def _get_date(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location = self._resolve(arguments)
        result = get_current_date(
            self._now_provider(),
            timezone_name=location.timezone,
            requested_location=location.requested_location,
            resolved_location=location.resolved_location,
            is_home=location.is_home,
        ).to_dict()
        logger.info(
            "Date tool succeeded for %s: %s",
            _log_location(location),
            result["display_date"],
        )
        return {"ok": True, **result}

    def _get_day(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location = self._resolve(arguments)
        result = get_day_of_week(
            self._now_provider(),
            timezone_name=location.timezone,
            requested_location=location.requested_location,
            resolved_location=location.resolved_location,
            is_home=location.is_home,
        ).to_dict()
        logger.info(
            "Day tool succeeded for %s: %s",
            _log_location(location),
            result["day_of_week"],
        )
        return {"ok": True, **result}

    def _get_weather(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location = self._resolve(arguments)
        forecast_day = arguments.get("forecast_day", "today")
        if not isinstance(forecast_day, str):
            forecast_day = "today"
        logger.info(
            "Weather request started for %s (%s).",
            _log_location(location),
            forecast_day,
        )
        try:
            weather = self._weather.get_current_weather(location, forecast_day)
        except WeatherError as exc:
            logger.warning(
                "Weather request failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return {
                "ok": False,
                "error": "weather_unavailable",
                "requested_location": location.requested_location,
                "resolved_location": location.resolved_location,
                "message": (
                    "Weather could not be retrieved for the resolved location. "
                    "Do not guess; tell the user you couldn't get the weather "
                    "there right now in one short sentence. Do not offer to retry "
                    "or ask a follow-up question."
                ),
            }

        logger.info(
            "Weather request succeeded for %s: period=%s condition=%s high=%.1fF low=%.1fF",
            _log_location(location),
            weather.forecast_day,
            weather.condition,
            weather.high_f,
            weather.low_f,
        )
        result = weather.to_dict()
        for field in (
            "temperature_f",
            "apparent_temperature_f",
            "humidity_percent",
            "precipitation_probability_percent",
            "high_f",
            "low_f",
        ):
            if result[field] is not None:
                result[field] = round(result[field])
        if result["wind_speed_mph"] is not None:
            result["wind_speed_mph"] = round(result["wind_speed_mph"], 1)
        if result["precipitation_inches"] is not None:
            result["precipitation_inches"] = round(
                result["precipitation_inches"], 3
            )
        return {"ok": True, **result}

    def _resolve(self, arguments: dict[str, Any]) -> ResolvedLocation:
        raw_location = arguments.get("location")
        if raw_location is not None and not isinstance(raw_location, str):
            raise LocationNotFoundError(str(raw_location))
        return self._locations.resolve(raw_location)

    @staticmethod
    def _location_error(exc: LocationError) -> dict[str, Any]:
        logger.warning("Location resolution failed (%s): %s", type(exc).__name__, exc)
        if isinstance(exc, LocationAmbiguousError):
            return {
                "ok": False,
                "error": "ambiguous_location",
                "requested_location": exc.requested_location,
                "candidates": list(exc.candidates),
                "message": (
                    "The location is ambiguous. Ask one short clarification "
                    "question using the candidate regions; do not choose one."
                ),
            }
        if isinstance(exc, LocationNotFoundError):
            return {
                "ok": False,
                "error": "location_not_found",
                "requested_location": exc.requested_location,
                "message": "That location could not be found. Do not guess it.",
            }
        if isinstance(exc, LocationConfigurationError):
            return {
                "ok": False,
                "error": "home_location_not_configured",
                "message": "The home location is not configured.",
            }
        if isinstance(exc, LocationUnavailableError):
            error_code = "geocoding_unavailable"
        elif isinstance(exc, LocationResponseError):
            error_code = "geocoding_response_error"
        else:
            error_code = "location_unavailable"
        return {
            "ok": False,
            "error": error_code,
            "message": (
                "The location could not be resolved right now. Do not guess; "
                "tell the user you couldn't look up that location."
            ),
        }


def _log_location(location: ResolvedLocation) -> str:
    return "home" if location.is_home else location.resolved_location
