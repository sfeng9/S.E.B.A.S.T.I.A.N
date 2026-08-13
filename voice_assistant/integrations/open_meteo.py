from __future__ import annotations

import json
import math
from datetime import date
from dataclasses import asdict, dataclass
from typing import Any
from urllib import error, parse, request

from voice_assistant.config import WeatherConfig
from voice_assistant.http_utils import validated_http_url
from voice_assistant.integrations.location import ResolvedLocation


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherError(RuntimeError):
    """Base error for weather configuration, transport, and response failures."""


class WeatherConfigurationError(WeatherError):
    pass


class WeatherUnavailableError(WeatherError):
    pass


class WeatherResponseError(WeatherError):
    pass


@dataclass(frozen=True)
class CurrentWeather:
    requested_location: str | None
    resolved_location: str
    latitude: float
    longitude: float
    timezone: str
    is_home: bool
    forecast_day: str
    forecast_date: str
    observed_at: str | None
    temperature_f: float | None
    apparent_temperature_f: float | None
    condition: str
    weather_code: int
    humidity_percent: float | None
    precipitation_inches: float | None
    precipitation_probability_percent: float | None
    wind_speed_mph: float | None
    high_f: float
    low_f: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def location_name(self) -> str:
        return self.resolved_location


class OpenMeteoClient:
    def __init__(
        self,
        config: WeatherConfig,
        forecast_url: str = OPEN_METEO_FORECAST_URL,
    ) -> None:
        self.config = config
        self.forecast_url = validated_http_url(forecast_url, require_https=True)

    def get_current_weather(
        self,
        location: ResolvedLocation,
        forecast_day: str = "today",
    ) -> CurrentWeather:
        self._validate_config(location, forecast_day)
        query = parse.urlencode(
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "apparent_temperature",
                        "relative_humidity_2m",
                        "precipitation",
                        "weather_code",
                        "wind_speed_10m",
                    ]
                ),
                "daily": ",".join(
                    [
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                    ]
                ),
                "temperature_unit": self.config.temperature_unit,
                "wind_speed_unit": self.config.wind_speed_unit,
                "precipitation_unit": self.config.precipitation_unit,
                "timezone": location.timezone,
                "forecast_days": 7,
            }
        )
        http_request = request.Request(
            f"{self.forecast_url}?{query}",
            headers={"User-Agent": "SebastianVoiceAssistant/0.1"},
        )
        try:
            # The configurable endpoint was validated as HTTPS during initialization.
            with request.urlopen(  # nosec B310
                http_request,
                timeout=self.config.timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise WeatherUnavailableError(
                f"Open-Meteo returned HTTP {exc.code}."
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise WeatherUnavailableError("Could not connect to Open-Meteo.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WeatherResponseError("Open-Meteo returned malformed JSON.") from exc

        return self._parse_weather(payload, location, forecast_day)

    def _validate_config(
        self,
        location: ResolvedLocation,
        forecast_day: str,
    ) -> None:
        latitude = location.latitude
        longitude = location.longitude
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise WeatherConfigurationError("Weather latitude must be between -90 and 90.")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise WeatherConfigurationError(
                "Weather longitude must be between -180 and 180."
            )
        if self.config.temperature_unit != "fahrenheit":
            raise WeatherConfigurationError(
                "This assistant currently expects Fahrenheit weather results."
            )
        if self.config.wind_speed_unit != "mph":
            raise WeatherConfigurationError(
                "This assistant currently expects wind speed in mph."
            )
        if self.config.precipitation_unit != "inch":
            raise WeatherConfigurationError(
                "This assistant currently expects precipitation in inches."
            )
        if not location.timezone.strip():
            raise WeatherConfigurationError("Location timezone is not configured.")
        if self.config.timeout_seconds <= 0:
            raise WeatherConfigurationError("Weather timeout must be greater than zero.")
        if not forecast_day.strip():
            raise WeatherConfigurationError(
                "Weather forecast day is empty."
            )

    def _parse_weather(
        self,
        payload: Any,
        location: ResolvedLocation,
        forecast_day: str,
    ) -> CurrentWeather:
        if not isinstance(payload, dict):
            raise WeatherResponseError("Open-Meteo response was not an object.")
        if payload.get("error"):
            raise WeatherUnavailableError("Open-Meteo reported a forecast error.")

        daily = _require_object(payload, "daily")
        daily_dates = _require_string_list(daily, "time")
        day_index = _forecast_day_index(forecast_day, daily_dates)
        forecast_date = _require_indexed_string(daily, "time", day_index)
        high_f = _require_indexed_number(daily, "temperature_2m_max", day_index)
        low_f = _require_indexed_number(daily, "temperature_2m_min", day_index)
        probability = _optional_indexed_number(
            daily, "precipitation_probability_max", day_index
        )

        if day_index > 0:
            weather_code = int(
                _require_indexed_number(daily, "weather_code", day_index)
            )
            return CurrentWeather(
                requested_location=location.requested_location,
                resolved_location=location.resolved_location,
                latitude=location.latitude,
                longitude=location.longitude,
                timezone=location.timezone,
                is_home=location.is_home,
                forecast_day=forecast_day,
                forecast_date=forecast_date,
                observed_at=None,
                temperature_f=None,
                apparent_temperature_f=None,
                condition=weather_condition(weather_code),
                weather_code=weather_code,
                humidity_percent=None,
                precipitation_inches=None,
                precipitation_probability_percent=probability,
                wind_speed_mph=None,
                high_f=high_f,
                low_f=low_f,
            )

        current = _require_object(payload, "current")
        weather_code = int(_require_number(current, "weather_code"))
        return CurrentWeather(
            requested_location=location.requested_location,
            resolved_location=location.resolved_location,
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
            is_home=location.is_home,
            forecast_day=forecast_day,
            forecast_date=forecast_date,
            observed_at=_require_string(current, "time"),
            temperature_f=_require_number(current, "temperature_2m"),
            apparent_temperature_f=_require_number(current, "apparent_temperature"),
            condition=weather_condition(weather_code),
            weather_code=weather_code,
            humidity_percent=_require_number(current, "relative_humidity_2m"),
            precipitation_inches=_require_number(current, "precipitation"),
            precipitation_probability_percent=probability,
            wind_speed_mph=_require_number(current, "wind_speed_10m"),
            high_f=high_f,
            low_f=low_f,
        )


def weather_condition(code: int) -> str:
    conditions = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "foggy with rime",
        51: "light drizzle",
        53: "drizzle",
        55: "heavy drizzle",
        56: "light freezing drizzle",
        57: "heavy freezing drizzle",
        61: "light rain",
        63: "rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "light snow",
        73: "snow",
        75: "heavy snow",
        77: "snow grains",
        80: "light rain showers",
        81: "rain showers",
        82: "heavy rain showers",
        85: "light snow showers",
        86: "heavy snow showers",
        95: "thunderstorms",
        96: "thunderstorms with light hail",
        99: "thunderstorms with heavy hail",
    }
    return conditions.get(code, "unknown conditions")


def _require_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise WeatherResponseError(f"Open-Meteo response is missing {key!r}.")
    return value


def _require_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherResponseError(f"Open-Meteo field {key!r} is not numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise WeatherResponseError(f"Open-Meteo field {key!r} is not finite.")
    return number


def _require_indexed_number(data: dict[str, Any], key: str, index: int) -> float:
    value = data.get(key)
    if not isinstance(value, list) or len(value) <= index:
        raise WeatherResponseError(f"Open-Meteo daily field {key!r} is missing.")
    return _require_number({key: value[index]}, key)


def _optional_indexed_number(
    data: dict[str, Any], key: str, index: int
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) <= index:
        raise WeatherResponseError(f"Open-Meteo daily field {key!r} is malformed.")
    item = value[index]
    if item is None:
        return None
    return _require_number({key: item}, key)


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WeatherResponseError(f"Open-Meteo field {key!r} is not text.")
    return value


def _require_indexed_string(data: dict[str, Any], key: str, index: int) -> str:
    value = data.get(key)
    if not isinstance(value, list) or len(value) <= index:
        raise WeatherResponseError(f"Open-Meteo daily field {key!r} is missing.")
    return _require_string({key: value[index]}, key)


def _require_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise WeatherResponseError(f"Open-Meteo daily field {key!r} is missing.")
    return [_require_string({key: item}, key) for item in value]


def _forecast_day_index(forecast_day: str, daily_dates: list[str]) -> int:
    requested = forecast_day.strip().casefold()
    if requested in {"today", "current", "now"}:
        return 0
    if requested == "tomorrow":
        if len(daily_dates) < 2:
            raise WeatherResponseError("Tomorrow is missing from the forecast.")
        return 1

    parsed_dates: list[date] = []
    try:
        parsed_dates = [date.fromisoformat(value) for value in daily_dates]
    except ValueError as exc:
        raise WeatherResponseError("Open-Meteo returned an invalid forecast date.") from exc

    for index, forecast_date in enumerate(parsed_dates):
        if requested in {
            forecast_date.isoformat().casefold(),
            forecast_date.strftime("%A").casefold(),
        }:
            return index

    raise WeatherConfigurationError(
        f"Forecast period {forecast_day!r} is outside the available seven-day forecast."
    )
