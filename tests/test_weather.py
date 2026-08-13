from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.error import URLError

from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import LocationConfig, load_assistant_config
from voice_assistant.integrations.location import ResolvedLocation
from voice_assistant.integrations.open_meteo import (
    OpenMeteoClient,
    WeatherConfigurationError,
    WeatherResponseError,
    WeatherUnavailableError,
)


WEATHER_PAYLOAD = {
    "current": {
        "time": "2026-08-12T06:30",
        "temperature_2m": 68.2,
        "apparent_temperature": 67.1,
        "relative_humidity_2m": 71,
        "precipitation": 0.0,
        "weather_code": 2,
        "wind_speed_10m": 4.5,
    },
    "daily": {
        "time": [
            "2026-08-12", "2026-08-13", "2026-08-14", "2026-08-15",
            "2026-08-16", "2026-08-17", "2026-08-18"
        ],
        "weather_code": [2, 61, 3, 0, 1, 2, 63],
        "temperature_2m_max": [82.4, 79.2, 77.0, 80.0, 81.0, 82.0, 75.0],
        "temperature_2m_min": [64.8, 63.1, 62.0, 63.0, 64.0, 65.0, 61.0],
        "precipitation_probability_max": [10, 70, 20, 0, 5, 10, 80],
    },
}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FailingWeatherClient:
    def get_current_weather(
        self,
        location: ResolvedLocation,
        forecast_day: str = "today",
    ) -> None:
        raise WeatherUnavailableError("Simulated network failure.")


class WeatherTests(unittest.TestCase):
    def setUp(self) -> None:
        base_config = load_assistant_config()
        self.config = replace(
            base_config,
            home_location=LocationConfig(
                name="Test home",
                latitude=35.0,
                longitude=-78.0,
                timezone="America/New_York",
            ),
        )
        self.location = ResolvedLocation(
            requested_location="Tokyo",
            resolved_location="Tokyo, Japan",
            latitude=35.6762,
            longitude=139.6503,
            timezone="Asia/Tokyo",
            is_home=False,
            country="Japan",
            country_code="JP",
            admin1="Tokyo",
        )

    def test_parses_requested_current_and_daily_fields(self) -> None:
        client = OpenMeteoClient(self.config.weather)
        with patch(
            "voice_assistant.integrations.open_meteo.request.urlopen",
            return_value=FakeResponse(WEATHER_PAYLOAD),
        ) as urlopen:
            result = client.get_current_weather(self.location)

        self.assertEqual(result.requested_location, "Tokyo")
        self.assertEqual(result.resolved_location, "Tokyo, Japan")
        self.assertEqual(result.timezone, "Asia/Tokyo")
        self.assertEqual(result.condition, "partly cloudy")
        self.assertEqual(result.temperature_f, 68.2)
        self.assertEqual(result.apparent_temperature_f, 67.1)
        self.assertEqual(result.humidity_percent, 71)
        self.assertEqual(result.precipitation_inches, 0.0)
        self.assertEqual(result.precipitation_probability_percent, 10)
        self.assertEqual(result.wind_speed_mph, 4.5)
        self.assertEqual(result.high_f, 82.4)
        self.assertEqual(result.low_f, 64.8)
        requested_url = urlopen.call_args.args[0].full_url
        self.assertIn("latitude=35.6762", requested_url)
        self.assertIn("longitude=139.6503", requested_url)
        self.assertIn("timezone=Asia%2FTokyo", requested_url)

    def test_parses_tomorrow_forecast_without_current_values(self) -> None:
        client = OpenMeteoClient(self.config.weather)
        with patch(
            "voice_assistant.integrations.open_meteo.request.urlopen",
            return_value=FakeResponse(WEATHER_PAYLOAD),
        ):
            result = client.get_current_weather(self.location, "tomorrow")

        self.assertEqual(result.forecast_date, "2026-08-13")
        self.assertEqual(result.condition, "light rain")
        self.assertEqual(result.high_f, 79.2)
        self.assertEqual(result.low_f, 63.1)
        self.assertEqual(result.precipitation_probability_percent, 70)
        self.assertIsNone(result.temperature_f)

    def test_selects_weekday_from_seven_day_forecast(self) -> None:
        client = OpenMeteoClient(self.config.weather)
        with patch(
            "voice_assistant.integrations.open_meteo.request.urlopen",
            return_value=FakeResponse(WEATHER_PAYLOAD),
        ):
            result = client.get_current_weather(self.location, "Friday")

        self.assertEqual(result.forecast_date, "2026-08-14")
        self.assertEqual(result.condition, "overcast")
        self.assertEqual(result.high_f, 77.0)

    def test_network_failure_is_wrapped(self) -> None:
        client = OpenMeteoClient(self.config.weather)
        with patch(
            "voice_assistant.integrations.open_meteo.request.urlopen",
            side_effect=URLError("offline"),
        ):
            with self.assertRaises(WeatherUnavailableError):
                client.get_current_weather(self.location)

    def test_malformed_response_is_rejected(self) -> None:
        client = OpenMeteoClient(self.config.weather)
        with patch(
            "voice_assistant.integrations.open_meteo.request.urlopen",
            return_value=FakeResponse({"current": {}}),
        ):
            with self.assertRaises(WeatherResponseError):
                client.get_current_weather(self.location)

    def test_invalid_coordinates_are_rejected_when_weather_is_called(self) -> None:
        invalid = replace(self.location, latitude=100.0)
        with self.assertRaises(WeatherConfigurationError):
            OpenMeteoClient(self.config.weather).get_current_weather(invalid)

    def test_router_returns_tool_error_instead_of_raising(self) -> None:
        router = AssistantToolRouter(
            self.config,
            weather_client=FailingWeatherClient(),
        )
        result = router.execute("get_current_weather", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "weather_unavailable")


if __name__ == "__main__":
    unittest.main()
