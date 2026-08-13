from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.error import URLError

from voice_assistant.assistant.tool_router import AssistantToolRouter
from voice_assistant.config import LocationConfig, load_assistant_config
from voice_assistant.integrations.location import (
    GeocodingCandidate,
    LocationAmbiguousError,
    LocationNotFoundError,
    LocationResolver,
    LocationResponseError,
    LocationUnavailableError,
    OpenMeteoGeocodingClient,
    ResolvedLocation,
)
from voice_assistant.integrations.open_meteo import CurrentWeather


HOME = LocationConfig(
    name="Test home",
    latitude=35.0,
    longitude=-78.0,
    timezone="America/New_York",
)


def candidate(
    name: str,
    admin1: str,
    country: str,
    timezone_name: str,
    population: int,
    latitude: float = 1.0,
    longitude: float = 2.0,
    feature_code: str = "PPL",
) -> GeocodingCandidate:
    return GeocodingCandidate(
        name=name,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone_name,
        country=country,
        country_code={"United States": "US", "Japan": "JP"}.get(country),
        admin1=admin1,
        population=population,
        feature_code=feature_code,
    )


TOKYO = candidate(
    "Tokyo",
    "Tokyo",
    "Japan",
    "Asia/Tokyo",
    8_336_599,
    latitude=35.6895,
    longitude=139.6917,
    feature_code="PPLC",
)


class FakeGeocoder:
    def __init__(self, results: tuple[GeocodingCandidate, ...]) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(self, location: str, count: int = 10) -> tuple[GeocodingCandidate, ...]:
        self.calls.append(location)
        return self.results


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class CapturingWeatherClient:
    def __init__(self) -> None:
        self.calls: list[tuple[ResolvedLocation, str]] = []

    def get_current_weather(
        self,
        location: ResolvedLocation,
        forecast_day: str = "today",
    ) -> CurrentWeather:
        self.calls.append((location, forecast_day))
        return CurrentWeather(
            requested_location=location.requested_location,
            resolved_location=location.resolved_location,
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
            is_home=location.is_home,
            forecast_day=forecast_day,
            forecast_date="2026-08-12",
            observed_at="2026-08-12T06:30",
            temperature_f=72.0,
            apparent_temperature_f=72.0,
            condition="clear",
            weather_code=0,
            humidity_percent=50.0,
            precipitation_inches=0.0,
            precipitation_probability_percent=0.0,
            wind_speed_mph=3.0,
            high_f=80.0,
            low_f=60.0,
        )


class LocationResolverTests(unittest.TestCase):
    def test_no_location_and_home_aliases_use_home_without_geocoding(self) -> None:
        geocoder = FakeGeocoder((TOKYO,))
        resolver = LocationResolver(HOME, geocoder=geocoder)  # type: ignore[arg-type]

        for query in (None, "", "home", "here", "outside", "my location"):
            resolved = resolver.resolve(query)
            self.assertTrue(resolved.is_home)
            self.assertEqual(resolved.resolved_location, "Test home")
            self.assertEqual(resolved.timezone, "America/New_York")

        self.assertEqual(geocoder.calls, [])

    def test_explicit_location_is_cached_and_does_not_change_home(self) -> None:
        geocoder = FakeGeocoder((TOKYO,))
        resolver = LocationResolver(HOME, geocoder=geocoder)  # type: ignore[arg-type]

        first = resolver.resolve("Tokyo")
        second = resolver.resolve(" tokyo ")
        home = resolver.resolve()

        self.assertEqual(first, second)
        self.assertEqual(geocoder.calls, ["Tokyo"])
        self.assertFalse(first.is_home)
        self.assertEqual(first.resolved_location, "Tokyo, Japan")
        self.assertTrue(home.is_home)
        self.assertEqual(home.resolved_location, "Test home")

    def test_dominant_population_selects_well_known_location(self) -> None:
        results = (
            candidate(
                "Raleigh", "North Carolina", "United States", "America/New_York", 467_665
            ),
            candidate(
                "Raleigh", "Mississippi", "United States", "America/Chicago", 1_000
            ),
        )
        resolver = LocationResolver(HOME, geocoder=FakeGeocoder(results))  # type: ignore[arg-type]

        resolved = resolver.resolve("Raleigh")

        self.assertEqual(resolved.resolved_location, "Raleigh, North Carolina, United States")

    def test_ambiguous_location_requires_clarification(self) -> None:
        results = (
            candidate(
                "Springfield", "Missouri", "United States", "America/Chicago", 169_176
            ),
            candidate(
                "Springfield", "Massachusetts", "United States", "America/New_York", 155_929
            ),
            candidate(
                "Springfield", "Illinois", "United States", "America/Chicago", 114_394
            ),
        )
        resolver = LocationResolver(HOME, geocoder=FakeGeocoder(results))  # type: ignore[arg-type]

        with self.assertRaises(LocationAmbiguousError) as context:
            resolver.resolve("Springfield")

        self.assertEqual(len(context.exception.candidates), 3)

    def test_state_qualifier_disambiguates_location(self) -> None:
        results = (
            candidate(
                "Springfield", "Missouri", "United States", "America/Chicago", 169_176
            ),
            candidate(
                "Springfield", "Illinois", "United States", "America/Chicago", 114_394
            ),
        )
        geocoder = FakeGeocoder(results)
        resolver = LocationResolver(HOME, geocoder=geocoder)  # type: ignore[arg-type]

        resolved = resolver.resolve("Springfield, IL")

        self.assertEqual(geocoder.calls, ["Springfield"])
        self.assertEqual(resolved.admin1, "Illinois")

    def test_unknown_location_is_reported(self) -> None:
        resolver = LocationResolver(HOME, geocoder=FakeGeocoder(()))  # type: ignore[arg-type]
        with self.assertRaises(LocationNotFoundError):
            resolver.resolve("FakeCityThatDoesNotExist")

    def test_location_cache_is_bounded(self) -> None:
        geocoder = FakeGeocoder((TOKYO,))
        resolver = LocationResolver(
            HOME,
            geocoder=geocoder,  # type: ignore[arg-type]
            cache_size=2,
        )

        resolver.resolve("Tokyo one")
        resolver.resolve("Tokyo two")
        resolver.resolve("Tokyo three")
        resolver.resolve("Tokyo one")

        self.assertEqual(
            geocoder.calls,
            ["Tokyo one", "Tokyo two", "Tokyo three", "Tokyo one"],
        )


class GeocodingClientTests(unittest.TestCase):
    def test_parses_open_meteo_result(self) -> None:
        payload = {
            "results": [
                {
                    "name": "Tokyo",
                    "latitude": 35.6895,
                    "longitude": 139.6917,
                    "timezone": "Asia/Tokyo",
                    "country": "Japan",
                    "country_code": "JP",
                    "admin1": "Tokyo",
                    "population": 8_336_599,
                    "feature_code": "PPLC",
                }
            ]
        }
        client = OpenMeteoGeocodingClient()
        with patch(
            "voice_assistant.integrations.location.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            results = client.search("Tokyo")

        self.assertEqual(results, (TOKYO,))

    def test_network_failure_is_wrapped(self) -> None:
        client = OpenMeteoGeocodingClient()
        with patch(
            "voice_assistant.integrations.location.request.urlopen",
            side_effect=URLError("offline"),
        ):
            with self.assertRaises(LocationUnavailableError):
                client.search("Tokyo")

    def test_malformed_response_is_rejected(self) -> None:
        client = OpenMeteoGeocodingClient()
        with patch(
            "voice_assistant.integrations.location.request.urlopen",
            return_value=FakeResponse({"results": "invalid"}),
        ):
            with self.assertRaises(LocationResponseError):
                client.search("Tokyo")


class LocationToolRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        base = load_assistant_config()
        self.config = replace(base, home_location=HOME)
        self.geocoder = FakeGeocoder((TOKYO,))
        self.resolver = LocationResolver(HOME, geocoder=self.geocoder)  # type: ignore[arg-type]
        self.weather = CapturingWeatherClient()
        self.now = datetime(2026, 8, 12, 10, 30, tzinfo=timezone.utc)
        self.router = AssistantToolRouter(
            self.config,
            weather_client=self.weather,
            location_resolver=self.resolver,
            now_provider=lambda: self.now,
        )

    def test_time_uses_explicit_timezone_then_home_timezone(self) -> None:
        tokyo = self.router.execute(
            "get_current_local_time", {"location": "Tokyo"}
        )
        home = self.router.execute("get_current_local_time", {})

        self.assertEqual(tokyo["display_time"], "7:30 PM")
        self.assertEqual(tokyo["resolved_location"], "Tokyo, Japan")
        self.assertEqual(home["display_time"], "6:30 AM")
        self.assertTrue(home["is_home"])

    def test_weather_receives_resolved_explicit_coordinates(self) -> None:
        result = self.router.execute(
            "get_current_weather",
            {"location": "Tokyo", "forecast_day": "tomorrow"},
        )

        called_location, called_day = self.weather.calls[0]
        self.assertTrue(result["ok"])
        self.assertEqual(called_location.latitude, 35.6895)
        self.assertEqual(called_location.longitude, 139.6917)
        self.assertEqual(called_day, "tomorrow")

    def test_ambiguous_location_becomes_structured_tool_error(self) -> None:
        ambiguous = (
            candidate(
                "Springfield", "Missouri", "United States", "America/Chicago", 169_176
            ),
            candidate(
                "Springfield", "Massachusetts", "United States", "America/New_York", 155_929
            ),
        )
        router = AssistantToolRouter(
            self.config,
            location_resolver=LocationResolver(
                HOME, geocoder=FakeGeocoder(ambiguous)  # type: ignore[arg-type]
            ),
        )

        result = router.execute(
            "get_current_weather", {"location": "Springfield"}
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "ambiguous_location")
        self.assertEqual(len(result["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
