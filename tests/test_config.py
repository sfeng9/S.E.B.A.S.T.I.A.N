from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voice_assistant.config import load_assistant_config, load_device_config


class LocalConfigTests(unittest.TestCase):
    def test_device_local_file_overrides_nested_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "microphone": {
                            "name_query": "placeholder",
                            "sample_rate": 48000,
                            "channels": 1,
                        },
                        "speaker": {
                            "name_query": "placeholder",
                            "sample_rate": 48000,
                            "channels": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            path.with_name("devices.local.json").write_text(
                json.dumps({"microphone": {"name_query": "My microphone"}}),
                encoding="utf-8",
            )

            config = load_device_config(path)

        self.assertEqual(config.microphone.name_query, "My microphone")
        self.assertEqual(config.microphone.sample_rate, 48000)
        self.assertEqual(config.speaker.name_query, "placeholder")

    def test_assistant_local_file_preserves_base_weather_units(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "assistant.json"
            path.write_text(
                json.dumps(
                    {
                        "home_location": {
                            "latitude": None,
                            "longitude": None,
                            "name": None,
                            "timezone": None,
                        },
                        "weather": {
                            "temperature_unit": "fahrenheit",
                            "wind_speed_unit": "mph",
                            "precipitation_unit": "inch",
                        }
                    }
                ),
                encoding="utf-8",
            )
            path.with_name("assistant.local.json").write_text(
                json.dumps(
                    {
                        "home_location": {
                            "latitude": 35.0,
                            "longitude": -78.0,
                            "name": "Private location",
                            "timezone": "America/New_York",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_assistant_config(path)

        self.assertEqual(config.home_location.latitude, 35.0)
        self.assertEqual(config.home_location.longitude, -78.0)
        self.assertEqual(config.home_location.name, "Private location")
        self.assertEqual(config.home_location.timezone, "America/New_York")
        self.assertEqual(config.weather.temperature_unit, "fahrenheit")
        self.assertEqual(config.llm.keep_alive, "5m")
        self.assertEqual(config.conversation.session_ttl_minutes, 30.0)
        self.assertEqual(config.conversation.max_context_tokens, 2400)
        self.assertEqual(config.resources.stt_idle_unload_minutes, 15.0)
        self.assertTrue(config.web_search.enabled)
        self.assertEqual(config.web_search.provider, "duckduckgo")
        self.assertEqual(config.web_search.max_results, 5)
        self.assertEqual(config.web_search.timeout_seconds, 8.0)
        self.assertEqual(config.gmail.max_results, 5)
        self.assertEqual(config.gmail.important_candidate_limit, 12)
        self.assertEqual(config.gmail.spoken_result_limit, 2)
        self.assertEqual(config.gmail.list_snippet_character_limit, 140)
        self.assertTrue(config.reminders.calendar_sync_enabled)
        self.assertEqual(config.reminders.calendar_reminder_minutes_before, 30)
        self.assertEqual(config.reminders.calendar_sync_interval_seconds, 300.0)
        self.assertEqual(config.reminders.calendar_sync_lookahead_hours, 168)
        self.assertEqual(config.reminders.calendar_sync_max_results, 2500)

    def test_legacy_weather_location_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "assistant.json"
            path.write_text(
                json.dumps(
                    {
                        "weather": {
                            "latitude": 35.0,
                            "longitude": -78.0,
                            "location_name": "Legacy home",
                            "timezone": "America/New_York",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_assistant_config(path)

        self.assertEqual(config.home_location.name, "Legacy home")
        self.assertEqual(config.home_location.latitude, 35.0)


if __name__ == "__main__":
    unittest.main()
