from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVICE_CONFIG = PROJECT_ROOT / "config" / "devices.json"
DEFAULT_ASSISTANT_CONFIG = PROJECT_ROOT / "config" / "assistant.json"


@dataclass(frozen=True)
class AudioEndpointConfig:
    name_query: str | None
    device_id: int | None
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class DeviceConfig:
    microphone: AudioEndpointConfig
    speaker: AudioEndpointConfig


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    base_url: str
    temperature: float
    context_window: int
    keep_alive: str


@dataclass(frozen=True)
class SpeechToTextConfig:
    model: str
    device: str
    compute_type: str
    language: str | None
    beam_size: int


@dataclass(frozen=True)
class TextToSpeechConfig:
    provider: str
    voice: str
    model_path: Path
    volume: float
    length_scale: float


@dataclass(frozen=True)
class WakeWordConfig:
    phrase: str
    model_path: Path
    threshold: float
    vad_threshold: float
    frame_ms: int


@dataclass(frozen=True)
class CommandRecordingConfig:
    max_seconds: float
    speech_start_timeout: float
    silence_seconds: float
    pre_roll_ms: int
    frame_ms: int
    vad_mode: int


@dataclass(frozen=True)
class LocationConfig:
    name: str | None
    latitude: float | None
    longitude: float | None
    timezone: str | None


@dataclass(frozen=True)
class WeatherConfig:
    temperature_unit: str
    wind_speed_unit: str
    precipitation_unit: str
    timeout_seconds: float


@dataclass(frozen=True)
class WebSearchConfig:
    enabled: bool
    provider: str
    max_results: int
    timeout_seconds: float
    region: str
    safesearch: str


@dataclass(frozen=True)
class ConversationConfig:
    session_ttl_minutes: float
    max_context_tokens: int


@dataclass(frozen=True)
class ResourceConfig:
    stt_idle_unload_minutes: float
    maintenance_interval_seconds: float


@dataclass(frozen=True)
class GoogleConfig:
    credentials_path: Path
    gmail_token_path: Path
    calendar_token_path: Path


@dataclass(frozen=True)
class CalendarConfig:
    calendar_id: str
    default_event_duration_minutes: int
    confirm_updates: bool
    confirm_deletes: bool
    max_results: int


@dataclass(frozen=True)
class GmailConfig:
    max_results: int
    important_candidate_limit: int
    spoken_result_limit: int
    list_snippet_character_limit: int
    detail_body_character_limit: int


@dataclass(frozen=True)
class ReminderConfig:
    database_path: Path
    poll_interval_seconds: float
    calendar_sync_enabled: bool
    calendar_reminder_minutes_before: int
    calendar_sync_interval_seconds: float
    calendar_sync_lookahead_hours: int
    calendar_sync_max_results: int


@dataclass(frozen=True)
class PersonalDataConfig:
    database_path: Path
    search_result_limit: int
    spoken_item_limit: int


@dataclass(frozen=True)
class ApplicationConfig:
    identifier: str
    aliases: tuple[str, ...]
    executable_names: tuple[str, ...]
    process_names: tuple[str, ...]
    start_menu_names: tuple[str, ...]
    configured_path: Path | None


@dataclass(frozen=True)
class PcControlConfig:
    enabled: bool
    confirmation_timeout_seconds: float
    graceful_close_timeout_seconds: float
    screenshot_directory: Path
    volume_step_percent: int
    applications: tuple[ApplicationConfig, ...]


@dataclass(frozen=True)
class HomeAssistantEntityConfig:
    identifier: str
    entity_id: str
    aliases: tuple[str, ...]
    room: str | None
    allow_control: bool


@dataclass(frozen=True)
class HomeAssistantConfig:
    enabled: bool
    url: str | None
    token_env_var: str
    token_path: Path
    timeout_seconds: float
    default_room: str | None
    max_results: int
    brightness_step_percent: int
    color_temperature_step_kelvin: int
    entities: tuple[HomeAssistantEntityConfig, ...]


@dataclass(frozen=True)
class AssistantConfig:
    llm: LlmConfig
    speech_to_text: SpeechToTextConfig
    text_to_speech: TextToSpeechConfig
    wake_word: WakeWordConfig
    command_recording: CommandRecordingConfig
    home_location: LocationConfig
    weather: WeatherConfig
    web_search: WebSearchConfig
    conversation: ConversationConfig
    resources: ResourceConfig
    google: GoogleConfig
    calendar: CalendarConfig
    gmail: GmailConfig
    reminders: ReminderConfig
    personal_data: PersonalDataConfig
    pc_control: PcControlConfig
    home_assistant: HomeAssistantConfig


def _endpoint_from_dict(data: dict[str, Any], default_rate: int) -> AudioEndpointConfig:
    return AudioEndpointConfig(
        name_query=data.get("name_query"),
        device_id=data.get("device_id"),
        sample_rate=int(data.get("sample_rate", default_rate)),
        channels=int(data.get("channels", 1)),
    )


def load_device_config(path: Path = DEFAULT_DEVICE_CONFIG) -> DeviceConfig:
    raw = _load_with_local_override(path)

    return DeviceConfig(
        microphone=_endpoint_from_dict(raw.get("microphone", {}), 16000),
        speaker=_endpoint_from_dict(raw.get("speaker", {}), 24000),
    )


def load_assistant_config(path: Path = DEFAULT_ASSISTANT_CONFIG) -> AssistantConfig:
    raw = _load_with_local_override(path)

    llm = raw.get("llm", {})
    speech_to_text = raw.get("speech_to_text", {})
    text_to_speech = raw.get("text_to_speech", {})
    wake_word = raw.get("wake_word", {})
    command_recording = raw.get("command_recording", {})
    home_location = raw.get("home_location", {})
    weather = raw.get("weather", {})
    web_search = raw.get("web_search", {})
    conversation = raw.get("conversation", {})
    resources = raw.get("resources", {})
    google = raw.get("google", {})
    calendar = raw.get("calendar", {})
    gmail = raw.get("gmail", {})
    reminders = raw.get("reminders", {})
    personal_data = raw.get("personal_data", {})
    pc_control = raw.get("pc_control", {})
    home_assistant = raw.get("home_assistant", {})
    tts_model_path = Path(
        str(text_to_speech.get("model_path", "data/voices/en_US-ryan-medium.onnx"))
    )
    if not tts_model_path.is_absolute():
        tts_model_path = PROJECT_ROOT / tts_model_path
    wake_model_path = Path(
        str(wake_word.get("model_path", "data/wake_words/sebastian.onnx"))
    )
    if not wake_model_path.is_absolute():
        wake_model_path = PROJECT_ROOT / wake_model_path
    return AssistantConfig(
        llm=LlmConfig(
            provider=str(llm.get("provider", "ollama")),
            model=str(llm.get("model", "qwen3:8b")),
            base_url=str(llm.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
            temperature=float(llm.get("temperature", 0.2)),
            context_window=int(llm.get("context_window", 4096)),
            keep_alive=str(llm.get("keep_alive", "5m")),
        ),
        speech_to_text=SpeechToTextConfig(
            model=str(speech_to_text.get("model", "small.en")),
            device=str(speech_to_text.get("device", "cuda")),
            compute_type=str(speech_to_text.get("compute_type", "int8_float16")),
            language=speech_to_text.get("language", "en"),
            beam_size=int(speech_to_text.get("beam_size", 1)),
        ),
        text_to_speech=TextToSpeechConfig(
            provider=str(text_to_speech.get("provider", "piper")),
            voice=str(text_to_speech.get("voice", "en_US-ryan-medium")),
            model_path=tts_model_path,
            volume=float(text_to_speech.get("volume", 1.0)),
            length_scale=float(text_to_speech.get("length_scale", 1.0)),
        ),
        wake_word=WakeWordConfig(
            phrase=str(wake_word.get("phrase", "Sebastian")),
            model_path=wake_model_path,
            threshold=float(wake_word.get("threshold", 0.5)),
            vad_threshold=float(wake_word.get("vad_threshold", 0.0)),
            frame_ms=int(wake_word.get("frame_ms", 80)),
        ),
        command_recording=CommandRecordingConfig(
            max_seconds=float(command_recording.get("max_seconds", 12.0)),
            speech_start_timeout=float(
                command_recording.get("speech_start_timeout", 5.0)
            ),
            silence_seconds=float(command_recording.get("silence_seconds", 0.9)),
            pre_roll_ms=int(command_recording.get("pre_roll_ms", 300)),
            frame_ms=int(command_recording.get("frame_ms", 30)),
            vad_mode=int(command_recording.get("vad_mode", 2)),
        ),
        home_location=LocationConfig(
            name=_optional_string(
                home_location.get("name", weather.get("location_name"))
            ),
            latitude=_optional_float(
                home_location.get("latitude", weather.get("latitude"))
            ),
            longitude=_optional_float(
                home_location.get("longitude", weather.get("longitude"))
            ),
            timezone=_optional_string(
                home_location.get("timezone", weather.get("timezone"))
            ),
        ),
        weather=WeatherConfig(
            temperature_unit=str(weather.get("temperature_unit", "fahrenheit")),
            wind_speed_unit=str(weather.get("wind_speed_unit", "mph")),
            precipitation_unit=str(weather.get("precipitation_unit", "inch")),
            timeout_seconds=float(weather.get("timeout_seconds", 10.0)),
        ),
        web_search=WebSearchConfig(
            enabled=bool(web_search.get("enabled", True)),
            provider=str(web_search.get("provider", "duckduckgo")),
            max_results=max(1, min(10, int(web_search.get("max_results", 5)))),
            timeout_seconds=max(
                1.0, min(30.0, float(web_search.get("timeout_seconds", 8.0)))
            ),
            region=str(web_search.get("region", "us-en")),
            safesearch=str(web_search.get("safesearch", "moderate")),
        ),
        conversation=ConversationConfig(
            session_ttl_minutes=max(
                0.01, float(conversation.get("session_ttl_minutes", 30.0))
            ),
            max_context_tokens=max(
                128, int(conversation.get("max_context_tokens", 2400))
            ),
        ),
        resources=ResourceConfig(
            stt_idle_unload_minutes=max(
                0.0, float(resources.get("stt_idle_unload_minutes", 15.0))
            ),
            maintenance_interval_seconds=max(
                0.25, float(resources.get("maintenance_interval_seconds", 5.0))
            ),
        ),
        google=GoogleConfig(
            credentials_path=_project_path(
                google.get("credentials_path", "secrets/google/client_secret.json")
            ),
            gmail_token_path=_project_path(
                google.get("gmail_token_path", "secrets/google/gmail_token.json")
            ),
            calendar_token_path=_project_path(
                google.get("calendar_token_path", "secrets/google/calendar_token.json")
            ),
        ),
        calendar=CalendarConfig(
            calendar_id=str(calendar.get("calendar_id", "primary")),
            default_event_duration_minutes=max(
                1, int(calendar.get("default_event_duration_minutes", 60))
            ),
            confirm_updates=bool(calendar.get("confirm_updates", True)),
            confirm_deletes=bool(calendar.get("confirm_deletes", True)),
            max_results=max(1, min(100, int(calendar.get("max_results", 20)))),
        ),
        gmail=GmailConfig(
            max_results=max(1, min(20, int(gmail.get("max_results", 5)))),
            important_candidate_limit=max(
                1, min(50, int(gmail.get("important_candidate_limit", 12)))
            ),
            spoken_result_limit=max(
                1, min(5, int(gmail.get("spoken_result_limit", 2)))
            ),
            list_snippet_character_limit=max(
                80, min(500, int(gmail.get("list_snippet_character_limit", 140)))
            ),
            detail_body_character_limit=max(
                500, int(gmail.get("detail_body_character_limit", 6000))
            ),
        ),
        reminders=ReminderConfig(
            database_path=_project_path(
                reminders.get("database_path", "data/reminders.sqlite3")
            ),
            poll_interval_seconds=max(
                0.25, float(reminders.get("poll_interval_seconds", 5.0))
            ),
            calendar_sync_enabled=bool(
                reminders.get("calendar_sync_enabled", True)
            ),
            calendar_reminder_minutes_before=max(
                0,
                min(
                    1440,
                    int(reminders.get("calendar_reminder_minutes_before", 30)),
                ),
            ),
            calendar_sync_interval_seconds=max(
                15.0,
                float(reminders.get("calendar_sync_interval_seconds", 300.0)),
            ),
            calendar_sync_lookahead_hours=max(
                1,
                min(
                    720,
                    int(reminders.get("calendar_sync_lookahead_hours", 168)),
                ),
            ),
            calendar_sync_max_results=max(
                1,
                min(
                    2500,
                    int(reminders.get("calendar_sync_max_results", 2500)),
                ),
            ),
        ),
        personal_data=PersonalDataConfig(
            database_path=_project_path(
                personal_data.get("database_path", "data/sebastian.db")
            ),
            search_result_limit=max(
                1, min(20, int(personal_data.get("search_result_limit", 8)))
            ),
            spoken_item_limit=max(
                1, min(20, int(personal_data.get("spoken_item_limit", 5)))
            ),
        ),
        pc_control=PcControlConfig(
            enabled=bool(pc_control.get("enabled", True)),
            confirmation_timeout_seconds=max(
                5.0,
                min(
                    300.0,
                    float(pc_control.get("confirmation_timeout_seconds", 30.0)),
                ),
            ),
            graceful_close_timeout_seconds=max(
                0.25,
                min(
                    10.0,
                    float(pc_control.get("graceful_close_timeout_seconds", 2.0)),
                ),
            ),
            screenshot_directory=_project_path(
                pc_control.get("screenshot_directory", "data/screenshots")
            ),
            volume_step_percent=max(
                1,
                min(100, int(pc_control.get("volume_step_percent", 10))),
            ),
            applications=_applications_from_config(pc_control.get("applications")),
        ),
        home_assistant=HomeAssistantConfig(
            enabled=bool(home_assistant.get("enabled", False)),
            url=_optional_string(home_assistant.get("url")),
            token_env_var=str(
                home_assistant.get("token_env_var", "HOME_ASSISTANT_TOKEN")
            ).strip()
            or "HOME_ASSISTANT_TOKEN",
            token_path=_project_path(
                home_assistant.get(
                    "token_path", "secrets/home_assistant/token.txt"
                )
            ),
            timeout_seconds=max(
                1.0,
                min(30.0, float(home_assistant.get("timeout_seconds", 5.0))),
            ),
            default_room=_optional_string(home_assistant.get("default_room")),
            max_results=max(
                1, min(20, int(home_assistant.get("max_results", 10)))
            ),
            brightness_step_percent=max(
                1,
                min(
                    100,
                    int(home_assistant.get("brightness_step_percent", 10)),
                ),
            ),
            color_temperature_step_kelvin=max(
                100,
                min(
                    2000,
                    int(
                        home_assistant.get(
                            "color_temperature_step_kelvin", 500
                        )
                    ),
                ),
            ),
            entities=_home_assistant_entities_from_config(
                home_assistant.get("entities")
            ),
        ),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _project_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _applications_from_config(value: Any) -> tuple[ApplicationConfig, ...]:
    if value is None:
        value = _DEFAULT_APPLICATIONS
    if not isinstance(value, list):
        raise ValueError("pc_control.applications must be a JSON array.")

    applications: list[ApplicationConfig] = []
    identifiers: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each pc_control application must be a JSON object.")
        identifier = str(item.get("id", "")).strip().casefold()
        if not identifier or not identifier.replace("_", "").isalnum():
            raise ValueError(f"Invalid pc_control application id: {identifier!r}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate pc_control application id: {identifier}")
        identifiers.add(identifier)

        configured_path = _optional_string(item.get("path"))
        applications.append(
            ApplicationConfig(
                identifier=identifier,
                aliases=_string_tuple(item.get("aliases"), fallback=(identifier,)),
                executable_names=_string_tuple(item.get("executable_names")),
                process_names=_string_tuple(item.get("process_names")),
                start_menu_names=_string_tuple(item.get("start_menu_names")),
                configured_path=(
                    _project_path(configured_path) if configured_path is not None else None
                ),
            )
        )
    return tuple(applications)


def _string_tuple(value: Any, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return fallback
    if not isinstance(value, list):
        raise ValueError("Application name collections must be JSON arrays.")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    return result or fallback


def _home_assistant_entities_from_config(
    value: Any,
) -> tuple[HomeAssistantEntityConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("home_assistant.entities must be a JSON array.")

    entities: list[HomeAssistantEntityConfig] = []
    identifiers: set[str] = set()
    entity_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each Home Assistant entity must be a JSON object.")
        identifier = str(item.get("id", "")).strip().casefold()
        entity_id = str(item.get("entity_id", "")).strip().casefold()
        if not identifier or not identifier.replace("_", "").isalnum():
            raise ValueError(f"Invalid Home Assistant entity id: {identifier!r}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z0-9_]+", entity_id):
            raise ValueError(f"Invalid Home Assistant entity_id: {entity_id!r}")
        domain = entity_id.split(".", 1)[0]
        if domain not in {
            "light",
            "switch",
            "fan",
            "sensor",
            "binary_sensor",
            "climate",
            "scene",
        }:
            raise ValueError(
                f"Home Assistant domain is disabled for safety: {domain!r}"
            )
        if identifier in identifiers:
            raise ValueError(f"Duplicate Home Assistant registry id: {identifier}")
        if entity_id in entity_ids:
            raise ValueError(f"Duplicate Home Assistant entity_id: {entity_id}")
        identifiers.add(identifier)
        entity_ids.add(entity_id)
        aliases = _string_tuple(
            item.get("aliases"),
            fallback=(identifier.replace("_", " "),),
        )
        entities.append(
            HomeAssistantEntityConfig(
                identifier=identifier,
                entity_id=entity_id,
                aliases=aliases,
                room=_optional_string(item.get("room")),
                allow_control=bool(item.get("allow_control", True)),
            )
        )
    return tuple(entities)


_DEFAULT_APPLICATIONS = [
    {
        "id": "chrome",
        "aliases": ["chrome", "google chrome", "browser"],
        "executable_names": ["chrome.exe"],
        "process_names": ["chrome.exe"],
        "start_menu_names": ["Google Chrome", "Chrome"],
    },
    {
        "id": "spotify",
        "aliases": ["spotify", "music"],
        "executable_names": ["spotify.exe"],
        "process_names": ["spotify.exe"],
        "start_menu_names": ["Spotify"],
    },
    {
        "id": "discord",
        "aliases": ["discord"],
        "executable_names": ["discord.exe"],
        "process_names": ["discord.exe"],
        "start_menu_names": ["Discord"],
    },
    {
        "id": "steam",
        "aliases": ["steam"],
        "executable_names": ["steam.exe"],
        "process_names": ["steam.exe"],
        "start_menu_names": ["Steam"],
    },
]


def _load_with_local_override(path: Path) -> dict[str, Any]:
    base = _load_json_object(path)
    local_path = path.with_name(f"{path.stem}.local{path.suffix}")
    if not local_path.exists():
        return base
    return _deep_merge(base, _load_json_object(local_path))


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration file must contain a JSON object: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
