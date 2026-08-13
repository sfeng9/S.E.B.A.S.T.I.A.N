from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib import error, parse, request

from voice_assistant.config import LocationConfig
from voice_assistant.http_utils import validated_http_url


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOME_REFERENCES = frozenset(
    {
        "",
        "here",
        "home",
        "at home",
        "back home",
        "around here",
        "my home",
        "my location",
        "current location",
        "outside",
    }
)


class LocationError(RuntimeError):
    """Base error for location configuration, lookup, and response failures."""


class LocationConfigurationError(LocationError):
    pass


class LocationNotFoundError(LocationError):
    def __init__(self, requested_location: str) -> None:
        self.requested_location = requested_location
        super().__init__(f"No location matched {requested_location!r}.")


class LocationAmbiguousError(LocationError):
    def __init__(self, requested_location: str, candidates: tuple[str, ...]) -> None:
        self.requested_location = requested_location
        self.candidates = candidates
        super().__init__(f"Location {requested_location!r} is ambiguous.")


class LocationUnavailableError(LocationError):
    pass


class LocationResponseError(LocationError):
    pass


@dataclass(frozen=True)
class ResolvedLocation:
    requested_location: str | None
    resolved_location: str
    latitude: float
    longitude: float
    timezone: str
    is_home: bool
    country: str | None = None
    country_code: str | None = None
    admin1: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeocodingCandidate:
    name: str
    latitude: float
    longitude: float
    timezone: str
    country: str | None
    country_code: str | None
    admin1: str | None
    population: int | None
    feature_code: str | None

    @property
    def display_name(self) -> str:
        parts = [self.name]
        for value in (self.admin1, self.country):
            if value and value.casefold() not in {part.casefold() for part in parts}:
                parts.append(value)
        return ", ".join(parts)

    def resolve(self, requested_location: str) -> ResolvedLocation:
        return ResolvedLocation(
            requested_location=requested_location,
            resolved_location=self.display_name,
            latitude=self.latitude,
            longitude=self.longitude,
            timezone=self.timezone,
            is_home=False,
            country=self.country,
            country_code=self.country_code,
            admin1=self.admin1,
        )


class OpenMeteoGeocodingClient:
    def __init__(
        self,
        timeout_seconds: float = 10.0,
        geocoding_url: str = OPEN_METEO_GEOCODING_URL,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Geocoding timeout must be greater than zero.")
        self.timeout_seconds = timeout_seconds
        self.geocoding_url = validated_http_url(geocoding_url, require_https=True)

    def search(self, location: str, count: int = 10) -> tuple[GeocodingCandidate, ...]:
        query = parse.urlencode(
            {
                "name": location,
                "count": count,
                "language": "en",
                "format": "json",
            }
        )
        http_request = request.Request(
            f"{self.geocoding_url}?{query}",
            headers={"User-Agent": "SebastianVoiceAssistant/0.2"},
        )
        try:
            # The configurable endpoint was validated as HTTPS during initialization.
            with request.urlopen(  # nosec B310
                http_request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise LocationUnavailableError(
                f"Open-Meteo geocoding returned HTTP {exc.code}."
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise LocationUnavailableError(
                "Could not connect to Open-Meteo geocoding."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LocationResponseError(
                "Open-Meteo geocoding returned malformed JSON."
            ) from exc

        return self._parse_results(payload)

    def _parse_results(self, payload: Any) -> tuple[GeocodingCandidate, ...]:
        if not isinstance(payload, dict):
            raise LocationResponseError("Geocoding response was not an object.")
        if payload.get("error"):
            raise LocationUnavailableError("Open-Meteo reported a geocoding error.")
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise LocationResponseError("Geocoding results were not a list.")

        return tuple(self._parse_candidate(item) for item in raw_results)

    @staticmethod
    def _parse_candidate(item: Any) -> GeocodingCandidate:
        if not isinstance(item, dict):
            raise LocationResponseError("A geocoding result was not an object.")
        name = _required_text(item, "name")
        latitude = _required_number(item, "latitude")
        longitude = _required_number(item, "longitude")
        timezone = _required_text(item, "timezone")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise LocationResponseError("Geocoding returned invalid coordinates.")
        population_value = item.get("population")
        population = (
            int(population_value)
            if isinstance(population_value, (int, float))
            and not isinstance(population_value, bool)
            and population_value >= 0
            else None
        )
        return GeocodingCandidate(
            name=name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            country=_optional_text(item.get("country")),
            country_code=_optional_text(item.get("country_code")),
            admin1=_optional_text(item.get("admin1")),
            population=population,
            feature_code=_optional_text(item.get("feature_code")),
        )


class LocationResolver:
    def __init__(
        self,
        home_location: LocationConfig,
        geocoder: OpenMeteoGeocodingClient | None = None,
        timeout_seconds: float = 10.0,
        cache_size: int = 128,
    ) -> None:
        self.home_location = home_location
        self.geocoder = geocoder or OpenMeteoGeocodingClient(timeout_seconds)
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, ResolvedLocation] = OrderedDict()

    def resolve(self, location: str | None = None) -> ResolvedLocation:
        requested = location.strip() if isinstance(location, str) else None
        if requested is None or _normalize(requested) in HOME_REFERENCES:
            return self._resolve_home()

        cache_key = _normalize(requested)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        search_name, qualifiers = _split_query(requested)
        candidates = self.geocoder.search(search_name)
        selected = _select_candidate(requested, search_name, qualifiers, candidates)
        resolved = selected.resolve(requested)
        self._cache[cache_key] = resolved
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return resolved

    def _resolve_home(self) -> ResolvedLocation:
        config = self.home_location
        if not config.name or not config.timezone:
            raise LocationConfigurationError(
                "Home location name and timezone are not configured."
            )
        if config.latitude is None or config.longitude is None:
            raise LocationConfigurationError(
                "Home location latitude and longitude are not configured."
            )
        if not math.isfinite(config.latitude) or not -90 <= config.latitude <= 90:
            raise LocationConfigurationError(
                "Home location latitude must be between -90 and 90."
            )
        if not math.isfinite(config.longitude) or not -180 <= config.longitude <= 180:
            raise LocationConfigurationError(
                "Home location longitude must be between -180 and 180."
            )
        return ResolvedLocation(
            requested_location=None,
            resolved_location=config.name,
            latitude=config.latitude,
            longitude=config.longitude,
            timezone=config.timezone,
            is_home=True,
        )


def _select_candidate(
    requested: str,
    search_name: str,
    qualifiers: tuple[str, ...],
    candidates: tuple[GeocodingCandidate, ...],
) -> GeocodingCandidate:
    if not candidates:
        raise LocationNotFoundError(requested)

    normalized_name = _normalize(search_name)
    exact = tuple(
        candidate
        for candidate in candidates
        if _normalize(candidate.name) == normalized_name
    )
    plausible = exact or candidates

    if qualifiers:
        qualified = tuple(
            candidate
            for candidate in plausible
            if all(_qualifier_matches(qualifier, candidate) for qualifier in qualifiers)
        )
        if qualified:
            plausible = qualified
        else:
            raise LocationNotFoundError(requested)

    ranked = sorted(
        plausible,
        key=lambda candidate: candidate.population or 0,
        reverse=True,
    )
    if len(ranked) == 1:
        return ranked[0]

    top, second = ranked[:2]
    top_population = top.population or 0
    second_population = second.population or 0
    if top_population >= 100_000 and top_population >= max(3 * second_population, 1):
        return top
    if top.feature_code == "PPLC" and top_population >= max(2 * second_population, 1):
        return top

    if not exact:
        top_similarity = SequenceMatcher(
            None, normalized_name, _normalize(top.name)
        ).ratio()
        second_similarity = SequenceMatcher(
            None, normalized_name, _normalize(second.name)
        ).ratio()
        if top_similarity >= 0.8 and top_similarity - second_similarity >= 0.15:
            return top

    options = tuple(candidate.display_name for candidate in ranked[:3])
    raise LocationAmbiguousError(requested, options)


def _split_query(query: str) -> tuple[str, tuple[str, ...]]:
    parts = tuple(part.strip() for part in query.split(",") if part.strip())
    if len(parts) > 1:
        return parts[0], tuple(_canonical_qualifier(part) for part in parts[1:])

    words = query.split()
    if len(words) > 1 and words[-1].upper() in US_STATE_NAMES:
        return " ".join(words[:-1]), (US_STATE_NAMES[words[-1].upper()],)
    return query, ()


def _qualifier_matches(qualifier: str, candidate: GeocodingCandidate) -> bool:
    normalized = _normalize(_canonical_qualifier(qualifier))
    values = (
        candidate.admin1,
        candidate.country,
        candidate.country_code,
    )
    return any(
        normalized == _normalize(value)
        or normalized in _normalize(value).split()
        for value in values
        if value
    )


def _canonical_qualifier(value: str) -> str:
    stripped = value.strip()
    return US_STATE_NAMES.get(stripped.upper(), stripped)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _required_text(data: dict[str, Any], key: str) -> str:
    value = _optional_text(data.get(key))
    if value is None:
        raise LocationResponseError(f"Geocoding field {key!r} is missing.")
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _required_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocationResponseError(f"Geocoding field {key!r} is not numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise LocationResponseError(f"Geocoding field {key!r} is not finite.")
    return number


US_STATE_NAMES = {
    abbreviation: name
    for abbreviation, name in (
        ("AL", "Alabama"), ("AK", "Alaska"), ("AZ", "Arizona"),
        ("AR", "Arkansas"), ("CA", "California"), ("CO", "Colorado"),
        ("CT", "Connecticut"), ("DE", "Delaware"), ("FL", "Florida"),
        ("GA", "Georgia"), ("HI", "Hawaii"), ("ID", "Idaho"),
        ("IL", "Illinois"), ("IN", "Indiana"), ("IA", "Iowa"),
        ("KS", "Kansas"), ("KY", "Kentucky"), ("LA", "Louisiana"),
        ("ME", "Maine"), ("MD", "Maryland"), ("MA", "Massachusetts"),
        ("MI", "Michigan"), ("MN", "Minnesota"), ("MS", "Mississippi"),
        ("MO", "Missouri"), ("MT", "Montana"), ("NE", "Nebraska"),
        ("NV", "Nevada"), ("NH", "New Hampshire"), ("NJ", "New Jersey"),
        ("NM", "New Mexico"), ("NY", "New York"), ("NC", "North Carolina"),
        ("ND", "North Dakota"), ("OH", "Ohio"), ("OK", "Oklahoma"),
        ("OR", "Oregon"), ("PA", "Pennsylvania"), ("RI", "Rhode Island"),
        ("SC", "South Carolina"), ("SD", "South Dakota"), ("TN", "Tennessee"),
        ("TX", "Texas"), ("UT", "Utah"), ("VT", "Vermont"),
        ("VA", "Virginia"), ("WA", "Washington"), ("WV", "West Virginia"),
        ("WI", "Wisconsin"), ("WY", "Wyoming"), ("DC", "District of Columbia"),
    )
}
