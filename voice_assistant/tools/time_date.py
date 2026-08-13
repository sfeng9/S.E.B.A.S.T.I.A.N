from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TimezoneError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalTimeResult:
    requested_location: str | None
    resolved_location: str | None
    is_home: bool
    display_time: str
    iso_datetime: str
    timezone: str
    timezone_abbreviation: str
    utc_offset: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurrentDateResult:
    requested_location: str | None
    resolved_location: str | None
    is_home: bool
    display_date: str
    iso_date: str
    timezone: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DayOfWeekResult:
    requested_location: str | None
    resolved_location: str | None
    is_home: bool
    day_of_week: str
    iso_date: str
    timezone: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_current_local_time(
    now: datetime | None = None,
    *,
    timezone_name: str | None = None,
    requested_location: str | None = None,
    resolved_location: str | None = None,
    is_home: bool = True,
) -> LocalTimeResult:
    local_now = _location_now(now, timezone_name)
    hour = local_now.strftime("%I").lstrip("0") or "12"
    return LocalTimeResult(
        requested_location=requested_location,
        resolved_location=resolved_location,
        is_home=is_home,
        display_time=f"{hour}:{local_now:%M %p}",
        iso_datetime=local_now.isoformat(timespec="seconds"),
        timezone=timezone_name or str(local_now.tzinfo or "local time"),
        timezone_abbreviation=local_now.tzname() or "local time",
        utc_offset=local_now.strftime("%z"),
    )


def get_current_date(
    now: datetime | None = None,
    *,
    timezone_name: str | None = None,
    requested_location: str | None = None,
    resolved_location: str | None = None,
    is_home: bool = True,
) -> CurrentDateResult:
    local_now = _location_now(now, timezone_name)
    return CurrentDateResult(
        requested_location=requested_location,
        resolved_location=resolved_location,
        is_home=is_home,
        display_date=f"{local_now:%B} {local_now.day}, {local_now.year}",
        iso_date=local_now.date().isoformat(),
        timezone=timezone_name or str(local_now.tzinfo or "local time"),
    )


def get_day_of_week(
    now: datetime | None = None,
    *,
    timezone_name: str | None = None,
    requested_location: str | None = None,
    resolved_location: str | None = None,
    is_home: bool = True,
) -> DayOfWeekResult:
    local_now = _location_now(now, timezone_name)
    return DayOfWeekResult(
        requested_location=requested_location,
        resolved_location=resolved_location,
        is_home=is_home,
        day_of_week=local_now.strftime("%A"),
        iso_date=local_now.date().isoformat(),
        timezone=timezone_name or str(local_now.tzinfo or "local time"),
    )


def _location_now(now: datetime | None, timezone_name: str | None) -> datetime:
    if now is None:
        instant = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        instant = now.astimezone()
    else:
        instant = now

    if timezone_name is None:
        return instant.astimezone()
    try:
        target_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise TimezoneError(f"Timezone {timezone_name!r} is unavailable.") from exc
    return instant.astimezone(target_timezone)
