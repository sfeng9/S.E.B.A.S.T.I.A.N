from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class DateTimeParseError(ValueError):
    pass


TIME_ONLY_PATTERN = re.compile(
    r"^\s*(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\s*$",
    re.IGNORECASE,
)


def home_now(timezone_name: str, now: datetime | None = None) -> datetime:
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise DateTimeParseError(f"Unknown timezone: {timezone_name}") from exc
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(zone)


def parse_datetime(
    value: str,
    timezone_name: str,
    now: datetime | None = None,
    prefer_future: bool = True,
) -> datetime:
    text = value.strip()
    if not text:
        raise DateTimeParseError("A date and time are required.")
    local_now = home_now(timezone_name, now)
    if TIME_ONLY_PATTERN.match(text):
        normalized = text.casefold().replace(".", "").strip()
        for format_string in ("%I:%M %p", "%I %p", "%H:%M", "%H"):
            try:
                parsed_time = datetime.strptime(normalized.upper(), format_string).time()
                parsed = datetime.combine(
                    local_now.date(), parsed_time, tzinfo=local_now.tzinfo
                )
                return parsed + timedelta(days=1) if parsed <= local_now else parsed
            except ValueError:
                continue
        raise DateTimeParseError(f"Could not understand time: {value}")
    try:
        parsed_iso = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed_iso = None
    if parsed_iso is not None:
        if parsed_iso.tzinfo is None:
            parsed_iso = parsed_iso.replace(tzinfo=local_now.tzinfo)
        return parsed_iso.astimezone(local_now.tzinfo)

    try:
        import dateparser
    except ImportError as exc:
        raise DateTimeParseError(
            "Date parsing dependency is missing. Run pip install -r requirements.txt."
        ) from exc
    parsed = dateparser.parse(
        text,
        settings={
            "TIMEZONE": timezone_name,
            "TO_TIMEZONE": timezone_name,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": local_now,
            "PREFER_DATES_FROM": "future" if prefer_future else "current_period",
        },
    )
    if parsed is None:
        raise DateTimeParseError(f"Could not understand date/time: {value}")
    parsed = parsed.astimezone(local_now.tzinfo)
    return parsed


def parse_period(
    value: str,
    timezone_name: str,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    local_now = home_now(timezone_name, now)
    normalized = " ".join(value.casefold().split())
    if normalized in {"today", "today's", "my day"}:
        day = local_now.date()
    elif normalized == "tomorrow":
        day = local_now.date() + timedelta(days=1)
    elif normalized in {"this afternoon", "afternoon", "today afternoon"}:
        return _day_window(local_now, time(12), time(17))
    elif normalized in {"this evening", "evening", "tonight", "today evening"}:
        return _day_window(local_now, time(17), time.max)
    else:
        parsed = parse_datetime(value, timezone_name, now=local_now)
        day = parsed.date()
    start = datetime.combine(day, time.min, tzinfo=local_now.tzinfo)
    return start, start + timedelta(days=1)


def _day_window(
    local_now: datetime,
    start_time: time,
    end_time: time,
) -> tuple[datetime, datetime]:
    start = datetime.combine(local_now.date(), start_time, tzinfo=local_now.tzinfo)
    if end_time == time.max:
        end = datetime.combine(
            local_now.date() + timedelta(days=1), time.min, tzinfo=local_now.tzinfo
        )
    else:
        end = datetime.combine(local_now.date(), end_time, tzinfo=local_now.tzinfo)
    return start, end
