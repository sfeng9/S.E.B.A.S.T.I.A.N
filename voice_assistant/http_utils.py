from __future__ import annotations

from urllib.parse import urlsplit


def validated_http_url(value: str, *, require_https: bool = False) -> str:
    url = value.strip()
    parsed = urlsplit(url)
    allowed_schemes = {"https"} if require_https else {"http", "https"}
    if parsed.scheme.casefold() not in allowed_schemes or not parsed.hostname:
        expected = "HTTPS" if require_https else "HTTP or HTTPS"
        raise ValueError(f"URL must use {expected} and include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials must not be embedded in URLs.")
    return url.rstrip("/")
