from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from voice_assistant.config import WebSearchConfig


logger = logging.getLogger(__name__)

NEWS_QUERY_CUE = re.compile(
    r"\b(?:news|headlines?|breaking|current events?|what happened|what's happening|"
    r"been doing recently)\b",
    re.IGNORECASE,
)


class SearchError(RuntimeError):
    """Base error for search provider failures."""


class SearchUnavailableError(SearchError):
    """The provider could not complete a search."""


class SearchResponseError(SearchError):
    """The provider returned no usable structured results."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class DuckDuckGoSearchProvider:
    name = "duckduckgo (ddgs auto fallback)"

    def __init__(
        self,
        timeout_seconds: float = 8.0,
        region: str = "us-en",
        safesearch: str = "moderate",
        client_factory: Any | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._region = region
        self._safesearch = safesearch
        self._client_factory = client_factory

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        client_factory = self._client_factory
        if client_factory is None:
            from ddgs import DDGS

            client_factory = DDGS
        raw_results: Any = None
        last_error: Exception | None = None
        search_method = "news" if NEWS_QUERY_CUE.search(query) else "text"
        provider_query = _news_search_query(query) if search_method == "news" else query
        for attempt, backend in enumerate(("duckduckgo", "auto"), start=1):
            try:
                client = client_factory(timeout=self._timeout_seconds)
                method = getattr(client, search_method)
                search_options: dict[str, Any] = {
                    "region": self._region,
                    "safesearch": self._safesearch,
                    "max_results": max_results,
                    "backend": backend,
                }
                if search_method == "news":
                    search_options["timelimit"] = "m"
                raw_results = method(provider_query, **search_options)
                if backend != "duckduckgo":
                    logger.info("DuckDuckGo unavailable; ddgs automatic fallback succeeded.")
                break
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "DuckDuckGo provider %s attempt %d failed (backend=%s)",
                    search_method,
                    attempt,
                    backend,
                    exc_info=True,
                )
        else:
            raise SearchUnavailableError(
                "DuckDuckGo search failed or timed out."
            ) from last_error

        if not isinstance(raw_results, list):
            try:
                raw_results = list(raw_results)
            except (TypeError, ValueError) as exc:
                raise SearchResponseError(
                    "Search provider returned a malformed response."
                ) from exc

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title"), 240)
            url = _clean_url(item.get("href") or item.get("url"))
            snippet = _clean_text(item.get("body") or item.get("snippet"), 600)
            if not title or not url or not snippet or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=_clean_text(item.get("source"), 120) or _source_name(url),
                )
            )
            if len(results) >= max_results:
                break

        if not results:
            raise SearchResponseError("Search returned no usable results.")
        return results


def create_search_provider(config: WebSearchConfig) -> SearchProvider:
    provider = config.provider.casefold().strip()
    if provider == "duckduckgo":
        return DuckDuckGoSearchProvider(
            timeout_seconds=config.timeout_seconds,
            region=config.region,
            safesearch=config.safesearch,
        )
    raise ValueError(f"Unsupported web search provider: {config.provider!r}")


def _clean_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _clean_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _source_name(url: str) -> str:
    host = (urlparse(url).hostname or "").casefold()
    return host.removeprefix("www.")


def _news_search_query(query: str) -> str:
    concise = query
    patterns = (
        r"\bwhat(?:'s| is) the latest news (?:about|on)\b",
        r"\blatest news (?:about|on)\b",
        r"\b(?:latest|recent|breaking) (?:news|headlines?|updates?)\b",
        r"\b(?:news|headlines?|updates?) (?:today|right now|this week)\b",
        r"\bwhat(?:'s| is) happening (?:with|in)\b",
        r"\bwhat has\b",
        r"\bbeen doing recently\b",
        r"\b(?:today|yesterday|right now|currently)\b",
        r"\b(?:20\d{2})\b",
    )
    for pattern in patterns:
        concise = re.sub(pattern, " ", concise, flags=re.IGNORECASE)
    concise = " ".join(concise.strip(" ?.,!").split())
    return concise or "world"
