from __future__ import annotations

import unittest

from voice_assistant.assistant.tool_router import (
    AssistantToolRouter,
    requires_web_search,
)
from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.web_search import (
    DuckDuckGoSearchProvider,
    SearchResult,
    SearchUnavailableError,
)


class FakeDdgsClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.call: dict[str, object] = {}

    def text(self, query: str, **kwargs):
        self.call = {"query": query, **kwargs}
        return [
            {
                "title": " Python 3.14.1 released ",
                "href": "https://www.python.org/downloads/release/python-3141/",
                "body": " The latest maintenance release. ",
            },
            {"title": "Malformed", "href": "javascript:alert(1)", "body": "bad"},
        ]

    def news(self, query: str, **kwargs):
        self.call = {"query": query, "method": "news", **kwargs}
        return [
            {
                "title": "Recent coverage",
                "url": "https://example.com/news/story",
                "body": "A recent event was reported.",
                "source": "Example News",
            }
        ]


class FakeSearchProvider:
    name = "fake-duckduckgo"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.calls.append((query, max_results))
        if self.fail:
            raise SearchUnavailableError("offline")
        return [
            SearchResult(
                title="Official release notes",
                url="https://python.org/downloads/",
                snippet="Python published a new stable release.",
                source="python.org",
            ),
            SearchResult(
                title="Coverage from Reuters",
                url="https://reuters.com/technology/example",
                snippet="Reuters reported the announcement.",
                source="reuters.com",
            ),
        ]


class FailingDdgsClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def text(self, query: str, **kwargs):
        raise RuntimeError("rate limited")


class FakeProductivity:
    schemas = ()
    reminder_store = object()

    def tool_requirement(self, prompt, history):
        return None

    def execute(self, name, arguments):
        return None

    def reset_session_context(self):
        return None

    def begin_turn(self):
        return None

    def spoken_override_for(self, called_tools):
        return None


class WebSearchProviderTests(unittest.TestCase):
    def test_duckduckgo_provider_normalizes_structured_results(self) -> None:
        created: list[FakeDdgsClient] = []

        def factory(timeout: float):
            client = FakeDdgsClient(timeout)
            created.append(client)
            return client

        provider = DuckDuckGoSearchProvider(timeout_seconds=4.0, client_factory=factory)
        results = provider.search("latest Python release", max_results=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "python.org")
        self.assertEqual(results[0].title, "Python 3.14.1 released")
        self.assertEqual(created[0].timeout, 4.0)
        self.assertEqual(created[0].call["backend"], "duckduckgo")

    def test_news_query_uses_duckduckgo_news_results(self) -> None:
        created: list[FakeDdgsClient] = []

        def factory(timeout: float):
            client = FakeDdgsClient(timeout)
            created.append(client)
            return client

        provider = DuckDuckGoSearchProvider(client_factory=factory)
        results = provider.search("latest news about Jungkook", max_results=3)

        self.assertEqual(created[0].call["method"], "news")
        self.assertEqual(created[0].call["query"], "Jungkook")
        self.assertEqual(results[0].source, "Example News")

    def test_uses_bounded_auto_fallback_after_duckduckgo_failure(self) -> None:
        created: list[object] = []

        def factory(timeout: float):
            client = (
                FailingDdgsClient(timeout)
                if not created
                else FakeDdgsClient(timeout)
            )
            created.append(client)
            return client

        provider = DuckDuckGoSearchProvider(client_factory=factory)
        results = provider.search("latest Python release", max_results=2)

        self.assertEqual(len(created), 2)
        self.assertEqual(created[1].call["backend"], "auto")
        self.assertEqual(results[0].source, "python.org")


class WebSearchRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FakeSearchProvider()
        self.router = AssistantToolRouter(
            load_assistant_config(),
            productivity=FakeProductivity(),
            search_provider=self.provider,
        )

    def test_stable_knowledge_does_not_require_search(self) -> None:
        prompts = (
            "Who is Jungkook?",
            "Who was Albert Einstein?",
            "What is photosynthesis?",
            "Explain a black hole.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(requires_web_search(prompt))
                self.assertIsNone(self.router.tool_requirement(prompt, ()))

    def test_current_information_requires_search(self) -> None:
        prompts = (
            "What's the latest news about Jungkook?",
            "Who's the president of the United States right now?",
            "What's the newest stable Python release?",
            "What happened in the news today?",
            "Who won the Lakers game last night?",
            "Who won the Super Bowl?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(requires_web_search(prompt))
                requirement = self.router.tool_requirement(prompt, ())
                self.assertEqual(requirement["tools"], ("web_search",))

    def test_existing_tool_domains_do_not_require_web_search(self) -> None:
        prompts = (
            "What's the weather today?",
            "What time is it right now?",
            "Do I have important emails today?",
            "What's on my calendar today?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(requires_web_search(prompt))
                names = {
                    schema["function"]["name"]
                    for schema in self.router.schemas_for(prompt, ())
                }
                self.assertNotIn("web_search", names)

    def test_search_returns_structured_results_and_preserves_sources(self) -> None:
        self.router.schemas_for("What's the latest Python release?", ())
        result = self.router.execute(
            "web_search", {"query": "latest stable Python release", "max_results": 2}
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "fake-duckduckgo")
        self.assertEqual(result["results"][0]["source"], "python.org")
        self.assertIn("untrusted external content", result["security_notice"])
        sources = self.router.execute("get_last_web_sources", {})
        self.assertEqual(len(sources["sources"]), 2)
        self.assertEqual(sources["sources"][0]["title"], "Official release notes")
        self.assertEqual(
            self.router.spoken_override_for(("get_last_web_sources",)),
            "I used python.org and reuters.com.",
        )

    def test_source_context_clears_with_session_context(self) -> None:
        self.router.schemas_for("What's the latest Python release?", ())
        self.router.execute("web_search", {"query": "latest stable Python release"})
        self.router.reset_session_context()

        result = self.router.execute("get_last_web_sources", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "no_web_sources")
        self.assertEqual(
            self.router.spoken_override_for(("get_last_web_sources",)),
            "I don't have sources from a recent web search in this session.",
        )

    def test_source_follow_up_requires_source_tool(self) -> None:
        requirement = self.router.tool_requirement("Which source said that?", ())
        self.assertEqual(requirement["tools"], ("get_last_web_sources",))

    def test_provider_failure_does_not_raise(self) -> None:
        router = AssistantToolRouter(
            load_assistant_config(),
            productivity=FakeProductivity(),
            search_provider=FakeSearchProvider(fail=True),
        )
        router.schemas_for("What's the latest Python release?", ())

        result = router.execute("web_search", {"query": "latest Python release"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "web_search_unavailable")

    def test_failed_new_search_does_not_leave_stale_sources(self) -> None:
        self.router.schemas_for("What's the latest Python release?", ())
        self.router.execute("web_search", {"query": "latest stable Python release"})
        self.provider.fail = True

        self.router.execute("web_search", {"query": "latest NVIDIA news"})
        sources = self.router.execute("get_last_web_sources", {})

        self.assertFalse(sources["ok"])
        self.assertEqual(sources["error"], "no_web_sources")

    def test_private_data_not_present_in_user_request_is_blocked(self) -> None:
        self.router.schemas_for("Search for NVIDIA news", ())
        result = self.router.execute(
            "web_search", {"query": "edward@example.com NVIDIA private meeting"}
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "private_search_query_blocked")
        self.assertEqual(self.provider.calls, [])


if __name__ == "__main__":
    unittest.main()
