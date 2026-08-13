from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.web_search import SearchError, create_search_provider


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Sebastian's configured web search provider.")
    parser.add_argument("query", nargs="?", default="latest stable Python release")
    parser.add_argument("--max-results", type=int, default=None)
    args = parser.parse_args()

    config = load_assistant_config().web_search
    if not config.enabled:
        print("Web search is disabled in config/assistant.json or its local override.")
        return 1
    provider = create_search_provider(config)
    max_results = args.max_results or config.max_results
    print(f"Provider: {provider.name}")
    print(f"Query: {args.query}")
    try:
        results = provider.search(args.query, max_results=max_results)
    except SearchError as exc:
        print(f"Result: web search failed ({type(exc).__name__}).")
        return 1

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.title}")
        print(f"   Source: {result.source}")
        print(f"   {result.snippet}")
        print(f"   {result.url}")
    print(f"Result: web search succeeded with {len(results)} result(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
