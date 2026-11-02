"""Unit tests: tool schemas and result shaping, with searchify replaced by fakes (no network)."""

from typing import Any, Dict, List, Type

import pytest
from langchain_tests.unit_tests import ToolsUnitTests
from searchify import SearchBackendError

from langchain_searchify import SearchifyExtract, SearchifySearch, _utilities
from langchain_searchify import searchify_extract, searchify_search

LONG = "word " * 20
HITS = [
    {"title": "A", "url": "https://a.example/1", "content": "Snippet A", "engine": "duckduckgo"},
    {"title": "B", "url": "https://b.example/2", "content": "Snippet B", "engine": "duckduckgo"},
]
PAGES = [
    {
        "url": "https://a.example/1",
        "title": "Page A",
        "content": f"Short line\nAlpha {LONG}\nAlpha {LONG}\nBeta {LONG}",
    },
    {"url": "https://b.example/2", "title": "Page B", "content": f"Gamma {LONG}"},
]


class TestSearchifySearchStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[SearchifySearch]:
        return SearchifySearch

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "Ballon d'Or 2026", "topic": "news", "time_range": "month"}


class TestSearchifyExtractStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[SearchifyExtract]:
        return SearchifyExtract

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"urls": ["https://en.wikipedia.org/wiki/Ballon_d%27Or"], "query": "2026 winner"}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Replace searchify's network calls with fakes and record what they received.

    The fake reranker reverses the input order, so reranked output is easy to predict.
    """
    calls: Dict[str, Any] = {}

    def fake_search(query: str, max_results: int, **kwargs: Any) -> List[Dict]:
        calls["search"] = {"query": query, "max_results": max_results, **kwargs}
        return HITS[:max_results]

    def fake_extract(urls: List[str]) -> List[Dict]:
        calls["extract"] = urls
        return [p for p in PAGES if p["url"] in urls]

    def fake_read_pages(urls: List[str], fmt: str, timeout: Any) -> tuple:
        known = {p["url"] for p in PAGES}
        return (
            [p for p in PAGES if p["url"] in urls],
            [{"url": u, "error": "unreadable"} for u in urls if u not in known],
        )

    def fake_rerank(query: str, items: List[Dict], top_k: Any = None) -> List[Dict]:
        ranked = [dict(it, rerank_score=float(i)) for i, it in enumerate(items)][::-1]
        return ranked[:top_k] if top_k else ranked

    monkeypatch.setattr(searchify_search, "search", fake_search)
    monkeypatch.setattr(searchify_search, "extract", fake_extract)
    monkeypatch.setattr(searchify_extract, "read_pages", fake_read_pages)
    monkeypatch.setattr(_utilities, "rerank", fake_rerank)
    monkeypatch.setattr(_utilities, "rerank_available", lambda: True)
    return calls


def test_search_returns_reranked_passages(calls: Dict[str, Any]) -> None:
    result = SearchifySearch(top_k=2).invoke({"query": "q"})

    assert result["query"] == "q"
    assert [r["content"].split()[0] for r in result["results"]] == ["Gamma", "Beta"]
    assert [r["score"] for r in result["results"]] == [2.0, 1.0]
    assert result["results"][0] == {
        "title": "Page B", "url": "https://b.example/2", "content": f"Gamma {LONG}".strip(), "score": 2.0,
    }


def test_search_skips_short_and_repeated_paragraphs(calls: Dict[str, Any]) -> None:
    result = SearchifySearch(rerank=False, top_k=10).invoke({"query": "q"})

    assert [r["content"].split()[0] for r in result["results"]] == ["Alpha", "Beta", "Gamma"]
    assert all("score" not in r for r in result["results"])


def test_search_passes_filters_and_settings(calls: Dict[str, Any]) -> None:
    SearchifySearch(max_results=1, backend="searxng").invoke(
        {"query": "q", "topic": "news", "time_range": "week", "language": "fr",
         "include_domains": ["lequipe.fr"], "exclude_domains": ["x.com"]}
    )

    assert calls["search"] == {
        "query": "q", "max_results": 1, "backend": "searxng", "topic": "news",
        "time_range": "week", "language": "fr",
        "include_domains": ["lequipe.fr"], "exclude_domains": ["x.com"],
    }
    assert calls["extract"] == ["https://a.example/1"]


def test_search_snippets_only_reads_no_pages(calls: Dict[str, Any]) -> None:
    result = SearchifySearch(include_page_content=False).invoke({"query": "q"})

    assert "extract" not in calls
    assert [r["content"] for r in result["results"]] == ["Snippet A", "Snippet B"]


def test_search_falls_back_to_snippets_when_no_page_is_readable(
    calls: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(searchify_search, "extract", lambda urls: [])

    result = SearchifySearch().invoke({"query": "q"})

    assert [r["url"] for r in result["results"]] == [h["url"] for h in HITS]


def test_search_trims_long_passages(calls: Dict[str, Any]) -> None:
    result = SearchifySearch(max_chars=30).invoke({"query": "q"})

    assert all(len(r["content"]) <= 32 and r["content"].endswith(" …") for r in result["results"])


def test_search_without_results_tells_the_agent(
    calls: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(searchify_search, "search", lambda *a, **k: [])

    result = SearchifySearch().invoke({"query": "q"})

    assert "No search results for 'q'" in result


def test_search_backend_error_tells_the_agent(
    calls: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_search(*args: Any, **kwargs: Any) -> List[Dict]:
        raise SearchBackendError("ddgs error: RatelimitException")

    monkeypatch.setattr(searchify_search, "search", failing_search)

    result = SearchifySearch().invoke({"query": "q"})

    assert result == "Web search failed: ddgs error: RatelimitException"


def test_rerank_defaults_to_installed_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_utilities, "rerank_available", lambda: False)
    assert SearchifySearch().rerank is False
    assert SearchifyExtract().rerank is False
    with pytest.raises(ImportError, match=r"langchain-searchify\[rerank\]"):
        SearchifySearch(rerank=True)

    monkeypatch.setattr(_utilities, "rerank_available", lambda: True)
    assert SearchifySearch().rerank is True


def test_extract_reports_failed_and_skipped_urls(calls: Dict[str, Any]) -> None:
    urls = ["https://a.example/1", "https://missing.example/", "https://b.example/2"]

    result = SearchifyExtract(max_urls=2, max_chars=40).invoke({"urls": urls})

    assert [r["url"] for r in result["results"]] == ["https://a.example/1"]
    assert result["results"][0]["title"] == "Page A"
    assert len(result["results"][0]["content"]) <= 42
    assert result["failed_urls"] == ["https://missing.example/"]
    assert result["skipped_urls"] == ["https://b.example/2"]


def test_extract_with_query_puts_matching_paragraphs_first(calls: Dict[str, Any]) -> None:
    result = SearchifyExtract().invoke({"urls": ["https://a.example/1"], "query": "beta"})

    content = result["results"][0]["content"]
    assert content.startswith("Beta") and "Alpha" in content
    assert "failed_urls" not in result and "skipped_urls" not in result


def test_extract_ignores_query_without_reranking(calls: Dict[str, Any]) -> None:
    result = SearchifyExtract(rerank=False).invoke({"urls": ["https://a.example/1"], "query": "beta"})

    assert result["results"][0]["content"].startswith("Short line")


def test_extract_nothing_readable_tells_the_agent(calls: Dict[str, Any]) -> None:
    result = SearchifyExtract().invoke({"urls": ["https://missing.example/"]})

    assert "None of these pages could be read" in result
