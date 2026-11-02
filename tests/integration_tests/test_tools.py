"""Integration tests: real web searches and page reads (needs internet)."""

from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_searchify import SearchifyExtract, SearchifySearch

WIKIPEDIA_PYTHON = "https://en.wikipedia.org/wiki/Python_(programming_language)"


class TestSearchifySearchStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[SearchifySearch]:
        return SearchifySearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"max_results": 3, "top_k": 3}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "Python programming language creator"}


class TestSearchifyExtractStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[SearchifyExtract]:
        return SearchifyExtract

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"urls": [WIKIPEDIA_PYTHON]}


def test_search_returns_passages_with_sources() -> None:
    result = SearchifySearch(max_results=3, top_k=3).invoke({"query": "Python programming language creator"})

    assert len(result["results"]) == 3
    assert all(r["url"].startswith("http") and r["content"] for r in result["results"])


def test_extract_with_query_finds_the_relevant_part() -> None:
    result = SearchifyExtract(max_chars=1500).invoke(
        {"urls": [WIKIPEDIA_PYTHON], "query": "Who designed Python?"}
    )

    assert "Guido van Rossum" in result["results"][0]["content"]
