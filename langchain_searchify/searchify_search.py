"""Tool for keyless web search with searchify."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, model_validator
from searchify import SearchBackendError, extract, search

from langchain_searchify._utilities import (
    best_passages,
    resolve_rerank,
    split_passages,
    trim,
)


class SearchifySearchInput(BaseModel):
    """Input for [SearchifySearch]."""

    # searchify has no index of its own: the query goes to public search engines as written and
    # the passages are reranked against it, so a full question finds more than keywords
    query: str = Field(
        description="The user's question as one complete, standalone question, in the user's "
        "language and worded the way they asked it. Do not shorten it to keywords and do not "
        "translate it. Only add what the conversation makes clear, such as names, events or "
        "years. Always replace relative times with the actual year or date from today's date: "
        '"this year", "cette année" or "dieses Jahr" become the year, "last weekend" becomes '
        'the dates. Example: after a question about the Champions League final, "and who '
        'scored?" becomes "Who scored in the Champions League final between Paris '
        'Saint-Germain and Arsenal?"'
    )
    topic: Optional[Literal["general", "news"]] = Field(
        default="general",
        description='Use "news" for recent events and current affairs, "general" otherwise.',
    )
    # no "day" or "week": small models pick them for questions about past events
    # (e.g. "this year's winner") and the filter then hides the articles that answer them
    time_range: Optional[Literal["month", "year"]] = Field(
        default=None,
        description="Only return results published within this period before today. Leave empty "
        "unless the user explicitly asks for recent results.",
    )
    language: Optional[Literal["en", "de", "fr", "es"]] = Field(
        default=None,
        description="Prefer results in this language. Leave empty for no preference.",
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        description='Only return results from these domains, e.g. ["wikipedia.org"].',
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Never return results from these domains.",
    )


class SearchifySearch(BaseTool):  # type: ignore[override]
    """Keyless web search with searchify.

    Searches the web (DuckDuckGo by default), reads the top result pages and
    returns their most relevant passages, each with its source URL.

    Setup:
        .. code-block:: bash

            pip install "langchain-searchify[rerank] @ git+https://github.com/<owner>/langchain-searchify"

    Instantiate:
        .. code-block:: python

            from langchain_searchify import SearchifySearch

            tool = SearchifySearch(max_results=5, top_k=8)

    Invoke directly:
        .. code-block:: python

            tool.invoke({"query": "Ballon d'Or 2026 winner"})

    Use with an agent:
        .. code-block:: python

            model.bind_tools([tool])
    """

    name: str = "searchify_search"
    description: str = (
        "A web search engine. Useful for questions about current events or facts "
        "that may have changed. Returns the most relevant passages from the top "
        "result pages, each with its source URL."
    )
    args_schema: Type[BaseModel] = SearchifySearchInput
    handle_tool_error: bool = True

    max_results: int = 5
    """Number of search hits to look at."""
    include_page_content: bool = True
    """Read the result pages and return their best passages. If False, return only
    the search snippets (faster, no pages are fetched)."""
    top_k: int = 8
    """Number of passages to return when include_page_content is True."""
    rerank: Optional[bool] = None
    """Order passages by relevance with searchify's local cross-encoder. None uses it
    if the [rerank] extra is installed or a `reranker` is given, otherwise passages keep
    their page order."""
    reranker: Optional[Any] = None
    """A loaded reranker to use instead of searchify's default model, e.g. a
    sentence-transformers CrossEncoder created with trust_remote_code=True. Anything with
    a predict(pairs) method that returns one score per (query, text) pair works."""
    min_words: int = 12
    """Paragraphs with fewer words are not used as passages."""
    max_chars: int = 600
    """Each passage or snippet is trimmed to this many characters."""
    backend: Optional[str] = None
    """searchify backend: "duckduckgo", "searxng", "serper" or "tavily". None uses
    searchify's configured chain (SEARCH_BACKEND, then SEARCH_FALLBACK)."""

    @model_validator(mode="after")
    def _resolve_rerank(self) -> "SearchifySearch":
        self.rerank = resolve_rerank(self.rerank, custom=self.reranker is not None)
        return self

    def _run(
        self,
        query: str,
        topic: Optional[Literal["general", "news"]] = "general",
        time_range: Optional[Literal["month", "year"]] = None,
        language: Optional[Literal["en", "de", "fr", "es"]] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool."""
        try:
            hits = search(
                query,
                self.max_results,
                backend=self.backend,
                topic=topic or "general",
                time_range=time_range,
                language=language,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
        except SearchBackendError as exc:
            raise ToolException(f"Web search failed: {exc}") from exc
        if not hits:
            raise ToolException(
                f"No search results for '{query}'. Try fewer or different keywords, "
                "or remove the time and domain filters."
            )

        results = self._passages(query, hits) if self.include_page_content else []
        if not results:  # snippets only, or none of the pages could be read
            results = [
                {"title": h["title"], "url": h["url"], "content": trim(h["content"], self.max_chars)}
                for h in hits
            ]
        return {"query": query, "results": results}

    def _passages(self, query: str, hits: List[Dict]) -> List[Dict]:
        pages = extract([h["url"] for h in hits])
        passages = split_passages(pages, self.min_words)
        best = best_passages(query, passages, self.top_k, bool(self.rerank), self.reranker)
        for passage in best:
            passage["content"] = trim(passage["content"], self.max_chars)
        return best
