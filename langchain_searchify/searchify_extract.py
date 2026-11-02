"""Tool for reading web pages as clean text with searchify."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, model_validator
from searchify import read_pages

from langchain_searchify._utilities import (
    best_passages,
    resolve_rerank,
    split_passages,
    trim,
)


class SearchifyExtractInput(BaseModel):
    """Input for [SearchifyExtract]."""

    urls: List[str] = Field(
        description="URLs of the web pages to read, for example from search results."
    )
    query: Optional[str] = Field(
        default=None,
        description="Optional. What you are looking for on these pages. If given, the "
        "most relevant parts of each page come first instead of the page's beginning.",
    )


class SearchifyExtract(BaseTool):  # type: ignore[override]
    """Read web pages as clean main text with searchify.

    Setup:
        .. code-block:: bash

            pip install "langchain-searchify[rerank] @ git+https://github.com/<owner>/langchain-searchify"

    Instantiate:
        .. code-block:: python

            from langchain_searchify import SearchifyExtract

            tool = SearchifyExtract(max_urls=3, max_chars=4000)

    Invoke directly:
        .. code-block:: python

            tool.invoke({"urls": ["https://en.wikipedia.org/wiki/Ballon_d%27Or"]})
    """

    name: str = "searchify_extract"
    description: str = (
        "Reads web pages and returns their clean main text. Use it when you already "
        "know the URL, for example from a search result, and need more than a short passage."
    )
    args_schema: Type[BaseModel] = SearchifyExtractInput
    handle_tool_error: bool = True

    max_urls: int = 3
    """At most this many URLs are read per call; the rest are reported as skipped."""
    max_chars: int = 4000
    """Each page's text is trimmed to this many characters."""
    format: Literal["text", "markdown"] = "text"
    """Output format of the page text."""
    rerank: Optional[bool] = None
    """Use searchify's local cross-encoder to put the parts matching `query` first.
    None uses it if the [rerank] extra is installed or a `reranker` is given; without
    either, `query` is ignored."""
    reranker: Optional[Any] = None
    """A loaded reranker to use instead of searchify's default model, e.g. a
    sentence-transformers CrossEncoder created with trust_remote_code=True. Anything with
    a predict(pairs) method that returns one score per (query, text) pair works."""
    min_words: int = 12
    """Paragraphs with fewer words are not ranked separately when a query is given."""
    timeout: Optional[float] = None
    """Fetch timeout per page in seconds. None uses searchify's FETCH_TIMEOUT."""

    @model_validator(mode="after")
    def _resolve_rerank(self) -> "SearchifyExtract":
        self.rerank = resolve_rerank(self.rerank, custom=self.reranker is not None)
        return self

    def _run(
        self,
        urls: List[str],
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool."""
        to_read, skipped = urls[: self.max_urls], urls[self.max_urls :]
        docs, failed = read_pages(to_read, self.format, self.timeout)
        if not docs:
            raise ToolException(
                f"None of these pages could be read: {to_read}. They may block bots "
                "or need JavaScript; try other URLs from the search results."
            )

        focused = self._focus(query, docs) if query and self.rerank else {}
        results = [
            {
                "title": doc.get("title") or "",
                "url": doc["url"],
                "content": trim(focused.get(doc["url"]) or doc["content"], self.max_chars),
            }
            for doc in docs
        ]
        output: Dict[str, Any] = {"results": results}
        if failed:
            output["failed_urls"] = [f["url"] for f in failed]
        if skipped:
            output["skipped_urls"] = skipped
        return output

    def _focus(self, query: str, docs: List[Dict]) -> Dict[str, str]:
        """Each page's paragraphs, most relevant to the query first."""
        ranked = best_passages(query, split_passages(docs, self.min_words), None, True, self.reranker)
        by_url: Dict[str, List[str]] = {}
        for passage in ranked:
            by_url.setdefault(passage["url"], []).append(passage["content"])
        return {url: "\n".join(paragraphs) for url, paragraphs in by_url.items()}
