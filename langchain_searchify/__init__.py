"""LangChain tools for searchify: keyless web search, clean page text and local reranking."""

from importlib import metadata
from typing import List

from langchain_searchify.searchify_extract import SearchifyExtract
from langchain_searchify.searchify_search import SearchifySearch

try:
    __version__: str = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__: List[str] = ["SearchifySearch", "SearchifyExtract", "__version__"]
