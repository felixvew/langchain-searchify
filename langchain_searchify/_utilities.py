"""Helpers shared by the searchify tools: passage splitting, trimming and reranking."""

import importlib.util
from typing import Any, Dict, List, Optional

from searchify import rerank

RERANK_INSTALL_HINT = (
    'Reranking needs sentence-transformers: pip install "langchain-searchify[rerank]", '
    "or create the tool with rerank=False."
)


def rerank_available() -> bool:
    """True if searchify's cross-encoder can be loaded."""
    return importlib.util.find_spec("sentence_transformers") is not None


def resolve_rerank(setting: Optional[bool], custom: bool = False) -> bool:
    """None means "rerank if a reranker is available"; True fails loudly if none is.

    `custom` is True when the tool was given its own reranker, which needs no extra.
    """
    if custom:
        return setting is not False
    if setting is None:
        return rerank_available()
    if setting and not rerank_available():
        raise ImportError(RERANK_INSTALL_HINT)
    return setting


def trim(text: str, max_chars: int) -> str:
    """Cut text to max_chars at a word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def split_passages(pages: List[Dict], min_words: int) -> List[Dict]:
    """One item per paragraph of each page, skipping short lines and repeats."""
    passages: List[Dict] = []
    seen = set()
    for page in pages:
        for paragraph in (page.get("content") or "").split("\n"):
            paragraph = paragraph.strip()
            if len(paragraph.split()) < min_words or paragraph in seen:
                continue
            seen.add(paragraph)
            passages.append(
                {"title": page.get("title") or "", "url": page["url"], "content": paragraph}
            )
    return passages


def best_passages(
    query: str,
    passages: List[Dict],
    top_k: Optional[int],
    use_rerank: bool,
    reranker: Optional[Any] = None,
) -> List[Dict]:
    """The top_k passages (all if None), most relevant first when reranking is on.

    `reranker` replaces searchify's default model: anything with a predict(pairs) method
    that returns one score per (query, text) pair, such as a sentence-transformers CrossEncoder.
    """
    if not use_rerank:
        return passages[:top_k] if top_k else passages
    if reranker is None:
        ranked = rerank(query, passages, top_k=top_k)
        for passage in ranked:
            passage["score"] = round(passage.pop("rerank_score"), 3)
        return ranked
    if not passages:
        return []
    scores = reranker.predict([(query, p["content"]) for p in passages])
    scored = sorted(zip(passages, map(float, scores)), key=lambda ps: -ps[1])
    ranked = [dict(p, score=round(s, 3)) for p, s in scored]
    return ranked[:top_k] if top_k else ranked
