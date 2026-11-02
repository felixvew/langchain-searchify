# langchain-searchify

LangChain tools for [searchify](https://github.com/sami-mag07/searchify-l): web search, clean page text and local reranking, with no paid search API or API key.

| Tool | What the model can do with it |
|---|---|
| `SearchifySearch` | Search the web and get the most relevant passages from the top result pages, each with its source URL |
| `SearchifyExtract` | Read specific URLs and get their clean main text |

Both are standard LangChain `BaseTool`s, so they work with `bind_tools()`, `create_agent()` and LangGraph's `ToolNode`.

## Install

```bash
pip install "langchain-searchify[rerank] @ git+https://github.com/felixvew/langchain-searchify"
```

The `[rerank]` extra adds `sentence-transformers` (and PyTorch) for searchify's local cross-encoder. Without it the tools still work, but passages come back in page order instead of by relevance.

## Use

Call a tool directly:

```python
from langchain_searchify import SearchifyExtract, SearchifySearch

search = SearchifySearch(max_results=5, top_k=8)
search.invoke({"query": "Ballon d'Or 2026 winner"})
# {"query": "...", "results": [{"title": "...", "url": "...", "content": "...", "score": 2.36}, ...]}
```

Give the tools to an agent:

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=[SearchifySearch(), SearchifyExtract()])
agent.invoke({"messages": [{"role": "user", "content": "Who won the Ballon d'Or this year?"}]})
```

Or wire them into your own LangGraph graph:

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [SearchifySearch(), SearchifyExtract()]
model_with_tools = model.bind_tools(tools)
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", tools_condition)
```

> [!IMPORTANT]
> searchify reads pages in a `spawn` process pool, on every operating system. Each worker re-imports your main script, so put your script's top-level code under `if __name__ == "__main__":`. Without the guard, that code runs again in every worker.

## Configuration

Settings you choose when you create a tool. The model can't change them.

| `SearchifySearch` | Default | Meaning |
|---|---|---|
| `max_results` | `5` | Search hits to look at |
| `include_page_content` | `True` | Read the result pages and return their best passages; `False` returns only search snippets |
| `top_k` | `8` | Passages to return |
| `rerank` | `None` | Rank passages with the local cross-encoder; `None` means "if `[rerank]` is installed" |
| `min_words` | `12` | Shorter paragraphs are not used as passages |
| `max_chars` | `600` | Each passage or snippet is trimmed to this length |
| `backend` | `None` | `"duckduckgo"`, `"searxng"`, `"serper"` or `"tavily"`; `None` uses searchify's configured chain |

| `SearchifyExtract` | Default | Meaning |
|---|---|---|
| `max_urls` | `3` | URLs read per call; the rest are reported as `skipped_urls` |
| `max_chars` | `4000` | Each page's text is trimmed to this length |
| `format` | `"text"` | `"text"` or `"markdown"` |
| `rerank` | `None` | When the model passes a `query`, put the matching paragraphs first |
| `timeout` | `None` | Fetch timeout per page in seconds |

Everything the tools return goes back into the model's context, so lower `top_k` and `max_chars` for models with small context windows.

searchify itself is configured through environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_BACKEND` | `duckduckgo` | Primary search backend |
| `SEARCH_FALLBACK` | `searxng` | Backend to try if the first finds nothing |
| `SEARXNG_URL` | | Your own SearXNG instance |
| `SERPER_API_KEY`, `TAVILY_API_KEY` | | Optional cloud fallbacks, used only if set |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Hugging Face name or local path of the reranker |
| `FETCH_TIMEOUT` | `15` | Page fetch timeout in seconds |

## Privacy

"No API key" doesn't mean offline. With the default setup, the model's search queries go to DuckDuckGo and the result pages are fetched from their websites. If `SERPER_API_KEY` or `TAVILY_API_KEY` is set, queries can also go to those services. Don't send confidential information in queries. To keep searches inside your own network, point `SEARCH_BACKEND` at a self-hosted SearXNG instance.

## Development

```bash
pip install -e ".[rerank]" pytest langchain-tests
pytest                          # unit tests, no network
pytest tests/integration_tests  # real searches, needs internet
```

## Why it's not on PyPI

searchify isn't on PyPI yet, and PyPI rejects packages that depend on a Git URL. Until searchify is published there, install this package from GitHub.
