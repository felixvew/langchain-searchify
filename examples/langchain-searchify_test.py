"""
@autor: Félix Fautsch
date: 02.11.2026

Simple langgraph agent with forced websearch using the langchain-searchify library with a local llm hosted through vLLM
Web Search results get stored inside the state and used inside the generate_response node to create the final answer
No APi key needed!
"""
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

import operator
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import BaseModel, Field

import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

from langchain_searchify import SearchifySearch

load_dotenv(Path(__file__).resolve().parents[1] / "vllm" / ".env")  # the same file docker compose reads
model = ChatOpenAI(
        model=os.environ['VLLM_MODEL'],
        base_url="http://localhost:8001/v1",
        api_key="not_needed",
        temperature=0.0
    )

search_tool = SearchifySearch(max_results=5, top_k=5, max_chars=500)
query_writer = model.bind_tools([search_tool], tool_choice=search_tool.name) # tool_choice forces a tool call

class WebSearchAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    web_search_results: list
    generation: str

# Using a small local model whos training data only goes up to 2023, we need to add todays date in the system prompt
QUERY_PROMPT = """Today's date is {today}.
Search the web for the information needed to answer the user's question."""

def local_search_tool(state: WebSearchAgentState) -> dict:
    """Let the model write a search query, run SearchifySearch and put the passages into the state."""
    system_prompt = SystemMessage(content=QUERY_PROMPT.format(today=date.today().isoformat()))
    query_call = query_writer.invoke([system_prompt] + state["messages"])
    search_args = query_call.tool_calls[0]["args"]           # e.g. {"query": "Ballon d'Or 2026 winner", "topic": "news"}
    print(search_args)
    result = search_tool.invoke(search_args)                  # a dict, or an error message if the search failed
    if isinstance(result, str):
        print(result)
        return {"web_search_results": []}
    return {"web_search_results": result["results"]}


GENERATE_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

Question: {question}

Context: {context}
"""

def generate_response(state: WebSearchAgentState):
    question = state["messages"][-1].content
    context = state["web_search_results"]
    generate_prompt_formatted = GENERATE_PROMPT.format(question=question, context=context)
    answer = model.invoke([HumanMessage(content=generate_prompt_formatted)])

    return {"generation": answer}

builder = StateGraph(WebSearchAgentState)

builder.add_node("web_search", local_search_tool)
builder.add_node("generate", generate_response)

builder.add_edge(START, "web_search")
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# searchify extracts pages in a "spawn" process pool, which re-imports this file in every worker
if __name__ == "__main__":
    graph_input = {"messages": [HumanMessage(content="Qui a gagné le ballon d'or cette année ?")]}
    print(graph.invoke(graph_input))
    #print(local_search_tool(state=graph_input))