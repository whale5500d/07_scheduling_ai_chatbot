"""
langgraph_pipeline: 공유 상태 정의

두 가지 StateGraph가 사용하는 상태를 한 파일에 정의한다.

RAGState — RAG 파이프라인용 (graph.py)
  각 필드는 reducer 없이 노드 반환값으로 overwrite된다.

AgentState — AI Agent용 (agent.py)
  messages 필드에 add_messages reducer를 적용해 메시지를 누적한다.
  RAGState의 overwrite 방식과 달리, 새 메시지가 기존 리스트에 append된다.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    question: str
    retrieved: list[tuple[Document, float]]
    answer: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
