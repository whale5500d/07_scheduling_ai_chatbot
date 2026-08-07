from datetime import datetime
from enum import StrEnum
from typing import Annotated, TypedDict, NotRequired, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ResponseVerdict(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = 'negative'
    UNCLEAR = 'unclear' # 응답이 불분명한 상태, 추후 RAG 사용 시점으로 사용

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_question: NotRequired[str | None]
    response_verdict: NotRequired[Literal["positive", "negative", "unclear"] | None]
    resolved_date: NotRequired[datetime | None]