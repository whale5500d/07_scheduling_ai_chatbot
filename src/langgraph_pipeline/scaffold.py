from enum import StrEnum
from typing import Annotated, TypedDict, NotRequired, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

def judge_schedule(state: AgentState):
    """가장 최근 사용자 메시지에 물음표가 있으면 일정 질문으로 간주하는
    최소 규칙 버전. 사례집 기반 RAG는 이후 단계에서 추가 예정."""
    last_message = state['messages'][-1]
    is_question = "?" in last_message.content

    if is_question:
        return {
            "messages": [AIMessage(content="일정 질문 O")],
            "pending_question": last_message.content
        }

    return {
        "messages": [AIMessage(content="일정 질문 X")],
        "pending_question": None
    }

def judge_response(state: AgentState):
    """가장 최근 사용자 메시지를 긍정/부정 키워드로 판정하는 최소 규칙 버전.
    사례집 기반 RAG는 이후 단계에서 추가 예정."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_message = human_messages[-1]
    text = last_message.content

    positive_keywords = ["응", "좋아", "당연", "그러자", "찬성"]
    negative_keywords = ["아니", "싫어", "별로", "그러지 말자", "반대"]

    if any(k in text for k in positive_keywords):
        answer, verdict = "긍정 응답", ResponseVerdict.POSITIVE
    elif any(k in text for k in negative_keywords):
        answer, verdict = "부정 응답", ResponseVerdict.NEGATIVE
    else:
        answer, verdict = "응답 판정 불가", ResponseVerdict.UNCLEAR

    return {
            "messages": [AIMessage(content=answer)],
            "response_verdict": verdict.value
        }

def judge_date(state: AgentState):
    """pending_question에서 상대적 날짜 표현을 절대 날짜로 정규화하는
    최소 규칙 버전. 사례집 기반 정교화는 이후 단계에서 추가 예정."""
    question = state["pending_question"]

    if question and "내일" in question:
        answer = "날짜: 내일로 판정됨 (정규화 로직은 추후 정교화 예정)"
    else:
        answer = "날짜 표현을 찾을 수 없습니다"

    return {"messages": [AIMessage(content=answer)]}

def save_rdb(state: AgentState):
    """일정을 RDB에 저장하는 최소 규칙 버전. 실제 POST 요청은 이후
    단계에서 추가 예정."""
    question = state["pending_question"]

    if question:
        answer = f"저장 완료: '{question}'"
    else:
        answer = "저장할 일정이 없습니다"

    return {
        "messages": [AIMessage(content=answer)],
        "pending_question": None,
        "response_verdict": None,
    }

def route_from_start(state: AgentState):
    return "judge_response" if state.get("pending_question") else "judge_schedule"

def route_after_schedule(state: AgentState):
    return "judge_response" if state.get("pending_question") else END

def route_after_response(state: AgentState):
    return "judge_date" if state.get("response_verdict") == ResponseVerdict.POSITIVE.value else END

# print
def print_result(label: str, result: dict):
    print(f"\n[{label}]")
    for m in result["messages"]:
        print(f"  {type(m).__name__}: {m.content}")
    print(f"  pending_question: {result['pending_question']}")
    print("===================================================")

# 1. declare state
class ResponseVerdict(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = 'negative'
    UNCLEAR = 'unclear' # 응답이 불분명한 상태, 추후 RAG 사용 시점으로 사용

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_question: NotRequired[str | None]
    response_verdict: NotRequired[Literal["positive", "negative", "unclear"] | None]

graph = StateGraph(AgentState)

# 2. design workflow
graph.add_node(judge_schedule)
graph.add_node(judge_response)
graph.add_node(judge_date)
graph.add_node(save_rdb)
graph.add_conditional_edges(START, route_from_start, {"judge_schedule": "judge_schedule", "judge_response": "judge_response"},)
graph.add_conditional_edges("judge_schedule", route_after_schedule, {"judge_response": "judge_response", END: END})
graph.add_conditional_edges("judge_response", route_after_response, {"judge_date": "judge_date", END: END})
graph.add_edge("judge_date", "save_rdb")
graph.add_edge("save_rdb", END)

# 3. create graph
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)

# 4. run graph
config: RunnableConfig = {"configurable": {"thread_id": "test-thread-1"}}

turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
print_result("1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content="응 좋아")],
    }, config=config)
print_result("2턴", turn2)

turn3 = graph.invoke({
        "messages": [HumanMessage(content="모레 영화 볼래?")],
    }, config=config)
print_result("3턴", turn3)