from langchain_core.messages import HumanMessage, AIMessage
from langgraph_pipeline_2.state import AgentState, ResponseVerdict

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
    question = state.get("pending_question")

    if question and "내일" in question:
        answer = "날짜: 내일로 판정됨 (정규화 로직은 추후 정교화 예정)"
    else:
        answer = "날짜 표현을 찾을 수 없습니다"

    return {"messages": [AIMessage(content=answer)]}

def save_rdb(state: AgentState):
    """일정을 RDB에 저장하는 최소 규칙 버전. 실제 POST 요청은 이후
    단계에서 추가 예정."""
    question = state.get("pending_question")

    if question:
        answer = f"저장 완료: '{question}'"
    else:
        answer = "저장할 일정이 없습니다"

    return {
        "messages": [AIMessage(content=answer)],
        "pending_question": None,
        "response_verdict": None,
    }