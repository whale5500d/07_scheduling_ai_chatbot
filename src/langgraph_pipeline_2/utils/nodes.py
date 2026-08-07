from typing import cast
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_pipeline_2.state import AgentState, ResponseVerdict
from langgraph_pipeline_2.utils.llm import get_bound_agent, get_date_agent

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
    today = datetime.now().strftime("%Y-%m-%d")
    input_prompt = f"오늘은 {today}입니다. 다음 표현의 절대 날짜를 구하세요: {question}"

    llm = get_date_agent(question)
    response = llm.invoke(input_prompt)
    casted_content = cast(str, response.content)

    if casted_content == "표현 불가":
        return {
            "messages": [AIMessage(content=casted_content)],
            "resolved_date": None
        }

    return {
        "messages": [AIMessage(content=casted_content)],
        "resolved_date": datetime.strptime(cast(str, casted_content), "%Y-%m-%d")
    }

def save_rdb(state: AgentState):
    """일정을 RDB에 저장하는 최소 규칙 버전. 실제 POST 요청은 이후
    단계에서 추가 예정."""
    # TODO: question을 content로 변경하기
    question = state.get('pending_question')
    resolved_date = state.get("resolved_date")

    if question and resolved_date:
        saved_payload = {'content': question, 'date': resolved_date}
        answer = f"저장 완료: '{question}' → {resolved_date}"
    elif question:
        saved_payload = {'content': question, 'date': None}
        answer = f"저장 완료: '{question}' (날짜 미정)"
    else:
        saved_payload = None
        answer = "저장할 일정이 없음."

    return {
        "messages": [AIMessage(content=str(answer))],
        "pending_question": None,
        "response_verdict": None,
        "resolved_date": None,
    }

def call_model(state: AgentState):
    is_already_searched = isinstance(state["messages"][-1], ToolMessage)

    llm = get_bound_agent(is_already_searched)
    response = llm.invoke(state["messages"])

    # 2. tool_calls가 있을 경우, 기존값 유지
    if len(response.tool_calls) > 0:
        return {"messages": [response]}
    
    # 3. tool_calls가 없을 경우, Structured Output으로 최종 판정
    return {
        "messages": [response],
        "response_verdict": response.content
    }
