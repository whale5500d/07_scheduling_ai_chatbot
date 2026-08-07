# llm.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_pipeline_2.state import ResponseVerdict, AgentState

_FIRST_JUDGED_RESULT = 'unclear' # 'positive', 'negative', 'unclear'
_STUB_NEGATIVE_KEYWORDS = ["꼭 가야 돼?", "배고파"]
_DATE_EXPRESSION_KEYWORDS = ["오늘", "내일", "모레", "이번주", "다음주"]
DEFAULT_MODEL = "gemini-2.0-flash-lite"

def get_bound_agent(is_already_searched: bool, state: AgentState):
    """
    도구 목록을 인자로 받습니다.
    도구들이 바인딩된 것처럼 동작하는 Fake LLM 객체를 돌려주는 역할을 합니다.
    """
    result = fake_get_bound_agent(is_already_searched, state)
    return result

    # gemini-2.0-flash_lite
    # llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL)
    # return llm.bind_tools(tools)

def get_date_agent(question: str | None):
    # (단기) 날짜 표현 키워드 포함 여부로만 판정하는 규칙 기반 stub.
    # TODO: (장기) 실제 LLM 연결 후 상대 날짜 표현 전반을 정규화하도록 교체 예정.
    # fake chat models
    result = fake_get_date_agent(question)
    return result

def fake_get_bound_agent(is_already_searched: bool, state: AgentState):
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    query_text = human_messages[-1].content if human_messages else None

    # 1차 진입(검색 전): LLM 1차 판단
    if not is_already_searched:
        if _FIRST_JUDGED_RESULT == 'positive':
            return GenericFakeChatModel(messages=iter([AIMessage(content=ResponseVerdict.POSITIVE.value)]))
        elif _FIRST_JUDGED_RESULT == 'negative':
            return GenericFakeChatModel(messages=iter([AIMessage(content=ResponseVerdict.NEGATIVE.value)]))
        else:
            return GenericFakeChatModel(messages=iter([AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "search_case_examples",
                    "args": {"query": query_text or "판정 불가 사례"},
                }],
            )]))

    # 2차 진입(검색 후): LLM 2차 판단 - 키워드 기반 최종 판정
    if query_text and any(keyword in query_text for keyword in _STUB_NEGATIVE_KEYWORDS):
        return GenericFakeChatModel(messages=iter([AIMessage(content=ResponseVerdict.NEGATIVE.value)]))
    return GenericFakeChatModel(messages=iter([AIMessage(content=ResponseVerdict.POSITIVE.value)]))

def fake_get_date_agent(question: str | None):
    if question and any(keyword in question for keyword in _DATE_EXPRESSION_KEYWORDS):
        return GenericFakeChatModel(messages=iter([AIMessage(content="2026-08-08")]))
    return GenericFakeChatModel(messages=iter([AIMessage(content="표현 불가")]))