# llm.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_pipeline_2.state import ResponseVerdict

_is_accuracy = True # 긍정/부정 답변의 LLM 판단 결과
DEFAULT_MODEL = "gemini-2.0-flash-lite"

def get_bound_agent(is_already_searched: bool):
    """
    도구 목록을 인자로 받습니다.
    도구들이 바인딩된 것처럼 동작하는 Fake LLM 객체를 돌려주는 역할을 합니다.
    """
    # (단기) 모든 UNCLEAR를 LLM에게 위임해 전수 처리(exhaustiveness)를 보장함.
    # TODO: (장기) 사용자 패턴 기반 벡터 유사도 로직을 점진 추가해 LLM 호출 비용, 응답 속도를 개선 예정
    # fake chat models
    if _is_accuracy or is_already_searched: # 최종 판정 (tool_calls 없음)
        return GenericFakeChatModel(messages=iter([AIMessage(content=ResponseVerdict.POSITIVE.value)]))
    else: # 애매한 답변일 경우, tool_calls 있음. search_case_examples 호출 (예정)
        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "search_case_examples",
                "args": {"query": "판정 불가 사례"},
            }],)
        return GenericFakeChatModel(messages=iter([ai_message]))

    # gemini-2.0-flash_lite
    # llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL)
    # return llm.bind_tools(tools)

def get_date_agent(question: str | None):
    # fake chat models
    if question:
         return GenericFakeChatModel(messages=iter([AIMessage(content="2026-08-08")]))
    return GenericFakeChatModel(messages=iter([AIMessage(content="표현 불가")]))