"""
langgraph_pipeline: DaySync 도메인 도구 정의

Agent가 호출할 수 있는 도구 3개를 정의한다.

도구 분류:
  순수 함수 도구 (store 불필요):
    - calc_preference_score : 선호도 점수(float) → 선호도 등급 문자열
    - check_schedule_conflict : 두 응답 → SC-114 위험 판단

  store 의존 도구 (클로저로 store를 캡처):
    - make_search_tool(store) → search_daysync_docs

  조합:
    - get_all_tools(store) → 세 도구를 리스트로 반환

[설계 결정] store 의존 도구를 클로저로 만드는 이유:
  @tool 데코레이터가 붙은 함수는 모듈 로딩 시 도구로 등록된다.
  서버 시작 시 구성되는 InMemoryVectorStore를 모듈 최상단에서 받을 방법이 없으므로,
  store를 인자로 받는 팩토리 함수 안에서 @tool을 선언해 store를 클로저로 캡처한다.
  이는 make_retrieve_node(store) 패턴(langgraph_pipeline/nodes.py)과 동일하다.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.prompt import format_docs

# ── 선호도 분류 기준 (DaySync 매뉴얼 §2 기반) ──────────────────────
_PREFERENCE_HIGH = 0.65   # 이 이상 → 선호 활동
_PREFERENCE_LOW = 0.35    # 이 미만 → 비선호 활동
# _PREFERENCE_LOW 이상 _PREFERENCE_HIGH 미만 → 보류(neutral)


@tool
def calc_preference_score(score: float) -> str:
    """선호도 점수(0.0~1.0)를 받아 DaySync 분류 기준에 따른 등급을 반환합니다.

    DaySync 매뉴얼에 따르면 선호도 점수가 0.65 이상이면 '선호 활동'으로 분류됩니다.

    Args:
        score: 0.0에서 1.0 사이의 선호도 점수.

    Returns:
        "선호 활동 (점수: {score})", "보류 활동 (점수: {score})", "비선호 활동 (점수: {score})" 중 하나.
    """
    if not (0.0 <= score <= 1.0):
        return f"유효하지 않은 점수입니다: {score}. 선호도 점수는 0.0~1.0 사이여야 합니다."

    if score >= _PREFERENCE_HIGH:
        grade = "선호 활동"
    elif score >= _PREFERENCE_LOW:
        grade = "보류 활동"
    else:
        grade = "비선호 활동"

    return f"{grade} (점수: {score})"


@tool
def check_schedule_conflict(first_response: str, second_response: str) -> str:
    """같은 활동에 대한 두 응답을 비교하여 일정 충돌(SC-114) 가능성을 판단합니다.

    DaySync 매뉴얼에 따르면 SC-114는 동일 요일에 거절(아니) 응답과
    수락(응) 응답이 동시에 기록된 경우 발생합니다. 충돌 시 마지막 응답이 채택됩니다.

    Args:
        first_response: 첫 번째 응답 ("수락" 또는 "거절" 계열 텍스트).
        second_response: 두 번째 응답 ("수락" 또는 "거절" 계열 텍스트).

    Returns:
        SC-114 충돌 여부와 DaySync의 처리 방식을 설명하는 문자열.
    """
    accept_keywords = {"수락", "응", "예", "yes", "ok", "확인", "동의", "승인"}
    reject_keywords = {"거절", "아니", "아니요", "no", "취소", "반려", "거부"}

    def classify(response: str) -> str:
        r = response.strip().lower()
        if any(k in r for k in accept_keywords):
            return "수락"
        if any(k in r for k in reject_keywords):
            return "거절"
        return "불명확"

    c1 = classify(first_response)
    c2 = classify(second_response)

    if c1 == "불명확" or c2 == "불명확":
        return (
            f"응답을 분류할 수 없습니다. "
            f"첫 번째: '{first_response}' → {c1}, "
            f"두 번째: '{second_response}' → {c2}. "
            "수락/거절 키워드를 포함한 응답을 입력해 주세요."
        )

    if c1 != c2:
        return (
            f"SC-114 충돌 감지: 첫 번째 응답({c1})과 두 번째 응답({c2})이 다릅니다. "
            f"DaySync는 마지막 응답인 '{second_response}'({c2})을 최종 값으로 채택합니다."
        )

    return (
        f"충돌 없음: 두 응답 모두 '{c1}'입니다. SC-114 코드가 발생하지 않습니다."
    )


def make_search_tool(store: InMemoryVectorStore, k: int = 3):
    """store를 클로저로 캡처한 search_daysync_docs 도구를 반환합니다.

    Args:
        store: 서버 시작 시 구성된 InMemoryVectorStore.
        k: 검색할 상위 문서 개수.

    Returns:
        @tool 데코레이터가 적용된 search_daysync_docs 함수.
    """
    @tool
    def search_daysync_docs(query: str) -> str:
        """DaySync 운영 매뉴얼에서 질문과 관련된 정보를 검색합니다.

        DaySync API, 설정 항목, 일정 충돌 처리, 선호도 점수 등에 대해
        모를 때 이 도구를 사용하세요.

        Args:
            query: 검색할 질문 또는 키워드.

        Returns:
            관련 문서 내용. 찾을 수 없으면 안내 문자열.
        """
        results = store.similarity_search_with_score(query, k=k)
        if not results:
            return "관련 정보를 찾을 수 없습니다."
        documents = [doc for doc, _ in results]
        return format_docs(documents)

    return search_daysync_docs


def get_all_tools(store: InMemoryVectorStore, k: int = 3) -> list:
    """Agent에 바인딩할 도구 전체 목록을 반환합니다.

    Args:
        store: 서버 시작 시 구성된 InMemoryVectorStore.
        k: 검색 도구의 상위 문서 개수.

    Returns:
        [search_daysync_docs, calc_preference_score, check_schedule_conflict]
    """
    return [
        make_search_tool(store, k),
        calc_preference_score,
        check_schedule_conflict,
    ]


_DEFAULT_MODEL = "gemini-2.0-flash-lite"


def get_agent_llm(tools: list, model: str = _DEFAULT_MODEL):
    """도구가 바인딩된 Gemini ChatModel을 반환합니다.

    bind_tools()로 도구 스키마를 LLM에 주입한다.
    이후 llm.invoke()가 반환하는 AIMessage는 LLM이 도구 호출을 결정했을 때
    tool_calls 필드를 채운다 — 이것이 "LLM이 판단한다"는 동작의 실체다.

    BaseLLM(HuggingFacePipeline)과의 차이:
      BaseLLM은 bind_tools()를 지원하지 않아 tool_calls를 생성할 수 없다.
      ChatModel(ChatGoogleGenerativeAI)만 native tool calling을 지원한다.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model)
    return llm.bind_tools(tools)


def make_call_model_node(llm):
    """call_model 노드 함수를 반환합니다.

    state["messages"] 전체(대화 히스토리)를 LLM에 전달하고,
    반환된 AIMessage를 {"messages": [response]}로 감싸 반환한다.
    add_messages reducer가 이 새 메시지를 히스토리에 append한다.
    """
    def call_model(state) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    return call_model
