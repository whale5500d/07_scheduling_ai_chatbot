"""
langgraph_pipeline.utils.llm: Agent용 LLM 생성

[분리 이유]
LLM 선택 지점은 2차 리팩토링(포트/어댑터 패턴 적용) 대상으로 확정되었다
(docs/retrospective, "의사결정 정리 - 리팩토링 방법론 선정" 참고).
외부 SDK 세부사항(에러 타입, 응답 스키마 등)이 도메인 로직에 새어들지
않도록 격리해야 하는 경계이기 때문이다.

이 파일을 tools.py/nodes.py와 미리 분리해 둔 이유는, 2차 리팩토링 시
변경 범위를 이 파일 하나로 국소화(local change)하기 위함이다. 즉 이번
1차(계층형) 리팩토링 시점에는 아직 포트/어댑터를 도입하지 않지만,
다음 리팩토링에서 손댈 대상을 미리 격리해 둔 것이다.
"""
from __future__ import annotations

_DEFAULT_MODEL = "gemini-2.0-flash-lite"


def get_agent_llm(tools: list, model: str = _DEFAULT_MODEL):
    """도구가 바인딩된 Gemini ChatModel을 반환합니다.

    bind_tools()로 도구 스키마를 LLM에 주입한다.
    BaseLLM(HuggingFacePipeline)과의 차이:
      BaseLLM은 bind_tools()를 지원하지 않아 tool_calls를 생성할 수 없다.
      ChatModel(ChatGoogleGenerativeAI)만 native tool calling을 지원한다.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model)
    return llm.bind_tools(tools)