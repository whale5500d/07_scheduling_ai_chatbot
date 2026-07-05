"""
langgraph_pipeline: RAG 파이프라인 → Agent 전환

[핵심 변경: 왜 Agent로 바꿨는가]

기존 고정 파이프라인(graph.py 이전 버전):
  START → [retrieve] → _route_after_retrieve → (generate | no_results) → END

  문제: 사용자 질문이 무엇이든 항상 retrieve를 먼저 실행한다.
  "선호도 점수 0.8은 몇 등급이야?" 처럼 검색이 필요 없는 질문에도
  불필요하게 벡터 스토어를 조회한다.

변경 후 Agent 구조:
  START → [call_model] → tools_condition
    → "tools" → [tool_node] → [call_model]  (반복)
    → END

  LLM이 매 단계 "도구가 필요한가"를 스스로 결정한다.

[LLM이 판단하는 방법 — bind_tools()]

  tools.py의 get_agent_llm()이 ChatGoogleGenerativeAI.bind_tools(tools)를 호출한다.
  bind_tools()는 도구 목록의 스키마(이름·설명·파라미터)를 LLM에 등록한다.
  이후 LLM은 응답을 생성할 때 두 가지 중 하나를 선택한다:
    (a) AIMessage(content="직접 답변")      → tools_condition → END
    (b) AIMessage(tool_calls=[...])         → tools_condition → "tools" 노드 실행

  고정 파이프라인은 (a)/(b) 분기 자체가 없었다.
  Agent에서는 이 분기를 LLM이 질문의 의도를 보고 결정한다.

[스트리밍]
  ChatGoogleGenerativeAI + stream_mode="messages"는 call_model 노드에서
  토큰 단위 AIMessageChunk를 직접 yield한다.
  기존 stream_rag_answer()의 "retrieve 완료 포착 → llm.stream() 직접 호출" 혼합 방식이
  필요 없어진다 — ChatModel이 graph 내부에서 완결되기 때문이다.
"""
from __future__ import annotations

from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_pipeline.state import AgentState
from langgraph_pipeline.tools import _DEFAULT_MODEL, get_agent_llm, get_all_tools, make_call_model_node


def build_rag_graph(
    store: InMemoryVectorStore,
    model: str = _DEFAULT_MODEL,
    k: int = 3,
):
    """RAG Agent CompiledStateGraph를 반환한다.

    기존 build_rag_graph(store, llm, k)와 시그니처가 다르다:
      이전: llm을 외부에서 주입 (HuggingFacePipeline)
      이후: model 이름만 받고 내부에서 Gemini ChatModel 생성

    LLM을 외부에서 주입하지 않는 이유:
      bind_tools()를 지원하는 ChatModel이어야 하므로, 호출부에서 잘못된 LLM을
      주입할 가능성을 원천 차단한다. main.py의 lg_llm(Gemma HuggingFacePipeline)은
      더 이상 이 그래프에 전달되지 않는다.

    Args:
        store: build_vector_store()로 만든 InMemoryVectorStore.
        model: Gemini 모델 이름 (기본값: gemini-2.0-flash-lite).
        k: search_daysync_docs 도구의 검색 개수.

    Returns:
        invoke({"messages": [HumanMessage(content=question)]}) →
            AgentState {"messages": [..., AIMessage(content=final_answer)]}
    """
    tools = get_all_tools(store, k=k)
    llm = get_agent_llm(tools, model=model)

    graph = StateGraph(AgentState)
    graph.add_node("call_model", make_call_model_node(llm))
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition)
    graph.add_edge("tools", "call_model")

    return graph.compile()


def _extract_answer(result: AgentState) -> str:
    """AgentState의 마지막 AIMessage에서 답변 문자열을 추출한다."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def run_rag_agent(
    question: str,
    store: InMemoryVectorStore,
    model: str = _DEFAULT_MODEL,
    k: int = 3,
) -> str:
    """질문을 Agent에 전달하고 최종 답변 문자열을 반환한다.

    기존 build_rag_graph().invoke()를 대체한다.
    /query 엔드포인트(비스트리밍)에 대응한다.
    """
    graph = build_rag_graph(store, model=model, k=k)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return _extract_answer(result)


def stream_rag_agent(
    question: str,
    store: InMemoryVectorStore,
    model: str = _DEFAULT_MODEL,
    k: int = 3,
) -> Iterator[str]:
    """질문을 Agent에 전달하고 call_model 노드의 토큰을 순서대로 yield한다.

    기존 stream_rag_answer()를 대체한다.
    /query/stream 엔드포인트(SSE 스트리밍)에 대응한다.

    stream_mode="messages"는 ChatModel(AIMessageChunk)을 토큰 단위로 yield한다.
    metadata["langgraph_node"]로 call_model 노드의 청크만 필터링해,
    ToolMessage(도구 실행 결과)나 중간 상태 업데이트를 제외한다.

    기존 혼합 방식(graph.stream → retrieve 포착 → llm.stream 직접 호출)과의 차이:
      기존: BaseLLM이 stream_mode="messages"를 지원하지 않아 우회가 필요했다.
      현재: ChatModel은 graph 내부에서 바로 토큰 스트리밍이 완결된다.
    """
    graph = build_rag_graph(store, model=model, k=k)
    initial_state: AgentState = {"messages": [HumanMessage(content=question)]}

    for chunk, metadata in graph.stream(initial_state, stream_mode="messages"):
        if (
            metadata.get("langgraph_node") == "call_model"
            and isinstance(chunk, AIMessage)
            and chunk.content
        ):
            yield chunk.content


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.embedding import get_embeddings_model
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.vector_store import build_vector_store

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)
    store = build_vector_store(chunks, get_embeddings_model())

    questions = [
        "DaySync의 내부 코드네임이 뭐야?",
        "선호도 점수가 0.72면 선호 활동인가요?",
        "SC-114 코드가 뭐야? 그리고 선호도 점수 0.4는 어떤 등급이에요?",
    ]

    for q in questions:
        print(f"\n[질문] {q}")
        print(f"[답변] {run_rag_agent(q, store)}")
