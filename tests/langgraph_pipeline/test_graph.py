"""
Test for langgraph_pipeline: graph.py (Agent)

기존 고정 RAG 파이프라인 테스트에서 Agent 동작 테스트로 교체한다.

핵심 검증:
1. 도구 호출 없는 응답 → tools 노드를 거치지 않고 END로 종료
2. 도구 호출 1회 → ToolNode 실행 후 다시 call_model로 돌아와 최종 답변
3. invoke() 결과가 AgentState 형태(messages 키)인가
4. HumanMessage가 최종 히스토리에 보존되는가
5. 도구를 두 번 호출하는 루프 시나리오
6. run_rag_agent() → 문자열 반환
7. run_rag_agent() → content가 list인 경우도 문자열로 변환 (Gemini multi-part 대응)

[테스트 전략]
실제 Gemini API를 호출하지 않고 GenericFakeChatModel로 시나리오를 재현한다.
build_rag_graph()는 내부에서 get_agent_llm()을 호출해 Gemini LLM을 생성하므로,
테스트에서는 build_rag_graph() 대신 make_call_model_node()를 직접 주입한다.
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_pipeline.vector_store import build_vector_store
from langgraph_pipeline.graph import run_rag_agent
from langgraph_pipeline.state import AgentState
from langgraph_pipeline.tools import get_all_tools, make_call_model_node


@pytest.fixture
def fake_embedding():
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="DaySync의 기본 API 포트는 9221입니다.", metadata={"idx": 0}),
        Document(page_content="선호도 점수가 0.65 이상이면 선호 활동입니다.", metadata={"idx": 1}),
    ]


@pytest.fixture
def store(sample_documents, fake_embedding):
    return build_vector_store(sample_documents, fake_embedding)


def _build_test_graph(fake_llm, store):
    """테스트용: GenericFakeChatModel을 직접 주입한 Agent 그래프를 빌드한다.

    GenericFakeChatModel은 bind_tools()를 지원하지 않으므로
    fake_llm을 make_call_model_node()에 직접 전달한다.
    ToolNode는 AIMessage.tool_calls[].name으로 도구를 dispatch하므로,
    도구 이름이 get_all_tools() 목록과 일치하면 정상 동작한다.
    """
    tools = get_all_tools(store, k=2)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("call_model", make_call_model_node(fake_llm))
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition)
    graph.add_edge("tools", "call_model")
    return graph.compile()


class TestAgentGraph:
    def test_no_tool_call_ends_immediately(self, store):
        """도구 호출 없는 AIMessage → tools 노드를 거치지 않고 END로 종료."""
        fake_llm = GenericFakeChatModel(
            messages=iter([AIMessage(content="바로 답변합니다.")])
        )
        graph = _build_test_graph(fake_llm, store)
        result = graph.invoke({"messages": [HumanMessage(content="테스트 질문")]})

        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        assert last.content == "바로 답변합니다."

    def test_tool_call_routes_to_tools_node(self, store):
        """도구 호출 AIMessage → ToolNode 실행 후 최종 답변."""
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_001",
                "name": "calc_preference_score",
                "args": {"score": 0.75},
            }],
        )
        final_msg = AIMessage(content="선호 활동입니다.")

        fake_llm = GenericFakeChatModel(messages=iter([tool_call_msg, final_msg]))
        graph = _build_test_graph(fake_llm, store)
        result = graph.invoke({"messages": [HumanMessage(content="0.75점은?")]})

        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert "선호 활동" in tool_messages[0].content

        last = messages[-1]
        assert isinstance(last, AIMessage)
        assert last.content == "선호 활동입니다."

    def test_result_has_messages_key(self, store):
        """invoke() 결과가 AgentState 형태(messages 키)인가."""
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="답변")]))
        graph = _build_test_graph(fake_llm, store)
        result = graph.invoke({"messages": [HumanMessage(content="질문")]})

        assert "messages" in result
        assert isinstance(result["messages"], list)

    def test_human_message_preserved_in_history(self, store):
        """HumanMessage가 최종 messages 히스토리에 보존되는가."""
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="답변")]))
        graph = _build_test_graph(fake_llm, store)
        question = "DaySync API 포트가 뭐야?"
        result = graph.invoke({"messages": [HumanMessage(content=question)]})

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert human_messages[0].content == question

    def test_multiple_tool_calls_loop(self, store):
        """도구를 두 번 호출하는 시나리오 — 루프가 올바르게 반복되는가."""
        tool_call_1 = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "calc_preference_score", "args": {"score": 0.8}}],
        )
        tool_call_2 = AIMessage(
            content="",
            tool_calls=[{"id": "c2", "name": "calc_preference_score", "args": {"score": 0.3}}],
        )
        final_msg = AIMessage(content="두 점수 모두 분석 완료.")

        fake_llm = GenericFakeChatModel(messages=iter([tool_call_1, tool_call_2, final_msg]))
        graph = _build_test_graph(fake_llm, store)
        result = graph.invoke({"messages": [HumanMessage(content="두 점수 비교해줘")]})

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        assert result["messages"][-1].content == "두 점수 모두 분석 완료."


class TestRunRagAgent:
    def test_returns_string(self, store):
        """run_rag_agent()는 문자열을 반환해야 한다."""
        from unittest.mock import patch

        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="문자열 답변")]))
        fake_graph = _build_test_graph(fake_llm, store)

        with patch("langgraph_pipeline.graph.build_rag_graph", return_value=fake_graph):
            result = run_rag_agent("테스트", store)

        assert isinstance(result, str)
        assert result == "문자열 답변"

    def test_list_content_is_joined_to_string(self, store):
        """AIMessage.content가 list[dict]인 경우도 문자열로 변환 (Gemini multi-part 대응)."""
        from unittest.mock import patch

        list_content_msg = AIMessage(
            content=[{"type": "text", "text": "파트1"}, {"type": "text", "text": "파트2"}]
        )
        fake_llm = GenericFakeChatModel(messages=iter([list_content_msg]))
        fake_graph = _build_test_graph(fake_llm, store)

        with patch("langgraph_pipeline.graph.build_rag_graph", return_value=fake_graph):
            result = run_rag_agent("테스트", store)

        assert result == "파트1파트2"
