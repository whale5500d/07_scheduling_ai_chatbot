"""
Test for langgraph_pipeline: tools.py

검증 항목:
1. calc_preference_score: 0.65 이상 → 선호, 0.35~0.65 → 보류, 0.35 미만 → 비선호
2. calc_preference_score: 범위 밖 점수 → 오류 메시지 반환
3. check_schedule_conflict: 수락+거절 조합 → SC-114 감지
4. check_schedule_conflict: 수락+수락 조합 → 충돌 없음
5. check_schedule_conflict: 분류 불가 응답 → 안내 메시지
6. make_search_tool: 문서가 있는 store → 관련 내용 반환
7. make_search_tool: 빈 store → "찾을 수 없습니다" 반환
8. get_all_tools: 도구 3개를 반환하는가
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.vector_store import build_vector_store
from langgraph_pipeline.tools import (
    calc_preference_score,
    check_schedule_conflict,
    get_all_tools,
    make_search_tool,
)


@pytest.fixture
def fake_embedding():
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="DaySync의 기본 API 포트는 9221입니다.", metadata={"idx": 0}),
        Document(page_content="선호도 점수가 0.65 이상이면 선호 활동입니다.", metadata={"idx": 1}),
        Document(page_content="SC-114는 동일 요일에 거절과 수락이 동시에 기록된 경우입니다.", metadata={"idx": 2}),
    ]


@pytest.fixture
def store(sample_documents, fake_embedding):
    return build_vector_store(sample_documents, fake_embedding)


@pytest.fixture
def empty_store(fake_embedding):
    return InMemoryVectorStore(embedding=fake_embedding)


class TestCalcPreferenceScore:
    def test_high_score_is_preferred(self):
        result = calc_preference_score.invoke({"score": 0.8})
        assert "선호 활동" in result
        assert "0.8" in result

    def test_boundary_score_is_preferred(self):
        result = calc_preference_score.invoke({"score": 0.65})
        assert "선호 활동" in result

    def test_mid_score_is_neutral(self):
        result = calc_preference_score.invoke({"score": 0.5})
        assert "보류 활동" in result

    def test_low_score_is_not_preferred(self):
        result = calc_preference_score.invoke({"score": 0.2})
        assert "비선호 활동" in result

    def test_boundary_low_score_is_not_preferred(self):
        result = calc_preference_score.invoke({"score": 0.34})
        assert "비선호 활동" in result

    def test_out_of_range_score_returns_error_message(self):
        result = calc_preference_score.invoke({"score": 1.5})
        assert "유효하지 않은" in result

    def test_negative_score_returns_error_message(self):
        result = calc_preference_score.invoke({"score": -0.1})
        assert "유효하지 않은" in result


class TestCheckScheduleConflict:
    def test_accept_then_reject_detects_sc114(self):
        result = check_schedule_conflict.invoke(
            {"first_response": "수락", "second_response": "거절"}
        )
        assert "SC-114" in result
        assert "거절" in result  # 마지막 응답이 채택됨을 명시

    def test_reject_then_accept_detects_sc114(self):
        result = check_schedule_conflict.invoke(
            {"first_response": "아니요", "second_response": "예"}
        )
        assert "SC-114" in result

    def test_both_accept_no_conflict(self):
        result = check_schedule_conflict.invoke(
            {"first_response": "수락", "second_response": "응"}
        )
        assert "충돌 없음" in result
        # 메시지에 "SC-114 코드가 발생하지 않습니다"처럼 단어가 포함될 수 있으므로
        # "발생"이 있는지로 충돌 없음을 검증한다
        assert "감지" not in result

    def test_both_reject_no_conflict(self):
        result = check_schedule_conflict.invoke(
            {"first_response": "거절", "second_response": "아니"}
        )
        assert "충돌 없음" in result

    def test_unclassifiable_response_returns_guidance(self):
        result = check_schedule_conflict.invoke(
            {"first_response": "글쎄요", "second_response": "수락"}
        )
        assert "분류할 수 없습니다" in result


class TestMakeSearchTool:
    def test_returns_document_content(self, store):
        # DeterministicFakeEmbedding은 의미 없는 벡터를 생성하므로 특정 단어 매칭 대신
        # "문서 내용이 반환되었다"는 사실만 검증한다 — fixture의 세 문서 중 하나는 반드시 나온다.
        search_tool = make_search_tool(store, k=2)
        result = search_tool.invoke({"query": "API 포트"})
        fixture_words = ["9221", "0.65", "SC-114"]
        assert any(w in result for w in fixture_words)

    def test_empty_store_returns_not_found_message(self, empty_store):
        search_tool = make_search_tool(empty_store, k=3)
        result = search_tool.invoke({"query": "아무 질문"})
        assert "찾을 수 없습니다" in result

    def test_result_contains_document_markers(self, store):
        search_tool = make_search_tool(store, k=2)
        result = search_tool.invoke({"query": "선호도"})
        assert "[문서 1]" in result

    def test_k_limits_document_count(self, store):
        search_tool = make_search_tool(store, k=1)
        result = search_tool.invoke({"query": "DaySync"})
        assert "[문서 1]" in result
        assert "[문서 2]" not in result


class TestGetAllTools:
    def test_returns_three_tools(self, store):
        tools = get_all_tools(store)
        assert len(tools) == 3

    def test_tool_names_are_unique(self, store):
        tools = get_all_tools(store)
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_expected_tool_names_present(self, store):
        tools = get_all_tools(store)
        names = {t.name for t in tools}
        assert "search_daysync_docs" in names
        assert "calc_preference_score" in names
        assert "check_schedule_conflict" in names
