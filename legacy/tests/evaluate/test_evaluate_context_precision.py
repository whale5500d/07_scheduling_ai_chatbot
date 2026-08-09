"""
Test for Level 1: evaluate_context_precision() in debugs/evaluate_context_precision.py

검증 항목:
1. 빈 리스트 입력 시 처리 (LLM 호출 없이 즉시 0.0 반환되는가)
2. 명백히 관련 있는 chunk에 대해 "관련 있음"으로 판단하는가
3. 명백히 무관한 chunk에 대해 "관련 없음"으로 판단하는가
4. judgments에 raw_response가 포함되는가 (LLM 응답 추적 가능성)
5. 점수 범위 검증 (0.0 ~ 1.0)

주의: 이 테스트들은 실제 Gemma 4 E2B-it 모델을 로딩하므로,
      최초 실행 시 모델 로딩 시간이 걸린다. scope="module"로 모델을
      한 번만 로딩하여 테스트 전체가 공유하도록 한다.
"""

import pytest

from .evaluate_context_precision import evaluate_context_precision  # noqa: E402
from rag_pipeline.generator import TextGenerator  # noqa: E402


@pytest.fixture(scope="module")
def generator():
    """모듈 전체에서 TextGenerator를 한 번만 로딩하여 재사용한다."""
    return TextGenerator()

@pytest.mark.slow
class TestEvaluateContextPrecision:
    """evaluate_context_precision() 함수에 대한 테스트 그룹 (Level 1: LLM-as-a-Judge)"""

    def test_empty_retrieved_chunks_returns_zero_score_without_llm_call(self):
        """
        예외 케이스: 빈 리스트를 넣으면 LLM을 호출하지 않고도 즉시 0.0을 반환해야 한다.
        (TextGenerator() 인스턴스화 자체를 하지 않으므로 모델 로딩 없이 빠르게 통과해야 함)
        """
        result = evaluate_context_precision(
            question="dummy question",
            retrieved_chunks=[],
            generator=None # type: ignore
        )
        assert result["context_precision_score"] == 0.0
        assert result["judgments"] == []

    def test_score_is_in_valid_range(self, generator):
        """정상 케이스: 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = [
            "DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다.",
            "오늘 날씨는 맑고 바람이 약하게 분다.",
        ]

        result = evaluate_context_precision(question, retrieved_chunks, generator)

        assert 0.0 <= result["context_precision_score"] <= 1.0

    def test_clearly_relevant_chunk_is_judged_relevant(self, generator):
        """
        의미적 검증: 질문에 직접 답하는 chunk는 "관련 있음"으로 판단되어야 한다.
        (질문과 chunk의 표현이 거의 동일하므로, LLM이 헷갈릴 여지가 거의 없는 케이스)
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다."]

        result = evaluate_context_precision(question, retrieved_chunks, generator)

        assert result["judgments"][0]["is_relevant"] is True

    def test_clearly_irrelevant_chunk_is_judged_irrelevant(self, generator):
        """
        의미적 검증: 질문과 전혀 무관한 chunk는 "관련 없음"으로 판단되어야 한다.
        (API 포트 질문에 날씨 정보는 명백히 무관한 케이스)
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["오늘 날씨는 맑고 바람이 약하게 분다."]

        result = evaluate_context_precision(question, retrieved_chunks, generator)

        assert result["judgments"][0]["is_relevant"] is False

    def test_judgments_contain_raw_response_for_traceability(self, generator):
        """
        추적 가능성 검증: 각 judgment에 LLM의 원본 응답(raw_response)이 포함되어,
        판단 근거를 사람이 직접 확인할 수 있어야 한다.
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다."]

        result = evaluate_context_precision(question, retrieved_chunks, generator)

        assert "raw_response" in result["judgments"][0]
        assert len(result["judgments"][0]["raw_response"]) > 0