"""
Test for Level 1: evaluate_answer_relevancy() in debugs/evaluate_answer_relevancy.py

검증 항목:
1. 빈 답변 입력 시 처리 (LLM 호출 없이 즉시 0.0 반환되는가)
2. 질문에 직접 대응하는 답변은 관련 있음으로 판단하는가
3. 질문과 무관한 답변은 관련 없음으로 판단하는가
4. 점수가 0.0 또는 1.0 중 하나로만 반환되는가 (이진 판단 설계 검증)
"""

import pytest

from .evaluate_answer_relevancy import evaluate_answer_relevancy  # noqa: E402
from rag_pipeline.generator import TextGenerator  # noqa: E402


@pytest.fixture(scope="module")
def generator():
    """모듈 전체에서 TextGenerator를 한 번만 로딩하여 재사용한다."""
    return TextGenerator()

@pytest.mark.slow
class TestEvaluateAnswerRelevancy:
    """evaluate_answer_relevancy() 함수에 대한 테스트 그룹 (Level 1: LLM-as-a-Judge)"""

    def test_empty_answer_returns_zero_score_without_llm_call(self):
        """예외 케이스: 빈 답변을 넣으면 LLM 호출 없이 즉시 0.0을 반환해야 한다."""
        result = evaluate_answer_relevancy(
            question="기본 API 포트는 무엇인가?",
            answer="",
            generator=None # type: ignore
        )
        assert result["answer_relevancy_score"] == 0.0
        assert result["is_relevant"] is False

    def test_score_is_binary(self, generator):
        """설계 검증: 점수가 0.0 또는 1.0 중 하나로만 반환되어야 한다 (이진 판단)."""
        question = "DaySync의 기본 API 포트는 무엇인가?"
        answer = "DaySync의 기본 API 포트는 9221이다."

        result = evaluate_answer_relevancy(question, answer, generator)

        assert result["answer_relevancy_score"] in (0.0, 1.0)

    def test_answer_directly_addressing_question_is_relevant(self, generator):
        """
        의미적 검증: 질문에 직접 대응하는 답변은 관련 있다고 판단되어야 한다.
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        answer = "DaySync의 기본 API 포트는 9221이다."

        result = evaluate_answer_relevancy(question, answer, generator)

        assert result["is_relevant"] is True
        assert result["answer_relevancy_score"] == 1.0

    def test_answer_unrelated_to_question_is_not_relevant(self, generator):
        """
        의미적 검증: 질문과 전혀 무관한 답변은 관련 없다고 판단되어야 한다.
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        answer = "오늘 날씨는 맑고 바람이 약하게 분다."

        result = evaluate_answer_relevancy(question, answer, generator)

        assert result["is_relevant"] is False
        assert result["answer_relevancy_score"] == 0.0