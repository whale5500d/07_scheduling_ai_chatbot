"""
Test for Level 1: evaluate_context_recall() in debugs/evaluate_context_recall.py

검증 항목:
1. 빈 Ground Truth 입력 시 처리 (LLM 호출 없이 즉시 0.0 반환되는가)
2. context에 명확히 존재하는 정보는 "포함됨"으로 판단하는가
3. context에 명확히 없는 정보는 "미포함"으로 판단하는가
4. 표현이 달라도(같은 의미를 다른 단어로 쓴 경우) 의미 기반으로 판단 가능한가
   (Level 0의 구조적 한계였던 "단어 일치 여부" 의존성에 대한 회귀 테스트)
5. 점수 범위 검증 (0.0 ~ 1.0)

DaySync 도메인 전환에 따른 변경:
- 기존에는 Ground Truth(한국어) vs context(영어)라는 언어 불일치 시나리오로
  Level 0의 한계를 검증했으나, 이제 둘 다 한국어가 되어 그 시나리오 자체가
  사라졌다. 같은 취지(표면적 단어 일치가 아니라 의미로 판단해야 한다)를
  "같은 의미를 다른 표현으로 쓴 경우"로 재구성하여 유지한다.
"""

import pytest

from .evaluate_context_recall import evaluate_context_recall  # noqa: E402
from rag_pipeline.generator import TextGenerator  # noqa: E402


@pytest.fixture(scope="module")
def generator():
    """모듈 전체에서 TextGenerator를 한 번만 로딩하여 재사용한다."""
    return TextGenerator()

@pytest.mark.slow
class TestEvaluateContextRecall:
    """evaluate_context_recall() 함수에 대한 테스트 그룹 (Level 1: LLM-as-a-Judge)"""

    def test_empty_ground_truth_returns_zero_score_without_llm_call(self):
        """예외 케이스: Ground Truth가 빈 리스트이면 LLM 호출 없이 즉시 0.0을 반환해야 한다."""
        result = evaluate_context_recall(
            question="dummy question",
            retrieved_chunks=["some context"],
            ground_truth=[], 
            generator=None # type: ignore
        )
        assert result["context_recall_score"] == 0.0
        assert result["judgments"] == []

    def test_score_is_in_valid_range(self, generator):
        """정상 케이스: 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다."]
        ground_truth = ["기본 포트는 9221이다.", "이 정보는 존재하지 않는다."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert 0.0 <= result["context_recall_score"] <= 1.0

    def test_information_present_in_context_is_matched(self, generator):
        """
        의미적 검증: context에 명확히 존재하는 정보는 "포함됨"으로 판단되어야 한다.
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다."]
        ground_truth = ["기본 포트는 9221이다."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert result["judgments"][0]["is_matched"] is True

    def test_information_absent_from_context_is_not_matched(self, generator):
        """
        의미적 검증: context에 전혀 없는 정보는 "미포함"으로 판단되어야 한다.
        (Level 1 실제 실행 결과에서 확인된 패턴: Level 0의 키워드 매칭 오탐을
         Level 1이 정확히 교정한 사례를 재현한다)
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다."]
        ground_truth = ["DaySync는 팀 일정 관리 시스템이다."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert result["judgments"][0]["is_matched"] is False

    def test_paraphrased_ground_truth_against_context_is_matched(self, generator):
        """
        회귀 테스트: Ground Truth가 context와 다른 단어로 표현되어도
        LLM이 단어 일치가 아닌 의미를 기반으로 판단할 수 있어야 한다.
        (Level 0은 단어 자체가 겹치지 않으면 매칭에 실패하는 구조적 한계가
         있었다 — Level 1에서 이 한계가 해소되었는지 확인한다)
        """
        question = "DaySync의 기본 API 포트는 무엇인가?"
        retrieved_chunks = ["DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다."]
        # "포트"라는 단어를 직접 쓰지 않고, 의미는 같지만 표현을 다르게 한 문장
        ground_truth = ["일정 조회 기능은 9221번에서 외부 요청을 받는다."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert result["judgments"][0]["is_matched"] is True