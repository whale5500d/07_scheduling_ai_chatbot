"""
Test for Step B: Prompt Augmentation (src/rag_pipeline/prompt_builder.py)

검증 항목:
1. 구조 검증: Instruction, Context, Question이 모두 prompt에 포함되는가
2. Context 번호 매김 검증: chunk 개수만큼 [문서 N] 형태가 생성되는가
3. 내용 포함 검증: 각 chunk의 텍스트가 prompt 안에 그대로 들어가는가
4. 질문 포함 검증: 사용자 질문이 정확히 포함되는가
5. 예외 케이스: retrieved_chunks가 빈 리스트일 때 ValueError 발생 여부
"""

from pathlib import Path

import pytest

from rag_pipeline.prompt_builder import build_prompt  # noqa: E402


class TestBuildPrompt:
    """build_prompt() 함수에 대한 테스트 그룹"""

    def test_prompt_contains_instruction(self):
        """구조 검증: prompt에 '문서에 근거해 답하라'는 지시문이 포함되어야 한다."""
        question = "기본 포트는 무엇인가?"
        retrieved_chunks = [("기본 API 포트는 9221이다.", 0.9)]

        prompt = build_prompt(question, retrieved_chunks)

        assert "문서에만 근거하여" in prompt
        assert "답을 찾을 수 없습니다" in prompt

    def test_prompt_contains_question(self):
        """질문 포함 검증: 사용자 질문이 prompt에 정확히 포함되어야 한다."""
        question = "DaySync의 내부 코드네임은 무엇인가?"
        retrieved_chunks = [("어떤 chunk 텍스트.", 0.8)]

        prompt = build_prompt(question, retrieved_chunks)

        assert question in prompt

    def test_each_chunk_is_labeled_with_document_number(self):
        """Context 번호 매김 검증: chunk 3개를 넣으면 [문서 1]~[문서 3]이 모두 생성되어야 한다."""
        question = "dummy question"
        retrieved_chunks = [
            ("첫 번째 chunk 텍스트", 0.9),
            ("두 번째 chunk 텍스트", 0.8),
            ("세 번째 chunk 텍스트", 0.7),
        ]

        prompt = build_prompt(question, retrieved_chunks)

        assert "[문서 1]" in prompt
        assert "[문서 2]" in prompt
        assert "[문서 3]" in prompt

    def test_chunk_text_content_is_preserved(self):
        """내용 포함 검증: 각 chunk의 실제 텍스트가 prompt 안에 그대로 포함되어야 한다."""
        question = "dummy question"
        retrieved_chunks = [
            ("내부 코드네임은 Project Dawnstar이다.", 0.95),
            ("기본 API 포트는 9221이다.", 0.85),
        ]

        prompt = build_prompt(question, retrieved_chunks)

        assert "내부 코드네임은 Project Dawnstar이다." in prompt
        assert "기본 API 포트는 9221이다." in prompt

    def test_empty_retrieved_chunks_raises_value_error(self):
        """예외 케이스: retrieved_chunks가 빈 리스트이면 ValueError가 발생해야 한다."""
        question = "dummy question"

        with pytest.raises(ValueError):
            build_prompt(question, [])

    def test_prompt_with_real_daysync_chunk(self):
        """실제 데이터 형태와 유사한 통합 검증: DaySync 매뉴얼 스타일의 chunk로 prompt가 정상 조립되는지 확인."""
        question = "DaySync의 내부 코드네임은 무엇인가?"
        retrieved_chunks = [
            (
                '개발 초기 단계에서는 내부적으로 "프로젝트 새벽별(Project Dawnstar)"이라는 코드네임으로 불렸다.',
                0.81,
            ),
        ]

        prompt = build_prompt(question, retrieved_chunks)

        assert "Project Dawnstar" in prompt
        assert question in prompt
        assert "[문서 1]" in prompt