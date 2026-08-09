"""
Test for langchain_pipeline 5단계: Prompt Augmentation (src/langchain_pipeline/prompt.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 5단계):
1. format_docs(): Document 리스트가 [문서 N] 라벨이 붙은 context 문자열로 변환되는가,
   빈 리스트면 ValueError인가
2. get_prompt_template(): instruction/context/question이 모두 최종 결과에 포함되는가
   (기존 test_prompt_builder.py의 검증 항목을 ChatPromptTemplate 기준으로 재작성)
3. [회귀 테스트] ChatPromptValue.to_string()이 "Human: " 접두사를 붙이는 공식 동작을
   고정해 둔다 — 이 동작 때문에 8단계(chain.py)에서는 to_string()이 아니라
   to_messages()[0].content를 사용해야 한다는 것을 코드로 증명한다.
"""

import pytest
from langchain_core.documents import Document

from langchain_pipeline.prompt import format_docs, get_prompt_template  # noqa: E402


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="첫 번째 chunk 텍스트"),
        Document(page_content="두 번째 chunk 텍스트"),
        Document(page_content="세 번째 chunk 텍스트"),
    ]


class TestFormatDocs:
    """format_docs() 함수에 대한 테스트 그룹"""

    def test_each_document_is_labeled_with_document_number(self, sample_documents):
        """Document 3개를 넣으면 [문서 1]~[문서 3]이 모두 생성되어야 한다."""
        context = format_docs(sample_documents)

        assert "[문서 1]" in context
        assert "[문서 2]" in context
        assert "[문서 3]" in context

    def test_document_content_is_preserved(self, sample_documents):
        """각 Document의 page_content가 context 문자열 안에 그대로 포함되어야 한다."""
        context = format_docs(sample_documents)

        assert "첫 번째 chunk 텍스트" in context
        assert "두 번째 chunk 텍스트" in context
        assert "세 번째 chunk 텍스트" in context

    def test_empty_documents_raises_value_error(self):
        """빈 리스트를 넣으면 ValueError가 발생해야 한다 (기존 build_prompt()와 동일한 제약)."""
        with pytest.raises(ValueError):
            format_docs([])


class TestGetPromptTemplate:
    """get_prompt_template()이 반환하는 ChatPromptTemplate에 대한 테스트 그룹"""

    def test_formatted_result_contains_instruction(self, sample_documents):
        """구조 검증: 결과에 '문서에 근거해 답하라'는 지시문이 포함되어야 한다."""
        question = "기본 포트는 무엇인가?"
        prompt_value = get_prompt_template().invoke(
            {"context": format_docs(sample_documents), "question": question}
        )
        content = prompt_value.to_messages()[0].content

        assert "문서에만 근거하여" in content
        assert "답을 찾을 수 없습니다" in content

    def test_formatted_result_contains_question(self, sample_documents):
        """질문 포함 검증: 사용자 질문이 결과에 정확히 포함되어야 한다."""
        question = "DaySync의 내부 코드네임은 무엇인가?"
        prompt_value = get_prompt_template().invoke(
            {"context": format_docs(sample_documents), "question": question}
        )
        content = prompt_value.to_messages()[0].content

        assert question in content

    def test_formatted_result_contains_context_with_document_labels(self):
        """Context 번호 매김 검증: format_docs()로 만든 [문서 N] 라벨이 최종 결과에도 남아 있어야 한다."""
        documents = [
            Document(page_content="내부 코드네임은 Project Dawnstar이다."),
            Document(page_content="기본 API 포트는 9221이다."),
        ]
        question = "dummy question"

        prompt_value = get_prompt_template().invoke(
            {"context": format_docs(documents), "question": question}
        )
        content = prompt_value.to_messages()[0].content

        assert "[문서 1]" in content
        assert "[문서 2]" in content
        assert "내부 코드네임은 Project Dawnstar이다." in content
        assert "기본 API 포트는 9221이다." in content

    def test_prompt_with_real_daysync_style_chunk(self):
        """실제 데이터 형태와 유사한 통합 검증: DaySync 매뉴얼 스타일의 chunk로 prompt가 정상 조립되는지 확인."""
        documents = [
            Document(
                page_content='개발 초기 단계에서는 내부적으로 "프로젝트 새벽별(Project Dawnstar)"이라는 코드네임으로 불렸다.'
            )
        ]
        question = "DaySync의 내부 코드네임은 무엇인가?"

        prompt_value = get_prompt_template().invoke(
            {"context": format_docs(documents), "question": question}
        )
        content = prompt_value.to_messages()[0].content

        assert "Project Dawnstar" in content
        assert question in content
        assert "[문서 1]" in content


class TestToStringHumanPrefixRegression:
    """
    [회귀 테스트] ChatPromptValue.to_string()의 "Human: " 접두사 동작을 고정해 둔다.

    이 테스트가 실패한다면(즉, to_string()이 더 이상 "Human: "을 붙이지 않는다면),
    8단계(chain.py)에서 to_messages()[0].content 대신 to_string()을 사용해도 되는지
    재검토해야 한다는 신호다.
    """

    def test_to_string_adds_human_prefix(self, sample_documents):
        question = "dummy question"
        prompt_value = get_prompt_template().invoke(
            {"context": format_docs(sample_documents), "question": question}
        )

        assert prompt_value.to_string().startswith("Human: ")

    def test_to_messages_content_has_no_human_prefix(self, sample_documents):
        """반면 to_messages()[0].content에는 'Human: ' 접두사가 없다 — chain.py가 이걸 써야 하는 이유."""
        question = "dummy question"
        prompt_value = get_prompt_template().invoke(
            {"context": format_docs(sample_documents), "question": question}
        )

        content = prompt_value.to_messages()[0].content
        assert not content.startswith("Human: ")
        assert "문서에만 근거하여" in content
