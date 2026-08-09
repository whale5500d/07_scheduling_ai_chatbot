"""
Test for langchain_pipeline 2단계: 청킹 (src/langchain_pipeline/splitter.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 2단계):
1. 정상 케이스: 의도한 chunk_size, chunk_overlap에 맞게 분할되는가
2. Overlap 검증: 인접한 두 chunk가 실제로 겹치는 부분을 공유하는가
3. 예외 케이스: chunk_overlap >= chunk_size일 때 ValueError가 발생하는가
4. metadata 보존: source(원본 파일 경로)가 청크에도 그대로 남는가
5. 섹션 헤더는 page_content가 아니라 metadata["Header 2"]에 담긴다 (MarkdownHeaderTextSplitter 기본 동작)
6. 실제 데이터 통합 검증: daysync_manual.md가 정상적으로 분할되는가
"""

from paths import DATA_DIR
import pytest
from langchain_core.documents import Document

from langchain_pipeline.splitter import split_by_section, split_fixed_size  # noqa: E402
from langchain_pipeline.loader import load_document  # noqa: E402


class TestSplitFixedSize:
    """split_fixed_size() 함수에 대한 테스트 그룹"""

    def test_splits_into_expected_count(self):
        """정상 케이스: 텍스트 길이와 step(=chunk_size - chunk_overlap)으로 chunk 개수를 예측할 수 있어야 한다."""
        # 길이 25인 텍스트, chunk_size=10, chunk_overlap=0 -> 3개 chunk 예상 (각 10, 10, 5자)
        documents = [Document(page_content="a" * 25, metadata={"source": "test.md"})]
        chunks = split_fixed_size(documents, chunk_size=10, chunk_overlap=0)

        assert len(chunks) == 3
        assert len(chunks[0].page_content) == 10
        assert len(chunks[1].page_content) == 10
        assert len(chunks[2].page_content) == 5

    def test_chunk_overlap_is_actually_shared_between_adjacent_chunks(self):
        """Overlap 검증: chunk_overlap만큼의 문자가 인접 chunk 사이에 실제로 겹쳐야 한다."""
        text = "0123456789" * 3  # 길이 30
        chunk_size, chunk_overlap = 10, 4
        documents = [Document(page_content=text, metadata={"source": "test.md"})]

        chunks = split_fixed_size(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        first_tail = chunks[0].page_content[-chunk_overlap:]
        second_head = chunks[1].page_content[:chunk_overlap]
        assert first_tail == second_head

    def test_overlap_greater_than_or_equal_to_chunk_size_raises_value_error(self):
        """예외 케이스: chunk_overlap >= chunk_size이면 ValueError가 발생해야 한다.

        RecursiveCharacterTextSplitter 자체는 '==' 케이스를 막지 않고(공식 동작 확인됨),
        대신 거의 중복뿐인 chunk를 과도하게 생성하므로, 기존 모듈과 동일한 가드를 직접 적용한다.
        """
        documents = [Document(page_content="some sample text", metadata={"source": "test.md"})]

        with pytest.raises(ValueError):
            split_fixed_size(documents, chunk_size=10, chunk_overlap=10)  # overlap == size

        with pytest.raises(ValueError):
            split_fixed_size(documents, chunk_size=10, chunk_overlap=15)  # overlap > size

    def test_metadata_is_preserved_across_chunks(self):
        """metadata 보존 검증: 원본 Document의 source가 분할된 모든 chunk에 그대로 남아야 한다."""
        documents = [Document(page_content="a" * 25, metadata={"source": "manual.md"})]
        chunks = split_fixed_size(documents, chunk_size=10, chunk_overlap=0)

        assert all(chunk.metadata.get("source") == "manual.md" for chunk in chunks)

    def test_empty_document_returns_empty_list(self):
        """경계 케이스: 내용이 빈 Document는 결과에서 제외되어야 한다 (에러가 나면 안 됨)."""
        documents = [Document(page_content="", metadata={"source": "test.md"})]
        chunks = split_fixed_size(documents, chunk_size=10, chunk_overlap=2)
        assert chunks == []

    def test_document_shorter_than_chunk_size_returns_single_chunk(self):
        """경계 케이스: 텍스트가 chunk_size보다 짧으면 chunk 1개만 반환해야 한다."""
        text = "short text"  # 10자
        documents = [Document(page_content=text, metadata={"source": "test.md"})]
        chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0].page_content == text

    def test_split_real_daysync_manual(self):
        """실제 데이터 통합 검증: daysync_manual.md가 정상적으로 분할되고, 핵심 정보가 보존되는지 확인."""
        real_data_path = DATA_DIR / "daysync_manual.md"
        documents = load_document(str(real_data_path))

        chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

        assert len(chunks) > 0
        combined_lower = " ".join(c.page_content for c in chunks).lower()
        assert "daysync" in combined_lower
        assert "sc-114" in combined_lower
        assert "project dawnstar" in combined_lower


class TestSplitBySection:
    """split_by_section() 함수에 대한 테스트 그룹"""

    def test_short_sections_become_single_documents_with_header_metadata(self):
        """섹션 미초과 시 단일 chunk: 헤더 텍스트는 본문이 아니라 metadata["Header 2"]에 담긴다."""
        text = "## Section A\nshort content A\n## Section B\nshort content B"
        documents = [Document(page_content=text, metadata={"source": "test.md"})]

        chunks = split_by_section(documents, chunk_size=300, chunk_overlap=50)

        assert len(chunks) == 2
        assert chunks[0].metadata["Header 2"] == "Section A"
        assert chunks[1].metadata["Header 2"] == "Section B"
        # 헤더 줄 자체는 page_content에서 제거된다 (MarkdownHeaderTextSplitter 기본 동작)
        assert "Section A" not in chunks[0].page_content
        assert "short content A" in chunks[0].page_content

    def test_section_boundaries_are_not_mixed(self):
        """섹션 경계 보존 검증: 서로 다른 섹션의 내용이 같은 chunk에 섞이지 않아야 한다."""
        text = "## Topic X\ncontent about X\n## Topic Y\ncontent about Y"
        documents = [Document(page_content=text, metadata={"source": "test.md"})]

        chunks = split_by_section(documents, chunk_size=300, chunk_overlap=50)

        for chunk in chunks:
            if chunk.metadata["Header 2"] == "Topic X":
                assert "Topic Y" not in chunk.page_content and "content about Y" not in chunk.page_content
            if chunk.metadata["Header 2"] == "Topic Y":
                assert "Topic X" not in chunk.page_content and "content about X" not in chunk.page_content

    def test_long_section_is_split_internally_without_crossing_boundary(self):
        """
        섹션 초과 시 내부 재분할 검증: chunk_size를 초과하는 섹션은 여러 chunk로 나뉘지만,
        그 어떤 chunk도 다른 섹션의 metadata나 내용을 포함하지 않아야 한다.
        """
        long_content = "word " * 100  # 500자 — chunk_size(50)를 초과하도록 구성
        text = f"## Long Section\n{long_content}\n## Short Section\nshort content"
        documents = [Document(page_content=text, metadata={"source": "test.md"})]

        chunks = split_by_section(documents, chunk_size=50, chunk_overlap=10)

        long_chunks = [c for c in chunks if c.metadata["Header 2"] == "Long Section"]
        short_chunks = [c for c in chunks if c.metadata["Header 2"] == "Short Section"]

        # Long Section은 여러 chunk로 나뉘어야 한다 (1개로는 부족함)
        assert len(long_chunks) > 1
        # Short Section 내용을 포함한 chunk에는 "word"(Long Section의 내용)가 섞이지 않아야 한다
        for chunk in short_chunks:
            assert "word" not in chunk.page_content

    def test_overlap_greater_than_or_equal_to_chunk_size_raises_value_error(self):
        """예외 케이스: chunk_overlap >= chunk_size이면 ValueError가 발생해야 한다."""
        documents = [Document(page_content="## A\ncontent", metadata={"source": "test.md"})]

        with pytest.raises(ValueError):
            split_by_section(documents, chunk_size=10, chunk_overlap=10)

    def test_empty_documents_raises_value_error(self):
        """예외 케이스: documents가 비어 있거나 내용이 모두 빈 경우 ValueError가 발생해야 한다."""
        with pytest.raises(ValueError):
            split_by_section([], chunk_size=300, chunk_overlap=50)

        with pytest.raises(ValueError):
            split_by_section([Document(page_content="   ", metadata={})], chunk_size=300, chunk_overlap=50)

    def test_source_metadata_is_preserved_alongside_header(self):
        """metadata 보존 검증: MarkdownHeaderTextSplitter.split_text()는 source를 보존하지 않으므로
        split_by_section()이 직접 합쳐서 보존해야 한다."""
        text = "## Topic X\ncontent about X"
        documents = [Document(page_content=text, metadata={"source": "manual.md"})]

        chunks = split_by_section(documents, chunk_size=300, chunk_overlap=50)

        assert chunks[0].metadata["source"] == "manual.md"
        assert chunks[0].metadata["Header 2"] == "Topic X"

    def test_section_chunking_separates_unrelated_topics_in_real_manual(self):
        """
        실제 데이터 통합 검증 (트러블슈팅 #8 회귀 테스트, DaySync 도메인으로 재현):
        "9221"(API 사용법 섹션)이 "SC-114"(일정 충돌 처리 섹션)와
        서로 다른 chunk에 분리되어 있어야 한다 — fixed-size chunking에서 섞였던 문제가
        section-based chunking에서는 해결되었는지 확인한다.
        """
        real_data_path = DATA_DIR / "daysync_manual.md"
        documents = load_document(str(real_data_path))

        chunks = split_by_section(documents, chunk_size=300, chunk_overlap=50)

        chunks_with_port = [c for c in chunks if "9221" in c.page_content]
        assert len(chunks_with_port) > 0

        for chunk in chunks_with_port:
            assert "SC-114" not in chunk.page_content
