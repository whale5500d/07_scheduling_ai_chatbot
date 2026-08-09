"""
Test for langchain_pipeline 1단계: 문서 로딩 (src/langchain_pipeline/loader.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 1단계):
1. 정상 케이스: 존재하는 파일을 정상적으로 로드하여 Document 리스트로 반환하는가
2. 메타데이터: Document.metadata["source"]에 파일 경로가 담기는가 (TextLoader 기본 동작)
3. 예외 케이스 1: 존재하지 않는 파일 -> FileNotFoundError
4. 예외 케이스 2: 빈 파일 -> ValueError (TextLoader 자체는 예외를 던지지 않으므로 직접 검증)
5. 내용 검증: 로드된 Document.page_content가 실제 파일 내용과 정확히 일치하는가
"""

from paths import DATA_DIR
import pytest
from langchain_core.documents import Document

from langchain_pipeline.loader import load_document  # noqa: E402


class TestLoadDocument:
    """load_document() 함수에 대한 테스트 그룹"""

    def test_load_existing_file_returns_document_list(self, tmp_path):
        """정상 케이스 + 내용 검증: Document 리스트를 반환하고 page_content가 정확히 일치해야 한다."""
        # Arrange
        test_content = "# 샘플 제목\n\n이것은 DaySync를 위한 테스트 문서입니다."
        test_file = tmp_path / "sample.md"
        test_file.write_text(test_content, encoding="utf-8")

        # Act
        result = load_document(str(test_file))

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].page_content == test_content

    def test_load_existing_file_sets_source_metadata(self, tmp_path):
        """TextLoader 기본 동작 검증: metadata["source"]에 파일 경로가 담겨야 한다."""
        test_file = tmp_path / "sample.md"
        test_file.write_text("내용", encoding="utf-8")

        result = load_document(str(test_file))

        assert result[0].metadata["source"] == str(test_file)

    def test_load_nonexistent_file_raises_file_not_found_error(self, tmp_path):
        """예외 케이스 1: 존재하지 않는 경로를 넘기면 FileNotFoundError가 발생해야 한다."""
        nonexistent_path = tmp_path / "does_not_exist.md"

        with pytest.raises(FileNotFoundError):
            load_document(str(nonexistent_path))

    def test_load_empty_file_raises_value_error(self, tmp_path):
        """예외 케이스 2: 파일은 존재하지만 내용이 비어 있으면 ValueError가 발생해야 한다.

        TextLoader.load() 자체는 빈 파일에 예외를 던지지 않고
        page_content=""인 Document를 그대로 반환하므로 (공식 동작 확인됨),
        load_document()가 직접 검사해 ValueError로 변환한다.
        """
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError):
            load_document(str(empty_file))

    def test_load_whitespace_only_file_raises_value_error(self, tmp_path):
        """추가 예외 케이스: 공백/줄바꿈만 있는 파일도 '비어 있음'으로 간주해야 한다."""
        whitespace_file = tmp_path / "whitespace.md"
        whitespace_file.write_text("   \n\n   ", encoding="utf-8")

        with pytest.raises(ValueError):
            load_document(str(whitespace_file))

    def test_load_real_daysync_manual(self):
        """실제 프로젝트 데이터 파일(daysync_manual.md)이 정상적으로 로드되는지 확인."""
        real_data_path = DATA_DIR / "daysync_manual.md"

        result = load_document(str(real_data_path))

        assert len(result) == 1
        assert len(result[0].page_content) > 0
        assert "DaySync" in result[0].page_content
        assert "Project Dawnstar" in result[0].page_content
