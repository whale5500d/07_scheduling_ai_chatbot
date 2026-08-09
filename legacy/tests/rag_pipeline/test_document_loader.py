"""
Test for Step A-1: Document Loading (src/rag_pipeline/document_loader.py)

검증 항목 (표 24):
1. 정상 케이스: 존재하는 파일을 정상적으로 로드하는가
2. 예외 케이스 1: 존재하지 않는 파일 -> FileNotFoundError
3. 예외 케이스 2: 빈 파일 -> ValueError
4. 내용 검증: 로드된 텍스트가 실제 파일 내용과 정확히 일치하는가
"""

from paths import DATA_DIR
import pytest

from rag_pipeline.document_loader import load_document  # noqa: E402


class TestLoadDocument:
    """load_document() 함수에 대한 테스트 그룹"""

    def test_load_existing_file_returns_correct_content(self, tmp_path):
        """정상 케이스 + 내용 검증: 실제 작성한 내용과 로드된 내용이 정확히 일치해야 한다."""
        # Arrange: 임시 파일 생성
        test_content = "# 샘플 제목\n\n이것은 DaySync를 위한 테스트 문서입니다."
        test_file = tmp_path / "sample.md"
        test_file.write_text(test_content, encoding="utf-8")

        # Act: 함수 실행
        result = load_document(str(test_file))

        # Assert: 반환값이 작성한 내용과 정확히 같아야 한다
        assert result == test_content

    def test_load_nonexistent_file_raises_file_not_found_error(self, tmp_path):
        """예외 케이스 1: 존재하지 않는 경로를 넘기면 FileNotFoundError가 발생해야 한다."""
        nonexistent_path = tmp_path / "does_not_exist.md"

        with pytest.raises(FileNotFoundError):
            load_document(str(nonexistent_path))

    def test_load_empty_file_raises_value_error(self, tmp_path):
        """예외 케이스 2: 파일은 존재하지만 내용이 비어 있으면 ValueError가 발생해야 한다."""
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

        # 비어 있지 않아야 하고, 문서의 알려진 키워드를 포함해야 한다
        assert len(result) > 0
        assert "DaySync" in result
        assert "Project Dawnstar" in result