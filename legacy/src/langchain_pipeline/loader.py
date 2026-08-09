"""
langchain_pipeline 1단계: 문서 로딩

기존 rag_pipeline/document_loader.py의 LangChain 대응 모듈 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 2, 1행).

기존 load_document()는 raw text(str)를 반환했지만, LangChain의 TextLoader는
Document(문서) 객체 리스트(list[Document])를 반환한다. 이후 단계(splitter 등)가
공통으로 Document를 입력으로 받으므로, 이 시점부터 텍스트가 아닌 Document로 다룬다.

TextLoader.load()의 실제 동작(공식 소스코드로 확인):
- 파일이 존재하지 않으면 FileNotFoundError가 아니라 RuntimeError로 감싸서 던진다.
- 빈 파일은 예외를 던지지 않고 page_content=""인 Document를 그대로 반환한다.

기존 모듈과 동일한 예외 계약(FileNotFoundError/ValueError)을 유지하기 위해,
파일 존재 여부와 내용이 비어있는지는 이 함수가 직접 검사한다.
"""
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_document(file_path: str) -> list[Document]:
    """
    단일 텍스트 파일(.txt, .md)을 LangChain의 TextLoader로 읽어
    Document(문서) 객체 리스트로 반환한다.

    Args:
        file_path: 읽을 파일의 경로

    Returns:
        길이 1인 Document 리스트. documents[0].page_content에 파일 전체 내용이 담긴다.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 파일 내용이 비어 있을 경우
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()

    if not documents or not documents[0].page_content.strip():
        raise ValueError(f"파일 내용이 비어 있습니다: {file_path}")

    return documents


if __name__ == "__main__":
    # 동작 확인용 테스트
    from paths import DATA_DIR

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))

    print(f"[로딩 성공] 문서 수: {len(documents)}, 총 문자 수: {len(documents[0].page_content)}")
    print(f"[미리보기]\n{documents[0].page_content[:300]}")
    print(f"[메타데이터] {documents[0].metadata}")
