"""
langchain_pipeline 2단계: 청킹(Chunking)

기존 rag_pipeline/chunker.py의 LangChain 대응 모듈 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 2, 2~3행).

전략 1: split_fixed_size() — RecursiveCharacterTextSplitter.split_documents()
전략 2: split_by_section() — MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
        2단계 구성. 공식 가이드가 제시하는 패턴이다:
        https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
        1단계로 `##` 헤더 기준 섹션을 나누고, 2단계로 split_documents()를 적용해
        chunk_size를 초과하는 섹션만 내부적으로 추가 분할한다. overlap은
        섹션 경계를 넘지 않는다(공식 동작, 직접 확인됨).

기존과의 차이 (직접 확인된 사실, 가정이 아님):
1. 입력/출력이 raw text(str)가 아니라 Document 리스트다 (1단계 loader.py 출력과 연결).
2. MarkdownHeaderTextSplitter.split_text()는 헤더 줄("## ...")을 page_content에서
   제거하고 metadata["Header 2"]로 옮긴다 — 기존처럼 헤더 텍스트가 청크 본문에
   남지 않는다. 또한 split_text()는 원본 Document의 metadata(예: source)를
   전달받지 않으므로, 이 함수가 직접 합쳐서 보존한다.
3. RecursiveCharacterTextSplitter는 chunk_overlap == chunk_size를 막지 않는다
   (내부적으로 무한 루프는 나지 않지만, 거의 중복뿐인 chunk를 과도하게 생성한다 —
   예: 길이 25 텍스트에 chunk_size=10, overlap=10을 주면 3개가 아니라 16개가 나온다).
   기존 모듈의 가드(>=이면 ValueError)를 그대로 유지해 이 함정을 막는다.
"""
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _validate_chunk_args(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap})은 chunk_size({chunk_size})보다 작아야 합니다."
        )


def split_fixed_size(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Document 리스트를 고정 길이(chunk_size) 단위로 분할한다. 인접 chunk는 chunk_overlap만큼 겹친다.

    Args:
        documents: loader.load_document()가 반환한 Document 리스트
        chunk_size: 한 chunk에 포함될 최대 문자 수
        chunk_overlap: 인접한 두 chunk가 공유하는 문자 수

    Returns:
        고정 길이로 분할된 Document(청크) 리스트. 원본 metadata(예: source)는 그대로 보존된다.
        내용이 빈 Document는 결과에서 제외된다 (RecursiveCharacterTextSplitter 기본 동작).

    Raises:
        ValueError: chunk_overlap이 chunk_size보다 크거나 같을 경우
    """
    _validate_chunk_args(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def split_by_section(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Document 리스트를 마크다운 `##` 헤더 기준으로 섹션 단위로 분할한다.
    섹션이 chunk_size보다 길면, 그 섹션 내부에서만 RecursiveCharacterTextSplitter로 추가 분할한다.

    Args:
        documents: loader.load_document()가 반환한 Document 리스트 (마크다운 텍스트)
        chunk_size: 섹션이 이 길이를 초과하면 내부적으로 추가 분할한다
        chunk_overlap: 내부 추가 분할 시 사용할 chunk_overlap

    Returns:
        섹션 경계를 보존하며 분할된 Document(청크) 리스트.
        metadata["Header 2"]에 소속 섹션의 헤더 텍스트가, metadata["source"]에는
        원본 문서의 source가 그대로 담긴다.

    Raises:
        ValueError: chunk_overlap이 chunk_size보다 크거나 같을 경우,
                    또는 documents가 비어 있거나 모든 내용이 비어 있을 경우
    """
    _validate_chunk_args(chunk_size, chunk_overlap)

    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError("documents가 비어 있습니다. 분할할 내용이 필요합니다.")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Header 2")])

    section_documents: list[Document] = []
    for doc in documents:
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            # split_text()는 원본 Document의 metadata(예: source)를 보존하지 않으므로 직접 합친다.
            section.metadata = {**doc.metadata, **section.metadata}
        section_documents.extend(sections)

    char_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return char_splitter.split_documents(section_documents)


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.loader import load_document

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))

    print("=" * 60)
    print("[전략 1] split_fixed_size")
    print("=" * 60)
    fixed_chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)
    print(f"총 chunk 개수: {len(fixed_chunks)}\n")
    for i, chunk in enumerate(fixed_chunks):
        preview = chunk.page_content[:60].replace("\n", " ")
        print(f"--- Chunk {i} (길이: {len(chunk.page_content)}자) ---\n{preview}...\n")

    print("=" * 60)
    print("[전략 2] split_by_section")
    print("=" * 60)
    section_chunks = split_by_section(documents, chunk_size=300, chunk_overlap=50)
    print(f"총 chunk 개수: {len(section_chunks)}\n")
    for i, chunk in enumerate(section_chunks):
        preview = chunk.page_content[:60].replace("\n", " ")
        print(f"--- Chunk {i} ({chunk.metadata.get('Header 2')}, 길이: {len(chunk.page_content)}자) ---\n{preview}...\n")
