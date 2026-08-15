import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings

FILE_PATH = "./data/project_upgrade_retrospective.md"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SECTION_HEADER_PATTERN = re.compile(r"(?=^### )", re.MULTILINE)


def build_vector_store() -> InMemoryVectorStore:
    # 1. 로딩
    markdown_text = Path(FILE_PATH).read_text(encoding="utf-8")

    # 2. 파싱
    raw_sections = SECTION_HEADER_PATTERN.split(markdown_text)

    # 3. 청킹: 헤더로 시작하지 않는 조각(파일 최상단 제목 등)을 제외하고, "### 카테고리 - 제목" 단위를 하나의 청크로 확정함
    section_texts = [section.strip() for section in raw_sections if section.strip().startswith("### ")]

    documents: list[Document] = []
    for section_text in section_texts:
        title_line = section_text.splitlines()[0]
        title = title_line.removeprefix("### ").strip()
        documents.append(Document(page_content=section_text, metadata={"title": title}))

    # 4. 임베딩
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 5. 벡터DB 저장
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    return vector_store