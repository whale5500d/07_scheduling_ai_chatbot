from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 초기화
# (현재) txt/md로 시작 (파싱 난이도가 가장 낮음).
# CSV → JSON → HTML → DOCX → PDF 순(난이도 순)으로 형식을 경험하며 확장 예정
# 이 과정에서 경험한 노하우는 문서로 기록 필요
_RESPONSE_CASE_EXAMPLES_PATH = Path("data/response_case_examples.md")
_EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def build_vector_store():
    # 1. 문서 로딩
    text = _RESPONSE_CASE_EXAMPLES_PATH.read_text(encoding="utf-8")

    # 2. 파싱
    document = Document(page_content=text, metadata={"source": str(_RESPONSE_CASE_EXAMPLES_PATH)})

    # 3. 청킹
    # 현재는 청크를 분리없이 파일 전체를 하나의 청크로 사용.
    # 추후 "카테고리 단위" → "행(사례) 단위" 순으로 청크를 점진 적용 예정.
    # 이 과정으로 검색 정확도 변화를 비교 분석해보기
    chunks = [document]

    # 4. 임베딩
    embeddings_model = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)

    # 5. 벡터 DB
    # 현재는 InMemoryVectorStore 사용 (프로젝트 규모, 설정 난이도 고려).
    # 추후 ChromaDB → PostgreSQL + pgvector → Pinecone, Weaviate, Qdrant로 확장 예정.
    # 이 과정으로 메모리, DB 동작 원리와 메모리 성능이 어느 정도 개선되는지 분석해보기
    store = InMemoryVectorStore(embeddings_model)
    store.add_documents(chunks)

    return store