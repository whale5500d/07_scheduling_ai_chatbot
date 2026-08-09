"""
langchain_pipeline 4단계: 저장(Storage) + 검색(Retrieval)

기존 rag_pipeline/vector_store.py(저장) + retriever.py(검색)의 LangChain 대응 모듈
(docs/LANGCHAIN_MIGRATION_PLAN.md 표 2, 5~6행 / 표 1 #5, #6 결정).

기존 구조는 "저장"(InMemoryVectorStore.add)과 "검색"(retrieve_top_k)을
서로 다른 두 모듈로 분리했다. LangChain의 VectorStore는 이 둘을 하나의
인터페이스로 결합한다 — §4.1/§4.2에서 "절충안 없이 그대로 채택"하기로
결정했으므로, 이 모듈은 InMemoryVectorStore/as_retriever의 동작을
오버라이드하거나 재구현하지 않는다 (예: 저장소가 비어 있을 때
similarity_search/as_retriever는 예외 없이 빈 리스트를 반환한다 — 기존
retrieve_top_k()의 "저장소가 비어 있으면 ValueError" 동작과 다르지만,
이것이 LangChain의 의도된 동작이므로 그대로 둔다).

이 모듈이 추가하는 것은 함수 경계에서의 입력 검증뿐이다(1, 2단계와 동일한
패턴) — "빈 문서 리스트로 저장소를 만들려는 호출 자체가 잘못된 호출"이라는
점만 막아준다. InMemoryVectorStore/VectorStoreRetriever 내부 동작에는
관여하지 않는다.

임베딩 소유권(표 1 #6): InMemoryVectorStore(embedding=...) 생성자에 Embeddings
인스턴스를 주입하는 방식 — 3단계 embedding.py가 만든 인스턴스를 그대로 전달받는다.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever


def build_vector_store(
    documents: list[Document],
    embedding: Embeddings,
) -> InMemoryVectorStore:
    """
    문서 리스트를 임베딩하여 InMemoryVectorStore에 저장한다.

    Args:
        documents: 저장할 문서 리스트 (2단계 splitter.py의 출력)
        embedding: 임베딩에 사용할 Embeddings 인스턴스 (3단계 embedding.py의 출력)

    Returns:
        문서가 모두 추가된 InMemoryVectorStore 인스턴스.

    Raises:
        ValueError: documents가 빈 리스트일 경우 (저장할 내용이 없는 호출을 막는다)
    """
    if not documents:
        raise ValueError("documents가 비어 있습니다. 저장할 문서가 필요합니다.")

    store = InMemoryVectorStore(embedding=embedding)
    store.add_documents(documents)
    return store


def get_retriever(store: InMemoryVectorStore, k: int = 3) -> VectorStoreRetriever:
    """
    VectorStore로부터 상위 k개를 검색하는 Retriever(Runnable)를 만든다.

    Args:
        store: build_vector_store()로 만든 InMemoryVectorStore 인스턴스
        k: 검색 시 반환할 최대 결과 개수

    Returns:
        .invoke(query) 호출로 유사도 검색을 수행하는 VectorStoreRetriever.
    """
    return store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.embedding import get_embeddings_model

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    store = build_vector_store(chunks, embeddings_model)
    retriever = get_retriever(store, k=3)

    question = "What is the default API port for NimbusFlow?"
    results = retriever.invoke(question)

    print(f"[질문] {question}\n")
    for rank, doc in enumerate(results, start=1):
        print(f"--- Rank {rank} ---")
        print(doc.page_content[:150].replace("\n", " "))
        print()
