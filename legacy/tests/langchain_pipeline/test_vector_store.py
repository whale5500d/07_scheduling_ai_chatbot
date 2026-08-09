"""
Test for langchain_pipeline 4단계: 저장(Storage) + 검색(Retrieval) (src/langchain_pipeline/vector_store.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 4단계):
1. build_vector_store(): 문서가 모두 저장되는가, 빈 리스트면 ValueError인가
2. get_retriever(): k개를 반환하는가, k가 전체 문서 수보다 크면 전체만 반환하는가
3. 인덱스 정합성: 저장한 문서의 내용과 메타데이터가 검색 결과에서 그대로 보존되는가
4. 빈 저장소 검색은 예외 없이 빈 리스트를 반환하는가 (LangChain 고유 동작, §4.1 "그대로 채택")
5. 통합 테스트: 1단계(loader) + 2단계(splitter) + 4단계(vector_store)를 실제
   daysync_manual.md로 엮어서 검증

주의:
- DeterministicFakeEmbedding(langchain_core.embeddings)을 사용한다 — 해시 기반의
  가벼운 가짜 임베딩으로, torch/sentence-transformers 없이도 결정론적인 벡터를
  만들어준다 (표 5, 4단계의 "식별 가능한 mock 벡터 대신 DeterministicFakeEmbedding
  사용 검토" 항목을 반영). 다만 의미 있는 유사도를 보장하지 않으므로,
  "의미적으로 가장 가까운 문서를 찾는가"가 아니라 "저장/검색 메커니즘 자체가
  올바르게 동작하는가"만 검증한다. 실제 임베딩(all-MiniLM-L6-v2)을 사용한
  의미적 검색 검증은 3단계 test_embedding.py와 사용자 로컬 환경에서의 수동
  통합 테스트로 별도 확인한다.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_pipeline.vector_store import build_vector_store, get_retriever  # noqa: E402


@pytest.fixture
def fake_embedding():
    """가벼운 결정론적 가짜 임베딩 (차원=8). 실제 모델 없이 저장/검색 로직만 검증한다."""
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The default API port is 8842.", metadata={"idx": 0}),
        Document(page_content="Conflicts are resolved by priority.", metadata={"idx": 1}),
        Document(page_content="NimbusFlow supports webhook retries.", metadata={"idx": 2}),
    ]


class TestBuildVectorStore:
    """build_vector_store()에 대한 테스트 그룹"""

    def test_stores_all_documents(self, sample_documents, fake_embedding):
        """문서 3개를 저장하면 내부 저장소에 정확히 3개가 들어 있어야 한다."""
        store = build_vector_store(sample_documents, fake_embedding)

        assert len(store.store) == 3

    def test_empty_documents_raises_value_error(self, fake_embedding):
        """빈 문서 리스트로 저장소를 만들면 ValueError가 발생해야 한다."""
        with pytest.raises(ValueError):
            build_vector_store([], fake_embedding)

    def test_preserves_content_and_metadata(self, sample_documents, fake_embedding):
        """저장된 문서의 page_content와 metadata가 원본과 동일하게 보존되어야 한다 (인덱스 정합성).

        InMemoryVectorStore.store는 {id: {"id", "vector", "text", "metadata"}} 형태의
        내부 딕셔너리다(공식 소스코드로 직접 확인됨) — Document 객체가 아니라 "text"/"metadata"
        키로 풀어서 저장한다.
        """
        store = build_vector_store(sample_documents, fake_embedding)

        stored_records = list(store.store.values())
        stored_contents = {record["text"] for record in stored_records}
        stored_metadatas = [record["metadata"] for record in stored_records]

        assert stored_contents == {doc.page_content for doc in sample_documents}
        assert {m["idx"] for m in stored_metadatas} == {0, 1, 2}


class TestGetRetriever:
    """get_retriever()에 대한 테스트 그룹"""

    def test_returns_runnable_retriever(self, sample_documents, fake_embedding):
        """get_retriever()는 .invoke()를 가진 Runnable(VectorStoreRetriever)을 반환해야 한다."""
        store = build_vector_store(sample_documents, fake_embedding)
        retriever = get_retriever(store, k=2)

        assert hasattr(retriever, "invoke")

    def test_returns_exactly_k_results_when_k_smaller_than_total(self, sample_documents, fake_embedding):
        """k가 전체 문서 수보다 작으면, 정확히 k개의 결과가 반환되어야 한다."""
        store = build_vector_store(sample_documents, fake_embedding)
        retriever = get_retriever(store, k=2)

        results = retriever.invoke("API port")

        assert len(results) == 2

    def test_returns_all_when_k_exceeds_total(self, sample_documents, fake_embedding):
        """k가 전체 문서 수보다 크면, 존재하는 문서 수만큼만 반환되어야 한다 (LangChain 기본 동작)."""
        store = build_vector_store(sample_documents, fake_embedding)
        retriever = get_retriever(store, k=10)

        results = retriever.invoke("API port")

        assert len(results) == 3

    def test_retrieved_documents_are_from_original_set(self, sample_documents, fake_embedding):
        """검색 결과의 page_content는 반드시 원본 문서 집합에 속해야 한다 (엉뚱한 텍스트가 나오면 안 된다)."""
        store = build_vector_store(sample_documents, fake_embedding)
        retriever = get_retriever(store, k=3)

        results = retriever.invoke("API port")
        original_contents = {doc.page_content for doc in sample_documents}

        assert all(doc.page_content in original_contents for doc in results)

    def test_empty_store_returns_empty_list_without_raising(self, fake_embedding):
        """
        빈 저장소에 대한 검색은 예외 없이 빈 리스트를 반환해야 한다.
        (기존 retrieve_top_k()는 빈 저장소에 ValueError를 던졌지만, LangChain의
         VectorStoreRetriever는 그렇지 않다 — §4.1 "절충안 없이 그대로 채택"에 따라
         이 차이를 그대로 받아들인다.)
        """
        from langchain_core.vectorstores import InMemoryVectorStore

        empty_store = InMemoryVectorStore(embedding=fake_embedding)
        retriever = get_retriever(empty_store, k=3)

        results = retriever.invoke("아무 질문")

        assert results == []


class TestIntegrationWithLoaderAndSplitter:
    """1단계(loader) + 2단계(splitter) + 4단계(vector_store)를 실제 데이터로 엮은 통합 테스트"""

    def test_real_document_chunks_are_retrievable(self, fake_embedding):
        """
        daysync_manual.md를 로딩 -> 청킹 -> 저장 후, 저장된 chunk 개수와
        검색 결과 개수가 청킹 단계의 출력과 일치하는지 확인한다.
        """
        from paths import DATA_DIR
        from langchain_pipeline.loader import load_document
        from langchain_pipeline.splitter import split_fixed_size

        sample_path = DATA_DIR / "daysync_manual.md"
        documents = load_document(str(sample_path))
        chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

        store = build_vector_store(chunks, fake_embedding)
        retriever = get_retriever(store, k=3)

        assert len(store.store) == len(chunks)

        results = retriever.invoke("What is the default API port for NimbusFlow?")
        assert len(results) == 3
        chunk_contents = {c.page_content for c in chunks}
        assert all(doc.page_content in chunk_contents for doc in results)
