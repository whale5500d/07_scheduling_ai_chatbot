"""
Test for langchain_pipeline 3단계: 임베딩 모델 준비 (src/langchain_pipeline/embedding.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 3단계):
1. Shape 검증: embed_documents() 출력이 (텍스트 개수, 384) 형태인가
2. 결정론적 동작 검증: 같은 텍스트를 두 번 인코딩하면 같은 벡터가 나오는가
3. 의미적 유사도 검증: 의미가 비슷한 문장끼리 cosine similarity가 더 높은가
4. embed_query()가 embed_documents([text])[0]과 동일한 결과를 내는가 (공식 구현 확인됨)

주의:
- 이 테스트들은 실제 sentence-transformers 모델(all-MiniLM-L6-v2)을 로딩하므로
  torch/transformers/sentence-transformers가 필요하고, 최초 실행 시 모델 다운로드
  시간이 걸린다 (기존 rag_pipeline 테스트와 동일한 무거운 의존성).
- 빈 리스트 입력에 대한 ValueError 검증은 기존 TextEmbedder.encode()에는 있었지만,
  이 모듈에는 없다 — §4.2(표 1 #6) 결정에 따라 인코딩 호출 책임 자체가 4단계
  vector_store.py로 이동했기 때문이다. 해당 검증은 4단계 테스트에서 다룬다.
"""

import numpy as np
import pytest

from langchain_pipeline.embedding import get_embeddings_model  # noqa: E402


def cosine_similarity(vec_a, vec_b) -> float:
    """두 벡터 간 cosine similarity를 계산한다. (검증 전용 헬퍼 함수)"""
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def embeddings_model():
    """
    모듈 전체에서 임베딩 모델을 한 번만 로딩하여 재사용한다.
    (scope="module": 매 테스트마다 모델을 다시 로딩하지 않아 테스트 속도가 빨라진다.)
    """
    return get_embeddings_model()


class TestGetEmbeddingsModel:
    """get_embeddings_model()이 반환하는 HuggingFaceEmbeddings 인스턴스에 대한 테스트 그룹"""

    def test_embed_documents_returns_correct_shape(self, embeddings_model):
        """Shape 검증: 텍스트 3개를 넣으면 384차원 벡터 3개가 반환되어야 한다."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vectors = embeddings_model.embed_documents(texts)

        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_embed_documents_is_deterministic(self, embeddings_model):
        """결정론적 동작 검증: 같은 텍스트를 두 번 인코딩하면 완전히 같은 벡터가 나와야 한다."""
        text = ["The default port is 8842."]

        first_result = embeddings_model.embed_documents(text)
        second_result = embeddings_model.embed_documents(text)

        np.testing.assert_array_equal(np.array(first_result), np.array(second_result))

    def test_similar_sentences_have_higher_similarity_than_unrelated_ones(self, embeddings_model):
        """
        의미적 유사도 검증: 의미가 비슷한 문장끼리의 유사도가,
        의미가 무관한 문장과의 유사도보다 높아야 한다.
        """
        anchor = "What is the default API port for NimbusFlow?"
        similar = "NimbusFlow exposes a REST API on port 8842 by default."
        unrelated = "The cat is sleeping on a sunny windowsill."

        vectors = embeddings_model.embed_documents([anchor, similar, unrelated])
        anchor_vec, similar_vec, unrelated_vec = vectors[0], vectors[1], vectors[2]

        sim_to_similar = cosine_similarity(anchor_vec, similar_vec)
        sim_to_unrelated = cosine_similarity(anchor_vec, unrelated_vec)

        assert sim_to_similar > sim_to_unrelated

    def test_embed_query_matches_embed_documents_single_result(self, embeddings_model):
        """embed_query()는 embed_documents([text])[0]과 동일한 결과를 내야 한다 (공식 소스코드로 확인된 구현)."""
        text = "What is the default API port for NimbusFlow?"

        query_vector = embeddings_model.embed_query(text)
        document_vector = embeddings_model.embed_documents([text])[0]

        np.testing.assert_array_equal(np.array(query_vector), np.array(document_vector))
