"""
Test for Step A-3: Embedding (src/model/embedder.py)

검증 항목 (표 38):
1. Shape 검증: 출력이 (텍스트 개수, 384) 형태인가
2. 결정론적 동작 검증: 같은 텍스트를 두 번 인코딩하면 같은 벡터가 나오는가
3. 의미적 유사도 검증: 의미가 비슷한 문장끼리 cosine similarity가 더 높은가
4. 예외 케이스: 빈 리스트 입력 시 ValueError 발생 여부

주의: 이 테스트들은 실제 sentence-transformers 모델을 로딩하므로
      최초 실행 시 모델 다운로드 시간이 걸리고, 이후에는 캐시를 사용한다.
"""

import numpy as np
import pytest

from rag_pipeline.embedder import TextEmbedder  # noqa: E402


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """두 벡터 간 cosine similarity를 계산한다. (검증 전용 헬퍼 함수)"""
    return float(
        np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    )


@pytest.fixture(scope="module")
def embedder():
    """
    모듈 전체에서 TextEmbedder를 한 번만 로딩하여 재사용한다.
    (scope="module": 이 파일 내 모든 테스트가 같은 embedder 인스턴스를 공유 -> 모델을
     매 테스트마다 다시 로딩하지 않아 테스트 속도가 빨라진다.)
    """
    return TextEmbedder()


class TestTextEmbedder:
    """TextEmbedder 클래스에 대한 테스트 그룹"""

    def test_encode_returns_correct_shape(self, embedder):
        """Shape 검증: 텍스트 3개를 넣으면 (3, 384) shape의 배열이 반환되어야 한다."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.encode(texts)

        assert embeddings.shape == (3, 384)

    def test_encode_is_deterministic(self, embedder):
        """결정론적 동작 검증: 같은 텍스트를 두 번 인코딩하면 완전히 같은 벡터가 나와야 한다."""
        text = ["The default port is 8842."]

        first_result = embedder.encode(text)
        second_result = embedder.encode(text)

        np.testing.assert_array_equal(first_result, second_result)

    def test_similar_sentences_have_higher_similarity_than_unrelated_ones(self, embedder):
        """
        의미적 유사도 검증: NimbusFlow 문서 맥락에 맞는 예시로,
        의미가 비슷한 문장끼리의 유사도가, 의미가 무관한 문장과의 유사도보다 높아야 한다.
        """
        anchor = "What is the default API port for NimbusFlow?"
        similar = "NimbusFlow exposes a REST API on port 8842 by default."
        unrelated = "The cat is sleeping on a sunny windowsill."

        vectors = embedder.encode([anchor, similar, unrelated])
        anchor_vec, similar_vec, unrelated_vec = vectors[0], vectors[1], vectors[2]

        sim_to_similar = cosine_similarity(anchor_vec, similar_vec)
        sim_to_unrelated = cosine_similarity(anchor_vec, unrelated_vec)

        assert sim_to_similar > sim_to_unrelated

    def test_encode_empty_list_raises_value_error(self, embedder):
        """예외 케이스: 빈 리스트를 넣으면 ValueError가 발생해야 한다."""
        with pytest.raises(ValueError):
            embedder.encode([])