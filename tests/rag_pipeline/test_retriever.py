"""
Test for Step B: Retrieval (src/model/retriever.py)

검증 항목:
1. cosine_similarity 정확성: 알려진 입력에 대해 정확한 값을 계산하는가
2. Top-k 정렬 검증: 결과가 실제로 점수 내림차순인가
3. Top-k 개수 검증: 요청한 k개만큼 정확히 반환하는가
4. 경계 케이스: k가 저장된 chunk 수보다 클 때의 처리
5. 예외 케이스: 빈 저장소에 대해 검색 시 ValueError 발생 여부
6. 의미적 검증: mock 벡터로 "가장 가까운 것"이 실제로 1위로 나오는가
"""

from pathlib import Path

import numpy as np
import pytest

from rag_pipeline.retriever import cosine_similarity, retrieve_top_k  # noqa: E402
from rag_pipeline.vector_store import InMemoryVectorStore  # noqa: E402


class TestCosineSimilarity:
    """cosine_similarity() 함수에 대한 테스트 그룹"""

    def test_identical_vectors_have_similarity_of_one(self):
        """완전히 같은 벡터끼리는 유사도가 1.0이어야 한다 (같은 방향, 각도 0도)."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_of_zero(self):
        """직각(90도)을 이루는 벡터끼리는 유사도가 0.0이어야 한다."""
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        similarity = cosine_similarity(vec_a, vec_b)

        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors_have_similarity_of_negative_one(self):
        """완전히 반대 방향(180도)인 벡터끼리는 유사도가 -1.0이어야 한다."""
        vec_a = np.array([1.0, 2.0])
        vec_b = np.array([-1.0, -2.0])
        similarity = cosine_similarity(vec_a, vec_b)

        assert similarity == pytest.approx(-1.0)


class TestRetrieveTopK:
    """retrieve_top_k() 함수에 대한 테스트 그룹"""

    def _build_store_with_mock_vectors(self):
        """
        테스트용 저장소를 만든다. 각 chunk의 벡터는 query_vector([1,0])와의
        예상 유사도가 chunk 번호와 반대로 매핑되도록 의도적으로 설계한다.

        chunk 0: [1.0, 0.0]  -> query와 동일 -> 유사도 1.0 (가장 유사)
        chunk 1: [0.7, 0.7]  -> 45도 -> 유사도 약 0.707
        chunk 2: [0.0, 1.0]  -> 90도 -> 유사도 0.0 (가장 무관)
        """
        chunks = ["chunk 0 (가장 유사해야 함)", "chunk 1 (중간)", "chunk 2 (가장 무관해야 함)"]
        vectors = np.array([
            [1.0, 0.0],
            [0.7, 0.7],
            [0.0, 1.0],
        ])

        store = InMemoryVectorStore()
        store.add(chunks, vectors)
        return store

    def test_results_are_sorted_in_descending_order(self):
        """Top-k 정렬 검증: 반환된 결과가 실제로 유사도 점수 내림차순이어야 한다."""
        store = self._build_store_with_mock_vectors()
        query_vector = np.array([1.0, 0.0])

        results = retrieve_top_k(query_vector, store, k=3)
        scores = [score for _, score in results]

        assert scores == sorted(scores, reverse=True)

    def test_most_similar_chunk_is_ranked_first(self):
        """의미적 검증: query와 동일한 방향의 벡터를 가진 chunk가 1위로 나와야 한다."""
        store = self._build_store_with_mock_vectors()
        query_vector = np.array([1.0, 0.0])

        results = retrieve_top_k(query_vector, store, k=3)
        top_chunk_text, top_score = results[0]

        assert top_chunk_text == "chunk 0 (가장 유사해야 함)"
        assert top_score == pytest.approx(1.0)

    def test_returns_exactly_k_results(self):
        """Top-k 개수 검증: k=2를 요청하면 정확히 2개만 반환해야 한다."""
        store = self._build_store_with_mock_vectors()
        query_vector = np.array([1.0, 0.0])

        results = retrieve_top_k(query_vector, store, k=2)

        assert len(results) == 2

    def test_k_larger_than_store_size_returns_all_available(self):
        """경계 케이스: k가 저장된 chunk 수보다 크면, 가능한 만큼만(전체) 반환해야 한다."""
        store = self._build_store_with_mock_vectors()  # chunk 3개
        query_vector = np.array([1.0, 0.0])

        results = retrieve_top_k(query_vector, store, k=10)

        assert len(results) == 3  # 에러 없이, 있는 만큼만 반환

    def test_empty_store_raises_value_error(self):
        """예외 케이스: 데이터가 하나도 없는 저장소에 검색을 시도하면 ValueError가 발생해야 한다."""
        empty_store = InMemoryVectorStore()
        query_vector = np.array([1.0, 0.0])

        with pytest.raises(ValueError):
            retrieve_top_k(query_vector, empty_store, k=3)