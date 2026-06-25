"""
Test for Step A-4: Storage (src/model/vector_store.py)

검증 항목 (표 47):
1. 정상 추가 검증: add() 호출 후 chunks, vectors, len()이 올바르게 반영되는가
2. 인덱스 정합성 검증: get_chunk(i)와 vectors[i]가 같은 원본에서 나온 것인가
3. 길이 불일치 예외 케이스: chunk 개수와 vector 개수가 다르면 ValueError 발생 여부
4. 다중 add() 호출 누적 검증: add()를 두 번 호출하면 데이터가 누적되는가

주의: 실제 embedding 모델을 쓰지 않고, 식별 가능한 가짜 벡터(mock vector)를 사용한다.
      이렇게 하면 "어떤 텍스트의 벡터인지"를 벡터 값 자체로 명확히 추적할 수 있어
      인덱스 정합성 검증이 쉬워진다.
"""

import numpy as np
import pytest

from rag_pipeline.vector_store import InMemoryVectorStore  # noqa: E402


def make_mock_vectors(n: int, dim: int = 4) -> np.ndarray:
    """
    테스트용 가짜 벡터를 생성한다.
    i번째 벡터는 모든 원소가 i로 채워진다 (예: 0번 벡터는 [0,0,0,0], 1번 벡터는 [1,1,1,1]).
    이렇게 하면 "이 벡터가 몇 번째였는지"를 값만 보고 바로 알 수 있다.
    """
    return np.array([[float(i)] * dim for i in range(n)])


class TestInMemoryVectorStore:
    """InMemoryVectorStore 클래스에 대한 테스트 그룹"""

    def test_add_stores_chunks_and_vectors_correctly(self):
        """정상 추가 검증: add() 후 chunks, vectors, len()이 입력과 정확히 일치해야 한다."""
        chunks = ["chunk A", "chunk B", "chunk C"]
        vectors = make_mock_vectors(n=3)

        store = InMemoryVectorStore()
        store.add(chunks, vectors)

        assert store.vectors is not None
        assert len(store) == 3
        assert store.chunks == chunks
        np.testing.assert_array_equal(store.vectors, vectors)

    def test_index_consistency_between_chunk_and_vector(self):
        """
        인덱스 정합성 검증: get_chunk(i)로 가져온 텍스트가
        vectors[i]와 실제로 같은 원본 데이터에서 나온 것인지 확인한다.
        """
        chunks = ["The default port is 8842.", "Drift Score threshold is 0.73."]
        vectors = make_mock_vectors(n=2)

        store = InMemoryVectorStore()
        store.add(chunks, vectors)

        assert store.vectors is not None

        # 0번 인덱스: chunk와 vector가 모두 "0번째 원본"을 가리켜야 한다
        assert store.get_chunk(0) == "The default port is 8842."
        np.testing.assert_array_equal(store.vectors[0], vectors[0])

        # 1번 인덱스도 동일하게 검증
        assert store.get_chunk(1) == "Drift Score threshold is 0.73."
        np.testing.assert_array_equal(store.vectors[1], vectors[1])

    def test_add_with_mismatched_length_raises_value_error(self):
        """예외 케이스: chunk 개수와 vector 개수가 다르면 ValueError가 발생해야 한다."""
        chunks = ["only one chunk"]
        vectors = make_mock_vectors(n=2)  # 개수 불일치 (chunks=1, vectors=2)

        store = InMemoryVectorStore()

        with pytest.raises(ValueError):
            store.add(chunks, vectors)

    def test_multiple_add_calls_accumulate_data(self):
        """
        다중 add() 호출 누적 검증: add()를 두 번 호출하면
        첫 번째 데이터를 덮어쓰지 않고 그 뒤에 누적되어야 한다.
        (여러 문서로 확장할 때를 대비한 핵심 동작)
        """
        store = InMemoryVectorStore()

        # 첫 번째 문서의 chunk/vector 추가
        first_chunks = ["doc1 chunk1", "doc1 chunk2"]
        first_vectors = make_mock_vectors(n=2)
        store.add(first_chunks, first_vectors)

        # 두 번째 문서의 chunk/vector 추가
        second_chunks = ["doc2 chunk1"]
        second_vectors = np.array([[9.0, 9.0, 9.0, 9.0]])  # 식별 가능한 값
        store.add(second_chunks, second_vectors)

        # 전체 개수는 2 + 1 = 3이어야 한다
        assert len(store) == 3
        assert store.vectors is not None

        # 순서대로 잘 누적되었는지 확인 (첫 데이터가 덮어써지지 않았어야 함)
        assert store.get_chunk(0) == "doc1 chunk1"
        assert store.get_chunk(1) == "doc1 chunk2"
        assert store.get_chunk(2) == "doc2 chunk1"

        # 벡터도 마찬가지로 순서대로 누적되었는지 확인
        np.testing.assert_array_equal(store.vectors[2], [9.0, 9.0, 9.0, 9.0])