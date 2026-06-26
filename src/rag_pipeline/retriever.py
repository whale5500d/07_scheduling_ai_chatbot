"""
Step B: Retrieval

책임(Responsibility): query_vector(질의 벡터)와 저장소(InMemoryVectorStore)를 받아,
가장 유사한 상위 k개의 (chunk 텍스트, 유사도 점수)를 반환한다.
이 단계에서는 prompt 조립이나 LLM 생성을 하지 않는다.

구현 방식: 순차 탐색(Sequential Search) — brute-force 전략
- 저장된 모든 chunk 벡터를 빠짐없이 하나씩 순회하며 query_vector와 비교한다.
- chunk 수가 많아지면(수만 개 이상) 느려질 수 있으나, 학습 목적상 동작 원리를
  명확히 보기 위해 벡터화 연산 없이 for-loop으로 먼저 구현한다.
"""

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 벡터 간 cosine similarity를 계산한다.

    Args:
        vec_a: 첫 번째 벡터
        vec_b: 두 번째 벡터 (vec_a와 같은 차원)

    Returns:
        -1.0 ~ 1.0 범위의 유사도 점수. 1에 가까울수록 의미가 유사함을 뜻한다.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return float(dot_product / (norm_a * norm_b))


def retrieve_top_k(
    query_vector: np.ndarray,
    store,
    k: int = 3,
) -> list[tuple[str, float]]:
    """
    저장소에 있는 모든 chunk 벡터를 순차적으로 탐색하여,
    query_vector와 가장 유사한 상위 k개의 (chunk 텍스트, 유사도 점수)를 반환한다.

    Args:
        query_vector: 사용자 질문을 임베딩한 벡터, shape (384,)
        store: chunks와 vectors를 보관하는 InMemoryVectorStore 인스턴스
        k: 반환할 상위 결과 개수

    Returns:
        (chunk 텍스트, 유사도 점수) 튜플의 리스트. 점수 내림차순으로 정렬되어 있다.

    Raises:
        ValueError: 저장소에 데이터가 하나도 없을 경우 (store.vectors가 None)
    """
    if store.vectors is None:
        raise ValueError("저장소가 비어 있습니다. 검색하기 전에 먼저 add()로 데이터를 추가해야 합니다.")

    # 2단계: 순차 비교 — 저장된 모든 벡터를 하나씩 순회하며 유사도 계산
    scores: list[tuple[int, float]] = []  # (인덱스, 점수) 쌍을 저장
    for index in range(len(store)):
        chunk_vector = store.vectors[index]
        score = cosine_similarity(query_vector, chunk_vector)
        scores.append((index, score))

    # 3단계: 점수 기준 내림차순 순위화 (인덱스 정보는 유지)
    scores.sort(key=lambda pair: pair[1], reverse=True)

    # 4단계: 상위 k개만 추출
    top_k_scores = scores[:k]

    # 5단계: 인덱스로 실제 텍스트 복원
    results: list[tuple[str, float]] = []
    for index, score in top_k_scores:
        chunk_text = store.get_chunk(index)
        results.append((chunk_text, score))

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag_pipeline.document_loader import load_document
    from rag_pipeline.chunker import chunk_fixed_size
    from rag_pipeline.embedder import TextEmbedder
    from rag_pipeline.vector_store import InMemoryVectorStore

    # Step A 재구성: Indexing
    sample_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "nimbusflow_manual.md"
    )
    document = load_document(str(sample_path))
    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

    embedder = TextEmbedder()
    vectors = embedder.encode(chunks)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    # Step B 시작: Query Embedding + Retrieval
    question = "What is the default API port for NimbusFlow?"
    query_vector = embedder.encode([question])[0]  # encode는 리스트를 받으므로 [0]으로 단일 벡터 추출

    top_k_results = retrieve_top_k(query_vector, store, k=3)

    print(f"[질문] {question}\n")
    for rank, (chunk_text, score) in enumerate(top_k_results, start=1):
        print(f"--- Rank {rank} (유사도: {score:.4f}) ---")
        print(chunk_text[:150].replace("\n", " "))
        print()