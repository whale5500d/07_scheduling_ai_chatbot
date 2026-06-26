"""
Step A-4: Storage (In-Memory 방식)

책임(Responsibility): chunk 리스트와 그에 대응하는 벡터 배열을 짝지어 메모리에 보관한다.
이 단계에서는 유사도 계산이나 검색(retrieval) 로직을 수행하지 않는다.

설계 원칙: 인덱스 정합성(index consistency)
- chunks[i]의 벡터는 반드시 vectors[i]에 있어야 한다.
- 이 1:1 대응이 깨지면 Retrieval 단계에서 잘못된 텍스트를 반환하는 치명적 버그가 발생한다.
"""

import numpy as np


class InMemoryVectorStore:
    """
    chunk 텍스트와 임베딩 벡터를 메모리(Python 객체)에 보관하는 가장 단순한 저장소.

    VectorDB(FAISS 등) 없이, numpy 배열과 list만으로 직접 구현한다.
    프로그램이 종료되면 데이터가 사라진다 (영속성 없음) — 이는 의도된 설계로,
    추후 File-based Storage 단계에서 영속성을 추가할 예정이다.
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None  # 아직 추가된 데이터가 없으면 None

    def add(self, chunks: list[str], vectors: np.ndarray) -> None:
        """
        chunk 리스트와 벡터 배열을 저장소에 추가한다.

        Args:
            chunks: 추가할 텍스트 chunk 리스트
            vectors: chunks와 동일한 순서로 대응하는 임베딩 벡터 배열, shape (len(chunks), dim)

        Raises:
            ValueError: chunks 개수와 vectors 개수가 일치하지 않을 경우
                        (인덱스 정합성이 깨지는 것을 방지)
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks 개수({len(chunks)})와 vectors 개수({len(vectors)})가 일치하지 않습니다. "
                "인덱스 정합성이 깨지므로 저장을 거부합니다."
            )

        self.chunks.extend(chunks)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def get_chunk(self, index: int) -> str:
        """
        주어진 인덱스에 해당하는 chunk 텍스트를 반환한다.
        (Retrieval 단계에서 "이 벡터의 인덱스가 5번이었다"는 결과를 받았을 때,
         실제 텍스트를 다시 찾기 위해 사용한다.)

        Args:
            index: chunk의 인덱스

        Returns:
            해당 인덱스의 chunk 텍스트
        """
        return self.chunks[index]

    def __len__(self) -> int:
        """저장된 chunk의 개수를 반환한다."""
        return len(self.chunks)


if __name__ == "__main__":
    from paths import DATA_DIR
    from rag_pipeline.document_loader import load_document
    from rag_pipeline.chunker import chunk_fixed_size
    from rag_pipeline.embedder import TextEmbedder

    sample_path = DATA_DIR / "daysync_manual.md"
    document = load_document(str(sample_path))
    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

    embedder = TextEmbedder()
    vectors = embedder.encode(chunks)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    assert store.vectors is not None

    print(f"[Storage 결과] 저장된 chunk 개수: {len(store)}")
    print(f"[Storage 결과] 저장된 벡터 배열 shape: {store.vectors.shape}")
    print(f"[Storage 결과] 0번 인덱스 chunk 미리보기: {store.get_chunk(0)[:80]}...")