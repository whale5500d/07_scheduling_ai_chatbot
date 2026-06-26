"""
Step A-3: Embedding

책임(Responsibility): chunk 리스트(list[str])를 받아 임베딩 벡터 배열(numpy.ndarray)을 반환한다.
이 단계에서는 벡터를 저장(storage)하거나 검색(retrieval)하지 않는다.

모델: sentence-transformers/all-MiniLM-L6-v2
- 384차원 벡터를 출력하는 영어 전용(monolingual) 경량 모델
- 최대 256 word piece까지 처리 가능 (이보다 길면 잘림 -> Chunking을 먼저 거치는 이유)
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    sentence-transformers 모델을 한 번만 로딩하고, 여러 번 재사용하여
    텍스트를 임베딩 벡터로 변환하는 클래스.

    모델 로딩(비용이 큰 작업)과 인코딩(반복적으로 호출되는 작업)을 분리하기 위해
    단순 함수 대신 클래스로 구현한다.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: 사용할 sentence-transformers 모델 이름
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 벡터 배열로 변환한다.

        Args:
            texts: 임베딩할 텍스트(chunk) 리스트

        Returns:
            shape (len(texts), embedding_dim)의 numpy 배열.
            embedding_dim은 all-MiniLM-L6-v2의 경우 384.

        Raises:
            ValueError: texts가 빈 리스트일 경우
        """
        if not texts:
            raise ValueError("texts가 빈 리스트입니다. 임베딩할 텍스트가 1개 이상 필요합니다.")

        # convert_to_numpy=True를 명시하여 반환 타입을 numpy.ndarray로 고정한다.
        # (명시하지 않으면 내부 설정에 따라 torch.Tensor가 반환될 수 있어
        #  정적 타입 검사기가 타입 불일치 경고를 보낸다.)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


if __name__ == "__main__":
    from paths import DATA_DIR
    from rag_pipeline.document_loader import load_document
    from rag_pipeline.chunker import chunk_fixed_size

    sample_path = DATA_DIR / "daysync_manual.md"
    document = load_document(str(sample_path))
    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

    embedder = TextEmbedder()
    embeddings = embedder.encode(chunks)

    print(f"[Embedding 결과] chunk 개수: {len(chunks)}")
    print(f"[Embedding 결과] 벡터 배열 shape: {embeddings.shape}")
    print(f"[Embedding 결과] 첫 벡터의 처음 5개 값: {embeddings[0][:5]}")