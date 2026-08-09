"""
langchain_pipeline 3단계: 임베딩(Embedding) 모델 준비

기존 rag_pipeline/embedder.py의 LangChain 대응 모듈
(docs/LANGCHAIN_MIGRATION_PLAN.md 표 2, 4행).

기존 TextEmbedder는 "모델 로딩"과 "encode() 호출"을 하나의 클래스로 묶었다.
LangChain에서는 임베딩 함수의 소유권이 VectorStore로 넘어간다(§4.2, 표 1 #6) —
이 모듈은 HuggingFaceEmbeddings 인스턴스를 생성하는 책임만 가지며, 실제 인코딩 호출
(embed_documents/embed_query)은 4단계 vector_store.py가 생성자에서 이 인스턴스를
주입받아 내부적으로 수행한다. 따라서 "빈 리스트 입력 시 ValueError" 같은 호출 단계의
검증 로직은 이 모듈에 없다 — 실제로 encode를 호출하는 책임 자체가 4단계로 이동했다.

모델: sentence-transformers/all-MiniLM-L6-v2 (기존과 동일, 384차원)

HuggingFaceEmbeddings의 표준 인터페이스(공식 소스코드로 직접 확인됨,
langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings):
- embed_documents(texts: list[str]) -> list[list[float]]
- embed_query(text: str) -> list[float]  (내부적으로 embed_documents([text])[0]과 동일)
(둘 다 numpy.ndarray가 아니라 list[float]를 반환한다 — 기존 TextEmbedder.encode()와의
 차이점이며, langchain_core.embeddings.Embeddings 인터페이스의 표준 반환 타입이다.)
"""
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    HuggingFaceEmbeddings 인스턴스를 생성한다.

    Args:
        model_name: 사용할 sentence-transformers 모델 이름

    Returns:
        주어진 모델로 구성된 HuggingFaceEmbeddings 인스턴스.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    vectors = embeddings_model.embed_documents([c.page_content for c in chunks])

    print(f"[Embedding 결과] chunk 개수: {len(chunks)}")
    print(f"[Embedding 결과] 벡터 개수: {len(vectors)}, 차원: {len(vectors[0])}")
    print(f"[Embedding 결과] 첫 벡터의 처음 5개 값: {vectors[0][:5]}")
