"""
Step B: Prompt Augmentation

책임(Responsibility): 사용자 질문과 Retrieval 결과(검색된 chunk들)를 받아,
LLM에게 전달할 하나의 완성된 prompt(문자열)를 조립한다.
이 단계에서는 LLM을 호출하지 않는다.

설계 원칙:
1. LLM에게 "주어진 문서를 근거로 답하라"는 지시를 명시하여 환각(hallucination)을 억제한다.
2. "문서에서 답을 찾을 수 없으면 모른다고 답하라"는 지시를 포함하여, 근거 없는 추측을 방지한다.
3. 검색된 chunk들을 번호로 구분하여 LLM이 각 정보의 출처를 인식할 수 있게 한다.
"""


def build_prompt(question: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    """
    사용자 질문과 검색된 chunk들을 결합하여 LLM에게 전달할 prompt를 생성한다.

    Args:
        question: 사용자의 원래 질문
        retrieved_chunks: retrieve_top_k()의 반환값. (chunk 텍스트, 유사도 점수) 리스트

    Returns:
        Instruction + Context + Question이 결합된 완성된 prompt 문자열

    Raises:
        ValueError: retrieved_chunks가 빈 리스트일 경우
    """
    if not retrieved_chunks:
        raise ValueError(
            "retrieved_chunks가 빈 리스트입니다. Prompt를 만들기 위해서는 최소 1개의 검색 결과가 필요합니다."
        )

    # Context 부분: 검색된 chunk들을 번호로 구분하여 나열
    context_blocks = []
    for i, (chunk_text, _score) in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[문서 {i}]\n{chunk_text}")
    context = "\n\n".join(context_blocks)

    # Instruction + Context + Question을 결합
    prompt = f"""당신은 주어진 문서에만 근거하여 질문에 답하는 유능한 어시스턴트입니다.

아래 문서를 참고하여 질문에 답하세요. 문서에서 답을 찾을 수 없다면 "주어진 문서에서 답을 찾을 수 없습니다."라고 답하세요. 문서에 없는 내용을 지어내지 마세요.

{context}

질문: {question}

답변:"""

    return prompt


if __name__ == "__main__":
    from paths import DATA_DIR
    from rag_pipeline.document_loader import load_document
    from rag_pipeline.chunker import chunk_fixed_size
    from rag_pipeline.embedder import TextEmbedder
    from rag_pipeline.vector_store import InMemoryVectorStore
    from rag_pipeline.retriever import retrieve_top_k

    sample_path = DATA_DIR / "daysync_manual.md"
    document = load_document(str(sample_path))
    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

    embedder = TextEmbedder()
    vectors = embedder.encode(chunks)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    question = "DaySync의 내부 코드네임은 무엇인가?"
    query_vector = embedder.encode([question])[0]
    retrieved = retrieve_top_k(query_vector, store, k=3)

    prompt = build_prompt(question, retrieved)

    print("=" * 60)
    print(prompt)
    print("=" * 60)