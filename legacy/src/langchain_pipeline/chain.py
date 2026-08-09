"""
langchain_pipeline 8단계: chain.py (LCEL 체인 조립)

기존 main.py의 절차적 호출(embed -> retrieve -> prompt -> generate)을 LangChain의
LCEL(`|` 연산자)로 선언적으로 조립한 Runnable로 옮긴다
(docs/LANGCHAIN_MIGRATION_PLAN.md 표 2 / 표 5, 8단계 / §4.5 / §5).

[설계 — 왜 함수 2개인가]
기존 main.py에는 두 엔드포인트가 있고, 각각 반환 형태가 다르다 (소스코드로 확인함):
  - POST /query        : {"answer": str, "retrieved_chunks": [{"text":, "score":}, ...]}
                          (검색에 사용된 chunk와 점수까지 함께 반환)
  - POST /query/stream  : 답변 텍스트(token)만 SSE로 스트리밍. retrieved_chunks는
                          응답에 포함되지 않는다 (query_stream()의 event_stream()이
                          generator.generate_stream(prompt)의 토큰만 yield함).

이 비대칭은 기존 모듈의 의도된 설계이지 누락이 아니므로, 그대로 승계한다:
  - build_rag_chain()        -> /query용. invoke() 결과가 dict.
  - build_answer_only_chain() -> /query/stream용. invoke()/stream() 결과가 str.

[설계 — 왜 RunnableParallel이 필요한가]
§4.5는 "retriever | prompt | llm처럼 선언적으로 연결"한다고 설명하지만, 단순
직렬(RunnableSequence)만으로는 build_rag_chain()의 요구사항을 만족할 수 없다.
retriever 단계의 출력(검색된 문서+점수)이 prompt 단계를 거치면서 사라지기
때문이다 — prompt 단계의 출력은 "문자열 1개"일 뿐, 원본 문서 정보를 더 이상
들고 있지 않다. 그런데 /query 응답은 "생성된 답변"과 "검색에 사용된 chunk+점수"를
동시에 필요로 한다. 따라서 RunnableParallel로 "검색 결과(retrieved)를 보존하면서
question도 함께 다음 단계로 넘기는" 분기 구조를 쓴다 — 이것이 정확히 §5에서
언급한 RunnableParallel/RunnablePassthrough의 용도다.

[설계 — InMemoryVectorStore.similarity_search_with_score() vs get_retriever()]
4단계(vector_store.py)에서 만든 get_retriever()는 VectorStoreRetriever를 반환하는데,
이 표준 Retriever 인터페이스는 Document만 반환하고 점수(score)를 버린다(LangChain
공식 동작 — VectorStoreRetriever._get_relevant_documents()가 score 없이
Document 리스트만 돌려줌). 그런데 기존 main.py의 /query 응답은 score가 반드시
필요하다(QueryResponse.retrieved_chunks[].score). 따라서 chain.py는 get_retriever()를
쓰지 않고, InMemoryVectorStore.similarity_search_with_score(query, k)
(list[tuple[Document, float]]을 반환하는 공식 메서드 — langchain_core 소스코드로
직접 확인함)를 직접 호출한다. 이는 4단계의 결정(§4.1 — thin wrapper 절충안 미채택,
LangChain의 결합된 VectorStore 인터페이스 그대로 사용)과 일치한다 — VectorStore
자체가 제공하는 다른 공식 메서드를 추가로 쓰는 것이지, 새로운 절충 계층을
만드는 게 아니다.

[기존 prompt_builder.build_prompt()와의 1:1 대응]
기존 build_prompt(question, retrieved_chunks)는 prompt 조립 시 점수를 쓰지 않고
텍스트만 context에 넣는다(소스코드로 확인함) — 5단계의 format_docs()도 동일하게
점수 없이 Document.page_content만 사용하므로, 그대로 대응된다.
"""

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.prompt import format_docs, get_prompt_template


def _prompt_text_from_documents(documents: list[Document], question: str) -> str:
    """5단계 prompt.py의 결과물을, 비-chat 모델(custom_transformer)/HuggingFacePipeline
    양쪽 모두가 받는 "순수 문자열" 형태로 변환한다.

    .to_string()이 아니라 .to_messages()[0].content를 쓰는 이유는 5단계에서 이미
    확인한 "Human: " 접두사 오염을 피하기 위함이다(prompt.py 모듈 docstring 참고).
    """
    context = format_docs(documents)
    prompt_value = get_prompt_template().invoke({"context": context, "question": question})
    return prompt_value.to_messages()[0].content


def build_rag_chain(
    store: InMemoryVectorStore,
    llm: Runnable,
    k: int = 3,
) -> Runnable:
    """
    기존 main.py POST /query와 동일한 절차(검색 -> prompt 조립 -> 생성)를 LCEL로
    조립한다.

    Args:
        store: 4단계 build_vector_store()로 만든 InMemoryVectorStore.
        llm: 6단계 get_gemma_llm() 또는 7단계 get_custom_transformer_llm()의 반환값
             (둘 다 Runnable이므로 이 함수는 어느 쪽이 오든 동일하게 동작한다 — §4.3).
        k: 검색할 상위 문서 개수 (기존 QueryRequest.k와 동일한 기본값 3).

    Returns:
        invoke(question: str) -> {"answer": str, "retrieved_chunks": [{"text": str, "score": float}, ...]}
        형태의 Runnable. 기존 QueryResponse와 동일한 정보를 담은 dict이다.
    """

    def _retrieve(question: str) -> list[tuple[Document, float]]:
        return store.similarity_search_with_score(question, k=k)

    def _to_prompt_text(inputs: dict) -> str:
        documents = [doc for doc, _score in inputs["retrieved"]]
        return _prompt_text_from_documents(documents, inputs["question"])

    def _to_retrieved_chunks(inputs: dict) -> list[dict]:
        return [{"text": doc.page_content, "score": score} for doc, score in inputs["retrieved"]]

    return RunnableParallel(retrieved=RunnableLambda(_retrieve), question=RunnablePassthrough()) | RunnableParallel(
        answer=RunnableLambda(_to_prompt_text) | llm,
        retrieved_chunks=RunnableLambda(_to_retrieved_chunks),
    )


def build_answer_only_chain(
    store: InMemoryVectorStore,
    llm: Runnable,
    k: int = 3,
) -> Runnable:
    """
    기존 main.py POST /query/stream과 동일한 절차를 LCEL로 조립한다.

    기존 query_stream()은 검색+prompt 조립까지는 동기적으로 끝내고, 생성
    단계만 토큰 단위로 스트리밍한다 — retrieved_chunks는 응답에 포함하지
    않는다(소스코드로 확인함). 이 함수가 반환하는 체인도 동일하게, 마지막
    단계(llm)만 진짜로 스트리밍되고, 그 앞(검색+prompt 조립)은 .stream() 호출
    시에도 즉시(eager) 실행된다 — 이것이 LCEL RunnableSequence.stream()의 표준
    동작이다(마지막 Runnable만 실제로 스트리밍되고, 앞 단계는 .invoke()처럼
    한 번에 계산됨).

    Returns:
        invoke(question: str) -> str (답변 전체)
        stream(question: str) -> Iterator[str] (답변 조각들)
    """

    def _retrieve(question: str) -> list[Document]:
        return [doc for doc, _score in store.similarity_search_with_score(question, k=k)]

    def _to_prompt_text(inputs: dict) -> str:
        return _prompt_text_from_documents(inputs["documents"], inputs["question"])

    return (
        RunnableParallel(documents=RunnableLambda(_retrieve), question=RunnablePassthrough())
        | RunnableLambda(_to_prompt_text)
        | llm
    )


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.embedding import get_embeddings_model
    from langchain_pipeline.llm import get_gemma_llm
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.vector_store import build_vector_store

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    store = build_vector_store(chunks, embeddings_model)

    print("[Gemma 4 E2B-it 로딩 중... 처음 실행 시 다운로드가 필요합니다]")
    llm = get_gemma_llm()

    question = "DaySync의 내부 코드네임은 무엇인가?"

    # /query 대응
    rag_chain = build_rag_chain(store, llm, k=3)
    result = rag_chain.invoke(question)
    print(f"[질문] {question}\n")
    print(f"[답변] {result['answer']}\n")
    print("[검색된 chunk]")
    for chunk in result["retrieved_chunks"]:
        print(f"  (score={chunk['score']:.4f}) {chunk['text'][:80]!r}")

    # /query/stream 대응
    answer_only_chain = build_answer_only_chain(store, llm, k=3)
    print("\n[스트리밍 답변] ", end="", flush=True)
    for token in answer_only_chain.stream(question):
        print(token, end="", flush=True)
    print()
