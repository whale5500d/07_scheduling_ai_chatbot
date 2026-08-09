"""
langchain_pipeline 5단계: Prompt Augmentation

기존 rag_pipeline/prompt_builder.py의 LangChain 대응 모듈
(docs/LANGCHAIN_MIGRATION_PLAN.md 표 2, 7행 / 표 5, 5단계 결정:
"ChatPromptTemplate, 기존 instruction 문구·포맷 그대로 유지").

입력 타입 변경: 기존 build_prompt(question, retrieved_chunks: list[tuple[str, float]])는
(chunk 텍스트, 유사도 점수) 튜플을 받았다. 4단계 vector_store.py의 VectorStoreRetriever는
.invoke(query)에서 list[Document]만 반환한다(점수는 similarity_search_with_score()를
별도로 호출해야 얻을 수 있음 — as_retriever()의 표준 인터페이스에는 없다).
기존 build_prompt()는 score를 실제로 prompt 텍스트에 사용하지 않았으므로(루프에서
_score를 버림), 이 정보 손실은 최종 prompt 내용에 영향이 없다. 따라서 이 모듈은
list[tuple[str, float]] 대신 list[Document]를 받는다.

[중요 — 공식 동작으로 확인된 주의사항]
ChatPromptTemplate.from_template(...)으로 만든 prompt는 단일 HumanMessage로 감싸진다.
이 결과를 BaseLLM 계열(6·7단계의 HuggingFacePipeline / custom LLM 서브클래스)에
그대로 흘려보내면, BaseLLM이 입력을 문자열로 변환할 때 ChatPromptValue.to_string()을
호출하는데, 이 메서드는 "Human: "을 맨 앞에 붙인다 (langchain_core의 공식 동작,
직접 코드로 확인됨). 즉 `prompt_template | llm`으로 바로 연결하면 LLM이 실제로 받는
텍스트 맨 앞에 "Human: "이 추가되어, "기존 instruction 문구·포맷 그대로 유지" 결정과
어긋난다.

따라서 8단계(chain.py)에서 LCEL 체인을 조립할 때는 prompt_template의 출력을 그대로
LLM에 넘기지 말고, `ChatPromptValue.to_messages()[0].content`(역할 prefix가 없는
원본 텍스트)를 추출해서 넘겨야 한다. 이 모듈은 ChatPromptTemplate 객체 자체만
제공하고, 그 추출 로직은 체인 조립 책임이므로 chain.py(8단계)에 둔다 — 이 모듈의
책임 범위는 "포맷팅"이지 "체인 연결"이 아니다.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

INSTRUCTION_TEMPLATE = """당신은 주어진 문서에만 근거하여 질문에 답하는 유능한 어시스턴트입니다.

아래 문서를 참고하여 질문에 답하세요. 문서에서 답을 찾을 수 없다면 "주어진 문서에서 답을 찾을 수 없습니다."라고 답하세요. 문서에 없는 내용을 지어내지 마세요.

{context}

질문: {question}

답변:"""


def format_docs(documents: list[Document]) -> str:
    """
    검색된 Document 리스트를 번호가 매겨진 context 문자열로 변환한다.
    (기존 build_prompt()의 context_blocks 조립 로직과 동일한 포맷: "[문서 N]\\n{content}")

    Args:
        documents: 검색된 문서 리스트 (4단계 vector_store.get_retriever()의 출력)

    Returns:
        "[문서 1]\\n...\\n\\n[문서 2]\\n..." 형태로 결합된 context 문자열

    Raises:
        ValueError: documents가 빈 리스트일 경우
            (Prompt를 만들기 위해서는 최소 1개의 검색 결과가 필요하다는 기존 제약을 유지)
    """
    if not documents:
        raise ValueError("documents가 빈 리스트입니다. Prompt를 만들기 위해서는 최소 1개의 검색 결과가 필요합니다.")

    context_blocks = [f"[문서 {i}]\n{doc.page_content}" for i, doc in enumerate(documents, start=1)]
    return "\n\n".join(context_blocks)


def get_prompt_template() -> ChatPromptTemplate:
    """
    기존 instruction 문구·포맷을 그대로 유지한 ChatPromptTemplate을 반환한다.

    사용 예: get_prompt_template().invoke({"context": format_docs(docs), "question": question})

    Returns:
        {context}, {question} 두 변수를 받는 ChatPromptTemplate.
    """
    return ChatPromptTemplate.from_template(INSTRUCTION_TEMPLATE)


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.embedding import get_embeddings_model
    from langchain_pipeline.vector_store import build_vector_store, get_retriever

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    store = build_vector_store(chunks, embeddings_model)
    retriever = get_retriever(store, k=3)

    question = "DaySync의 내부 코드네임은 무엇인가?"
    retrieved_docs = retriever.invoke(question)

    prompt_template = get_prompt_template()
    prompt_value = prompt_template.invoke({"context": format_docs(retrieved_docs), "question": question})

    print("=" * 60)
    print(prompt_value.to_messages()[0].content)
    print("=" * 60)
