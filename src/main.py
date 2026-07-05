"""
Step C-1: FastAPI로 RAG 파이프라인 배포

책임(Responsibility): 지금까지 만든 RAG 파이프라인(Step A + Step B)을
HTTP로 호출 가능한 REST API로 노출한다.

설계 결정:
1. 단일 엔드포인트(POST /query) — 검색+생성 전체를 한 번의 호출로 처리한다.
2. Indexing은 서버 시작 시 1회만 수행한다 (lifespan 이벤트 활용) — 데이터가
   고정되어 있으므로 매 요청마다 다시 인덱싱하는 것은 비효율적이다.
3. 응답에 최종 답변뿐 아니라 검색된 chunk와 유사도 점수도 함께 포함한다 —
   답변의 근거를 확인할 수 있게 하고, 추후 RAGAS 평가에서 재사용할 수 있게 한다.
4. 백엔드 분기(docs/LANGCHAIN_MIGRATION_PLAN.md §9 (c), 표 5 9단계): 환경 변수
   RAG_BACKEND 값에 따라 lifespan에서 기존 rag_pipeline(기본값) 또는 신규
   langchain_pipeline 중 하나로 리소스를 구성한다. 신규 엔트리포인트를 따로 만들지
   않고 기존 main.py에 분기만 추가하는 방식을 §9에서 그대로 결정했으므로, 기본값은
   기존 동작과 완전히 동일하게 유지한다(RAG_BACKEND를 설정하지 않으면 이 분기를
   추가하기 전과 100% 동일하게 동작해야 한다).

[langchain_pipeline 관련 import를 함수 내부(lazy import)로 둔 이유]
rag_pipeline 경로는 무거운 의존성(torch/transformers/sentence-transformers)이
없어도 동작하도록 만들어진 적이 없다(원래부터 그 의존성들을 직접 쓴다). 반면
langchain_pipeline.llm/embedding도 결국 같은 무거운 의존성을 쓰지만, 두 백엔드는
"동시에 둘 다 쓰이지 않는다"(환경 변수로 양자택일) — 따라서 모듈 최상단에서
langchain_pipeline을 무조건 import하면, RAG_BACKEND를 한 번도 "langchain"으로
설정하지 않는 환경(예: 이 마이그레이션 작업을 검증하는 sandbox)에서도 langchain
쪽 무거운 의존성 전체가 import 시점에 강제로 필요해진다. 실제로 선택된 분기에서만
import하면 이 결합을 없앨 수 있다 — 이것은 "필요할 때만 로딩"이라는 일반적인
지연 import(lazy import) 패턴이며, FastAPI/LangChain 고유의 특별한 기법이 아니다.
"""

import os
from contextlib import asynccontextmanager
from typing import Iterator

from paths import DATA_DIR

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_pipeline.chunker import chunk_by_section
from rag_pipeline.document_loader import load_document
from rag_pipeline.embedder import TextEmbedder
from rag_pipeline.generator import TextGenerator
from rag_pipeline.prompt_builder import build_prompt
from rag_pipeline.retriever import retrieve_top_k
from rag_pipeline.vector_store import InMemoryVectorStore

# 백엔드 선택 환경 변수. 값이 "langchain"(대소문자 무관)일 때만 신규 경로를 쓰고,
# 그 외 모든 값(미설정 포함)은 기존 rag_pipeline 경로를 그대로 쓴다 — 기본값을
# 기존 동작과 동일하게 유지하기 위한 의도적인 선택이다.
RAG_BACKEND_ENV_VAR = "RAG_BACKEND"
LANGCHAIN_BACKEND_VALUE = "langchain"
LANGGRAPH_BACKEND_VALUE = "langgraph"

# 서버 전체에서 공유할 리소스(모델, 저장소)를 담을 컨테이너.
# 전역 변수를 직접 쓰는 대신 객체 하나에 묶어, 어떤 리소스들이 공유되는지 명확히 한다.
resources: dict = {}


def _use_langchain_backend() -> bool:
    """RAG_BACKEND 환경 변수가 "langchain"이면 True. 테스트에서 직접 호출해 분기
    조건 자체도 검증한다."""
    return os.environ.get(RAG_BACKEND_ENV_VAR, "").strip().lower() == LANGCHAIN_BACKEND_VALUE


def _use_langgraph_backend() -> bool:
    """RAG_BACKEND 환경 변수가 "langgraph"이면 True."""
    return os.environ.get(RAG_BACKEND_ENV_VAR, "").strip().lower() == LANGGRAPH_BACKEND_VALUE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 1회 실행: 문서 로딩 -> 청킹 -> 임베딩 -> 저장,
    그리고 Generation에 사용할 모델까지 미리 로딩해 둔다.
    서버 종료 시 별도로 정리할 리소스는 없다.

    RAG_BACKEND=langchain이면 langchain_pipeline 경로로, 그 외에는 기존
    rag_pipeline 경로로 리소스를 구성한다. resources["backend"]에 어느 경로가
    선택되었는지 기록해 두고, 각 엔드포인트는 이 값으로 분기한다.
    """
    data_path = DATA_DIR / "daysync_manual.md"

    if _use_langgraph_backend():
        from langchain_pipeline.embedding import get_embeddings_model
        from langchain_pipeline.loader import load_document as lc_load_document
        from langchain_pipeline.splitter import split_fixed_size
        from langchain_pipeline.vector_store import build_vector_store

        lg_documents = lc_load_document(str(data_path))
        lg_chunks = split_fixed_size(lg_documents, chunk_size=300, chunk_overlap=50)

        embeddings_model = get_embeddings_model()
        lg_store = build_vector_store(lg_chunks, embeddings_model)

        # Agent는 내부에서 Gemini LLM을 직접 생성하므로 lg_llm을 주입하지 않는다.
        # (기존 HuggingFacePipeline은 bind_tools()를 지원하지 않아 Agent에 사용 불가)
        resources["backend"] = "langgraph"
        resources["lg_store"] = lg_store
    elif _use_langchain_backend():
        from langchain_pipeline.embedding import get_embeddings_model
        from langchain_pipeline.llm import get_gemma_llm
        from langchain_pipeline.loader import load_document as lc_load_document
        from langchain_pipeline.splitter import split_fixed_size
        from langchain_pipeline.vector_store import build_vector_store

        lc_documents = lc_load_document(str(data_path))
        lc_chunks = split_fixed_size(lc_documents, chunk_size=300, chunk_overlap=50)

        embeddings_model = get_embeddings_model()
        lc_store = build_vector_store(lc_chunks, embeddings_model)

        # 기존 rag_pipeline 경로와 동일한 모델을 쓴다(아래 generator = TextGenerator(
        # model_name="google/gemma-4-E2B-it")와 대응) — 두 경로의 차이를 "구현 방식"으로
        # 한정하고, "어떤 모델을 쓰는가"는 동일하게 유지한다.
        lc_llm = get_gemma_llm()

        resources["backend"] = "langchain"
        resources["lc_store"] = lc_store
        resources["lc_llm"] = lc_llm
    else:
        document = load_document(str(data_path))
        chunks = chunk_by_section(document, chunk_size=300, chunk_overlap=50)

        embedder = TextEmbedder()
        vectors = embedder.encode(chunks)

        store = InMemoryVectorStore()
        store.add(chunks, vectors)

        generator = TextGenerator(model_name="google/gemma-4-E2B-it")
        # generator = TextGenerator(model_name="customer_transformer")

        resources["backend"] = "rag_pipeline"
        resources["embedder"] = embedder
        resources["store"] = store
        resources["generator"] = generator

    yield  # 이 지점에서 서버가 요청을 받기 시작한다

    resources.clear()


app = FastAPI(title="DaySync RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    k: int = 3


class RetrievedChunk(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    질문을 받아 RAG 파이프라인(검색 -> prompt 조립 -> 생성)을 실행하고,
    최종 답변과 검색에 사용된 chunk 정보를 함께 반환한다.

    resources["backend"]가 "langchain"이면 8단계 chain.py의 build_rag_chain()을
    매 요청마다 만들어 호출한다 — request.k가 요청마다 달라질 수 있으므로(기존
    rag_pipeline 경로도 retrieve_top_k(..., k=request.k)처럼 매 요청 k를 그대로
    반영한다), k를 체인 생성 시점에 고정하지 않고 요청이 올 때마다 그 k로 체인을
    새로 구성한다. build_rag_chain()은 무거운 리소스(store, llm)를 새로 만들지
    않고 LCEL 그래프(RunnableParallel 등 가벼운 Python 객체)만 새로 구성하므로,
    매 요청 호출에 따른 비용은 무시할 수 있는 수준이다.
    """
    if resources.get("backend") == "langgraph":
        from langgraph_pipeline.graph import run_rag_agent

        answer = run_rag_agent(request.question, resources["lg_store"], k=request.k)
        return QueryResponse(answer=answer, retrieved_chunks=[])

    if resources.get("backend") == "langchain":
        from langchain_pipeline.chain import build_rag_chain

        chain = build_rag_chain(resources["lc_store"], resources["lc_llm"], k=request.k)
        result = chain.invoke(request.question)
        retrieved_chunks = [RetrievedChunk(**chunk) for chunk in result["retrieved_chunks"]]
        return QueryResponse(answer=result["answer"], retrieved_chunks=retrieved_chunks)

    embedder = resources["embedder"]
    store = resources["store"]
    generator = resources["generator"]

    query_vector = embedder.encode([request.question])[0]
    retrieved = retrieve_top_k(query_vector, store, k=request.k)

    prompt = build_prompt(request.question, retrieved)
    answer = generator.generate(prompt)

    retrieved_chunks = [
        RetrievedChunk(text=text, score=score) for text, score in retrieved
    ]

    return QueryResponse(answer=answer, retrieved_chunks=retrieved_chunks)


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    질문을 받아 RAG 파이프라인(검색 -> prompt 조립)을 실행한 뒤, 생성 단계만
    토큰 단위로 스트리밍하여 반환한다. Retrieval/Prompt Augmentation은
    /query와 동일하며, Generation 결과를 받는 방식만 다르다 (한 번에 vs 점진적으로).

    응답은 Server-Sent Events(SSE) 형식(text/event-stream)으로, 각 토큰을
    "data: <토큰>\\n\\n" 형태로 전송하고, 끝나면 "data: [DONE]\\n\\n"을 보낸다.

    langchain 백엔드에서는 8단계 chain.py의 build_answer_only_chain()을 쓴다 —
    기존 query_stream()과 동일하게, 이 경로의 응답에도 retrieved_chunks는
    포함하지 않는다(8단계 chain.py 모듈 docstring에서 이미 확인한 기존 동작의
    의도된 비대칭).
    """
    if resources.get("backend") == "langgraph":
        from langgraph_pipeline.graph import stream_rag_agent

        def event_stream() -> Iterator[str]:
            for token in stream_rag_agent(
                request.question, resources["lg_store"], k=request.k
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if resources.get("backend") == "langchain":
        from langchain_pipeline.chain import build_answer_only_chain

        chain = build_answer_only_chain(resources["lc_store"], resources["lc_llm"], k=request.k)

        def event_stream() -> Iterator[str]:
            for token in chain.stream(request.question):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    embedder = resources["embedder"]
    store = resources["store"]
    generator = resources["generator"]

    query_vector = embedder.encode([request.question])[0]
    retrieved = retrieve_top_k(query_vector, store, k=request.k)
    prompt = build_prompt(request.question, retrieved)

    def event_stream() -> Iterator[str]:
        for token in generator.generate_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class AgentQueryRequest(BaseModel):
    question: str
    k: int = 3


class AgentQueryResponse(BaseModel):
    answer: str


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(request: AgentQueryRequest) -> AgentQueryResponse:
    """
    DaySync Agent 엔드포인트.

    RAG_BACKEND=langgraph이면 lg_store를, 그 외에는 별도 초기화된 agent_store를 사용한다.
    GOOGLE_API_KEY가 없으면 503을 반환한다.
    """
    from fastapi import HTTPException

    store = resources.get("lg_store") or resources.get("agent_store")
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Agent를 사용하려면 GOOGLE_API_KEY 환경 변수를 설정해야 합니다.",
        )

    from langgraph_pipeline.graph import run_rag_agent

    answer = run_rag_agent(request.question, store, k=request.k)
    return AgentQueryResponse(answer=answer)


@app.get("/health")
def health() -> dict:
    """서버가 정상 동작 중인지, Indexing이 완료되었는지, 어느 백엔드가 활성화되어
    있는지 확인하는 간단한 헬스체크 엔드포인트."""
    backend = resources.get("backend", "rag_pipeline")
    if backend in ("langchain", "langgraph"):
        store_key = "lc_store" if backend == "langchain" else "lg_store"
        lc_store = resources.get(store_key)
        # InMemoryVectorStore(langchain_core)는 len()을 직접 지원하지 않으므로,
        # 내부 딕셔너리(store.store)의 길이를 쓴다 — 4단계 test_vector_store.py에서
        # 이미 같은 방식으로 검증한 속성이다.
        indexed_chunks = len(lc_store.store) if lc_store is not None else 0
    else:
        indexed_chunks = len(resources.get("store", []))
    return {"status": "ok", "indexed_chunks": indexed_chunks, "backend": backend}
