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
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_pipeline.chunker import chunk_by_section
from rag_pipeline.document_loader import load_document
from rag_pipeline.embedder import TextEmbedder
from model.generator import TextGenerator
from model.prompt_builder import build_prompt
from rag_pipeline.retriever import retrieve_top_k
from rag_pipeline.vector_store import InMemoryVectorStore

# 서버 전체에서 공유할 리소스(모델, 저장소)를 담을 컨테이너.
# 전역 변수를 직접 쓰는 대신 객체 하나에 묶어, 어떤 리소스들이 공유되는지 명확히 한다.
resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 1회 실행: 문서 로딩 -> 청킹 -> 임베딩 -> 저장,
    그리고 Generation에 사용할 모델까지 미리 로딩해 둔다.
    서버 종료 시 별도로 정리할 리소스는 없다.
    """
    data_path = Path(__file__).resolve().parent.parent / "data" / "nimbusflow_manual.md"
    document = load_document(str(data_path))
    chunks = chunk_by_section(document, chunk_size=300, chunk_overlap=50)

    embedder = TextEmbedder()
    vectors = embedder.encode(chunks)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    generator = TextGenerator()

    resources["embedder"] = embedder
    resources["store"] = store
    resources["generator"] = generator

    yield  # 이 지점에서 서버가 요청을 받기 시작한다

    resources.clear()


app = FastAPI(title="NimbusFlow RAG API", lifespan=lifespan)


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
    """
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
    """
    embedder = resources["embedder"]
    store = resources["store"]
    generator = resources["generator"]

    query_vector = embedder.encode([request.question])[0]
    retrieved = retrieve_top_k(query_vector, store, k=request.k)
    prompt = build_prompt(request.question, retrieved)

    def event_stream():
        for token in generator.generate_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    """서버가 정상 동작 중인지, Indexing이 완료되었는지 확인하는 간단한 헬스체크 엔드포인트."""
    return {"status": "ok", "indexed_chunks": len(resources.get("store", []))}