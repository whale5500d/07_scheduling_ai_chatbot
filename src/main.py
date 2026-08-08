from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph_pipeline_2.utils.indexing import build_vector_store
from langgraph_pipeline_2.graph import graph

# 배포 파이프라인 검증용 — 이미지가 실제로 갱신되는지 확인하기 위한 식별자.
# 검증 후에는 유지해도 무방하다 (버전 확인용으로 계속 활용 가능).
APP_VERSION = "v0.0.3-pipeline-test"

# 서버 전체에서 공유할 리소스(모델, 저장소)를 담을 컨테이너.
# 전역 변수를 직접 쓰는 대신 객체 하나에 묶어, 어떤 리소스들이 공유되는지 명확히 한다.
resources: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store = build_vector_store()
    resources['vector_store'] = vector_store
    yield

    resources.clear()

app = FastAPI(title="Scheduling AI Chatbot API", lifespan=lifespan)


class QueryRequest(BaseModel):
    thread_id: str
    message: str | None = None # 사용자 메시지
    confirm: bool | None = None # interrupt() 재개 여부
    # 참고: message와 confirm은 상호 배타적(동시에 채워지지 않음).
    # 지금은 필드 2개로 단순하게 두었으나, 클라이언트가 다양해지면(자유 입력 등)
    # {"type": "message"/"confirm", "value": ...} 형태로 통합하는 방안 재검토 필요.

class QueryResponse(BaseModel):
    is_interrupted: bool # interrupt 여부
    is_finished: bool # 그래프가 END에 도달해 더 실행할 노드가 없는지 여부
    date: str | None = None # is_interrupted가 True일 때, judge_date가 확정한 날짜 (그 외는 Null)


# 현재: request-response 모델로 구현.
# 확장: 여러 사용자가 동시에 참여하거나, 서버가 클라이언트 요청 없이 먼저 메시지를 보내야 하는 시나리오가 생기면 Websocket 기반 검토 필요.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:    
    # 1. config 구성
    config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}

    # 2. invoke 분기
    # confirm이 None이 아닐 경우, interrupt 재개
    # confirm이 None일 경우, 일반 메시지
    if request.confirm is not None:
        invoked_result = graph.invoke(Command(resume=request.confirm), config=config)
    else:
        invoked_result = graph.invoke({"messages": [HumanMessage(content=request.message)]}, config=config)

    # 3. is_interrupted 판단
    KEY_INTERRUPT = "__interrupt__"
    is_interrupted = KEY_INTERRUPT in invoked_result # dict 내 특정 키 하나를 찾는 문법

    # 4. date 추출
    if is_interrupted:
        interrupt_obj = invoked_result[KEY_INTERRUPT][0]
        raw_date = interrupt_obj.value["date"]
        date = raw_date.strftime("%Y-%m-%d") if raw_date is not None else None
    else:
        date = None

    # 5. is_finished 판단
    state_snapshot = graph.get_state(config)
    is_finished = not state_snapshot.next

    # 6. QueryResponse 반환
    return QueryResponse(
                is_interrupted=is_interrupted,
                is_finished=is_finished,
                date=date,
            )
    


# @app.post("/query/stream")
# def query_stream(request: QueryRequest) -> StreamingResponse:
#     """
#     질문을 받아 RAG 파이프라인(검색 -> prompt 조립)을 실행한 뒤, 생성 단계만
#     토큰 단위로 스트리밍하여 반환한다. Retrieval/Prompt Augmentation은
#     /query와 동일하며, Generation 결과를 받는 방식만 다르다 (한 번에 vs 점진적으로).

#     응답은 Server-Sent Events(SSE) 형식(text/event-stream)으로, 각 토큰을
#     "data: <토큰>\\n\\n" 형태로 전송하고, 끝나면 "data: [DONE]\\n\\n"을 보낸다.

#     langchain 백엔드에서는 8단계 chain.py의 build_answer_only_chain()을 쓴다 —
#     기존 query_stream()과 동일하게, 이 경로의 응답에도 retrieved_chunks는
#     포함하지 않는다(8단계 chain.py 모듈 docstring에서 이미 확인한 기존 동작의
#     의도된 비대칭).
#     """
#     if resources.get("backend") == "langgraph":
#         from langgraph_pipeline.graph import stream_rag_agent

#         def event_stream() -> Iterator[str]:
#             for token in stream_rag_agent(
#                 request.question, resources["lg_store"], k=request.k
#             ):
#                 yield f"data: {token}\n\n"
#             yield "data: [DONE]\n\n"

#         return StreamingResponse(event_stream(), media_type="text/event-stream")

#     if resources.get("backend") == "langchain":
#         from langchain_pipeline.chain import build_answer_only_chain

#         chain = build_answer_only_chain(resources["lc_store"], resources["lc_llm"], k=request.k)

#         def event_stream() -> Iterator[str]:
#             for token in chain.stream(request.question):
#                 yield f"data: {token}\n\n"
#             yield "data: [DONE]\n\n"

#         return StreamingResponse(event_stream(), media_type="text/event-stream")

#     embedder = resources["embedder"]
#     store = resources["store"]
#     generator = resources["generator"]

#     query_vector = embedder.encode([request.question])[0]
#     retrieved = retrieve_top_k(query_vector, store, k=request.k)
#     prompt = build_prompt(request.question, retrieved)

#     def event_stream() -> Iterator[str]:
#         for token in generator.generate_stream(prompt):
#             yield f"data: {token}\n\n"
#         yield "data: [DONE]\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    return {"status": "ok", "indexed_chunks": indexed_chunks, "backend": backend, "version": APP_VERSION}
