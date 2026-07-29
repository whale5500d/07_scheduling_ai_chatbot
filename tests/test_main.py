"""
Test for src/main.py — FastAPI 엔드포인트 + 9단계 백엔드 분기(RAG_BACKEND)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 9단계: "기존 엔드포인트 테스트에
분기 케이스 추가"):
1. _use_langchain_backend(): 환경 변수 RAG_BACKEND 값에 따라 올바르게 True/False를
   반환하는가 (미설정/대소문자 변형/무관한 값 포함)
2. /query, /query/stream, /health 각 엔드포인트가 resources["backend"]에 따라
   올바른 경로(기존 rag_pipeline 절차 vs 8단계 chain.py의 invoke()/stream())로
   분기하는가
3. 두 백엔드 모두에서 /query 응답이 동일한 스키마(QueryResponse)를 만족하는가
4. langchain 백엔드의 /query/stream 응답이 기존과 동일하게 retrieved_chunks를
   포함하지 않는가 (토큰만 SSE로 전송되는 기존 비대칭 유지 확인)
5. RAG_BACKEND를 설정하지 않았을 때(기본값), 기존 동작과 동일하게 rag_pipeline
   경로로 떨어지는가 (이번 분기 추가가 기존 동작을 깨지 않았는지의 핵심 회귀 검증)

[테스트 전략 — 왜 lifespan을 직접 실행하지 않는가]
실제 lifespan()은 두 백엔드 모두 무거운 실제 모델(rag_pipeline: sentence-transformers
+ transformers/torch, langchain_pipeline: 동일)을 로딩한다. 이 sandbox에는 해당
라이브러리들이 없으므로(6·7단계에서 이미 확인됨) lifespan()을 그대로 실행하는
end-to-end 테스트는 할 수 없다. 대신 main.query()/main.query_stream()/main.health()는
모두 평범한 Python 함수이고, 전역 딕셔너리 resources에서 필요한 리소스를 꺼내 쓸
뿐이다 — 따라서 lifespan()을 거치지 않고 resources를 테스트에서 직접 채운 뒤,
엔드포인트 함수를 직접 호출해서 "9단계에서 새로 추가한 분기 코드" 자체를
검증한다. TestClient/HTTP 계층 자체는 FastAPI/Starlette가 이미 검증하는 부분이므로
다시 검증하지 않는다.

각 백엔드의 "무거운 모델" 경계만 가짜로 대체한다:
- rag_pipeline 분기: rag_pipeline.vector_store.InMemoryVectorStore(순수 numpy,
  무거운 의존성 없음)는 실제 객체를 쓰고, embedder/generator만 duck-typed fake로
  대체한다.
- langchain 분기: langchain_pipeline.vector_store.build_vector_store +
  DeterministicFakeEmbedding(4단계에서 이미 검증된 패턴)은 실제로 쓰고,
  llm만 langchain_core.language_models.fake의 FakeListLLM/FakeStreamingListLLM로
  대체한다(8단계 test_chain.py와 동일한 패턴).

[StreamingResponse.body_iterator를 비동기로 모으는 이유]
main.py의 event_stream()은 동기(sync) generator다. Starlette의 StreamingResponse는
동기 generator를 받으면 내부적으로 iterate_in_threadpool()로 감싸 "비동기 이터레이터"로
변환한다(공식 소스코드로 직접 확인함) — 그래서 body_iterator를 그냥 list()로 모을 수
없고, asyncio.run()으로 비동기 순회해야 한다. _collect_sse_events() 헬퍼가 이를 처리한다.
"""

import asyncio

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM, FakeStreamingListLLM

import main
from main import APP_VERSION
from langchain_pipeline.vector_store import build_vector_store
from rag_pipeline.vector_store import InMemoryVectorStore as LegacyInMemoryVectorStore


def _collect_sse_events(streaming_response) -> list[str]:
    """StreamingResponse.body_iterator(비동기 이터레이터)를 동기 list로 모은다."""

    async def _collect() -> list[str]:
        events = []
        async for chunk in streaming_response.body_iterator:
            events.append(chunk)
        return events

    return asyncio.run(_collect())


@pytest.fixture(autouse=True)
def clear_resources():
    """매 테스트 전후로 main.resources를 깨끗하게 비운다 (전역 상태 오염 방지)."""
    main.resources.clear()
    yield
    main.resources.clear()


@pytest.fixture(autouse=True)
def clear_rag_backend_env(monkeypatch):
    """매 테스트 전 RAG_BACKEND 환경 변수를 제거해, 테스트 간 환경이 새지 않게 한다."""
    monkeypatch.delenv(main.RAG_BACKEND_ENV_VAR, raising=False)


class TestUseLangchainBackend:
    """_use_langchain_backend() — 환경 변수 분기 조건 자체에 대한 단위 테스트."""

    def test_returns_false_when_env_var_not_set(self):
        assert main._use_langchain_backend() is False

    def test_returns_true_when_env_var_is_langchain(self, monkeypatch):
        monkeypatch.setenv(main.RAG_BACKEND_ENV_VAR, "langchain")
        assert main._use_langchain_backend() is True

    def test_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(main.RAG_BACKEND_ENV_VAR, "LangChain")
        assert main._use_langchain_backend() is True

    def test_returns_false_for_unrelated_value(self, monkeypatch):
        """기존 동작을 깨지 않기 위해, 알 수 없는 값은 기본(rag_pipeline)으로 떨어져야 한다."""
        monkeypatch.setenv(main.RAG_BACKEND_ENV_VAR, "rag_pipeline")
        assert main._use_langchain_backend() is False


class _FakeEmbedder:
    """rag_pipeline.embedder.TextEmbedder의 duck-typed fake.

    실제 모델 없이, 텍스트 길이/공백 개수를 기반으로 한 결정론적 저차원 벡터를
    만든다 — 의미적 유사도는 보장하지 않지만(4단계 DeterministicFakeEmbedding과
    동일한 한계), retrieve_top_k()가 실제로 동작하는지(저장/검색 메커니즘)만
    확인하는 이 테스트의 목적에는 충분하다.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t)), float(t.count(" "))] for t in texts])


class _FakeGenerator:
    """rag_pipeline.generator.TextGenerator의 duck-typed fake."""

    def __init__(self, answer: str = "rag_pipeline 가짜 답변"):
        self.answer = answer
        self.received_prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        self.received_prompts.append(prompt)
        return self.answer

    def generate_stream(self, prompt: str, max_new_tokens: int = 80, fake_stream_delay: float = 0.0):
        self.received_prompts.append(prompt)
        yield from self.answer.split(" ")


@pytest.fixture
def legacy_resources():
    """resources를 rag_pipeline(기본) 백엔드로 직접 채운다 (lifespan을 거치지 않음)."""
    embedder = _FakeEmbedder()
    chunks = ["The default API port is 8842.", "Conflicts are resolved by priority."]
    vectors = embedder.encode(chunks)

    store = LegacyInMemoryVectorStore()
    store.add(chunks, vectors)

    generator = _FakeGenerator()

    main.resources["backend"] = "rag_pipeline"
    main.resources["embedder"] = embedder
    main.resources["store"] = store
    main.resources["generator"] = generator
    return generator


@pytest.fixture
def langchain_resources():
    """resources를 langchain_pipeline 백엔드로 직접 채운다 (lifespan을 거치지 않음)."""
    fake_embedding = DeterministicFakeEmbedding(size=8)
    documents = [
        Document(page_content="The default API port is 8842."),
        Document(page_content="Conflicts are resolved by priority."),
    ]
    store = build_vector_store(documents, fake_embedding)

    main.resources["backend"] = "langchain"
    main.resources["lc_store"] = store
    main.resources["lc_llm"] = FakeListLLM(responses=["langchain 가짜 답변"])
    return store


class TestQueryEndpointBranching:
    """POST /query (main.query()) — 백엔드별 분기 검증."""

    def test_rag_pipeline_backend_returns_query_response(self, legacy_resources):
        request = main.QueryRequest(question="API 포트가 뭐야?", k=2)

        response = main.query(request)

        assert isinstance(response, main.QueryResponse)
        assert response.answer == "rag_pipeline 가짜 답변"
        assert len(response.retrieved_chunks) == 2
        assert all(isinstance(c.score, float) for c in response.retrieved_chunks)

    def test_langchain_backend_returns_query_response(self, langchain_resources):
        request = main.QueryRequest(question="API 포트가 뭐야?", k=2)

        response = main.query(request)

        assert isinstance(response, main.QueryResponse)
        assert response.answer == "langchain 가짜 답변"
        assert len(response.retrieved_chunks) == 2
        assert all(isinstance(c.score, float) for c in response.retrieved_chunks)

    def test_langchain_backend_respects_per_request_k(self, langchain_resources):
        """k가 lifespan 시점에 고정되지 않고, 매 요청의 request.k를 그대로 따라야 한다
        (build_rag_chain()을 요청마다 새로 구성하는 설계의 검증 — main.py docstring 참고)."""
        request = main.QueryRequest(question="아무 질문", k=1)

        response = main.query(request)

        assert len(response.retrieved_chunks) == 1


class TestQueryStreamEndpointBranching:
    """POST /query/stream (main.query_stream()) — 백엔드별 분기 검증."""

    def test_rag_pipeline_backend_streams_tokens_ending_with_done(self, legacy_resources):
        request = main.QueryRequest(question="API 포트가 뭐야?", k=2)

        response = main.query_stream(request)
        events = _collect_sse_events(response)

        assert events[-1] == "data: [DONE]\n\n"
        joined = "".join(e[len("data: ") : -len("\n\n")] for e in events[:-1])
        assert joined == "rag_pipeline가짜답변"  # 단어 단위로 yield된 토큰들을 그대로 이어붙인 결과

    def test_langchain_backend_streams_tokens_ending_with_done(self, langchain_resources):
        # FakeListLLM은 진짜 스트리밍(여러 청크)을 지원하지 않으므로(8단계 test_chain.py와
        # 동일한 이유), 스트리밍 검증에는 FakeStreamingListLLM을 쓴다.
        main.resources["lc_llm"] = FakeStreamingListLLM(responses=["langchain스트리밍답변"])
        request = main.QueryRequest(question="API 포트가 뭐야?", k=2)

        response = main.query_stream(request)
        events = _collect_sse_events(response)

        assert events[-1] == "data: [DONE]\n\n"
        assert len(events) > 2  # [DONE] 제외하고도 여러 토큰으로 나뉘어 와야 한다 (진짜 스트리밍 확인)
        joined = "".join(e[len("data: ") : -len("\n\n")] for e in events[:-1])
        assert joined == "langchain스트리밍답변"

    def test_langchain_backend_stream_response_excludes_retrieved_chunks(self, langchain_resources):
        """기존 query_stream()과 동일하게, SSE 응답에는 retrieved_chunks가 전혀 포함되지
        않아야 한다(8단계 chain.py 모듈 docstring에서 이미 확인한 의도된 비대칭)."""
        main.resources["lc_llm"] = FakeStreamingListLLM(responses=["langchain스트리밍답변"])
        request = main.QueryRequest(question="API 포트가 뭐야?", k=2)

        response = main.query_stream(request)
        full_body = "".join(_collect_sse_events(response))

        assert "score" not in full_body
        assert "retrieved_chunks" not in full_body


class TestHealthEndpointBranching:
    """GET /health (main.health()) — 백엔드별 분기 검증."""

    def test_rag_pipeline_backend_reports_indexed_chunks_and_backend_name(self, legacy_resources):
        result = main.health()

        assert result == {"status": "ok", "indexed_chunks": 2, "backend": "rag_pipeline", "version": APP_VERSION}

    def test_langchain_backend_reports_indexed_chunks_and_backend_name(self, langchain_resources):
        result = main.health()

        assert result == {"status": "ok", "indexed_chunks": 2, "backend": "langchain", "version": APP_VERSION}

    def test_health_before_lifespan_runs_does_not_raise(self):
        """resources가 비어 있어도(lifespan이 아직 실행되지 않은 상태) 예외 없이
        기본값(rag_pipeline, 0개)을 반환해야 한다 — 기존 동작과 동일."""
        result = main.health()

        assert result == {"status": "ok", "indexed_chunks": 0, "backend": "rag_pipeline", "version": APP_VERSION}
