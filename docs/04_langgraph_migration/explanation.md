# RAG 파이프라인의 LangGraph 마이그레이션 계획

> `src/langchain_pipeline/`(LCEL 기반 RAG 파이프라인)의 오케스트레이션 계층을
> LangGraph `StateGraph`로 전환하는 설계 문서.
> 실제 구현은 이 문서의 결정 사항을 기반으로 완료되었다 (`feat/langgraph-pipeline` 브랜치).

## 0. 전제와 범위

- `src/rag_pipeline/`, `src/langchain_pipeline/`은 수정하지 않고 기준선(baseline)으로 유지한다.
- 로더·청커·임베딩·벡터스토어·프롬프트·LLM 등 **개별 컴포넌트 모듈은 재사용**한다. `langchain_pipeline/chain.py`(LCEL 체인 조립)에 해당하는 오케스트레이션 계층만 교체한다.
- 신규 코드는 `src/langgraph_pipeline/` 패키지에 작성한다.
- `tests/`는 `src/`와 1:1 미러링되는 기존 관례를 따른다.

## 1. 결정된 사항

**표 1. 마이그레이션 범위 결정 사항**
| # | 항목 | 결정 | 근거 |
|---|---|---|---|
| 1 | 재사용 범위 | `langchain_pipeline/{loader,splitter,embedding,vector_store,prompt,llm}.py`는 그대로 사용. `chain.py`만 `graph.py`로 교체 | LangGraph 마이그레이션은 "어떻게 조립하는가"의 변경이지, "무엇을 쓰는가"의 변경이 아니다 |
| 2 | 패키지명 | `langgraph_pipeline`으로 확정 | `langgraph` PyPI 패키지와의 shadowing 회피. `langchain_pipeline`과의 명확한 구분 |
| 3 | FastAPI 통합 방식 | 신규 엔트리포인트 없이 기존 `main.py`에 `RAG_BACKEND=langgraph` 분기 추가 | `03_LANGCHAIN_MIGRATION_PLAN.md` 표 1 #9와 동일한 원칙 |
| 4 | 빈 검색 결과 처리 | 조건부 라우팅(`no_results` 노드)으로 폴백 — LLM 미호출, 고정 메시지 반환 | LCEL 버전은 `format_docs([])` → `ValueError`를 던졌다. StateGraph에서는 이 경우를 명시적 분기로 처리하는 것이 자연스럽고, LLM 호출 비용도 절약된다 |
| 5 | 스트리밍 구현 방식 | `graph.stream(stream_mode="updates")`로 retrieve 완료 시점을 포착한 뒤, `llm.stream()`으로 직접 토큰 yield하는 혼합 방식 | LangGraph의 `stream_mode="messages"`는 `ChatModel`(AIMessage 반환)에서만 토큰 단위 스트리밍을 지원한다. 이 프로젝트의 LLM은 `BaseLLM`(`HuggingFacePipeline`, `CustomTransformerLLM`)이므로, `stream_mode="messages"`는 이 컨텍스트에서 사용 불가 |
| 6 | 체크포인팅(Checkpointing) | 이번 범위 제외 | 현재 구조는 단발성 질의응답이며, 대화 히스토리 보존이 필요한 멀티턴 시나리오가 없다 |
| 7 | GraphRAG | 이번 범위 제외 | `03_LANGCHAIN_MIGRATION_PLAN.md` 표 1 #2와 동일 |

이후 모든 절은 표 1의 결정을 전제로 한다.

## 2. 전체 구조

### 2.1 디렉터리 구조

```text
src/
├── rag_pipeline/            # 변경 없음 (기준선)
├── langchain_pipeline/      # 변경 없음 (기준선)
│   ├── loader.py            # ┐
│   ├── splitter.py          # │ langgraph_pipeline이 그대로 재사용하는 모듈들
│   ├── embedding.py         # │
│   ├── vector_store.py      # │
│   ├── prompt.py            # │
│   ├── llm.py               # ┘
│   └── chain.py             # LCEL 체인 조립 — langgraph_pipeline/graph.py로 교체됨
└── langgraph_pipeline/      # 신규 — chain.py에 해당하는 오케스트레이션만 교체
    ├── __init__.py
    ├── state.py             # RAGState TypedDict 정의
    ├── nodes.py             # 노드 함수 정의 (retrieve, generate, no_results)
    └── graph.py             # StateGraph 조립 + stream_rag_answer()

tests/
├── langchain_pipeline/      # 변경 없음
└── langgraph_pipeline/      # 신규
    ├── test_nodes.py
    └── test_graph.py
```

### 2.2 패키지명: `langgraph_pipeline` (확정)

`src/` 아래의 모든 폴더는 `pyproject.toml`의 `[tool.setuptools.packages.find] where = ["src"]`
설정에 의해 최상위 패키지로 import된다. `src/langgraph/`로 만들면
`import langgraph`가 PyPI의 실제 LangGraph 패키지를 가리키는 대신 이 폴더를 shadowing하게 된다.
패키지명은 `langgraph_pipeline`으로 확정한다 (표 1 #2).

## 3. 모듈 대응 매핑

**표 2. `langchain_pipeline` → LangGraph 대응 매핑**
| 기존 모듈 | 책임 | LangGraph 대응 | 변경 여부 |
|---|---|---|---|
| `loader.py` | 파일 → Document 리스트 | 재사용 | 변경 없음 |
| `splitter.py` | Document → 청크 분할 | 재사용 | 변경 없음 |
| `embedding.py` | HuggingFace 임베딩 모델 | 재사용 | 변경 없음 |
| `vector_store.py` | InMemoryVectorStore 구축 + retriever | 재사용 | 변경 없음 |
| `prompt.py` | ChatPromptTemplate 조립 | 재사용 | 변경 없음 |
| `llm.py` | Gemma / CustomTransformerLLM | 재사용 | 변경 없음 |
| `chain.py`의 LCEL 파이프 조립 | `retrieve \| prompt \| llm` | `state.py` + `nodes.py` + `graph.py`의 StateGraph로 교체 | **신규 작성** |

## 4. 핵심 아키텍처 차이 — LCEL vs StateGraph

**4.1 상태(State)의 암묵적 vs 명시적 표현**

LCEL에서는 단계 간에 전달되는 데이터가 코드에 명시되지 않는다. 파이프(`|`) 연산자가
앞 단계의 반환값을 다음 단계의 입력으로 그대로 흘려보낼 뿐이다.

```python
# LCEL — 상태가 암묵적 (파이프로 흐른다)
RunnableParallel(retrieved=retrieve, question=passthrough) | generate | llm
```

StateGraph에서는 모든 단계가 공유하는 데이터를 `TypedDict`로 명시적으로 선언한다.

```python
# StateGraph — 상태가 명시적 (TypedDict로 선언)
class RAGState(TypedDict):
    question: str
    retrieved: list[tuple[Document, float]]
    answer: str
```

이 차이는 단순한 코드 스타일의 차이가 아니다. 상태를 명시적으로 갖기 때문에
각 노드의 완료 시점에 상태를 검사하거나, 이후 설명할 조건부 라우팅을 구현하거나,
체크포인팅으로 실행을 재개하는 것이 가능해진다.

**4.2 조건부 라우팅 — StateGraph 마이그레이션의 핵심 이점**

LCEL 체인은 선형 파이프이다. 검색 결과가 없을 때 다른 경로로 분기할 방법이 없어,
`format_docs([])`가 `ValueError`를 던지는 방식으로 실패했다.

StateGraph는 노드 완료 후 **다음 노드를 함수로 결정**할 수 있다.

```python
def _route_after_retrieve(state: RAGState) -> str:
    return "generate" if state["retrieved"] else "no_results"

graph.add_conditional_edges("retrieve", _route_after_retrieve, {...})
```

실행 흐름:

```
START → [retrieve]
             ↓ _route_after_retrieve()
             ├── retrieved 있음 → [generate] → END   (LLM 호출)
             └── retrieved 없음 → [no_results] → END  (LLM 미호출, 폴백 메시지)
```

LCEL에서 이와 동일한 분기를 구현하려면 `RunnableLambda`로 분기 로직을 직접 구현하거나
`RunnableBranch`를 사용해야 하며, 어느 쪽이든 체인 구조가 복잡해진다. StateGraph에서는
`add_conditional_edges()` 하나로 표현된다.

**4.3 노드(Node)와 체인(Chain)의 책임 분리**

LCEL `chain.py`는 컴포넌트 조립과 실행 로직이 하나의 함수 안에 섞인다.

```python
# chain.py — 조립과 실행 로직이 뒤섞임
def build_rag_chain(store, llm, k):
    def _retrieve(question):  # 실행 로직
        return store.similarity_search_with_score(question, k=k)
    def _to_prompt_text(inputs):  # 실행 로직
        ...
    return RunnableParallel(...) | RunnableParallel(...)  # 조립
```

LangGraph에서는 **노드 함수(실행 로직)**와 **그래프(조립)**를 파일 단위로 분리한다.

```python
# nodes.py — 실행 로직만 (독립 테스트 가능)
def make_retrieve_node(store, k): ...
def make_generate_node(llm): ...
def no_results_node(state): ...

# graph.py — 조립만
graph.add_node("retrieve", make_retrieve_node(store, k))
graph.add_conditional_edges(...)
```

노드 함수는 `(state: RAGState) -> dict` 시그니처를 가지는 순수 함수이므로,
그래프를 띄우지 않고 단위 테스트로 독립적으로 검증할 수 있다.

**4.4 스트리밍 동작의 차이 — [결정: 혼합 방식]**

LangGraph의 스트리밍 모드는 세 가지다.

**표 3. LangGraph stream_mode별 동작과 이 프로젝트에서의 적합성**
| `stream_mode` | yield하는 것 | 이 프로젝트에서의 적합성 |
|---|---|---|
| `"values"` | 노드 완료마다 전체 state | 각 노드가 끝날 때마다 RAGState 전체를 받는다. 토큰 단위 스트리밍 불가 |
| `"updates"` | 노드 완료마다 `{node_name: state_update}` | retrieve 노드가 끝나는 시점을 포착할 수 있다 → **혼합 방식 구현에 사용** |
| `"messages"` | LLM이 생성하는 토큰 단위 AIMessageChunk | `ChatModel`(`ChatOpenAI` 등)에서만 동작. `BaseLLM`(`HuggingFacePipeline`, `CustomTransformerLLM`)에서는 토큰이 아닌 전체 응답이 한 번에 나온다 → **이 프로젝트에서 사용 불가** |

결정(표 1 #5): `stream_mode="updates"`로 retrieve 노드의 완료 시점을 포착한 뒤,
`llm.stream()`을 직접 호출하는 혼합 방식으로 구현한다.

```python
def stream_rag_answer(question, store, llm, k):
    for update in graph.stream(initial_state, stream_mode="updates"):
        if "retrieve" in update:
            retrieved = update["retrieve"]["retrieved"]
            break  # retrieve 완료 즉시 → generate 노드는 실행하지 않음
    # 이후 llm.stream()으로 직접 토큰 yield
    for token in llm.stream(prompt_text):
        yield token
```

이 방식은 LangGraph가 온전히 스트리밍을 처리하지 않는다는 한계가 있다 (§5 참고).
향후 `BaseLLM`을 `ChatModel`로 교체하면 `stream_mode="messages"`로 전환해
그래프 내부에서 토큰 스트리밍을 완결할 수 있다.

## 5. 한계점 및 향후 확장 방향

**표 4. 현재 구현의 한계와 향후 확장 방향**
| 한계 | 원인 | 향후 확장 방향 |
|---|---|---|
| 스트리밍이 그래프 외부에서 처리됨 | `BaseLLM`은 `stream_mode="messages"` 미지원 | LLM을 `ChatModel`(`ChatGoogleGenerativeAI` 등)로 교체 → `stream_mode="messages"` 전환 |
| 매 요청마다 그래프를 새로 컴파일 | LCEL `build_rag_chain()`의 패턴 승계. `k`가 요청마다 달라지는 구조 | `k`를 서버 시작 시 고정하거나, `k`를 state에 포함시켜 그래프를 1회만 컴파일 |
| 체크포인팅 미사용 | 단발성 질의응답이라 불필요 | 멀티턴 대화 시 `MemorySaver` 등 Checkpointer 주입으로 대화 히스토리 보존 가능 |
| 단순 선형 파이프라인 | 현재 데이터와 요구사항이 단순함 | 검색 품질 평가 후 재검색(Self-RAG), 다중 문서 소스, Human-in-the-loop 등 복잡한 흐름으로 확장 가능 |

## 6. 단계별 구현 순서

**표 5. 단계별 구현 순서 (구현 코드 + 테스트 코드 동시 작성)**
| 단계 | 대상 (`langgraph_pipeline/`) | 구현 내용 | 테스트 (`tests/langgraph_pipeline/`) | 선행 단계 |
|---|---|---|---|---|
| 1 | `state.py` | `RAGState` TypedDict 선언. `question`, `retrieved: list[tuple[Document, float]]`, `answer` 3개 필드 | 별도 테스트 파일 없음 — `test_nodes.py`, `test_graph.py`에서 fixture로 사용하며 간접 검증 | 없음 |
| 2 | `nodes.py` | `make_retrieve_node(store, k)`, `make_generate_node(llm)`, `no_results_node` 구현. 각 노드는 `(state) -> dict` 시그니처, 담당 필드만 반환 | `test_nodes.py` — `DeterministicFakeEmbedding` + `FakeListLLM` + echo `RunnableLambda`로 각 노드의 반환 형태, 필드 격리, 빈 store 처리를 독립적으로 검증 | 1 |
| 3 | `graph.py` (`build_rag_graph`) | `StateGraph(RAGState)`에 노드 3개(retrieve, generate, no_results) 등록. `add_conditional_edges`로 `_route_after_retrieve` 조건부 라우팅 연결. `graph.compile()` 반환 | `test_graph.py::TestBuildRagGraph` — `invoke()` 결과 형태, k 제한, 빈 store에서 no_results 분기(LLM 미호출 확인), echo LLM으로 prompt 내용 검증 | 2 |
| 4 | `graph.py` (`stream_rag_answer`) | `graph.stream(stream_mode="updates")`로 retrieve 완료 포착 → `llm.stream()`으로 직접 yield하는 혼합 스트리밍 함수 구현 | `test_graph.py::TestStreamRagAnswer` — `FakeStreamingListLLM`으로 다중 청크 yield 검증, 빈 store에서 `NO_RESULTS_ANSWER` 단일 yield + llm.stream 미호출 검증 | 3 |
| 5 | `main.py` 분기 추가 | `_use_langgraph_backend()` 함수 추가. `lifespan`, `/query`, `/query/stream`, `/health`에 `RAG_BACKEND=langgraph` 분기 추가 | 기존 `test_main.py`의 langchain 분기 테스트 패턴을 참고해 langgraph 분기 케이스 추가 | 3, 4 |

1~4는 `langchain_pipeline/`에 의존하는 컴포넌트를 재사용하므로 `03_LANGCHAIN_MIGRATION_PLAN.md`의 1~7단계 완료를 전제로 한다.

## 7. 신규 의존성

**표 6. `requirements.txt`에 추가한 패키지**
| 패키지 | 용도 |
|---|---|
| `langgraph` | `StateGraph`, `START`, `END`, `CompiledStateGraph`. 노드 등록·엣지 연결·컴파일, `graph.invoke()`/`graph.stream()` |

기존 `langchain-core`, `langchain-huggingface`, `langsmith` 등은 이미 설치되어 있으며 변경 없다.

## 8. 범위 제외 항목 (표 1 근거)

- **GraphRAG** (`graph_extractor.py`, `graph_retriever.py`): 이번 범위 제외 (표 1 #7)
- **체크포인팅**: 단발성 질의응답 구조에서 불필요. 멀티턴 요구 발생 시 추가 (표 1 #6)
- **LangSmith 평가 연동**: `langchain_pipeline` 기준선 평가는 `tests/evaluate/langsmith_eval.py`에서 이미 처리. `langgraph_pipeline`의 별도 평가 스크립트는 추후 과제

## 9. LCEL chain.py → StateGraph 전환 결정 사항 요약

- **(a) 재사용 범위**: 컴포넌트 6개(loader, splitter, embedding, vector_store, prompt, llm)는 그대로 사용. `chain.py`의 오케스트레이션만 교체 (표 1 #1)
- **(b) 빈 검색 결과 처리**: `no_results` 노드로 폴백 — `ValueError` 전파 대신 명시적 분기 (표 1 #4)
- **(c) 스트리밍**: `stream_mode="updates"` + `llm.stream()` 혼합 방식. `BaseLLM`의 `stream_mode="messages"` 미지원이 원인 (표 1 #5, §4.4)
- **(d) 패키지명**: `langgraph_pipeline` (표 1 #2, §2.2)
- **(e) FastAPI 통합**: 기존 `main.py`에 `RAG_BACKEND=langgraph` 분기 추가 (표 1 #3)

## 출처

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph — StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)
- [LangGraph — stream() 메서드 및 stream_mode](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [LangGraph — How to stream from your graph](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [LangGraph — Add node retry policies](https://langchain-ai.github.io/langgraph/how-tos/node-retries/)
- [LangGraph — Persistence / Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangChain — BaseLLM (stream 기본 동작)](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html)
- [LangChain — DeterministicFakeEmbedding](https://reference.langchain.com/python/langchain-core/embeddings/fake/DeterministicFakeEmbedding)
