# RAG 파이프라인의 LangChain 마이그레이션 계획

> 코드 작성 전 단계의 계획 문서. `src/rag_pipeline/`(현재 직접 구현한 RAG 파이프라인)의
> 각 모듈을 LangChain 표준 컴포넌트로 옮기는 전략을 다룬다. 실제 구현은 이 문서의
> 결정 사항을 확인한 뒤 진행한다.

## 0. 전제와 범위

- `src/rag_pipeline/`, `src/custom_transformer/`는 수정하지 않고 기준선(baseline)으로 유지한다.
- 신규 코드는 `langchain_pipeline` 패키지에 모듈별로 점진적으로 작성한다 (§2.2).
- `tests/`는 `src/`와 1:1 미러링되어 있는 기존 관례를 그대로 따른다.

## 1. 결정된 사항

**표 1. 마이그레이션 범위 결정 사항**
| # | 항목 | 결정 | 근거 |
|---|---|---|---|
| 1 | `custom_transformer` 처리 | LangChain의 LLM(언어 모델) 계층으로 wrapping하고, Gemma 4 E2B-it과 스위처블(switchable) 구조 유지 | 기존 `generator.py`의 `TextGenerator(model_name=...)` 스위치 설계를 그대로 계승 |
| 2 | GraphRAG(`graph_extractor.py`, `graph_retriever.py`) | 이번 범위 제외 | 현행 구현 그대로 유지, 추후 별도 작업 |
| 3 | 전환 방식 | `rag_pipeline/`는 그대로 두고, 신규 패키지에 모듈별로 점진적으로 작성 | 비교 가능한 기준선 보존 |
| 4 | 평가 코드(`tests/evaluate/*`, RAGAS 스타일) | 기존 평가 코드는 `rag_pipeline` 기준선용으로 보존, `langchain_pipeline` 평가는 LangSmith Tracing + Dataset 기반으로 전환 (§6 10단계) | LangGraph는 범위 밖, LangSmith는 활성화 |
| 5 | 저장-검색 결합 구조 (§4.1) | 절충안 없이 LangChain의 결합된 `VectorStore` 인터페이스를 그대로 채택 | thin wrapper는 다른 VectorStore 구현체(FAISS, Chroma 등)로 교체할 때 호환성을 해칠 수 있음 |
| 6 | 임베딩 함수 소유권 이전 (§4.2) | 절충안 없이 LangChain 방식(Embeddings를 VectorStore 생성자에 주입) 채택 | 표 1 #5와 동일한 원칙(절충 없이 LangChain 방식 사용) 적용 |
| 7 | RAG 체인 패턴 (§5) | LCEL Runnable 직접 조립 방식 채택. `create_agent` 기반 RAG Agent는 2차 확장 과제로 보류 | `custom_transformer`가 tool-calling 미지원, 검색 1회·생성 1회의 단일 패스 구조로 충분 |
| 8 | 패키지명 (§2.2) | `langchain_pipeline`으로 확정 | PyPI `langchain` 패키지와의 이름 충돌(shadowing) 회피 |
| 9 | FastAPI 통합 방식 (§6 9단계) | 신규 엔트리포인트 분리 없이 기존 `main.py`에 분기 추가 | — |

이후 모든 절은 표 1의 결정을 전제로 한다.

## 2. 전체 구조 (Top-down)

### 2.1 디렉터리 구조

```text
src/
├── rag_pipeline/        # 변경 없음 (기준선)
├── custom_transformer/  # 변경 없음
└── langchain_pipeline/  # 신규 — rag_pipeline의 LangChain 대응 구현
    ├── __init__.py
    ├── loader.py          # document_loader.py 대응
    ├── splitter.py        # chunker.py 대응
    ├── embedding.py       # embedder.py 대응
    ├── vector_store.py    # vector_store.py + retriever.py 대응
    ├── prompt.py          # prompt_builder.py 대응
    ├── llm.py             # generator.py 대응 (Gemma + custom_transformer 스위처블)
    └── chain.py           # main.py 오케스트레이션 로직 대응 (LCEL 체인 조립)

tests/
├── rag_pipeline/         # 변경 없음
└── langchain_pipeline/   # 신규, src/langchain_pipeline/과 1:1 미러링
    ├── test_loader.py
    ├── test_splitter.py
    ├── test_embedding.py
    ├── test_vector_store.py
    ├── test_prompt.py
    ├── test_llm.py
    └── test_chain.py
```

### 2.2 패키지명: `langchain_pipeline` (확정)

요청했던 폴더명은 `langchain`이었으나, 그대로 쓰면 실제 `langchain` 라이브러리와
이름이 충돌한다. `pyproject.toml`의 `[tool.setuptools.packages.find] where = ["src"]`
설정 때문에 `src/` 바로 아래의 모든 폴더는 (`rag_pipeline`, `custom_transformer`처럼)
최상위 패키지로 import된다. 즉 `src/langchain/`을 만들면 `import langchain`이
PyPI의 실제 LangChain 패키지 대신 이 폴더를 가리키게 되는 셰도잉(shadowing) 문제가
생긴다. 패키지명은 `langchain_pipeline`으로 확정한다(표 1 #8).

## 3. 모듈 대응 매핑

**표 2. `rag_pipeline` → LangChain 대응 매핑**
| 기존 모듈 | 책임 | LangChain 대응 | 패키지 | 공식 문서 |
|---|---|---|---|---|
| `document_loader.py` | 파일 → raw text | `TextLoader` | `langchain_community.document_loaders` | [TextLoader](https://reference.langchain.com/python/langchain-community/document_loaders/text/TextLoader) |
| `chunker.py::chunk_fixed_size` | 고정 길이 분할 | `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | [text splitters](https://reference.langchain.com/python/langchain-text-splitters) |
| `chunker.py::chunk_by_section` | `##` 헤더 기준 섹션 분할 | `MarkdownHeaderTextSplitter` (+ 섹션 초과 시 위 splitter로 재분할) | `langchain_text_splitters` | 위와 동일 |
| `embedder.py::TextEmbedder` | `all-MiniLM-L6-v2`로 인코딩 | `HuggingFaceEmbeddings` | `langchain_huggingface` | [HuggingFaceEmbeddings](https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html) |
| `vector_store.py::InMemoryVectorStore` | chunk·벡터 보관 | `InMemoryVectorStore` | `langchain_core.vectorstores` | [InMemoryVectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html) |
| `retriever.py::retrieve_top_k` | top-k 유사도 검색 | `vector_store.as_retriever(k=...)` / `similarity_search_with_score` | 위와 동일 클래스의 메서드 | 위와 동일 |
| `prompt_builder.py::build_prompt` | instruction + context + question 조립 | `PromptTemplate` / `ChatPromptTemplate` | `langchain_core.prompts` | [prompts 레퍼런스](https://reference.langchain.com/python/langchain-core) |
| `generator.py` (Gemma 경로) | HF causal LM 추론 | `HuggingFacePipeline` (직접 만든 `transformers.pipeline`을 감싸는 방식) | `langchain_huggingface` | [HuggingFacePipeline](https://reference.langchain.com/python/langchain-huggingface/llms/huggingface_pipeline/HuggingFacePipeline) |
| `generator.py` (custom_transformer 경로) | 직접 구현한 디코더 추론 | `LLM` 서브클래스 직접 작성 | `langchain_core.language_models.llms` | [LLM](https://reference.langchain.com/python/langchain-core/language_models/llms/LLM) |
| `main.py`의 절차적 호출 순서 | embed→retrieve→prompt→generate를 직접 호출 | LCEL(`|` 연산자)로 조립한 Runnable 체인 | `langchain_core.runnables` | [공식 RAG 튜토리얼](https://docs.langchain.com/oss/python/langchain/rag) |
| `graph_extractor.py`, `graph_retriever.py` | 그래프 기반 검색 | (제외, 표 1 #2) | — | — |
| `tests/evaluate/*` | RAGAS 스타일 평가 | (제외, 표 1 #4) | — | — |

## 4. 핵심 아키텍처 차이 — 한계점과 확장 방향

이 작업은 "0에서 1을 만드는" 작업이 아니라, 이미 동작하는 구현(1)을 표준 인터페이스로
확장하는 작업이다. 따라서 모듈별로 새로 배워야 할 개념을 늘어놓기보다, 기존 구현이
가진 한계와 LangChain이 그 한계를 어떤 방식으로 다루는지를 짝지어 정리한다.

**4.1 저장(storage)과 검색(retrieval)의 결합 — [결정: LangChain 방식 채택]**
기존 구조는 `vector_store.py`(저장)와 `retriever.py`(검색)를 책임에 따라 분리했다.
LangChain의 `VectorStore`는 `add_documents()`와 `similarity_search()`를 하나의
클래스에 묶는다. 이는 책임 분리의 후퇴가 아니라, FAISS·Chroma 등 다른 벡터 DB로
드롭인(drop-in) 교체가 가능하도록 인터페이스를 표준화한 결과다.

결정(표 1 #5): 기존의 분리된 구조를 유지하는 thin wrapper 절충안은 채택하지 않는다.
LangChain의 결합된 `VectorStore` 인터페이스를 그대로 채택하고, `langchain_pipeline/vector_store.py`
하나의 모듈이 기존 `vector_store.py` + `retriever.py`의 역할을 모두 담당한다.

**4.2 임베딩 함수의 소유권 이전 — [결정: LangChain 방식 채택]**
기존: `TextEmbedder.encode()`를 호출부에서 명시적으로 호출하고, 결과 벡터를
`store.add()`에 직접 전달한다. LangChain: `Embeddings` 객체를 `VectorStore`
생성 시점에 주입하면, 이후 `add_documents()`/`similarity_search()` 내부에서
암묵적으로 `embed_documents()`/`embed_query()`가 호출된다. 호출부 코드는
짧아지지만 "언제 임베딩이 실행되는지"가 명시적이지 않게 되는 트레이드오프가 있다.

결정(표 1 #6): 이 트레이드오프를 감수하고 LangChain 방식을 그대로 채택한다.
`langchain_pipeline/embedding.py`는 `HuggingFaceEmbeddings` 인스턴스를 생성하는
역할만 하고, `langchain_pipeline/vector_store.py`가 생성자에서 이를 주입받는다.

**4.3 모델 분기 로직의 소멸**
기존 `TextGenerator`는 `if model_name == "custom_transformer": ... else: ...`로
직접 분기하고, `generate()`/`generate_stream()` 시그니처를 손으로 통일했다.
LangChain의 모든 컴포넌트는 `Runnable` 인터페이스(`invoke`/`stream`/`batch`/
`ainvoke`)를 공통으로 구현하므로, `custom_transformer`를 `LLM` 서브클래스로,
Gemma를 `HuggingFacePipeline`으로 각각 감싸면 호출부에서는 더 이상 if/else가
필요 없다 — 어떤 `Runnable`이 들어와도 `.invoke(prompt)`로 동일하게 호출된다.

**4.4 스트리밍 동작의 차이 — [조사 완료, 구현 방향 확정]**
`LLM` 베이스 클래스는 `_stream()`을 오버라이드하지 않으면, `.stream()` 호출 시
`_call()` 결과를 한 번에 통째로 하나의 청크로 yield하는 방식으로 폴백(fallback)한다.
이는 공식 레퍼런스에 명시된 동작이다 — "stream() will use \_stream if provided,
otherwise it will use \_call and the output will arrive in one chunk"
([BaseLLM 레퍼런스](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html)).
즉 기존 `_generate_stream_custom_transformer()`가 구현한 단어 단위 가짜 스트리밍은
자동으로 따라오지 않는다.

공식 [Custom LLM 가이드](https://python.langchain.com/docs/how_to/custom_llm/)는
`_stream()`이 `Iterator[GenerationChunk]`를 반환하도록 안내한다. 예시 구현은 텍스트를
임의 단위(가이드 예시는 문자 단위)로 잘라 `GenerationChunk(text=...)`로 yield하고,
콜백이 연결되어 있으면 `run_manager.on_llm_new_token(chunk.text, chunk=chunk)`를
함께 호출한다. 즉 "토큰을 실시간으로 생성하지 못하는 모델을 임의 단위로 잘라
스트리밍처럼 보이게 하는 방식"은 LangChain이 별도 명칭("fake streaming" 등) 없이
`_stream()`의 표준 구현 방식으로 다룬다 — 기존 단어 단위 분할 로직을 그대로 이
형태로 옮기는 것이 정상적인 확장 지점 사용이며, 별도의 절충이 아니다.

실제로 `langchain_huggingface.llms.HuggingFacePipeline`의 `_stream()` 소스코드
([GitHub](https://github.com/langchain-ai/langchain/blob/master/libs/partners/huggingface/langchain_huggingface/llms/huggingface_pipeline.py))도
동일한 `GenerationChunk` + `run_manager.on_llm_new_token()` 패턴을 따르며,
`transformers.TextIteratorStreamer`를 별도 스레드(Thread)에서 실행해 실제 토큰
단위 스트리밍을 구현한다 — 이는 기존 Gemma 경로가 이미 사용 중인 방식과 동일하다.

결정: 표 3에 백엔드별 구현 방향을 정리했다.

**표 3. 백엔드별 스트리밍 구현 방향**
| 백엔드 | 방식 |
|---|---|
| Gemma | `HuggingFacePipeline`이 `TextIteratorStreamer` 기반 실시간 스트리밍을 이미 구현하므로 별도 작업 없이 그대로 사용 |
| custom_transformer | `LLM` 서브클래스에 `_stream()`을 오버라이드하여, 기존 `_generate_stream_custom_transformer()`의 단어 단위 분할 로직을 `GenerationChunk` yield + `run_manager.on_llm_new_token()` 호출 형태로 이전 구현 |

이 방향은 절충 없이 LangChain의 표준 확장 지점(`_stream()` 오버라이드)을 그대로
사용하는 것이므로, 표 1 #5·#6의 원칙과 일치한다.

**4.5 오케스트레이션 방식**
기존 `main.py`는 4단계를 절차적으로 직접 호출한다. LCEL은 이 4단계를
`retriever | prompt | llm`처럼 선언적으로 연결한다. 각 단계가 `Runnable`이므로
`invoke`/`stream`/`batch`가 체인 전체에 자동으로 전파된다.

## 5. RAG 체인 패턴 — [결정: LCEL Runnable 직접 조립]

LangChain 공식 튜토리얼([Build a RAG agent](https://docs.langchain.com/oss/python/langchain/rag))은
현재(v1.x) 두 가지 패턴을 제시한다.

**표 4. RAG Agent vs RAG Chain(LCEL) 비교**
| | RAG Agent (`create_agent` + tool-calling) | RAG Chain (LCEL Runnable 직접 조립) |
|---|---|---|
| 검색 시점 | LLM이 필요하다고 판단할 때만 (tool call) | 매 질문마다 항상 1회 |
| 호출 비용 | 검색 시 LLM 호출 2회(쿼리 생성 + 답변 생성) | LLM 호출 1회 |
| 적합한 LLM | tool-calling을 지원하는 `BaseChatModel` | 어떤 `Runnable`이든 가능 (tool-calling 불필요) |
| `custom_transformer`와의 적합성 | 낮음 — tool-calling 미지원, 메시지 기반 상태(`AgentState`) 도입 필요 | 높음 — 기존 6개 모듈과 1:1 매핑 |
| 확장성 | 멀티턴 대화, 다중 검색, LangSmith 트레이싱에 유리 | 단일 패스 질의응답에 충분 |

현재 `main.py`의 `/query`는 항상 검색을 1회 수행하는 단일 패스 구조이므로 RAG Agent보다
RAG Chain에 가깝다. **결정(표 1 #7)**: `create_agent`를 거치지 않고 `langchain_core.runnables`의
기본 단위(`RunnableSequence`, `RunnableLambda`, `RunnablePassthrough`)로 LCEL 체인을
직접 조립한다. 근거는 표 4의 적합성 행과 동일하다 — `create_agent` 기반 RAG Agent
도입은 2차 확장 과제로 명시적으로 분리한다.

## 6. 단계별 구현 순서

**표 5. 단계별 구현 순서 (구현 코드 + 테스트 코드 동시 작성)**
| 단계 | 대상 (`langchain_pipeline/`) | 구현 내용 | 테스트 (`tests/langchain_pipeline/`) | 선행 단계 |
|---|---|---|---|---|
| 1 | `loader.py` | `TextLoader`로 `daysync_manual.md` 로딩 | `Document.page_content`가 비어있지 않은지 검증 | 없음 |
| 2 | `splitter.py` | `RecursiveCharacterTextSplitter` + `MarkdownHeaderTextSplitter` | 기존 `test_chunker.py` 케이스를 `Document` 리스트 기준으로 재작성 | 1 |
| 3 | `embedding.py` | `HuggingFaceEmbeddings(all-MiniLM-L6-v2)` | 기존 `test_embedder.py`의 shape/결정론적/유사도 검증을 그대로 재사용 | 없음 (1·2와 독립) |
| 4 | `vector_store.py` | `InMemoryVectorStore(embedding=...)`, `add_documents()`, `as_retriever(k=3)` | 식별 가능한 mock 벡터 대신 [`DeterministicFakeEmbedding`](https://reference.langchain.com/python/langchain-core/embeddings/fake/DeterministicFakeEmbedding) 사용 검토 | 2, 3 |
| 5 | `prompt.py` | `ChatPromptTemplate`, 기존 instruction 문구·포맷 그대로 유지 | 포맷팅 결과에 instruction/context/question이 모두 포함되는지 검증 | 없음 |
| 6 | `llm.py` (Gemma) | 기존 MPS 디바이스 선택 로직 유지, `transformers.pipeline`을 만들어 `HuggingFacePipeline`으로 wrapping (§4.4, 표 3) | `invoke()` 결과가 문자열인지 검증, `stream()` 결과가 다중 청크로 도착하는지 검증 (모델 로딩이 무거우므로 `@pytest.mark.slow` 검토) | 없음 |
| 7 | `llm.py` (custom*transformer) | `LLM` 서브클래스 작성, `_call()`에 기존 `_generate_custom_transformer()` 로직 이전, `_stream()`에 기존 단어 단위 가짜 스트리밍 로직을 `GenerationChunk` yield 형태로 이전 (§4.4, 표 3) | 기존 `generator.py` 관련 테스트 케이스 재사용 가능 여부 확인, `_stream()`이 단일 청크가 아닌 다중 청크로 yield하는지 검증 | 없음 |
| 8 | `chain.py` | 4·5·6·7의 결과를 LCEL로 조립 | `invoke()` 결과가 기존 `main.py` `/query` 응답과 동일한 형태인지 통합 검증 | 1–7 |
| 9 | FastAPI 통합 (기존 `main.py`에 분기 추가, 표 1 #9) | `lifespan`에서 설정값(환경 변수 등)에 따라 `rag_pipeline` 체인 또는 `langchain_pipeline` 체인 중 하나를 생성, `/query`·`/query/stream`을 `chain.invoke()`/`chain.stream()`으로 교체 | 기존 엔드포인트 테스트에 분기 케이스 추가 | 8 |
| 10 | `tests/evaluate/langsmith_eval.py` (`langchain_pipeline/`이 아니라 기존 `tests/evaluate/evaluate*_.py`와 같은 위치 — 평가 전용 모듈이라는 성격이 동일하고, 4개 평가 함수를 같은 디렉터리에서 직접 import) | LangSmith Tracing(env var 기반, 코드 변경 없음) + 골든 Dataset 업로드(`Client.create*dataset`/`create_examples`) + 기존 4개 평가 함수(`evaluate*_.py`)를 LangSmith evaluator 시그니처 `(inputs, outputs, reference_outputs)`로 래핑, `Client.evaluate()`로 실행 | `tests/evaluate/test_langsmith_eval.py` — 골든셋 구조·`target()` 출력 형태·4개 evaluator 어댑터의 입출력 매핑을 fake(가짜) client/judge로 검증. 실제 LangSmith 연결과 실제 모델 로딩은 단위 테스트 범위 밖(수동 실행) | 8 |

1·3·5·6·7은 서로 독립이라 병렬로 진행할 수 있다. 4는 2·3 완료 후, 8은 1–7 모두 필요,
9·10은 8 완료 후 진행한다 (9와 10은 서로 독립).

## 7. 신규 의존성

**표 6. `requirements.txt`에 추가할 패키지**
| 패키지 | 용도 |
|---|---|
| `langchain-core` | `Runnable`/LCEL, `PromptTemplate`, `InMemoryVectorStore`, 커스텀 `LLM` 베이스 클래스 |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter` |
| `langchain-community` | `TextLoader` |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, `HuggingFacePipeline` (기존 `sentence-transformers`/`transformers` 의존성과 함께 사용) |
| `langchain` (메인 패키지) | `create_agent` 등 고수준 API. §5 결정(LCEL 직접 조립)에 따라 지금은 불필요 — 2차 확장 시 추가 |
| `langsmith` | `Client`로 골든 Dataset 생성/업로드 및 `evaluate()` 실행 (§6 10단계). Tracing 자체는 코드 변경 없이 `LANGSMITH_TRACING` 등 env var로 활성화 |

## 8. 범위 제외 항목 (표 1 근거)

GraphRAG(`graph_extractor.py`, `graph_retriever.py`)와 LangGraph는 이번 마이그레이션
범위 밖이다 (표 1 #2). `tests/evaluate/*`의 기존 RAGAS 스타일 평가 코드는 `rag_pipeline`
기준선용으로 그대로 보존하며 수정하지 않는다 — `langchain_pipeline`에 대한 신규 평가는
LangSmith Tracing + Dataset 기반으로 별도 구현한다(표 1 #4, §6 10단계).

## 9. 결정 사항 요약

이전 버전에서 "확인이 필요한 추가 사항"으로 남겨두었던 4개 항목은 모두 결정되어 표 1에 반영했다.

- (a) RAG 체인 패턴: LCEL 직접 조립으로 결정. `create_agent`는 2차 확장 과제로 보류 (표 1 #7, §5)
- (b) 패키지명: `langchain_pipeline`으로 확정 (표 1 #8, §2.2)
- (c) FastAPI 통합 방식: 신규 엔트리포인트 분리 없이 기존 `main.py`에 분기 추가 (표 1 #9, §6 9단계)
- (d) §4.1의 thin wrapper 절충안: 채택하지 않음. LangChain의 결합된 `VectorStore` 인터페이스를 그대로 사용 (표 1 #5, §4.1)

추가로, §4.2(임베딩 함수 소유권 이전)도 동일한 원칙에 따라 절충 없이 LangChain 방식을
채택했고(표 1 #6), §4.4(스트리밍 동작)는 LangChain 공식 패턴(`_stream()` 오버라이드 +
`GenerationChunk` yield)을 그대로 따르는 것으로 구현 방향을 확정했다(표 3).

- (e) 평가 코드 전환 범위: `langchain_pipeline`만 LangSmith로 평가한다(표 1 #4, §6 10단계).
  데이터셋은 `data/00_Design_Specification.md`의 "RAG 검증용 핵심 정보(Verification
  Anchors)" 표를 기반으로 한 신규 골든셋을 사용하고, 기존 `tests/evaluate/*`는
  `rag_pipeline` 기준선용으로 그대로 보존한다(`rag_pipeline`은 평가 대상에서 제외).

표 5의 1–9단계는 구현 코드와 테스트 코드를 함께 작성하는 방식으로 완료되었다.
다음 단계는 10단계(`tests/evaluate/langsmith_eval.py`)를 동일한 방식으로 작성하는 것이다.

## 출처

- [Build a RAG agent with LangChain](https://docs.langchain.com/oss/python/langchain/rag)
- [TextLoader](https://reference.langchain.com/python/langchain-community/document_loaders/text/TextLoader)
- [langchain_text_splitters 레퍼런스](https://reference.langchain.com/python/langchain-text-splitters)
- [HuggingFaceEmbeddings](https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html)
- [InMemoryVectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html)
- [HuggingFacePipeline](https://reference.langchain.com/python/langchain-huggingface/llms/huggingface_pipeline/HuggingFacePipeline)
- [LLM (커스텀 LLM 서브클래스)](https://reference.langchain.com/python/langchain-core/language_models/llms/LLM)
- [How to create a custom LLM class](https://python.langchain.com/docs/how_to/custom_llm/)
- [BaseLLM 레퍼런스 (stream 기본 동작)](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html)
- [HuggingFacePipeline 소스코드 (`_stream` 구현)](https://github.com/langchain-ai/langchain/blob/master/libs/partners/huggingface/langchain_huggingface/llms/huggingface_pipeline.py)
- [DeterministicFakeEmbedding](https://reference.langchain.com/python/langchain-core/embeddings/fake/DeterministicFakeEmbedding)
- [HuggingFace 통합 개요](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
- [LangSmith Tracing Quickstart (env var 기반 자동 계측)](https://docs.langchain.com/langsmith/observability-quickstart)
- [Trace LangChain applications](https://docs.langchain.com/langsmith/trace-with-langchain)
- [Evaluate an LLM application (Client.evaluate())](https://docs.langchain.com/langsmith/evaluate-llm-application)
