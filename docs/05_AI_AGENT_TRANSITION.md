# LangGraph RAG 파이프라인을 AI Agent로 전환하기

## 1. 배경 — 고정 파이프라인의 한계

- 기존에는 StateGraph로 구성된 고정 RAG 파이프라인이었습니다. 해당 구조는 아래와 같습니다.
  ```
    START → [retrieve] → _route_after_retrieve → (generate | no_results) → END
  ```
- 이 구조는 "retrieve 노드 실행 여부"를 제(개발자)가 직접 코드로 고정했습니다. 사용자의 질문 내용과 무관하게 항상 벡터 스토어(vector store, 임베딩 검색 저장소)를 조회한 뒤, 검색 결과의 존재 여부만으로 생성(generate) 또는 결과 없음(no_results) 두 경로 중 하나로 분기했습니다.
- 이 방식의 근본적인 한계는 검색이 필요 없는 질문에도 항상 벡터 스토어 조회 비용이 발생한다는 점입니다. 예를 들어 "선호도 점수 0.8은 몇 등급이야?"라는 질문은 순수 계산 함수만으로 답변이 가능하지만, 고정 파이프라인 구조에서는 검색 단계를 우회할 방법이 없습니다. 검색 단계와 생성 단계 중 무엇을 실행할지 판단하는 주체가 미리 작성한 조건문이라는 점이 해당 구조의 제약입니다.

## 2. 목표

- LLM이 매 단계 "도구(tool)가 필요한지"를 스스로 판단하도록 하는 것을 목표로 두었습니다. 이를 위해 LangChain의 `bind_tools()` API(도구의 이름, 설명, 파라미터 스키마를 LLM에 등록하는 기능)를 사용해서 LLM이 직접 도구 호출 여부와 호출할 도구를 결정합니다. 규칙 기반에서 LLM 자율 판단으로 판단 주체의 이동은 LangGraph 파이프라인과 AI Agent를 구분하는 핵심 기준임으로 해당 구현을 목표로 잡았습니다.

## 3. 아키텍처 변경

- 변경 후 구조는 아래와 같습니다.
  ```
    START → [call_model] → tools_condition → (반복 시작) "tools" → [tool_node] → [call_model] (반복) → END
  ```

**표 1. 그래프 구조 비교 (before(고정 RAG 파이프라인) / after(AI Agent))**
| 항목 | before(고정 RAG 파이프라인) | after(AI Agent) |
| --- | --- | --- |
| 노드 구성 | `retrieve`, `generate`, `no_results` | `call_model`, `tools` |
| 분기 조건 | `_route_after_retrieve`(retrieved 리스트의 존재 여부 검사) | `tools_condition`(prebuilt 함수, LLM의 tool_calls 필드 존재 여부로 분기) |
| 분기 주체 | 개발자가 작성한 규칙 | LLM의 자체 판단(`bind_tools()`)으로 등록된 스키마 기준 |
| 반복 구조 | 없음(단방향, 최대 1회 실행) | `tools → call_model` 엣지로 순환 가능(여러 도구를 순차 호출 가능) |
| LLM 종류 | `Runnable`(외부 주입, `HuggingFacePipeline`, `CustomTransformerLLM` 포함 가능) | `ChatGoogleGenerativeAI.bind_tools()`로 고정 - native tool calling 지원 모델만 허용 |
| 상태 | RAGState(question/retrieved/answer 필드, overwrite 방식) | AgentState(messages 단일 필드, append 방식) |
| 스트리밍 방식 | `graph.stream(stream_mode="updates")`로 retrieve 완료 포착 후 llm.stream() 직접 호출 (혼합 방식) | `graph.stream(stream_mode="messages")`만으로 완결 (call_model 노드에서 직접 토큰 스트리밍) |

## 4. 핵심 구현 변경

**4-1. 상태 모델: RAGState -> AgentState**

- 기존 RAGState는 `question`, `retrieved`, `answer` 세 필드를 `reducer`없이 노드 반환값으로 덮어쓰는 overwrite 방식이었습니다. 검색 결과가 몇 번 나올지, 도구를 몇 번 호출할지가 고정되어 있었기 때문에 이 방식으로 충분했습니다. Agent 구조에서는 도구 호출 횟수가 가변적이므로, AgentState는 messages 단일 필드에 `add_messages` reducer를 적용해 메시지를 누적(append)합니다.

**4-2. 도구 정의: tools.py**

- `tools.py`는 신규 파일이며, Agent가 호출할 수 있는 도구 3개를 정의합니다.

**표 2. 도구 분류**
| 분류 | 도구 | 설명 |
| --- | --- | --- |
| 순수 함수 도구 (store 불필요) | `calc_preference_score` | 선호도 점수(float) → 선호도 등급 문자열. 경계값: 0.65 이상 선호, 0.35 미만 비선호 |
| 순수 함수 도구 (store 불필요) | `check_schedule_conflict` | 두 응답 비교 → SC-114(일정 충돌 코드) 위험 판단 |
| store 의존 도구 (클로저로 store 캡처) | `search_daysync_docs` (`make_search_tool(store)`가 생성) | 벡터 스토어 검색 |

- store 의존 도구인 `search_daysync_docs` 함수는 Closure로 만들어져 있습니다. `@tool` 데코레이터가 붙은 함수가 모듈 로딩 시점에 도구로 등록하기 때문입니다. 서버 시작 시 구성되는 `InMemoryVectorStore`를 모듈 최상단에서 받을 수 없으므로, store를 인자로 받는 팩토리 함수 안에서 `@tool`을 선언해 store를 클로저로 캡처합니다.

**4-3. routing: `_route_after_retrieve` &rarr; `tools_condition`**

- 기존의 고정 파이프라인에는 경로 분기 자체가 없습니다. 그래서 항상 retrieve를 먼저 실행했습니다. AI Agent에서는 `tools_condition`를 사용해 AI 모델 호출을 분기 처리할 수 있습니다. `tools_condition`은 LangGraph가 제공하는 prebuilt 함수입니다. `call_model` 노드가 반환한 `AIMessage`의 `tool_calls` 필드가 존재하는지 여부로 분기 처리합니다.
  - `AIMessage(content="직접 답변")` → `tools_condition` → `END`
  - `AIMessage(tool_calls=[...])` → `tools_condition` → "tools" 노드 실행

**4-4. LLM 종류 제약: Runnable &rarr; ChatModel**

- 기존에는 `build_rag_graph(store, llm, k)`로 `llm`을 외부에서 주입받았습니다. 그래서 `HuggingFacePipeline`, `CustomTransformerLLM` 등 BaseLLM 계열을 모두 사용 가능했습니다. AI Agent는 `llm`을 주입받지 않고, 내부에서 `get_agent_llm()`을 호출해 `ChatGoogleGenerativeAI`를 직접 생성합니다.
- BaseLLM을 더 이상 허용하지 않는 이유는 `bind_tools()`가 native tool calling을 지원하는 ChatModel 계열에서만 동작하기 때문입니다. BaseLLM은 bind_tools()를 지원하지 않아 tool_calls를 생성할 수 없습니다. 이 제약을 그래프 내부로 옮겨서, 호출부에서 잘못된 LLM을 주입할 가능성을 원천 차단합니다.

## 5. API 계층 변경

**표 3. main.py 변경 사항**
| 항목 | before(고정 RAG 파이프라인) | after(AI Agent) |
| --- | --- | --- |
| lifespan의 langgraph 분기 | `lg_store`와 `lg_llm`(Gemma) 둘 다 생성 | `lg_store`만 생성 — LLM은 Agent 내부(`get_agent_llm()`)에서 생성 |
| `/query` 엔드포인트 | `build_rag_graph(store, llm).invoke()` 호출, `retrieved_chunks` 채워서 반환 | `run_rag_agent(question, store)` 호출, `retrieved_chunks=[]` 고정 반환 |
| `/query/stream` 엔드포인트 | `stream_rag_answer()` 호출 | `stream_rag_agent()` 호출 |
| 신규 엔드포인트 | 없음 | `/agent/query` (`AgentQueryRequest`/`AgentQueryResponse`) 추가 |

## 6. 테스트 전략 변경

**표 4. 테스트 전략 비교**
| 항목 | 변경 전 test_graph.py (고정 파이프라인 검증) | 변경 후 test_graph.py (Agent 검증) |
| --- | --- | --- |
| 검증 대상 | 고정 라우팅(`_route_after_retrieve`)의 조건 분기 | LLM의 자율적 도구 호출 결정(`tools_condition`) |
| 가짜 컴포넌트 | `FakeListLLM`, `FakeStreamingListLLM` (langchain_core 제공, 고정 응답 반환) | `GenericFakeChatModel` (tool_calls 필드를 포함한 AIMessage 시퀀스 재현 가능) |
| 그래프 구성 방식 | `build_rag_graph()`를 그대로 호출 | `_build_test_graph()`로 별도 구성 — `GenericFakeChatModel`이 `bind_tools()`를 지원하지 않으므로 `get_agent_llm()`을 우회하고 `make_call_model_node()`에 fake_llm을 직접 주입 |
| 반복(loop) 검증 | 없음 (고정 파이프라인은 반복 구조가 없음) | `test_multiple_tool_calls_loop` — 도구 2회 연속 호출 시나리오 |
| Mock 대상 | 없음 | `run_rag_agent()` 테스트에서 `patch("langgraph_pipeline.graph.build_rag_graph", ...)`로 내부 Gemini 생성을 우회 |

- 변경 전 `test_graph.py`는 `stream_rag_answer`, `build_rag_graph(store, llm, k)` 시그니처, `langgraph_pipeline.nodes.NO_RESULTS_ANSWER`를 참조했는데, 이는 변경 후 코드베이스에 존재하지 않거나 시그니처가 변경되었습니다. `test_graph.py`는 고정 파이프라인 검증 코드 전체가 Agent 검증 코드로 교체(replace)되었기 때문입니다.

- `test_tools.py`는 신규 파일이며, `tools.py`의 순수 함수 도구와 store 의존 도구를 분리하여 검증합니다. 경계값(boundary value) 테스트가 포함된 점이 특징이며, 이는 `tools.py`의 `_PREFERENCE_HIGH = 0.65`, `_PREFERENCE_LOW = 0.35` 상수 정의와 1:1로 대응합니다.

## 7. 한계 및 후속 과제

**표 5. 확인된 이슈**
| 번호 | 내용 | 근거 |
| --- | --- | --- |
| 1 | `/query`의 langgraph 백엔드 응답에서 `retrieved_chunks`가 항상 빈 리스트 — 검색 근거 확인 및 RAGAS 재사용이라는 기존 설계 결정 위반 | main.py 설계 결정 3번 vs after `/query` 구현 |
| 2 | `/agent/query`가 참조하는 `resources["agent_store"]`가 lifespan의 세 분기(langgraph/langchain/rag_pipeline) 어디에서도 설정되지 않아, `RAG_BACKEND=langgraph`가 아니면 항상 503 반환 | main.py lifespan 3개 분기 vs `/agent/query` |
| 3 | `/agent/query`의 503 에러 메시지가 "GOOGLE_API_KEY 미설정"을 안내하지만, 실제 구현은 환경 변수를 검사하지 않고 store 존재 여부만 검사함 (docstring과 구현 불일치) | `/agent/query` 코드 |
| 4 | `nodes.py`, `RAGState`, `NO_RESULTS_ANSWER`가 after 코드베이스 전체(`graph.py`, `main.py`)에서 더 이상 참조되지 않는 죽은 코드로 확인됨 | graph.py(after), main.py(after) import 목록 전수 확인 |
