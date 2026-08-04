# 개인 프로젝트 개선 확장

## 트러블 슈팅

### 개념 학습 정리 - docker run 생성/실행 단계 분리로 인한 잔여 컨테이너 발생

**문제 상황**

`.env` 값 반영을 위해 컨테이너 재기동 절차를 진행하던 중, `docker run --env-file .env -p 8000:8000 ...` 명령이 포트 충돌 에러(port is already allocated)를 반환함. 에러가 발생했으므로 컨테이너 생성 자체가 이루어지지 않았을 것으로 예상함.

**부족한 개념**

`docker run` 명령의 내부 실행 단계 구조. `docker run`을 단일 원자적(atomic) 동작으로 이해하고 있었으며, 컨테이너 생성(container creation)과 실행(execution, 네트워크 설정 및 프로세스 시작)이 서로 분리된 순차 단계라는 점을 인지하지 못함.

**알게 된 사실**

`docker ps -a`로 확인한 결과, 포트 충돌 에러가 발생했음에도 해당 컨테이너가 `STATUS: Created` 상태로 남아있음을 확인함. 이는 `docker run`이 (1) 컨테이너 객체 생성 (파일시스템, 설정 등록) → (2) 네트워크 설정 및 프로세스 시작의 두 단계로 진행되며, 포트 바인딩 실패는 2단계에서 발생하기 때문에 이미 완료된 1단계(생성)는 되돌아가지 않는다는 것을 의미함. 즉 `docker run` 명령은 하나이지만, 에러가 어느 단계에서 발생하느냐에 따라 컨테이너가 생성된 채 실행만 실패한 상태(`Created`)로 남을 수도, 생성 자체가 되지 않을 수도 있음.

**개념이 포함된 섹션**

OS

### 개념 학습 정리 - 오픈 가중치 모델과 API 모델의 서빙 방식 차이

**문제 상황**

LangGraph `/query` 테스트 중 클라이언트에는 500, 인스턴스 내부 로그에는 429(RESOURCE_EXHAUSTED) 에러가 찍힘. `gemini-2.0-flash-lite` 모델의 서비스가 종료되었음이 원인.
구현 당시, 기존에 사용 중이던 Hugging Face Transformers 라이브러리(로컬 모델)가 있음에도 Agent 구조에서 API 방식(Gemini)을 채택한 이유를 인지하지 못한 상태였음. 이를 계기로 API 장애 대응 방안과 모델 채택 시 확인해야 할 기준을 함께 검토함.

**부족한 개념**

오픈 가중치(open-weight) 모델을 로컬에서 서빙할 때, 라이브러리 설치·가중치 파일·메모리 적재가 각각 어느 시점에 어떻게 분리되어 일어나는지에 대한 구조적 이해.

**알게 된 사실**

근본적으로 RAG 파이프라인과 AI 모델이 상호작용하는 방식(모델 서빙 방식)을 정리하기 위해, Qwen GGUF를 예시로 오픈 가중치 모델의 서빙 흐름을 아래와 같이 확인함.

(기초 지식: 파이썬 인터프리터는 `main.py` 파일부터 위에서 아래로 한 줄씩 순서대로 파싱하고 실행함. `import` 문을 만나면 해당 모듈의 코드를 캐시에 저장하고 실행하며(추후 재import 시 캐시된 모듈을 재사용), 호출 대상이 모듈이 아니라 `def`/`class`라면 이름만 등록해둠. 이 동작 원리가 이후 4번 항목(서버 실행 시 모듈 로딩 순서)의 근거가 됨.)

1. `huggingface_hub` 라이브러리를 설치하면, Hugging Face 서버와 통신하는 코드가 설치됨. 가중치 자체는 이 설치와 무관하게 별도로 다운로드됨.
2. Hugging Face 서버에는 여러 모델의 가중치 데이터가 존재함. Qwen 모델로 학습한 가중치도 여기 포함됨.
3. `llama-cpp-python`을 설치하면 범용 GGUF 추론 엔진이 디스크(`.venv`)에 설치됨. 이 엔진은 Qwen 전용이 아니라 Llama, Mistral 등 여러 GGUF 모델에 공통으로 쓰이는 코드임.
4. 서버가 실행되면, 파이썬 인터프리터가 `main.py`부터 한 줄씩 파싱하고 실행함.
   1. `import llama_cpp` 시점에서 GGUF 엔진(코드)이 프로세스 메모리에 적재됨.
   2. `lifespan` 실행 시점에서 `TextGenerator(...)` 호출 문법이 실행되면, `__call__`이 트리거되고 그 결과로 `__init__`이 실행되며, `model_name` 조건에 따라 `_init_qwen_gguf`가 실행됨.
      1. `hf_hub_download()` 함수가 로컬 캐시에 가중치 데이터가 있는지 먼저 확인하고, 없으면 HTTP 통신으로 요청해 가져옴. 있으면 캐시된 파일을 그대로 사용함.
      2. `Llama(model_path=...)` 생성자가 실행되는 순간, 디스크의 가중치 파일이 읽혀 메모리에 적재됨.
5. `/query` 요청이 오면, 메모리에 이미 적재되어 있던 GGUF 추론 엔진이 마찬가지로 메모리에 적재된 가중치를 사용해 순전파 연산을 수행함. 디스크 재접근이나 재적재 없음.

대조적으로 API 모델(Gemini) 방식은 SDK(google-genai)가 HTTP 요청·응답 처리 코드만 제공하는 구조이므로, API key만 있으면 로컬 리소스(디스크, 메모리) 소비 없이 모델에 접근해서 사용할 수 있음.

**개념이 포함된 섹션**

AI, PL

### 의사결정 정리 - RAG 채택 배경

**문제 상황**

Pretrained Model만으로는 모델의 파라메트릭 지식과 실제 추론 시점에 필요한 지식 사이 간극을 메울 수 없음. 특수 데이터(최신, 비공개, 조직 내부 데이터 등)를 답변에 반영해야 하는 간극 문제를 해결할 방법이 필요함.

**고려한 옵션**

- Pretraining from Scratch: 소수의 글로벌 Foundation AI 기업을 제외한 대부분 회사는 From Scratch Model이 없으므로, 재학습은 비현실적인 수단임.
- Fine-Tuning: 설령 From Scratch Model이 있어도, 기본적으로 전체 파인튜닝은 상당한 비용이 듦. 따라서 항상 합리적인 선택일 수 없음(방식에 따라 편차는 존재).
- Prompt 직접 삽입: 단기적으로 특수 데이터 전체를 요청마다 프롬프트에 입력해야 해서 토큰이 낭비되고, 장기적으로 문서 양이 많아지면 컨텍스트 길이 제한(context length limit)을 초과하여, 답변 가능한 상황임에도 이론상 모델의 응답 성능이 떨어짐.
- RAG: 데이터(Data), 검색기(Retriever), LLM이라는 세 핵심 구성요소가 유연한 아키텍처 설계를 지원함.
  - 이미 존재하는 Pretrained Model을 사용하기 때문에, From Scratch Model이 없어도 AI 서비스를 개발 및 운영할 수 있으므로 현실적인 해법임.
  - 만약 From Scratch Model가 있더라도, 자체 서버와 검색기를 활용해 모델의 가중치를 직접 수정하지 않아도 되어 비용이 많이 소모되지 않는 방법으로 합리적인 선택임.
  - 검색기를 통해 필요한 부분만 선별적으로 검색하면, 토큰 낭비를 줄이고, 모델의 응답 성능을 유지할 수 있음.

**결정 및 이유**

RAG를 채택함. 이 유연성 덕분에 현재는 기존 Pretrained Model을 그대로 활용하면서, 사내 Foundation Model이 구축되더라도 LLM 구성요소만 교체하여 전환할 수 있음. 다른 해결책 대비 현실적인 대안임.

원 논문에 따르면 RAG로 얻을 수 있는 2가지 장점을 Updatability(갱신 가능성), Provenance(출처 추적)을 가진다고 언급함.

- Updatability가 있으므로, 회사에게 각종 경제적 비용 절감 효과(토큰 사용량 감소 등)를 제공할 수 있음.
- Provenance가 있으므로, 유저에게 정확하고 검증된 정보를 제공할 수 있음.

**참고 자료**

- [원 논문, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, Lewis et al., 2020](https://arxiv.org/pdf/2005.11401)
- [NVIDIA - What is Retrieval-Augmented-Generation?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)

### 의사결정 정리 - 리팩토링 방법론 선정

**문제 상황**

세 파이프라인(rag_pipeline/langchain_pipeline/langgraph_pipeline) 모두 RAG 프레임워크라는 공통점에서 출발해, 리팩토링에 적용할 구조 방법론을 정해야 했음. 현재 코드는 커리큘럼 단계에 따라 각 시점의 설계 결정으로 작성되었으나, RAG가 왜 필요한지에 대한 근본적인 검토 없이 진행됨. 그 결과 세 파이프라인이 통합된 기준 없이 병존하게 되어, 전체 구조를 일관되게 설명하기 어려움.

**고려한 옵션**

- 계층형 아키텍처(Layered Architecture), 포트와 어댑터(Ports and Adapters), 전략 패턴(Strategy Pattern) 세 가지를 비교함.
  - 계층형 아키텍처: 책임을 단계별 계층(Retrieval/Generation/Orchestration 등)으로 분리하는 방식
    - 장점: 책임 구분이 직관적이라 일관된 전체 구조를 가져가기에 적합함.
    - 단점: SOLID 원칙 중 의존성 역전 원칙(DIP)이 적용되지 않아, 상위 계층이 하위 계층에 직접 의존함. 하위 계층이 국소적으로 변경(local change, 모델 교체 등)되면 상위 계층까지 함께 수정해야 하며, 개발자가 이 변경 범위를 매번 추적해야 함(유지보수 비용 증가).
  - 포트와 어댑터 패턴: 핵심 로직(포트)과 외부 구현(어댑터)을 인터페이스로 분리하는 방식
    - 장점: 변경 범위 추적하지 않아도 국소적 변경(모델 교체 등)을 어댑터 교체만으로 빠르게 대응 가능(유지보수 비용 감소).
    - 단점: 인터페이스 설계에 사전 고민이 필요해 초기 구현 복잡도가 높음.
  - 전략 패턴: 동일 인터페이스로 여러 구현체를 런타임에 교체하는 방식
    - 장점: 포트와 어댑터 패턴보다 구현이 간단함.
    - 단점: 외부 구현을 숨기는 것이 설계 원칙으로 강제되지 않아 호출부까지 외부 타입이 노출될 수 있음.

**결정 및 이유**

- 우선 일관된 전체 구조를 가져가기 위해 계층형 아키텍처로 1차 리팩토링을 진행함. 이후 실무에서는 YAGNI, 변동성 예측에 따라 리팩토링하는 것이 중요하므로, 변동성이 큰 지점(LLM 선택)을 2차 리팩토링 대상으로 설정함.
- LLM 선택 지점은 외부 SDK의 세부사항(에러 타입, 응답 스키마 등)이 도메인 로직에 새어들어오지 않도록 완전히 격리해야 하는 경계라고 판단함. 포트와 어댑터는 이 격리를 설계 원칙으로 강제하므로, 2차 리팩토링은 포트와 어댑터 방식으로 진행함.

### 개념 학습 정리 - Makefile과 taskipy 차이

**문제 상황**

Python 프로젝트에서 커스텀 스크립트 방법을 찾다가 Makefile, taskipy 2가지 후보가 나왔으나 둘의 차이를 명확히 몰랐음.

**부족한 개념**

명령 실행기(task runner)의 언어 범위(범용 vs 특정 언어 전용)와, 기존 도구 체인(uv)과의 통합 여부에 따른 차이.

**알게 된 사실**

taskipy는 python 전용이고, Makefile의 경우 범용이라 여러 언어를 함께 오케스트레이션할 때 유리하다는 점을 알게 됨. 구체적으로는 다음과 같음.

- taskipy: `pyproject.toml` 안에 설정을 추가하며, `uv run task ...` 형태로 uv가 관리하는 가상환경 안에서 그대로 실행됨. 현재 프로젝트가 uv로 통일되어 있어 도구 체인 일관성 측면에서 적합함.
- Makefile: `make` 명령(GNU Make)으로 실행되며 언어 무관(language-agnostic). 프로젝트에 Python 외 다른 언어(프론트엔드 빌드, Docker 빌드 스텝 등)까지 함께 오케스트레이션해야 할 때 유리함. 파일 변경 감지 기반 재빌드 등 빌드 자동화 기능도 보유함.

**개념이 포함된 섹션**

PL

### 개념 학습 정리 - LangGraph 폴더 구조 방법론과 배포 호환성

**문제 상황**

4단계(`langgraph_pipeline/tools.py` 계층 재배치) 진행 과정에서 분리된 기능(도구, LLM 생성, 노드 함수)을 관리하는 방법에 대해 질문했고, LangGraph 공식 문서에 권장 구조(`utils/tools.py`, `nodes.py`, `state.py` + `agent.py`)가 있다는 것을 확인함.

**부족한 개념**

LangGraph 공식 권장 폴더 구조의 존재 여부, LangGraph와 LangChain의 관계, 현재 프로젝트 구조가 향후 LangSmith 배포와 호환되는지에 대한 이해가 부족했음.

**알게 된 사실**

- LangGraph 공식 문서는 애플리케이션 구조를 `my-app/my_agent/utils/`(tools.py, nodes.py, state.py) + `agent.py`로 권장함. 현재 프로젝트는 `src/langgraph_pipeline/utils/`로, 최상위 감싸는 폴더명만 프로젝트 맥락(여러 백엔드 공존)에 맞게 생략된 형태이며 원칙은 동일하게 적용됨.
- LangGraph는 LangChain 없이도 단독 사용 가능하지만, 실무에서는 LangChain의 표준 인터페이스(`ChatModel`, `bind_tools()`, `@tool` 데코레이터, `AIMessage.tool_calls`)를 LangGraph 노드 내부에서 그대로 재사용하는 것이 일반적임. `tools_condition` 같은 LangGraph prebuilt 함수 자체가 `tool_calls` 필드(LangChain이 정의)를 검사하도록 설계되어 있어, 둘은 서로를 전제로 만들어짐.
- LangGraph에는 "Application Structure"라는 전용 공식 가이드가 있지만, LangChain에는 이에 대응하는 애플리케이션 폴더 구조 공식 가이드가 없음. Repository Structure(LangChain 자체 소스 저장소 구조)와 LangChain Templates(배포용 참조 아키텍처)는 목적이 다름.
- `langgraph.json`의 `dependencies`/`graphs` 필드는 모노레포 및 중첩 경로를 지원하므로, 현재처럼 `src/` 아래 여러 파이프라인이 공존하는 구조에서도 추후 배포 시 `graphs` 필드에 정확한 경로(`./src/langgraph_pipeline/graph.py:build_rag_graph`)만 명시하면 됨. 지금 폴더 구조를 미리 공식 예시에 맞춰 바꿀 필요는 없음.

**개념이 포함된 섹션**

AI

### 의사결정 정리 - 기획안 v2 확정(RAG 적용 및 연속성 축 제거)

**문제 상황**

기존 todo.md는 규칙 기반 분류로만 설계되어 있어, 질문 표현(할래?/허쉴?/ㅎㅅ? 등 어미 변형)이나 애매한 응답(긍정/부정 키워드 목록 밖의 응답)처럼 규칙만으로는 판정이 불확실한 회색 지대를 처리할 방법이 없었음. 또한 대화 연속성을 판단할 필요가 있는데, 이를 어떤 기준으로 판단할지 미정이었음.

**고려한 옵션**

- 회색 지대 처리: 규칙을 계속 추가하는 방법 vs 과거 판정 사례를 검색해 LLM 판단 근거로 제공하는 RAG 방법. 사례집을 검색 대상으로 삼으면, 판정 결과가 누적될 때마다 자동으로 검색 대상이 갱신되어 별도 문서 수정 없이 지속적으로 대응 가능함.
- 연속성 판단: reducer(`add_messages`)를 기준으로 매번 messages 리스트 전체를 훑어 미확정 질문을 찾는 방법 vs 별도 State 변수(`pending_question`)로 관리하는 방법. reducer 방식은 대화가 길어질수록 처리 비용이 증가함.

**결정 및 이유**

질문 표현 판정과 애매한 응답 판정에 사례집 기반 RAG를 추가하기로 결정함. 이에 따라 평가 지표 체계도 실행 순서(1차 일정 여부 판단 → 3차 응답 판단 → 2차 날짜 판단 → 4차 RDB 저장) 기준으로 재배치함. RAG가 추가된 두 지점(질문 표현/애매한 응답 판정)은 RAGAS(문맥 정밀도, 문맥 재현율, 충실도, 답변 관련성)로, 나머지 최종 판단은 기존 작업 품질 지표(classification metrics, exact match, success rate)로 각각 평가함.

대화 연속성은 별도 축으로 두지 않고, State의 `pending_question` 필드로 흡수하기로 결정함. 처리 비용 측면에서 별도 변수 관리가 reducer 전체 탐색보다 적절하다고 판단함. 다만 `pending_question`이 응답 없이 잔존할 수 있으므로, 유효 기간(TTL) 산정 기준은 추후 논의 사항으로 남김.

### 개념 학습 정리 - LangGraph 공식 예시를 통한 StateGraph 동작 원리 이해

**문제 상황**

기획안을 LangGraph 신규 노드/도구로 설계하려 했으나, 기존 `graph.py`/`tools.py` 구조가 이해되지 않은 채 확장하려다 막힘. LangGraph 공식 최소 예시를 직접 실행하며 기초 동작 원리부터 확인함.

**부족한 개념**

`StateGraph` 선언 시점과 실행(invoke) 시점의 차이, State의 스키마와 실제 값의 구분, reducer가 노드 반환값을 State에 병합하는 방식.

**알게 된 사실**

- `graph = StateGraph(MessagesState)` 시점에는 State의 스키마(타입)만 등록되며, 실제 State 값은 아직 없음. `invoke()`가 호출되면 그때 초기 State 값이 생성되고, 실행되는 동안 노드 사이를 오가며 값이 갱신됨. `MessagesState`(Prebuilt)와 `AgentState`(직접 정의)는 이 State의 스키마를 표현하는 방식의 차이일 뿐, 동작 원리는 동일함.
- `add_node(mock_llm)`은 함수를 등록만 할 뿐 즉시 실행하지 않음. `invoke()` 실행 중 해당 노드에 도달했을 때 비로소 함수가 실행됨.
- `add_edge(START, "mock_llm")`, `add_edge("mock_llm", END)`는 각각 시작점→노드, 노드→도착점을 연결하는 고정 엣지임.
- `invoke({"messages": [HumanMessage(content="hi!")]})`에서 넘긴 `HumanMessage`는 초기 State 값이며, 그래프 실행 전 State의 시작점으로 먼저 리스트에 들어감. 이후 `mock_llm` 노드가 실행되어 반환한 `AIMessage`가 `add_messages` reducer에 의해 append 방식으로 누적됨(reducer는 값을 축소하는 것이 아니라 병합 규칙을 정의하는 함수).
- `invoke()`의 반환값은 자동으로 출력되지 않으므로, 결과를 확인하려면 별도로 `print()`가 필요함.

**개념이 포함된 섹션**

AI

### 에러 원인 규명 - langgraph_pipeline 파일명과 라이브러리명 충돌로 인한 ModuleNotFoundError

**문제 상황**

LangGraph 공식 예시 확장 2단계(`MessagesState` → `AgentState` 교체) 진행 중, 새로 만든 파일명을 `langgraph.py`로 지정함. `uv run python -m langgraph_pipeline.langgraph` 실행 시 다음 에러 발생.

```
ModuleNotFoundError: No module named 'langgraph.graph'; 'langgraph' is not a package
```

**원인 분석**

파일 내부의 `from langgraph.graph import StateGraph, START, END` 구문이, 설치된 실제 `langgraph` 패키지가 아니라 방금 만든 `langgraph.py` 파일 자기 자신을 가리키게 됨. `-m` 실행 시 해당 파일이 위치한 디렉토리가 `sys.path`에 포함되면서 발생한 셀프 셰도잉(self-shadowing)이 원인임.

**결정 및 대응**

파일명을 라이브러리명과 겹치지 않는 `scaffold.py`로 변경함. "점진적으로 실제 구조로 발전시킬 발판"이라는 의미를 담아 명명함.

### 개념 학습 정리 - TypedDict와 Annotated의 역할

**문제 상황**

LangGraph 공식 예시 확장 2단계에서 `class AgentState(TypedDict)`, `Annotated[list[BaseMessage], add_messages]` 문법을 사용하고 있었으나, 각 구성요소가 정확히 무엇을 하는지 이해하지 못한 채 사용함.

**부족한 개념**

`TypedDict`가 딕셔너리에 타입 정보를 붙이는 방식, `Annotated`가 타입에 메타데이터를 추가하는 방식, 그리고 이 메타데이터를 LangGraph가 reducer로 해석하는 과정.

**알게 된 사실**

- `typing`: 파이썬의 타입 시스템을 갖는 표준 모듈입니다.
- `Annotated[타입, 메타데이터]`: 기본 타입에 메타데이터를 추가하는 표준 문법입니다.
- `TypedDict`: 정적 타입 검사기(pyright 등)가 타입을 파악할 수 있도록, 딕셔너리의 키와 값의 타입을 미리 선언하는 문법입니다.
- `Annotated[list[BaseMessage], add_messages]`
  1. "BaseMessage 타입으로 구성된 리스트"가 실제 타입(필드)이라는 의미입니다.
  2. "add_messages"는 함수입니다. LangGraph에서 가져온 함수이므로, LangGraph가 이 메타 데이터를 실행할 때 읽습니다.
  3. AgentState는 "해당 필드를 갱신할 때마다 add_messages 함수를 reducer로 실행하라"는 의미가 됩니다.

**개념이 포함된 섹션**

PL

### 개념 학습 정리 - Checkpointer와 Thread를 통한 대화 연속성 확인

**문제 상황**

질문과 응답이 서로 다른 사람의, 서로 다른 시점 메시지로 온다는 실제 서비스 컨셉에서, 이전 턴의 `pending_question`을 다음 턴에 어떻게 넘겨줄지 구조 설계가 필요했음.

**부족한 개념**

LangGraph 고급 문법의 하나인 Checkpointer, Thread는 여러 번의 독립된 `invoke()` 호출 사이에서 State를 이어줌. 그리고 하나의 대화(스레드) 범위 메모리인지 여러 대화에 걸친 메모리인지의 구분함.

**알게 된 사실**

- Checkpointer는 매 실행마다 State을 스냅샷으로 저장하고, Thread는 이 스냅샷을 모아서 하나의 `thread_id`로 묶음. 같은 `thread_id`로 다시 `invoke()`를 호출하면 LangGraph가 마지막 체크포인트에서 State부터 이어서 실행함.
- LangGraph는 같은 메모리 저장도 하나의 스레드 범위 메모리(Checkpointer)와 여러 스레드에 걸친 메모리(Store)로 구분함. 현재 서비스는 하나의 대화창 안에서의 연속성이 필요하므로 Checkpointer가 적절함.
- `graph.compile(checkpointer=checkpointer)`로 MemorySaver를 연결하고, 같은 `thread_id`로 `invoke()`를 2회 호출함. 2턴의 messages에 1턴의 HumanMessage/AIMessage가 동일한 id로 유지되어 정상 동작을 검증함.
- 다만 그래프가 "`pending_question` 값이 있으면 `judge_response`로 가야 한다"는 조건부 라우팅이 없음. 다음 과제로 남김. (현재는 `pending_question`이 None으로 명시되어 있으므로, NotRequired를 사용하여 에러 상황을 들어내도록 전환 필요)
- 다만 그래프가 "`pending_question` 값이 있으면 `judge_response`로 가야 한다"는 조건부 라우팅이 없음. 다음 과제로 남김. (추가로, `invoke()` 호출 시 입력값으로 명시한 `pending_question: None`이 노드 실행 전에 먼저 값을 덮어씀. `NotRequired`로 필드를 선택적으로 전환해, 추후 덮어쓰지 않도록 개선 필요)

**개념이 포함된 섹션**

AI

### 추론 검증 정리 - graph.invoke() 실행 순서 오판 확인

**문제 상황**

`turn1 = graph.invoke(...)`와 `print_result("1턴", turn1)`으로 이어지는 코드에서, `turn1` 결과에 2턴 메시지("응 좋아")까지 포함된 것처럼 보이는 상황이 발생함.

**추론한 내용**

`print_result("1턴", turn1)`이 먼저 실행되고, 그 이후에 `invoke()` 호출로 `judge_schedule`/`judge_response`가 동작하면서 관련 메시지만 필터링되어 출력된다고 판단함.

**검증 결과**

- 파이썬 인터프리터의 순차 실행 원리(한 줄씩 위에서 아래로)로 코드를 따지지 않고, `print_result()` 함수가 먼저 실행되는 것으로 잘못 가정한 것이 원인이었음.
- `judge_response` 내부의 디버그 `print(state["messages"])`와 `print("=====")` 구분선 출력이 `[1턴]`/`[2턴]` 출력보다 먼저 실행되고 있었음을 확인함.
- `turn1 = graph.invoke(...)` 호출이 먼저 완료되어야 그 결과값을 `print_result`에 넘길 수 있으므로, 실행 순서를 잘못 파악하고 있었음. (휴먼 에러)

**결론**

- `turn1`에는 실제로 2턴 메시지가 포함되지 않았으며, 착시의 원인은 실행 순서에 대한 잘못된 가정이었음. 이후 `judge_response`를 `HumanMessage`만 필터링하도록 수정해 2턴에서 "긍정 응답"이 정상적으로 반환됨을 확인함.

**개념이 포함된 섹션**

PL

### 개념 학습 정리 - 엣지와 라우팅의 개념 구분

**문제 상황**

조건부 라우팅 구현하기 전, LangGraph 생태계에서 얘기하는 라우팅이 무엇인지 개념 정의가 필요했음.

**부족한 개념**

엣지(edge)와 라우팅(routing)이 서로 다른 시점(설계 시점 vs 실행 시점)을 가리키는 표현이라는 것, 그리고 "조건부"라는 수식어가 왜 중복되어 보이는데도 관례적으로 함께 쓰이는지.

**알게 된 사실**

라우팅은 "실행 시점"에 여러 갈래의 엣지 중 실행할지 결정하는 행위/로직이고, 엣지는 그래프 "설계 시점"에 정의하는 표현임. 같은 것을 지칭하더라도 설계 시점과 실행 시점에 따라 표현이 달랐음. "조건에 따라 분기처리한다"는 의미를 이미 내포하고 있지만, 관례상 조건부 라우팅, 조건부 엣지라고 표현함.

**개념이 포함된 섹션**

AI

### 에러 원인 규명 - NotRequired 전환 후, pending_question이 유지되지 않는 문제

**문제 상황**

"진행 중인 질문(`pending_question`)"을 선택값(`NotRequired`)으로 설정하고 테스트를 실행함. 여전히 고정된 단일 워크플로우로 2턴에 입력받은 "응답"은 `judge_schedule` 함수를 타기 때문에 "진행 중인 질문"이 `None`이 됨.

**원인 분석**

`NotRequired` 전환했어도, 그래프 구조 자체는 여전히 고정되어 있어서, 2턴에서도 `judge_schedule`이 무조건 다시 실행됨. `judge_schedule`은 사용자의 응답("응 좋아")에 물음표가 없으므로 일정 질문이 아니라고 판정하고, `pending_question`을 자체적으로 `None`으로 덮어씀.

**결정 및 대응**

START 노드와 judge_schedule 노드 간 실행 사이에 "조건부 라우팅"을 추가하여 문제를 해결함. `pending_question`이 이미 값을 가지고 있으면 `judge_schedule`을 건너뛰고 바로 `judge_response`로 라우팅되도록 구현함.

**인사이트**

디버그 출력으로 각 시점의 값을 직접 찍어 원인을 하나씩 분리해서 검증하는 것이 중요함.

### 의사결정 정리 - 조기 종료 판단 변수 설계

**문제 상황**

기획안에 따라 조기 종료를 위한 조건부 라우팅을 추가 확장함. 조기 종료 판단 근거를 어디에 둘지 결정이 필요했음.

**고려한 옵션**

- (기존) `AIMessage.content` 또는 `pending_question`을 기준으로 조기 종료 처리.
- (신규 1) `additional_kwargs`에 값을 추가하여 해당 필드 기준으로 조기 종료 처리.
- (신규 2) 별도 State 필드를 추가하여 해당 필드 기준으로 조기 종료 처리.

**결정 및 이유**

별도 State 필드(`response_verdict`)를 생성하는 것으로 선택함.

- (기존 방식의 한계) 사람이 읽는 결과 표시용이라, 문구가 정책에 따라 수정되면 이를 파싱해 판단하던 로직이 조용히 깨질 수 있어 에러 추적이 어려움.
- (신규 1 방식의 한계) 값이 암시적이라 `AgentState` 정의에 나타나지 않음. `AgentState`만 보고는 분기 처리 여부를 확인하기 어려움. 또한 이미 `pending_question`을 별도 State 필드로 저장하기로 한 것과도 저장 방식이 어긋남.
- (신규 2 방식 보완) 추가로 타입 범위를 3가지 값(`POSITIVE`/`NEGATIVE`/`UNCLEAR`)으로 좁힘. 오타, 미정의된 값을 타입 체커가 사전에 제외하도록 보완함.
