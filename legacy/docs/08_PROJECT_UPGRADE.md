# 개인 프로젝트 개선 확장(Upgrade)

## LangGraph

- 상태(State) 하나를 두고 해당값을 바꾸면서, 워크플로우와 에이전트를 그래프 구조로 갖는 저수준 오케스트레이션 프레임워크입니다.

### 공식 예시 코드

- [공식 문서 내 예시 코드 바로가기](https://docs.langchain.com/oss/python/langgraph/overview#install)

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(result)
```

1. 객체 생성 및 State 등록
   - `graph = StateGraph(MessagesState)` StateGraph 객체를 선언하여 graph에 할당합니다. 이 시점에는 State의 스키마가 등록됩니다. (invoke 실행 단계에서 노드 사이를 오가면서 값이 생성 및 변경됩니다.) MessagesState, AgentState는 이 State의 스키마(타입)을 표현합니다. MessagesState로 만들면 LangGraph가 미리 만들어둔 상태값(Prebuilt)로 스키마가 표현되고, AgentState로 만들면 스키마를 직접 정의할 수 있습니다.

2. 노드/엣지로 그래프 설계
   - `graph.add_node(mock_llm)` StateGraph의 add_node 메서드를 호출합니다. add_node는 노드를 추가하는 메서드로 입력받은 mock_llm 함수를 등록합니다. 나중에 invoke 단계에서 mock_llm 노드에 도달하면 mock_llm 함수가 실행됩니다.
   - `graph.add_edge(START, "mock_llm")` 시작점에서 mock_llm 노드로 연결하는 엣지를 구성합니다.
   - `graph.add_edge("mock_llm", END)` mock_llm 노드에서 도착점으로 연결하는 엣지를 구성합니다.

3. 그래프 객체 생성
   - `graph.complie()` 실행 가능한 그래프 객체를 생성합니다.

4. 그래프 실행
   - `graph.invoke(...)` 초기 State를 넣어서 그래프를 실행합니다. 앞서 등록한 State가 생성되고 초기값이 할당되고, mock_llm Node가 동작합니다.

5. 실행 결과
   - result는 `graph.invoke`가 실제로 실행된 후 결과가 할당됩니다.
   - `print(result)`: 결과는 messages에 리스트로 저장되어 있는 딕셔너리입니다. HumanMessage, AIMessage가 순서대로 들어있습니다. invoke 실행 과정을 돌이켜보면, 입력으로 받은 HumanMessage는 초기 State 값입니다. 그래서 먼저 리스트에 들어갑니다. 이후, 워크플로우를 타면서 node로 만들어진 mock_llm이 실행되고, AIMessage가 리스트에 append 방식으로 누적됩니다(reducer가 병합 규칙을 정의)

### 작업 순서

1. LangGraph 공식 문서 예시를 사용해 그래프 구조 최소 구현
2. 커스텀 스키마(타입) 생성 및 소스 코드 레벨 LangGraph 동작 원리 설명 문서 작성
3. "실행 중인 질문(`pending_question`)"을 스키마에 상태값으로 추가 (Default로 Required로 설정)
4. "일정 판단", "응답 판단", "날짜 판단", "서버 저장" 노드 및 엣지 추가
5. 개발 계획 상 질문/응답 query는 순서가 있음. 질문과 응답 사이 State는 유지되어야 함. 상태의 영속성(Persistence, 공식 문서 상 대화 연속성(Conversation Continuity)라 표현)를 위해 Thread, Checkpoint(er)를 적용함. 서로 다른 query가 요청하더라도 State가 자동 복원(restore)됨을 확인.
6. 고정된 단일 파이프라인으로 설계되어 있음. 호출 시 모든 노드가 실행되므로 장기적으로 연산 낭비가 예상됨. 계획 상 필요한 노드만 사용되도록 조건부 라우팅 처리가 필요함.

## 딥다이브 주제

- 동적 링킹(dynamic linking)
- 메타클래스(`__call__`) 동작 원리
- 파국적 망각(catastrophic forgetting)
- 포트/어댑터 vs 전략 패턴의 격리 강제성 차이
- uv(Rust) vs Poetry(Python) 속도 차이
- LangGraph Checkpointer 구현 종류별 차이(MemorySaver/SqliteSaver/PostgresSaver)
- python -m 모듈 실행 vs 스크립트 직접 실행의 차이
