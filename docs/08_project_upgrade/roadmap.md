# Roadmap

### 작업 순서

1. ✅ LangGraph 공식 문서 예시를 사용해 그래프 구조 최소 구현
2. ✅ 커스텀 스키마(타입) 생성 및 소스 코드 레벨 LangGraph 동작 원리 설명 문서 작성
3. ✅ "실행 중인 질문(`pending_question`)"을 스키마에 상태값으로 추가 (Default로 Required로 설정)
4. ✅ "일정 판단", "응답 판단", "날짜 판단", "서버 저장" 노드 및 엣지 추가
5. ✅ 개발 계획 상 질문/응답 query는 순서가 있음. 질문과 응답 사이 State는 유지되어야 함. 상태의 영속성(Persistence, 공식 문서 상 대화 연속성(Conversation Continuity)라 표현)를 위해 Thread, Checkpoint(er)를 적용함. 서로 다른 query가 요청하더라도 State가 자동 복원(restore)됨을 확인.
6. ✅ 고정된 단일 파이프라인으로 설계되어 있음. 호출 시 모든 노드가 실행되므로 장기적으로 연산 낭비가 예상됨. 계획 상 필요한 노드만 사용되도록 조건부 라우팅 처리가 필요함.

## 프로젝트 추후 과제

- rag_pipeline -> langchain -> langgraph 전환 필요성 순서대로 정리
- 미니 LangGraph 실행 프레임워크 직접 구현
- 합성 데이터 3종 프로젝트명 갱신(langgraph_pipeline_2 외 나머지 파이프라인)
- pending_question TTL 산정 기준 미정
- pyright 타입 오류 다수, 타입 정의 정리 필요
- langgraph_pipeline_2: 워킹 스켈레톤·사례집 완료, 사례집 VectorDB 임베딩·연결·store 인자화(lifespan 전환)·LLM 실제 연결 등 남음
- 함수 네이밍 커스텀 컨벤션 설계 보류 중
