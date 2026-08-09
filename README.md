# 메신저 기반 일정 자동 생성 AI 시스템(Automatic Scheduling AI System Based on Messenger)

사용자 간 대화(메신저 대화)를 입력받아 AI가 일정 정보를 추론하고 자동으로 일정을 생성하는 시스템.

기획안: [바로가기](./docs/00_plan/PLAN.md)

## 진행 현황

표 1. 단계별 진행 현황
| 순번 | 작업 | 상태 | 문서 |
| --- | --- | --- | --- |
| 1 | Transformer 구현 (from-scratch model) | 완료 ✅ | [설명](./docs/01_transformer/explanation.md) · [회고](./docs/01_transformer/retrospective.md) · [의사결정 기록](./docs/01_transformer/decision_record.md) |
| 2 | RAG 통합 | 완료 ✅ | [설명](./docs/02_rag_integration/explanation.md) · [회고](./docs/02_rag_integration/retrospective.md) |
| 3 | LangChain 마이그레이션 | 완료 ✅ | [설명](./docs/03_langchain_migration/explanation.md) |
| 4 | LangGraph 마이그레이션 | 완료 ✅ | [설명](./docs/04_langgraph_migration/explanation.md) |
| 5 | AI Agent 전환 | 완료 ✅ | [설명](./docs/05_ai_agent_transition/explanation.md) |
| 6 | 서버 환경 세팅 | 완료 ✅ | [설명](./docs/06_server_configuration/explanation.md) · [회고](./docs/06_server_configuration/retrospective.md) |
| 7 | CI/CD | 완료 ✅ | [설명](./docs/07_ci_cd/explanation.md) · [회고](./docs/07_ci_cd/retrospective.md) |
| 8 | 프로젝트 개선(langgraph_pipeline_2) | 진행 중 🚧 | [설명](./docs/08_project_upgrade/explanation.md) · [로드맵](./docs/08_project_upgrade/roadmap.md) · [회고](./docs/08_project_upgrade/retrospective.md) · [딥다이브 백로그](./docs/08_project_upgrade/deep_dive_backlog.md) |

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate # 1. 가상환경 활성화
pip install -r requirements.txt # 2. 의존성 설치
uv run task server-with-token # 3-1. 서버 실행 (HF Token 사용 O)
uv run task server-without-token # 3-2. 서버 실행 (HF Token 사용 X)
uv run task test # 4. 테스트 케이스 실행
```

## 프로젝트 구조

```bash
.
├── data/                     # 가상 데이터(사례집)
├── docs/                     # 단계별 설명, 회고록, 로드맵
└── src/
    ├── langgraph_pipeline_2/ # 현재 파이프라인 (LangGraph 기반)
    │   ├── graph.py          # StateGraph 정의 및 노드/엣지 구성
    │   ├── state.py          # AgentState 스키마 정의
    │   ├── test.py
    │   └── utils/
    │       ├── indexing.py   # VectorDB 인덱싱
    │       ├── llm.py        # LLM 연결
    │       ├── nodes.py      # 그래프 노드 정의
    │       └── tools.py      # 도구(tool) 정의
    ├── main.py               # FastAPI 진입점
    └── paths.py              # 프로젝트 전역 경로 중앙화
```
