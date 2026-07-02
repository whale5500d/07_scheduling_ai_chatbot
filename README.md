# 개인 프로젝트 - 한국어 Chatbot

## 요약

**프로젝트 진행 현황**
| 순번 | 작업 내용 | 진행 상태 | 상세 내용 |
| --- | --- | --- | --- |
| 1 | Transformer 구현 | ✅ | [바로가기](./docs/01_TRANSFORMER_IMPLEMENT.md) |
| 2 | Vanilla RAG Architecture 구현 | ✅ | [바로가기](./docs/02_RAG_INTEGRATION.md) |
| 3 | LangChain Migration | ✅ | [바로가기](./docs/03_LANGCHAIN_MIGRATION_PLAN.md) |
| 4 | LangGraph Migration | ✅ | 바로가기 🚧 |

## 실행 방법

```bash
python -m venv venv
source .venv/bin/activate # 1. 가상환경 활성화
pip install -r requirements.txt # 2. 의존성 설치
uvicorn main:app --reload --port 8000 # 3. 서버 실행
```

## 프로젝트 구조

```bash
.
├── data/                    # 가상 데이터 (DaySync 도메인)
├── docs/                    # 구현 문서, 회고록(트러블슈팅)
├── src/
│   ├── custom_transformer/  # 1. Transformer (Decoder-only)
│   ├── rag_pipeline/        # 2. Vanilla RAG 파이프라인 (Gemma 4 E2B-it 기반)
│   ├── langchain_pipeline/  # 3. LangChain 파이프라인
│   ├── langgraph_pipeline/  # 4. LangGraph 파이프라인
│   ├── main.py              # FastAPI 진입점
│   └── paths.py             # 프로젝트 전역 경로 중앙화
└── tests/                   # src/ 구조를 미러링한 테스트 트리
    ├── custom_transformer   # 1. Transformer
    ├── rag_pipeline         # 2. Vanilla RAG 파이프라인
    ├── langchain_pipeline   # 3. LangChain 파이프라인
    ├── langgraph_pipeline   # 4. LangGraph 파이프라인
    ├── debugs               # 진단 스크립트 (assert 없음)
    ├── evaluate             # 평가 로직 + 대응 테스트
    ├── conftest.py          # pytest 공유 fixture 정의
    └── test_main.py         # FastAPI 엔드포인트 통합 테스트
```
