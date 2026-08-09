# RAG 아키텍처 통합 (custom_transformer + RAG_Project)

## 개요

이 문서는 `README.md`(커스텀 Transformer 챗봇 구현 과정)가 다루는 범위 이후,
별도로 진행해온 RAG_Project(Gemma 4 E2B-it 기반)와 커스텀 Transformer를
하나의 패키지로 통합한 과정을 다룬다.

목표는 "커스텀 Transformer로 RAG의 Generation 단계(Gemma 자리)를 갈아끼울 수
있는가"를 기능적으로 검증하는 것이며, 답변 품질이 아니라 **에러 없이
retrieval → prompt 조립 → generation까지 파이프라인이 끝까지 도는가**를
1차 목표로 삼았다.

## 통합 전 상태

- `99_Personal_Project`: 커스텀 Transformer(BPE Tokenizer + 직접 구현한
  Decoder-only Transformer), instruction tuning 적용, EOS 토큰으로 반복
  생성 문제 해결까지 진행된 상태 (자세한 내용은 `README.md`, `docs/RESTROSPECTIVE.md` 참고).
- `rag_project`: Gemma 4 E2B-it 기반 RAG 파이프라인. Document Loading →
  Chunking → Embedding → Storage(Step A, Indexing)와 Query Embedding →
  Retrieval → Prompt Augmentation → Generation(Step B, Query)으로 구성.
  NimbusFlow(가상 소프트웨어 제품)를 검증용 가상 데이터로 사용.

## 통합 단계

### 1. 단일 패키지로 병합

두 프로젝트가 동일한 패키지명(`model`)을 사용하고 있어, 합치면 import가
충돌하는 문제가 있었다. Transformer 쪽은 결합도가 높아 그대로 두고,
RAG_Project 쪽 `model`을 `rag_pipeline`으로 이름을 바꿔 해결했다.

- `rag_project/`를 `99_Personal_Project/` 하위로 이동
- RAG_Project 단독 배포 시절 코드(Transformer 쪽 `main.py`, `schemas.py`)는
  RAG_Project의 `main.py`가 동일 책임을 이미 가지고 있어 제거
- 독립 모듈(`document_loader`, `chunker`, `embedder`, `vector_store`) →
  검색 모듈(`retriever`, `graph_extractor`, `graph_retriever`) →
  `prompt_builder`/`generator` 순서로, 의존성이 적은 것부터 단계적으로
  `rag_pipeline` 패키지로 이동
- 커스텀 Transformer 모델 본체 전체(`transformer_model.py`, `decoder_layer.py`
  등 구성 요소, tokenizer, 학습 스크립트, 학습된 가중치)를 `custom_transformer/`
  패키지로 재배치
- `tests/`는 `src/`와 동일한 구조로 미러링하여, 코드와 테스트의 1:1 대응을
  유지하면서도 배포 시 `tests/` 전체를 한 번에 제외할 수 있게 함

### 2. Gemma ↔ 커스텀 Transformer 전환 설계

RAG_Project의 `TextGenerator` 클래스에 `model_name` 파라미터를 추가해,
하나의 인터페이스로 두 모델을 전환할 수 있게 했다.

```python
generator = TextGenerator(model_name="google/gemma-4-E2B-it")  # 기본값
generator = TextGenerator(model_name="custom_transformer")     # 커스텀 모델
```

- `model_name == "custom_transformer"`일 때만 커스텀 모델(tokenizer 재학습,
  가중치 로드)을 초기화 — 두 모델을 동시에 메모리에 올리지 않음
- `generate(prompt)`는 두 모델 모두 "새로 생성된 부분만 반환"하도록 반환
  계약을 통일 (커스텀 모델 쪽은 입력 길이만큼 슬라이싱해서 맞춤)
- 커스텀 모델 쪽에 입력 길이 truncate(`CUSTOM_MAX_INPUT_TOKENS=400`)를
  추가해, 긴 RAG prompt(영어 지시문 + 문서 청크)가 positional encoding의
  `max_len`을 넘어 에러로 죽지 않도록 안전장치를 둠
- `generate_stream()`은 Gemma는 토큰이 실제로 생성되는 즉시 스트리밍하고,
  커스텀 모델은 `generate()`로 전체 텍스트를 먼저 만든 뒤 단어 단위로
  잘라 흘려보내는 가짜 스트리밍으로 구현 (모델 구조상 진짜 토큰 스트리밍이
  아직 불가능하기 때문)

실제 실행 검증: Gemma 경로는 `"What is the internal codename of NimbusFlow
during development?"` 질문에 `"Project Driftwood"`(정답)까지 정상 생성됨을
확인. 커스텀 Transformer 경로는 에러 없이 끝까지 돌고, 반복 생성 없이
멈추는 것까지 확인 (답변 내용 정확도는 범위 밖으로 분류).

### 3. 도메인 전환 (NimbusFlow → DaySync)

커스텀 Transformer가 한국어 "일정 묻기" 도메인으로 instruction tuning되어
있는데, RAG_Project의 데이터·prompt는 전부 영어였다. 두 컴포넌트의 언어를
일치시키기 위해 RAG_Project의 가상 데이터 도메인을 통째로 전환했다.

| 기존 (NimbusFlow)                                            | 전환 후 (DaySync)                          |
| ------------------------------------------------------------ | ------------------------------------------ |
| 가상 소프트웨어 제품 매뉴얼                                  | 가상 일정 관리 시스템 매뉴얼               |
| "Project Driftwood"                                          | "Project Dawnstar(프로젝트 새벽별)"        |
| API 포트 8842                                                | API 포트 9221                              |
| 에러 코드 NF-227                                             | 일정 충돌 코드 SC-114                      |
| `uses_engine_mode`, `experienced_error` (Graph RAG relation) | `prefers_activity`, `experienced_conflict` |

- 기존 데이터가 단순 텍스트가 아니라 RAG 효과를 검증하기 위한
  Verification Anchor(고유 코드네임, 설정값, 에러코드 등)로 설계되어
  있었기 때문에, 내용을 새로 쓰는 대신 같은 검증 의도를 보존하는 새
  anchor를 설계하는 방식으로 진행
- `prompt_builder.py`, Graph RAG의 `build_graph_prompt()`, 평가
  스크립트(`evaluate_*.py`)의 LLM judge prompt까지 전부 한국어로 전환
  (Yes/No 판단도 "예"/"아니오"로 변경)
- 데이터 내용에 직접 의존하던 테스트(`test_chunker.py` 등)를 새 anchor
  기준으로 재작성

### 4. 경로 계산 중앙화 (`paths.py`)

폴더를 여러 차례 재배치하면서, `Path(__file__).resolve().parent...` 형태로
경로를 계산하던 코드 26곳이 폴더 깊이가 바뀔 때마다 깨지는 문제가 있었다.
`src/paths.py`를 만들어, `pyproject.toml` 위치를 탐색 기반으로 찾아
`PROJECT_ROOT`/`SRC_DIR`/`DATA_DIR`을 한 곳에서만 계산하고, 나머지 모든
파일은 이를 가져다 쓰도록 일괄 교체했다.

## 현재 구조

```bash
src/
├── main.py                 # RAG_Project FastAPI 앱 (/query, /query/stream)
├── paths.py                 # 프로젝트 전역 경로 중앙화
├── custom_transformer/      # 커스텀 Transformer (모델, tokenizer, 학습 스크립트)
└── rag_pipeline/             # RAG_Project (retriever, generator, graph 등)
```

```bash
data/
├── 00_design_specification.md   # 가상 데이터 설계 명세 (Verification Anchor 정의)
├── daysync_manual.md
└── daysync_team_records.md
```

## 실행 방법

```bash
# 1. 커스텀 Transformer 학습 (먼저 한 번 실행해야 가중치 파일이 생성됨)
python custom_transformer/scripts/train.py

# 2. RAG 파이프라인 직접 실행 (FastAPI 없이 generator.py만 단독 확인)
python src/rag_pipeline/generator.py

# 3. FastAPI로 전체 파이프라인 실행
uvicorn main:app --reload --port 8000
```

`main.py`의 `TextGenerator()` 호출에서 `model_name`을 바꿔 Gemma/커스텀
Transformer를 전환할 수 있다.

## 알려진 한계 (이번 통합 범위 밖으로 분류한 것)

- 커스텀 Transformer는 24쌍 규모의 학습 데이터로 인해, RAG로 검색된 실제
  문서 내용에 기반한 의미 있는 답변을 생성하지 못한다. 이번 통합의 목표는
  "기능적 동작 확인"이었으므로, 답변 품질 개선은 범위 밖으로 분류했다.
- 커스텀 Transformer의 `generate_stream()`은 진짜 토큰 단위 스트리밍이
  아니라, 전체 생성 후 단어 단위로 잘라 흘려보내는 방식이다.
- `src/main.py`를 책임 단위(`RagPipeline` 클래스 등)로 재구성하는 작업은
  설계만 논의되었고 아직 코드에 반영되지 않았다.

## 관련 회고록

- [[바로가기](./retrospective/retrospective_integration.md)]
