## 트러블 슈팅 20 - RAG_Project와 Transformer 프로젝트의 단일 패키지 통합 (폴더 구조 재설계)

### 문제 상황

- 두 개의 독립된 프로젝트(`rag_project`: RAG 파이프라인, `99_Personal_Project`: 커스텀 Transformer)를 하나의 패키지로 합쳐야 했다.
- 두 프로젝트 모두 `src/model/`이라는 같은 이름의 패키지를 갖고 있었으나, 내용은 완전히 달랐다(RAG_Project의 `model`은 chunker/generator/retriever 등, Transformer 쪽의 `model`은 transformer_model/decoder_layer 등).
- 단순히 한 폴더 안에 다른 폴더를 복사해 넣으면, 같은 이름의 패키지가 충돌해 import가 깨질 상황이었다.
- 또한 Transformer 프로젝트 쪽에는 이미 FastAPI로 단독 배포까지 진행했던 시절의 코드(`src/main.py`, `src/schemas/schemas.py`)가 남아 있어, RAG_Project의 동일 역할 코드와 중복되는 문제가 있었다.

### 원인 분석

1. **패키지 이름 충돌**: 두 프로젝트가 독립 실행될 때는 문제가 없었지만, 하나의 `sys.path` 아래 합쳐지는 순간 `model`이라는 이름이 둘 중 하나만 가리킬 수 있었다.
2. **역할 중복 코드의 존재**: Transformer 프로젝트의 `main.py`(FastAPI 진입점), `schemas.py`(Pydantic 요청/응답 모델)는 RAG_Project가 이미 동일한 책임(모델 로딩, 엔드포인트, 헬스체크)을 갖고 있어 중복이었다.
3. **모델 본체와 학습 파이프라인이 한 패키지에 강하게 결합**: `transformer_model.py`는 `decoder_layer.py` 등 5개 하위 모듈에, `train.py`는 모델 본체 + tokenizer + 데이터 유틸에 모두 의존하고 있어, 폴더를 옮기려면 의존 관계 전체를 한 번에 추적해야 했다.
4. **테스트 구조를 어디에 둘지 기준이 없었음**: 테스트를 소스 코드 옆에 둘지(co-location), 별도 `tests/` 트리로 둘지에 대한 일관된 원칙이 없어, "대상 파일과 테스트가 1:1로 보이는가"와 "배포 시 가벼운가"가 상충하는 것처럼 보였다.

### 결정 및 대응

1. **0단계 — 폴더 복사**: `rag_project`를 `.venv`, `__pycache__`, `*.egg-info`를 제외하고 Transformer 프로젝트 루트 아래로 복사. 이 시점에는 두 `model` 패키지가 물리적으로 분리된 폴더에 있어 충돌이 실제로 발생하지 않음을 확인.

2. **충돌 회피 전략 결정**: Transformer 쪽 `model`/`tokenizer`는 이름을 바꾸지 않고 그대로 유지, RAG_Project 쪽 `model`만 실체에 맞는 이름(`rag_pipeline`)으로 변경하기로 결정. 변경 범위가 더 작은 쪽(이미 여러 곳에서 참조되고 있던 Transformer 쪽 대신, 상대적으로 결합이 적은 RAG_Project 쪽)을 바꾸는 것이 비용이 적다는 판단.

3. **비핵심 코드 제거 (커밋 분리)**:
   - 판단 기준: "커스텀 Transformer 모델 자체(구조, 학습, 추론)에 직접 관련된 코드"인지, 아니면 "그 모델을 단독 배포하기 위한 부가 인프라"인지로 분류.
   - `src/main.py`, `src/schemas/schemas.py`를 제거 — 두 파일의 책임(모델 로딩, 생성 호출, 헬스체크, 요청/응답 스키마) 모두 RAG_Project의 `main.py`가 이미 동일하게 수행하므로 중복으로 판단.
   - 제거 전 `test_full_pipeline.py`, `test_generate.py`가 이 두 파일에 의존하지 않음을 확인하여, 부수 피해(collateral damage) 없이 제거 가능함을 검증.

4. **단계별 이동 (의존성 적은 것부터)**:
   - 2단계: 독립 모듈 4개(`document_loader.py`, `chunker.py`, `embedder.py`, `vector_store.py`) — 서로 의존이 없어 가장 먼저 이동.
   - 3단계: 검색 관련(`retriever.py`, `graph_extractor.py`, `graph_retriever.py`) — `vector_store.py`에 의존하므로 2단계 이후 진행. `graph_extractor.py`/`graph_retriever.py`는 아직 옮기지 않은 `generator.py`에 의존해, 이 시점에는 `ModuleNotFoundError`가 나는 것을 의도된 상태로 받아들이고 진행.
   - 매 단계마다 옮긴 모듈에 대응하는 테스트를 같이 옮기고, import 경로(`from model.xxx` → `from rag_pipeline.xxx`)를 수정한 뒤 실제로 테스트를 실행해 결과를 확인.
   - 실제 데이터 파일(`nimbusflow_manual.md`)에 의존하는 일부 테스트는, `data/` 폴더 이동이 더 뒤 단계로 예정되어 있어 일시적으로 실패하는 것을 허용하고 진행 — "의도된 실패"와 "예상 못 한 실패"를 구분해서 기록.

5. **모델 본체 전체 재배치 결정**: 처음에는 "얇은 래퍼만 RAG 쪽에 두고 모델 본체는 그대로 둔다"는 절충안을 검토했으나, 최종적으로는 모델 본체(`transformer_model.py`, `decoder_layer.py` 등), tokenizer, 학습 파이프라인(`scripts/`), 학습된 가중치까지 전부 `custom_transformer/`라는 하나의 자기 완결적 폴더로 모으는 방향으로 결정. 이유: 폴더 구조만 보고도 "커스텀 Transformer와 관련된 모든 것"이 한곳에 있다는 게 명확해지는 것이 장기적으로 더 일관성 있는 구조라고 판단.
   - 데이터/결과물 폴더 이름을 패키지명과 헷갈리지 않게 정리: `checkpoints` → `scripts/trained_model`(학습 스크립트의 출력물이라는 것을 명확히), `scripts/data` → `scripts/raw_data`(학습 입력 데이터임을 명확히).
   - `__init__.py`는 import 대상(`.py` 모듈)이 있는 폴더에만 두고, 데이터만 담는 폴더(`trained_model/`, `raw_data/`)에는 불필요하다고 판단해 제거.

6. **테스트 구조 — 미러링(mirroring) 방식으로 절충**:
   - "코드 옆에 테스트를 두는 방식"과 "별도 tests/ 트리"의 장단점이 상충하는 것처럼 보였으나, `tests/` 트리를 `src/` 트리와 완전히 동일한 하위 구조로 유지하면 두 장점을 동시에 취할 수 있다는 결론에 도달.
   - 예: `src/custom_transformer/model/decoder_layer.py` ↔ `tests/custom_transformer/model/test_decoder_layer.py` — 폴더 경로가 1:1로 대응되어, 코드와 테스트가 물리적으로 같은 폴더에 있지 않아도 "테스트가 있는지" 즉시 확인 가능. 동시에 `tests/` 전체를 배포 시 한 번에 제외할 수 있어 가벼움도 유지.
   - 실제로 미러링 결과를 검토하는 과정에서 `scripts`(복수형) vs `script`(단수형) 이름 불일치를 발견해 수정.

7. **`generate.py`는 의도적으로 보류**: 모델 본체가 전부 옮겨지면서 `generate.py`의 import도 깨졌지만, 이 파일이 "RAG와 어떻게 연결될 것인가"(Gemma의 `generator.py`를 대체할지, 별도 어댑터로 둘지)에 대한 결정이 아직 안 끝난 상태라, 폴더 이동이라는 단일 종류의 변경에 集중하기 위해 일단 깨진 상태로 두고 다음 단계로 넘기기로 함.

### 인사이트

- 패키지 이름 충돌은 코드를 합치는 시점이 아니라, **합쳐질 가능성을 처음 설계할 때부터** 고려했어야 더 적은 비용으로 해결할 수 있었다는 것을 경험했다. 사후에 이름을 바꾸는 작업(`model` → `rag_pipeline`)이 그 자체로 별도의 트러블슈팅 단계가 될 만큼 비용이 있었다.
- "관련 없는 코드를 제거한다"는 작업은 생각보다 판단 기준이 필요했다 — 단순히 "안 쓰는 것 같다"가 아니라, "그 책임을 대체할 다른 코드가 이미 있는가"를 확인해야 안전하게 제거할 수 있었다. 제거 전 의존 관계(어떤 테스트가 이 파일을 참조하는지)를 먼저 확인하는 절차가 실제로 부수 피해를 막았다.
- 큰 구조 변경을 "한 번에" 하지 않고, 의존성이 적은 모듈부터 단계적으로 옮기면서 매번 테스트로 검증하는 방식이, 중간에 어디서 무엇이 깨졌는지 추적하기 훨씬 쉬웠다. "의도된 실패"(아직 안 옮긴 것 때문에 나는 에러)와 "의도하지 않은 실패"를 구분해서 기록해두는 것이, 다음 단계로 넘어갈 때 무엇을 신경 써야 하는지 명확하게 해줬다.
- 폴더/테스트 구조에 대한 결정도, 흑백으로 양자택일하기보다 "두 요구사항이 실제로 상충하는지"를 먼저 따져보면 절충안이 있는 경우가 많았다(테스트 미러링 구조가 그 예). 트레이드오프처럼 보이는 문제도, 그 트레이드오프가 어느 축에서 발생하는지 분해해보면 둘 다 만족시킬 수 있는 지점이 있을 수 있다.
- 일부 작업(이번 경우 `generate.py`)은 의도적으로 미완성 상태로 남겨두고 다음 단계로 넘기는 것이, 한 커밋/한 단계에 여러 종류의 결정(폴더 이동 + 역할 재설계)을 섞지 않기 위한 합리적인 선택이었다. "보류"를 명시적으로 기록해두면, 나중에 그 이유를 다시 찾아볼 필요가 없다.

## 트러블 슈팅 21 - Gemma/커스텀 Transformer 갈아끼우기 설계 및 실제 동작 검증

### 문제 상황

- `TextGenerator`에 `model_name` 파라미터로 Gemma와 커스텀 Transformer를 갈아끼울 수 있도록 설계한 뒤, 실제로 동작하는지 검증하는 과정에서 두 가지 별개의 문제가 발생했다.
- 첫째, `requirements.txt`에 `transformers`, `sentence-transformers`를 추가하고 `pip install -e .`를 실행했는데도 설치가 반영되지 않았다.
- 둘째, 설치 문제를 해결한 뒤 `pyright src/`를 돌리자 `generator.py`에서 5개의 타입 에러가 발생했다(`self.model.generate()` 접근 불가, `do_sample` 파라미터 인식 불가 등).

### 원인 분석

1. **`requirements.txt`와 `pyproject.toml`은 서로 독립된 파일**: `pip install -e .`는 `pyproject.toml`의 `dependencies`만 참조해서 설치를 수행한다. `requirements.txt`에 추가한 내용은 `pip install -r requirements.txt`로 별도 실행해야 적용되며, 두 파일이 자동으로 동기화되지 않는다는 것을 인지하지 못했다.
2. **`AutoModelForCausalLM`의 반환 타입이 합성 타입(Union)이라는 점**: `Auto` 계열 클래스는 실행 시점에 어떤 구체 모델 클래스든 반환할 수 있어, 정적 타입 분석기(pyright) 입장에서는 반환 타입을 `_BaseModelWithGenerate`라는 넓은 타입으로만 추론한다. 이 타입에는 `.generate()`, `.to()` 같은 실제 메서드가 정적으로 보장되지 않아, 런타임에는 정상 동작하는 코드를 pyright가 에러로 표시한다(`transformers` 라이브러리를 쓰는 프로젝트에서 흔히 나타나는 잘 알려진 타입 스텁 미비 문제).
3. **`self.model` 속성이 두 분기(Gemma/커스텀)에서 다른 타입으로 할당됨**: `_init_gemma()`에서는 `AutoModelForCausalLM`의 반환값을, `_init_custom_transformer()`에서는 `TransformerLanguageModel`을 `self.model`에 할당하다 보니, pyright는 `self.model`의 타입을 두 가능성을 합친 Union으로 추론하여, 커스텀 Transformer 분기의 `self.model.generate()` 호출에도 같은 종류의 에러를 띄웠다.

### 결정 및 대응

1. **의존성 관리 방식 정리**: `pyproject.toml`의 `[project.dependencies]`에 `transformers`, `sentence-transformers`를 직접 추가하고 `pip install -e .`를 재실행하여 해결. `requirements.txt`와 `pyproject.toml`을 둘 다 수동으로 맞추는 대신, `pyproject.toml`을 단일 진실 공급원(single source of truth)으로 삼기로 함.

2. **타입 에러는 코드 로직 문제와 분리해서 판단**: pyright 에러가 곧 런타임 에러를 의미하지 않는다는 점을 확인하기 위해, `pyright`와는 별개로 `python3 src/rag_pipeline/generator.py`를 직접 실행하여 실제 동작을 검증.
   - 실행 시 `python3 -m src/rag_pipeline/generator.py`처럼 파일 경로를 `-m` 옵션에 그대로 넘기는 실수가 있었음 — `-m`은 점(`.`) 구분 모듈 이름을 받아야 하므로, `python3 -m rag_pipeline.generator` 또는 `python3 src/rag_pipeline/generator.py`(파일 경로 직접 실행) 중 하나로 정정.
   - 직접 실행 결과: Gemma 4 E2B-it이 정상 로딩되고, retrieval → prompt 조립 → generation까지 전체 파이프라인이 끝까지 동작했으며, 질문("What is the internal codename of NimbusFlow during development?")에 대해 정확한 답변("Project Driftwood")까지 생성됨을 확인. 이는 RAG_Project가 원래 검증해뒀던 정답과 일치함.
   - 결론: 5개의 pyright 에러는 모두 타입 스텁 한계로 인한 오탐(false positive)이며, 코드 로직 자체는 정상임을 확정.

3. **타입 에러 억제 — `# type: ignore`로 명시적 처리**:
   - 에러가 발생한 4개 지점(Gemma 모델 로딩의 `.to(self.device)`, `_generate_gemma()`의 `model.generate()` 호출, `_generate_custom_transformer()`의 `model.generate()` 호출, `generate_stream()`의 `Thread(target=self.model.generate, ...)`)에 각각 `# type: ignore[arg-type]` 또는 `# type: ignore[attr-defined]`를 추가.
   - 에러 코드(`[arg-type]`, `[attr-defined]`)를 명시해서, "무조건 억제"가 아니라 "이 특정 종류의 에러만 의도적으로 무시한다"는 것을 코드에 남김 — 추후 다른 종류의 진짜 에러가 같은 줄에서 발생하면 여전히 잡히도록 함.
   - `do_sample` 파라미터 관련 에러(`reportCallIssue`)는 호출문이 여러 줄에 걸쳐 있어, 첫 줄에 붙인 `# type: ignore`가 안쪽 줄의 에러까지 억제하는지 pyright 재실행으로 추가 확인이 필요한 상태로 남김.

### 인사이트

- "설치가 안 된다"는 증상이 실제로는 "설치 자체의 실패"가 아니라 "설치 대상 파일을 잘못 지정함"이었다는 것을 짚어내려면, 에러 메시지 유무와 종류(설치 중 에러가 떴는지, 그냥 조용히 반영이 안 됐는지)를 먼저 구분해서 물어보는 게 원인 진단에 더 빠른 경로였다.
- 정적 타입 분석기(pyright)의 에러는 "코드가 잘못됐다"가 아니라 "분석기가 타입을 확신할 수 없다"는 신호일 수 있다는 것을, Auto 계열 클래스(`AutoModelForCausalLM`)처럼 반환 타입이 동적으로 결정되는 라이브러리 패턴에서 실제로 경험했다. 타입 에러가 나면 곧바로 코드를 의심하기보다, 먼저 실제 실행으로 런타임 동작을 확인하고 나서 타입 에러의 성격(진짜 버그 vs 스텁 한계)을 판단하는 순서가 더 효율적이었다.
- 하나의 속성(`self.model`)이 조건 분기에 따라 서로 다른 구체 타입을 가질 수 있는 설계(이번 경우 Gemma용 `AutoModelForCausalLM` vs 커스텀 `TransformerLanguageModel`)는, 기능적으로는 "갈아끼우기"라는 의도된 설계이지만, 정적 타입 검사기 입장에서는 Union 타입으로 인식되어 타입 좁히기(narrowing)가 어려워진다는 트레이드오프가 있다는 것을 확인했다. 이런 경우 `# type: ignore`로 처리하는 것이, 지금 규모(학습 목적, 두 가지 모델만 전환)에서는 별도의 Protocol이나 타입 가드를 도입하는 것보다 비용 대비 합리적인 선택이었다.
- `-m` 옵션과 파일 경로 직접 실행은 서로 다른 import 해석 방식(모듈 이름 기반 vs 파일 시스템 기반)을 쓴다는 것을 다시 확인했다 — 패키지 구조를 재배치하는 작업이 많았던 만큼, 실행 방식의 차이가 "코드가 잘못됐다"는 오해로 이어지지 않도록 주의가 필요했다.

## 트러블 슈팅 22 - RAG_Project 도메인 전환 (NimbusFlow → DaySync)

### 문제 상황

- 이전 트러블슈팅에서 커스텀 Transformer에 instruction tuning을 적용할 때, 학습 데이터와 prompt를 전부 한국어("일정 묻기" 도메인)로 구성했다.
- 반면 RAG_Project의 기존 데이터(`nimbusflow_manual.md` 등)와 prompt_builder, Graph RAG 관련 prompt는 전부 영어로 작성되어 있었다.
- 즉 RAG_Project 전체가 영어 도메인으로 설계되어 있는 상태에서, 그 안에 끼워 넣은 커스텀 Transformer는 한국어 전용 모델이라는 언어 불일치가 구조적으로 존재했다.
- 이 불일치를 해소하기 위해, RAG_Project가 검증하는 데이터·prompt의 언어/도메인 범주를 커스텀 Transformer가 학습한 "일정 묻기" 도메인과 일치시키기로 했다. (참고로 이 시점에 실행해본 결과가 "거야"처럼 의미 없는 출력이었던 것은 24쌍 규모 모델의 한계로 이미 예상했던 사실이며, 이번 도메인 전환의 직접적인 계기는 아니다.)
- 다만 RAG_Project의 데이터(`nimbusflow_manual.md` 등)는 단순 텍스트가 아니라, retrieval 정확도·chunking 전략·Graph RAG 추출까지 검증하기 위해 의도적으로 설계된 "가상 문서"였고, 이를 교체하면 그 문서 내용에 의존하는 모든 코드(prompt, 테스트, 평가 스크립트)가 함께 깨질 수밖에 없는 구조였다.

### 원인 분석

1. **데이터가 단순 콘텐츠가 아니라 설계된 검증 장치였음**: `00_Design_Specification.md`를 보면, NimbusFlow 문서의 모든 수치·코드네임·에러코드가 "RAG 효과를 검증하기 위한 Verification Anchor"로 의도적으로 배치되어 있었다. 단순히 내용을 바꾸는 게 아니라, 같은 검증 의도를 유지하는 새 anchor를 설계해야 했다.
2. **언어 일관성 문제**: 기존 prompt_builder, graph 관련 prompt가 전부 영어였는데, 새 데이터를 한국어로 만들면 prompt와 데이터의 언어가 어긋난다. 커스텀 Transformer(한국어 전용)와 Gemma(다국어 가능) 양쪽 모두와 일관되게 동작하려면 prompt도 한국어로 통일해야 했다.
3. **테스트가 데이터 내용 자체를 assert하고 있었음**: `test_chunker.py`, `test_document_loader.py` 등이 `"NimbusFlow"`, `"8842"`, `"Drift Score"` 같은 실제 문서 내용을 직접 검증 기준으로 사용하고 있어서, 데이터를 바꾸면 이 assert들이 전부 실패하는 구조였다.
4. **Graph RAG의 relation_type이 도메인 특화 용어였음**: `uses_engine_mode`, `experienced_error` 같은 relation_type 이름 자체가 NimbusFlow의 개념에 묶여 있어, 새 도메인에 맞는 의미로 재정의가 필요했다.
5. **평가 스크립트(`debugs/`)의 위치 결정 기준이 모호했음**: `evaluate_*.py`(테스트가 있는 평가 로직)와 `debug_retrieval.py`(테스트 없는 진단 스크립트)가 같은 폴더에 있었지만, 성격이 달라서 "src에 둘지 tests에 둘지" 판단에 추가 기준이 필요했다.

### 결정 및 대응

1. **가상 설정을 그대로 유지하며 도메인만 교체**: NimbusFlow(가상 소프트웨어 제품)를 DaySync(가상 일정 관리 시스템)로 바꾸고, 기존 Verification Anchor 구조를 1:1로 대응시켰다.
   - 내부 코드네임: "Project Driftwood" → "Project Dawnstar(프로젝트 새벽별)"
   - 설정값: `engine_mode`(solo) → `preference_threshold`(0.65)
   - API 포트: 8842 → 9221
   - 에러 코드: NF-227 → SC-114(일정 충돌 코드)
   - Graph RAG용 인물/팀 구조(Team Falcon, Mina Park 등 → Team Sunrise, 도윤/서연 등)도 동일하게 대응시키고, 의도적으로 동명이인("도윤")을 배치해 기존 문서가 검증하던 "섹션 경계를 넘어 정보가 섞이지 않는가"라는 포인트를 그대로 재현했다.

2. **언어 일관성 원칙 적용**: `prompt_builder.py`의 Instruction/Context/Question 템플릿, `graph_retriever.py`의 `build_graph_prompt()`, `debugs/evaluate_*.py`의 LLM 평가 prompt까지 전부 한국어로 전환했다. 판단 응답도 영어 Yes/No에서 한국어 "예"/"아니오"로 바꾸고, 파싱 로직(`response.strip().startswith("예")`)도 함께 수정했다.

3. **relation_type 의미 대응**:
   - `uses_engine_mode` → `prefers_activity` (선호 활동 — 여러 사람이 같은 활동을 공유할 수 있어 원래의 "값 노드 허브" 문제를 동일하게 재현)
   - `managed_by`, `changed_config`는 의미가 그대로 통해 이름 유지
   - `experienced_error` → `experienced_conflict`

4. **테스트 재작성 — 데이터 의존 assert를 새 anchor 기준으로 교체**:
   - `test_chunker.py`의 실제 데이터 통합 검증에서, 기존 "8842 vs Drift Score" 섹션 분리 검증을 "9221(API 사용법 섹션) vs SC-114(일정 충돌 처리 섹션)"로 교체. 두 값이 서로 다른 섹션에 있으면서도 텍스트가 고유해 우연한 매칭 위험이 적은 anchor를 선택했다.
   - `test_document_loader.py`, `test_prompt_builder.py`도 동일한 원칙으로 키워드를 교체.
   - `test_evaluate_context_recall.py`의 "한국어 Ground Truth vs 영어 context" 언어 불일치 회귀 테스트는, 데이터가 전부 한국어가 되면서 그 시나리오 자체가 사라졌다. 같은 취지(표면적 단어 일치가 아닌 의미 판단)를 "같은 의미를 다른 표현으로 쓴 경우"로 재구성해 검증 의도를 유지했다.

5. **`debugs/`와 `evaluate/` 폴더를 성격에 따라 분리**:
   - 판단 기준을 "테스트가 있는가"로 명확히 함: `evaluate_*.py`(4개)는 대응하는 `test_evaluate_*.py`가 있어 테스트 자산에 가깝다고 보고, 평가 로직과 테스트를 한 폴더(`tests/rag_pipeline/evaluate/`)에 같이 두기로 결정.
   - `debug_retrieval.py`는 assert 없이 사람이 콘솔 출력을 읽는 진단 스크립트라 테스트 대상이 아니므로, 별도 폴더(`tests/rag_pipeline/debugs/`)로 분리.
   - 둘 다 `src/`가 아니라 `tests/` 트리 안에 둔 이유: 배포 시 제외해야 하는 코드(운영 코드가 아닌 검증/진단 목적)라는 점에서, 앞서 정립한 "tests/ 전체를 배포 시 한 번에 제외" 원칙과 일관되게 가는 것이 맞다고 판단했다.
   - `evaluate/` 안에서는 평가 로직과 테스트가 같은 폴더에 있으므로, 테스트의 import를 상대 import(`from .evaluate_answer_relevancy import ...`)로 바꾸고 `evaluate/__init__.py`를 추가했다.

6. **(파생 문제, 해결은 보류) 경로 계산 패턴의 구조적 취약성 발견**: 폴더를 재배치하는 과정에서, 여러 파일에 흩어진 `Path(__file__).resolve().parent.parent...` 패턴이 "몇 단계 위로 올라가야 하는지"를 파일마다 암묵적으로 가정하고 있었고, 이 가정이 폴더 이동마다 깨진다는 것을 `debug_retrieval.py` 실행 에러로 확인했다. 같은 파일 안에서도 목적지(프로젝트 루트 vs `src/`)에 따라 단계 수가 다르게 계산되어 있었다. 근본 해결책으로 `paths.py`(프로젝트 루트를 `pyproject.toml` 위치 기준으로 한 번만 계산하고, 모든 파일이 이를 가져다 쓰는 방식)을 설계를 생각하고 있으며, 적용은 다음 작업으로 보류했다.

### 인사이트

- 가상 데이터가 단순한 "예시 텍스트"가 아니라 "검증을 위해 설계된 장치"일 경우, 도메인을 바꿀 때는 내용을 새로 쓰는 것보다 **기존 설계 의도(어떤 것을 검증하기 위해 이 값이 거기 있었는가)를 먼저 파악하고, 그 의도를 보존하는 새 값을 설계하는 순서**가 안전하다는 것을 확인했다. 이번에 동명이인 함정, 섹션 분리 검증용 두 값(9221/SC-114) 선정이 그 사례였다.
- "언어 일관성"이라는 원칙 하나를 정하고 나면, 그 적용 범위가 생각보다 넓게 퍼진다는 것을 경험했다 — prompt_builder 하나만 바꾸는 줄 알았는데, Graph RAG의 prompt, 평가 스크립트의 LLM judge prompt, 심지어 그 판단 결과를 파싱하는 로직(Yes/No → 예/아니오)까지 전부 연쇄적으로 영향을 받았다.
- 한 컴포넌트(커스텀 Transformer)의 언어를 결정하는 순간, 그 컴포넌트와 연결되는 다른 모든 컴포넌트(RAG_Project 전체)의 언어 정합성까지 같이 고려해야 한다는 것을 경험했다. 부분의 변경이 전체 시스템의 일관성 요구를 끌어올린 사례였다.
- 테스트가 "함수의 동작"이 아니라 "특정 데이터의 내용"을 검증 기준으로 삼고 있으면, 데이터 교체가 코드 변경 없이도 테스트 전체를 깨뜨릴 수 있다는 것을 다시 확인했다. 이런 테스트는 데이터에 강하게 결합되어 있다는 것을 미리 인지하고, 데이터를 바꿀 계획이 있다면 테스트도 같이 재작성할 비용을 처음부터 고려해야 한다.
- 폴더 위치를 결정할 때 "운영 코드인가, 테스트인가"라는 이분법보다, "테스트가 짝으로 존재하는가"라는 더 구체적인 기준이 실제 결정에 더 도움이 됐다 — `evaluate_*.py`와 `debug_retrieval.py`가 겉보기엔 비슷해 보였지만, 이 기준으로 보니 서로 다른 폴더에 둬야 한다는 게 명확해졌다.
- 폴더 구조를 자주 재배치하는 프로젝트에서는, "파일 위치 기준 상대경로"라는 패턴 자체가 구조적으로 깨지기 쉬운 설계라는 것을 실제 에러로 체감했다. 이런 패턴이 코드 여러 곳에 반복되어 있다면, 그 자체가 "다음 폴더 재배치에서 또 깨질 부채"라는 신호로 받아들이고, 가능한 한 빨리 중앙화된 해결책으로 옮기는 게 장기적으로 안전하다.

## 트러블 슈팅 23 - 경로 계산 중앙화 (paths.py 도입)

### 문제 상황

- 폴더 재배치를 여러 차례 진행하면서, `Path(__file__).resolve().parent.parent...` 형태로 데이터/패키지 경로를 계산하는 코드가 22곳 이상에 흩어져 있었다.
- 각 파일이 "나로부터 몇 단계 위로 가면 목적지(data/, custom_transformer/ 등)가 있다"를 직접 계산하고 있어서, 파일이 다른 폴더로 옮겨지거나 폴더 깊이가 바뀔 때마다 그 단계 수가 깨졌다.
- 실제로 `tests/debugs/debug_retrieval.py`를 실행했을 때 `FileNotFoundError`가 발생했는데, 원인은 해당 파일이 새 폴더 구조에서 더 깊은 위치로 옮겨졌지만 `.parent.parent`라는 기존 계산식은 옛 깊이를 기준으로 남아있었기 때문이었다.
- 같은 파일(`generator.py`) 안에서도 목적지가 다르면(`custom_transformer/` vs `data/`) `.parent`를 호출하는 횟수가 서로 달라, 코드를 볼 때마다 "이 줄은 어디를 기준으로 몇 단계 위로 가는 코드인가"를 매번 추론해야 하는 부담이 있었다.

### 원인 분석

1. **"위치 기준" 계산의 구조적 취약성**: `.parent`를 정해진 횟수만큼 호출하는 방식은, 그 파일이 정확히 현재 위치에 있다는 전제에 암묵적으로 의존한다. 폴더 구조가 자주 바뀌는 프로젝트에서는 이 전제가 계속 깨졌다.
2. **계산 로직이 한 곳에 모여있지 않음**: 같은 목적지(`data/`)를 가리키는 코드가 9개 이상의 파일에 중복 작성되어 있어서, 폴더 하나를 옮길 때마다 그 모든 파일을 찾아서 동시에 고쳐야 했다.
3. **`paths.py` 초안에서도 같은 실수가 재발할 뻔함**: 처음 `paths.py`를 설계할 때, `PROJECT_ROOT`는 탐색(loop) 방식으로 안전하게 계산했지만, `CUSTOM_TRANSFORMER_DIR`은 여전히 `Path(__file__).resolve().parent.parent / "custom_transformer"`처럼 "위치 기준" 계산을 그대로 남겨두었다 — 중앙화의 의미를 절반만 살린 상태였다.
4. **도메인 전환(트러블슈팅 22) 작업 중 일부 파일이 누락됨**: `chunker.py`, `embedder.py`, `retriever.py`, `vector_store.py`의 `__main__` 블록이 `nimbusflow_manual.md`를 그대로 참조하고 있던 것을 이번 경로 점검 과정에서 추가로 발견했다.

### 결정 및 대응

1. **`src/paths.py`를 신설하여 경로 계산을 한 곳으로 모음**:

```python
   from pathlib import Path

   def find_project_root(marker: str = "pyproject.toml") -> Path:
       current = Path(__file__).resolve()
       for parent in [current, *current.parents]:
           if (parent / marker).exists():
               return parent
       raise FileNotFoundError(...)

   PROJECT_ROOT = find_project_root()
   SRC_DIR = PROJECT_ROOT / "src"
   DATA_DIR = PROJECT_ROOT / "data"
```

- `PROJECT_ROOT`는 "몇 단계 위인지"를 미리 정하지 않고, `pyproject.toml`이 나올 때까지 한 칸씩 올라가며 탐색하는 방식으로 계산한다. 이로써 `paths.py` 자신이 어느 깊이에 있어도(`src/` 바로 아래든, 더 깊은 곳으로 옮겨지든) 깨지지 않는다.
- `SRC_DIR`, `DATA_DIR`은 `PROJECT_ROOT`에서 **이름으로 아래로 내려가는 방식**으로 정의한다 — "몇 단계 위로"가 아니라 "루트에서 이 이름의 폴더"라는 의미라서, 그 폴더 자체의 상위-하위 관계가 안 바뀌는 한 깨지지 않는다.

2. **`CUSTOM_TRANSFORMER_DIR`을 별도 상수로 만들지 않고, 호출하는 쪽에서 `SRC_DIR`로 직접 조합하도록 단순화**: 사용처가 `generator.py` 한 곳뿐이라, 미리 상수를 만들어두는 것보다 `SRC_DIR / "custom_transformer"`를 호출 지점에서 조합하는 게 추론하기 더 쉽다고 판단했다. 사용처가 늘어나면 그때 `paths.py`로 끌어올리는 것으로 결정.

3. **22곳 + 추가로 발견한 4곳, 총 26곳을 일괄 정리**:
   - 모든 `Path(__file__).resolve().parent...` 경로 계산 코드를 `from paths import DATA_DIR`(필요 시 `SRC_DIR`도) 형태로 교체.
   - `sys.path.append(str(Path(__file__).resolve().parent.parent))` 형태의 코드도 모두 제거 — `pip install -e .`로 패키지가 이미 설치되어 있어 더는 필요하지 않은 코드였다.
   - 도메인 전환 시 누락되었던 `chunker.py`, `embedder.py`, `retriever.py`, `vector_store.py`의 `nimbusflow_manual.md` 참조를 `daysync_manual.md`로 함께 수정.

### 인사이트

- "경로를 고치는 비용을 한 번 지불하느냐, 매번 반복해서 지불하느냐"가 핵심이었다. `paths.py` 도입이 지금 26곳을 고치는 수고 자체를 없애주지는 않았지만, **앞으로 폴더가 또 재배치될 때 이 26곳을 다시 건드릴 필요가 없게** 만들었다는 점에서 일회성 비용과 반복 비용을 분리한 효과가 있었다.
- 중앙화를 한다고 선언해놓고도, 그 안에서 다시 "위치 기준" 계산(`.parent.parent`)을 쓰면 중앙화의 의미가 반감된다는 것을 `CUSTOM_TRANSFORMER_DIR` 초안에서 직접 경험했다. 중앙화 모듈 내부조차 "탐색"과 "이름 기반 하위 경로"라는 두 가지 안전한 패턴만 쓰도록 일관되게 점검해야 한다.
- `pathlib.Path.parents`가 단순히 ".. 몇 개"가 아니라, **현재 경로에서 상위로 올라가는 모든 경로를 순서대로 담은 시퀀스**라는 것을 이해하고 나서야, "원하는 표지(marker)가 나올 때까지 위로 탐색한다"는 패턴을 정확히 짤 수 있었다. 위치를 모를 때는 위치를 미리 계산하지 않고 탐색으로 찾는 것이, 구조 변경에 더 안전한 설계라는 걸 다시 확인했다.
- 도메인 전환처럼 영향 범위가 넓은 작업을 할 때, 변경이 필요한 지점을 한 번에 다 찾았다고 확신하기보다, 이후 다른 작업(이번엔 경로 점검)을 하다가 추가로 누락분이 발견될 수 있다는 것을 받아들이고, 발견 즉시 같이 처리하는 것이 별도 트러블슈팅으로 분리하는 것보다 효율적이었다.

## 트러블 슈팅 0 - OOO

### 문제 상황

### 원인 분석

### 결정 및 대응

### 인사이트
