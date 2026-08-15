# vLLM 적용

## 트러블 슈팅

### 에러 원인 규명 - Tesla T4의 bfloat16 미지원으로 인한 vLLM 엔진 시작 실패

**문제 상황**

베이스라인(1단계)과 동일하게 vLLM 엔진 서버도 bfloat16(원본 정밀도)으로 실행하려 했으나, `vllm serve` 실행 시 EngineCore가 시작하지 못하고 종료됨.

**원인 분석**

vLLM은 엔진 시작 시점에 GPU가 지정된 dtype(정밀도 타입)을 지원하는지 명시적으로 검사함. Tesla T4(compute capability, GPU 세대별 연산 지원 수준을 나타내는 지표 7.5)는 bfloat16을 하드웨어 가속하지 않으며, bfloat16은 compute capability 8.0 이상부터 지원됨. Transformers는 이 제약을 검사하지 않아 베이스라인에서는 bfloat16으로도 에러 없이 동작했으나, vLLM은 이를 ValueError로 명시적으로 거부함.

**결정 및 대응**

vLLM 실행 시 `--dtype half`(float16)로 지정함. 서빙 엔진 비교라는 실험 목적상 정밀도가 통제 변인이어야 하므로, 베이스라인도 float16으로 재측정해 두 서빙 방식의 정밀도를 통일함.

**인사이트**

Transformers가 에러 없이 동작한다고 해서 해당 정밀도가 GPU에서 실제로 가속되고 있다는 보장은 아님. float16 재측정 결과 베이스라인 자체의 Latency도 크게 개선되어(Mean E2EL 53739.27ms → 36485.33ms), bfloat16이 T4에서 비효율적인 연산 경로를 타고 있었음이 확인됨.

### 에러 원인 규명 - vllm 설치로 인한 torchaudio-torch CUDA 버전 불일치

**문제 상황**

Colab에서 `pip install vllm` 실행 후 vLLM 관련 명령을 실행하면, torchaudio 임포트 단계에서 RuntimeError가 발생함.

**원인 분석**

`pip install vllm`이 의존성 해결 과정에서 torch를 CUDA 13.0 빌드로 업그레이드함. 반면 Colab에 사전 설치된 torchaudio는 CUDA 12.8 빌드 그대로 남아있어, 두 라이브러리의 CUDA 버전이 어긋남. transformers가 오디오 관련 손실 함수(loss_rnnt.py)를 로드하는 과정에서 torchaudio를 임포트하고, torchaudio가 자체적으로 CUDA 버전 일치 여부를 검사하다 이 에러를 발생시킴.

**결정 및 대응**

이 프로젝트는 오디오 기능을 쓰지 않으므로, vllm 설치 직후 `pip uninstall torchaudio`로 제거함.

**인사이트**

무거운 패키지(vllm) 설치가 사전 설치된 부속 라이브러리(torchaudio)의 버전과 어긋나는 경우가 있으며, 실제로 쓰지 않는 부속 라이브러리는 제거하는 것으로 간단히 해결됨.

### 에러 원인 규명 - HuggingFaceEmbeddings의 무거운 임포트 체인으로 인한 torch.\_dynamo Config 에러

**문제 상황**

vllm 설치 후 RAG 앱 서버(K 셀) 실행 시, lifespan에서 `HuggingFaceEmbeddings` 생성 도중 `TypeError: Config() got an unexpected keyword argument 'deprecated'` 에러가 발생함. `--force-reinstall`로 torch 재설치, torch 계열 완전 삭제 후 재설치, Colab 런타임 완전 초기화까지 시도했으나 모두 동일하게 재현됨.

**원인 분석**

에러 트레이스백을 추적한 결과, `HuggingFaceEmbeddings` → `sentence_transformers` → `transformers.modeling_utils` → `integrations.finegrained_fp8` → `deepgemm` → `torch._dynamo` 순으로 이어지는 임포트 체인에서 문제가 발생함. vllm이 요구하는 최신 transformers 버전의 `deepgemm.py`가 `torch._dynamo.allow_in_graph` 데코레이터를 무조건 적용하는데, 이 경로에서 torch 내부 `Config` 함수와 충돌함. 설치 꼬임, pip 캐시 손상, 파이썬 커널 메모리 캐시 등 여러 가설을 세우고 재설치·초기화를 반복했으나 전부 틀린 진단이었고, 완전히 새로운 런타임에서도 동일하게 재현되는 것을 확인하고 나서야 설치 문제가 아니라 임포트 체인 자체의 문제임을 특정함.

**결정 및 대응**

임베딩 모델 자체를 필요한 것보다 훨씬 무거운 `transformers`를 거치지 않는 방식으로 교체함. `langchain_community`의 `FastEmbedEmbeddings`(ONNX Runtime 기반)로 바꿔, RAG 앱 프로세스가 torch·transformers를 전혀 임포트하지 않도록 함.

**인사이트**

에러 트레이스백의 마지막 줄(Config 함수 호출 실패)에만 집중해 원인을 파고들다가, 트레이스백 전체가 보여주는 "이 임포트 경로 자체가 목적에 비해 과도하게 무겁다"는 근본 신호를 여러 번의 재설치 시도 끝에야 확인함. 필요한 기능(문장을 벡터로 변환하는 것)에 비해 과도하게 무거운 의존성을 끌어오면, 그 의존성이 안 쓰는 코드 경로의 버그에도 영향을 받게 됨.

### 의사결정 정리 - vLLM 엔진 서버를 별도 프로세스로 분리한 구조 결정

**문제 상황**

베이스라인은 `model.generate()`를 RAG 앱 프로세스 안에서 직접 호출하는 구조였음. vLLM으로 전환할 때, 이 구조를 유지한 채 vLLM을 라이브러리로 통합할지, 별도 서버로 분리할지 결정이 필요했음.

**고려한 옵션**

- 라이브러리 통합: `vllm.LLM` 클래스를 RAG 앱 프로세스 안에 직접 불러와 사용. 프로세스가 하나로 유지됨.
- 별도 서버 분리: `vllm serve`로 별도 포트(8001번)에 엔진 서버를 띄우고, RAG 앱(8000번)이 검색·증강까지 처리한 뒤 HTTP로 생성을 위임.

**결정 및 이유**

별도 서버로 분리함. `vllm serve`는 Continuous Batching, Paged Attention 같은 스케줄러·배칭 로직이 이미 완성된 형태로 구현되어 있음. 라이브러리로 통합하면 이 스케줄러·큐 관리 로직을 직접 재구현해야 해서 최소 의존성 원칙에 어긋남. 또한 RAG 앱(가볍고 CPU 위주)과 vLLM 엔진(무겁고 GPU 위주)의 재시작·교체 주기가 달라, 분리하는 것이 운영 관점에서도 유리함.

### 의사결정 정리 - 베이스라인·vLLM 모델 정밀도를 bfloat16에서 float16으로 통일

**문제 상황**

베이스라인은 bfloat16(Qwen2.5 모델의 기본 정밀도)으로 측정을 완료한 상태였는데, vLLM 엔진이 Tesla T4에서 bfloat16을 거부함(compute capability 7.5, bfloat16은 8.0 이상 필요).

**고려한 옵션**

- float16 통일: 베이스라인을 float16으로 재측정하고, vLLM도 float16으로 실행.
- 현재 상태 유지: 베이스라인은 bfloat16, vLLM은 float16으로 두고 이 차이를 실험의 한계로 문서에 기록.

**결정 및 이유**

float16 통일을 선택함. 서빙 엔진(베이스라인 vs vLLM) 비교 실험에서는 서빙 엔진 하나만 변인으로 남기고 나머지(정밀도)는 고정해야 함. 현재 상태를 유지하면 5번 단계(비교 분석)에서 관찰되는 성능 차이가 서빙 엔진 때문인지 정밀도 때문인지 구분할 수 없어, 재측정 비용을 감수하고 통일함.

### 의사결정 정리 - 임베딩 라이브러리를 FastEmbedEmbeddings로 교체

**문제 상황**

`HuggingFaceEmbeddings`가 vllm 설치와 임포트 충돌(`torch._dynamo` Config 에러)을 일으켜, RAG 앱 서버가 lifespan 단계에서 시작조차 못 하는 상태가 반복됨.

**고려한 옵션**

- transformers 버전 고정: vllm이 요구하는 버전과 충돌하지 않는 특정 transformers 버전을 찾아 고정.
- 임베딩 라이브러리 교체: torch·transformers를 거치지 않는 임베딩 라이브러리로 교체.

**결정 및 이유**

임베딩 라이브러리 교체(`FastEmbedEmbeddings`, ONNX Runtime 기반)를 선택함. transformers 버전 고정은 vllm의 요구 버전과 계속 충돌할 여지가 남아있는 시행착오적 접근인 반면, RAG 앱이 필요로 하는 기능은 "문장을 벡터로 바꾸는 것" 하나뿐이라 애초에 torch·transformers 전체를 끌어올 필요가 없었음. 이 교체로 RAG 앱 프로세스에서 torch·transformers 의존성 자체가 사라져, 문제의 근본 원인이 제거됨.

### 개념 학습 정리 - vLLM이 별도 서버로 떠야 하는 이유

**문제 상황**

vLLM 서빙 구조를 설계하며, "하나의 서버로 처리하면 안 되는가"라는 의문이 들어 개념 확인이 필요했음.

**부족한 개념**

vLLM의 핵심 최적화 기법(Continuous Batching)이 함수 호출 형태가 아니라 서버 형태로 구현되어 있는 이유, 그리고 베이스라인(단순 함수 호출)과의 구조적 차이.

**알게 된 사실**

베이스라인의 `model.generate()`는 파이썬 함수 하나를 호출하는 것과 같아, 동시 요청을 배치로 묶는 판단 로직이 전혀 없이 그냥 순서대로 하나씩 처리됨. 반면 vLLM의 Continuous Batching은 여러 클라이언트의 요청을 계속 받아 큐에 쌓아두고, 매 스텝마다 어떤 요청들을 묶어 GPU에 넣을지 판단하는 스케줄러 역할을 함. 이 스케줄러는 요청을 계속 받아들이는 상태를 유지해야 하므로 서버 형태로 구현되어 있으며, "vLLM을 쓴다"는 것 자체가 서버 하나를 띄운다는 뜻이 됨. 참고로 `vllm.LLM` 클래스로 같은 프로세스 안에 라이브러리 형태로 통합하는 방법도 있으나, 이 경우 배칭 관리 로직을 직접 구현해야 함.

**개념이 포함된 섹션**

AI

### 개념 학습 정리 - TPOT와 ITL의 계산 단위 차이

**문제 상황**

vLLM 측정 결과에서 TPOT(Time Per Output Token)와 ITL(Inter-token Latency)이 같은 것을 가리키는 지표라고 배웠는데, Mean 값은 비슷하지만 Median과 P99 값이 서로 다르게 나오는 것을 확인함.

**부족한 개념**

TPOT와 ITL이 이름과 정의상 같은 물리량(토큰 간 시간)을 재는 지표임에도, 통계값(Mean/Median/P99)이 달라지는 이유.

**알게 된 사실**

TPOT는 요청 1개당 값 1개로, "(전체 응답 시간 − TTFT) ÷ (출력 토큰 수 − 1)"로 계산한 요청 단위 평균값임. ITL은 토큰 하나하나 사이의 개별 간격을 전부 기록함. 즉 TPOT는 요청 단위로 한 번 평균 낸 값들의 분포를 보는 것이고, ITL은 뭉뚱그리지 않은 개별 간격 값들의 분포를 그대로 보는 것. Mean은 두 방식이 비슷하게 수렴하지만, 한 요청 안에서 특정 토큰 하나만 유독 오래 걸린 경우 TPOT는 그 요청의 나머지 정상 속도 토큰들과 평균 내며 희석되는 반면, ITL은 그 순간의 간격을 그대로 하나의 값으로 남겨 P99 같은 상위 분위수에 그대로 드러남. 실제로 베이스라인 측정에서 P99 TPOT(573.23ms) 대비 P99 ITL(4673.03ms)이 8배 이상 크게 나타난 것이 이 차이 때문으로 확인됨.

**개념이 포함된 섹션**

AI

### 추론 검증 정리 - vLLM 적용 후 처리량·지연시간 개선 여부 검증

**문제 상황**

베이스라인(1단계, float16)과 vLLM(4단계, float16, 동일 부하 조건) 측정을 각각 완료한 뒤, vLLM이 실제로 성능을 개선하는지 비교 검증이 필요했음.

**추론한 내용**

Continuous Batching, Paged Attention 등 vLLM의 배칭 최적화로, 베이스라인 대비 Output token throughput은 증가하고 Latency(TTFT, E2EL)는 전반적으로 감소할 것으로 예상함.

**검증 결과**

Output token throughput은 베이스라인 12.80 tok/s에서 vLLM 43.99 tok/s로 약 3.4배 증가함. Mean E2EL은 베이스라인 36485.33ms에서 vLLM 11092.01ms로 약 3.3배 감소함. Peak concurrent requests도 6.00에서 8.00으로 늘어남. 다만 Mean TTFT는 예상과 반대로 베이스라인(2370.07ms)보다 vLLM(3630.00ms)이 더 크게 나옴.

**결론**

Throughput 증가와 E2EL 감소는 추론대로 확인되어, vLLM의 배칭 최적화 효과가 측정으로 뒷받침됨. 다만 TTFT가 오히려 늘어난 부분은 예상과 다른 결과이며, 원인은 아직 확인되지 않아 별도 확인이 필요한 사항으로 남김.

**개념이 포함된 섹션**

AI
