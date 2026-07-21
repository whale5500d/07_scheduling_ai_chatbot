## Decision Record: Held-out Evaluation으로 Custom Transformer의 일반화 능력 검증

### 문제 상황

Training loss가 0.0013까지 수렴했으나(사실상 암기 수준), 이것이 진짜 일반화인지 단순 암기인지 구분할 방법이 없었음. Perplexity/exact match를 held-out set(unseen activity 3종: 캠핑/청소/베이킹)에 대해 측정해 이를 정량적으로 검증하기로 함.

### 진행 과정 및 발견된 문제 (순차적 진단)

진행 과정 및 발견된 문제
| 순서 | 발견 | 원인 | 조치 |
|---|---|---|---|
| 1 | Slot copy accuracy가 0.0000, held-out perplexity 46.57로 비정상적으로 높음 | BPE tokenizer의 vocabulary가 training corpus로만 구축되어, held-out 전용 글자가 전부 `<unk>`로 뭉개짐 | `tokenizer.train()`에 held-out corpus도 포함 (모델 가중치 학습에는 여전히 training만 사용 — data leakage 아님) |
| 2 | tokenizer 수정 후에도 slot copy 0.0000 유지 | `output_linear`이 held-out activity를 정답으로 한 번도 안 봐서, 해당 vocab 행이 억제 방향으로만 학습됨 | Weight tying (`output_linear.weight = embedding.weight`) 적용 |
| 3 | Weight tying 직후 held-out perplexity가 9.15 → 159049로 폭발, 생성 결과 mode collapse | `nn.Embedding`과 `nn.Linear`의 기본 초기화 스케일이 약 28배 차이나서, tying 시 logits 스케일이 붕괴 | embedding을 `std = d_model ** -0.5`로 재초기화 후 tying — 안정성 회복(held-out perplexity 9.79) |
| 4 | 구조적 문제 해소 후에도 slot copy 0.0000 유지 | held-out activity는 학습 루프(`for question, answer in qa_pairs`)에 애초에 한 번도 입력되지 않아, embedding 자체가 무작위 초기화 그대로였음 | training set에 held-out activity가 **질문(입력) 위치에만** 등장하고 **답변(정답)에는 등장하지 않는** 생략형 문장 6개 추가 (unseen 정의 유지) |

### 최종 결정

4번 조치 이후에도 slot copy accuracy는 0.0000으로 변화 없음(표 1 참고, perplexity/label accuracy는 각 단계에서 개선됨). Pointer/copy mechanism 같은 아키텍처 근본 변경 없이는 해결 불가능하다고 판단, 이번 챌린지 범위에서는 여기서 evaluation을 종료함.

### 인사이트

- **동일한 최종 지표(0.0000)가 서로 다른 원인에서 반복적으로 나타날 수 있다.** 4번의 독립적 원인(tokenizer OOV, output layer 미학습, 초기화 스케일 불일치, embedding gradient 부재)이 전부 "slot copy 0%"라는 동일한 증상으로 나타났음 — 지표 하나만 보고 원인을 단정하면 안 되고, perplexity·생성 결과 상세(details)를 함께 봐야 원인을 좁힐 수 있었음.
- **Weight tying은 "가능성의 문을 여는 것"이지 "정답을 보장하는 것"이 아니다.** Output layer가 unseen 토큰을 출력할 수 있게 만드는 것과, 실제로 그 토큰이 학습 신호를 충분히 받는 것은 별개의 문제였음.
- **평가 자체의 설계(무엇을 unseen으로 유지할지)가 데이터 보강 방법을 제약한다.** "정답에는 등장 안 시키되 질문에는 등장시킨다"는 절충안은, held-out의 시간 슬롯이 이미 전부 소진되어 있어 새 시간 표현(글피/다음 주)을 도입해야 했음. 데이터 분리(split) 설계와 보강(augmentation) 설계가 서로 얽혀 있음을 확인함.
- **24~33쌍 규모에서는 slot copy(항등함수) 능력의 일반화가 사실상 불가능함을 정량적으로 확인함.** 이는 실패가 아니라, "왜 더 많은 데이터 또는 다른 아키텍처(pointer mechanism)가 필요한가"에 대한 근거가 되는 유의미한 evaluation 결과임.
