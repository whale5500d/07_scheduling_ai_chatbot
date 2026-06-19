# 05 Chatbot - 단계 1 (Dummy)

## 현재 단계

- ✅ **단계 1**: FastAPI 기본 구조 + Dummy Generator 완료
- ✅ **단계 2**: 반복 호출 방식의 간이 생성기 완료 (규칙 기반)
- ✅ **단계 3**: PyTorch의 기본 모듈만 사용한 실제 모델 적용 (진행 예정)
  - 고수준 완성형 모듈(`nn.Transformer`, `nn.MultiHeadAttention` 등) 사용하지 않기
  - 기본 모듈(`nn.Linear`, `nn.Dropout`, `softmax` 등)만 사용하기
- 🚧 지속 진행 작업: 학습 루프 구현 및 품질 개선

## 현재 진행 상황 및 테스트 결과 (2026.06.19 기준)

### FastAPI + 실제 모델 연결 완료

- `generate.py`를 더미 로직에서 `TransformerLanguageModel` + `BPETokenizer` 기반으로 교체
- `main.py`에 lifespan을 적용하여 모델 로딩 구조 개선
- `/generate` 엔드포인트가 실제 모델을 호출하도록 연결 완료

### curl 테스트 결과 예시

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello how are you", "max_length": 20}'
```

### 응답 결과 예시

<img src="images/260619_curl_test_result.png" alt="260619_FastAPI curl 테스트 결과">

```bash
# 결과 JSON으로 표시
{
  "generated_text": "hello w you t new newer nowzhest nice meeheweayougoowm wo isyotodatet todaynewer hello how are you",
  "prompt": "hello how are you"
}
```

- ⚠️ 참고: 현재는 모델이 학습되지 않은 상태(random weights)이며, BPE vocabulary도 매우 제한적인 dummy corpus로 학습된 상태입니다. 따라서 생성 품질은 낮습니다. 학습 루프 적용 후 품질이 개선될 예정입니다.

## 프로젝트 구조

```bash
05_Chat_Bot/
├── main.py              # FastAPI 앱
├── generator.py         # 생성 로직 (단계 2 기준)
├── schemas.py           # Pydantic Request/Response
├── requirements.txt
├── README.md
└── test_generator.py    # generator 테스트용 (선택)
```

## 실행 방법

```bash
python -m venv venv
source .venv/bin/activate # 1. 가상환경 활성화
pip install -r requirements.txt # 2. 의존성 설치
uvicorn main:app --reload --port 8000 # 3. 서버 실행
python test_generator.py # 4. 테스트 방법
```

## Transformer 구현 깊이

| 항목                                     | 사용 여부      | 이유                                                   |
| ---------------------------------------- | -------------- | ------------------------------------------------------ |
| ❌ nn.Transformer nn.MultiheadAttention  | 사용하지 않음  | 지나치게 얕음. 고수준 완성품이라 직접 만든 의미가 없음 |
| ✅ nn.Module, nn.Linear, nn.Dropout      | 사용           | 기본적인 신경망 부품                                   |
| ✅ torch.matmul, torch.transpose         | 사용           | 행렬 연산 (필수)                                       |
| ✅ F.softmax, F.dropout                  | 사용           | 기본 연산 함수                                         |
| ⚠️ NumPy나 순수 파이썬으로 처음부터 구현 | 하지 않아도 됨 | 지나치게 깊음. 비효율적이고 실용적이지 않음            |

## Transformer 구현 순서

| 단계 | 내용                                   | 목표                                          | 예상 소요 |
| ---- | -------------------------------------- | --------------------------------------------- | --------- |
| ✅ 1 | Scaled Dot-Product Attention 직접 구현 | 가장 핵심적인 Attention 메커니즘 이해 및 구현 | 가장 중요 |
| ✅ 2 | Multi-Head Attention 구현              | 여러 Head를 다루는 방식 이해                  | 중요      |
| ✅ 3 | DecoderLayer 구현                      | Attention + FFN + LayerNorm + Residual 연결   | 중        |
| ✅ 4 | TransformerDecoder 구현                | 여러 개의 DecoderLayer를 쌓기                 | 중        |
| ✅ 5 | 전체 모델 조립 + Positional Encoding   | Embedding + Positional Encoding + 출력층 연결 | 중        |
| ✅ 6 | 학습 루프 + generate 함수 연결         | 모델 학습 및 텍스트 생성 동작 확인            | -         |
| ✅ 7 | FastAPI 연결                           | 기존 웹 구조와 연결                           | -         |

## 전체 모델의 흐름

```text
입력 토큰 ID
   ↓
[Embedding]
   ↓
[Positional Encoding]
   ↓
[TransformerDecoder] (여러 층의 DecoderLayer)
   ↓
[Linear] → Vocab Size (다음 토큰 예측)
```

## Development History

| 단계   | 커밋 메시지                             | 커밋 링크                                                                                            | 비고                                        |
| ------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 단계 1 | chore: Chat Bot - Web Dummy 생성        | [6c56644](https://github.com/whale2200d/05_Chat_Bot/commit/6c56644d14e089ea32e00a83692f72f571fb4d94) | FastAPI 기본 구조 + Dummy Generator 구현    |
| 단계 2 | feat: 반복 호출 방식의 간이 생성기 완료 | [d29e892](https://github.com/whale2200d/05_Chat_Bot/commit/d29e892f2d172991a4ca06ad5ea484eb43c0c3c4) | 반복 생성 로직 + 간단한 반복 방지 기능 추가 |

> **현재 상태**: 단계 2 완료 (2026.06.15 기준)  
> 총 2개의 주요 커밋으로 구성되어 있으며, 단계 3에서 PyTorch 모델을 적용할 예정입니다.
