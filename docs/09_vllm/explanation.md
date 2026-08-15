# 1단계 베이스라인 서빙 측정 결과

## 측정 목적

1단계(Transformers model.generate() 직접 호출 방식) RAG 서빙의 Throughput (처리량), Latency (지연 시간) 측정값을 기록한다. 이후 vLLM 서빙 측정 결과와 비교할 기준선으로 삼는다.

## 측정 조건

표 1. 부하 조건
| 파라미터 | 값 |
|---|---|
| 모델 | Qwen/Qwen2.5-3B-Instruct (bfloat16, 원본 정밀도) |
| 서빙 방식 | Transformers model.generate() 직접 호출 (배칭 없음) |
| 측정 도구 | vllm bench serve (backend openai-chat) |
| Dataset | random |
| Num Prompts | 20 |
| Max Concurrency | 4 |
| Request Rate | inf |
| Input Length | 128 |
| Output Length | 128 (요청 값. 서버가 반영하지 않음. 확인된 사실 참고) |
| 실행 환경 | Google Colab, T4 GPU |

## 측정 결과

표 2. 측정 결과
| 지표 | 값 |
|---|---|
| Successful requests | 20 |
| Failed requests | 0 |
| Benchmark duration (s) | 531.80 |
| Total input tokens | 2560 |
| Total generated tokens | 6296 |
| Request throughput (req/s) | 0.04 |
| Output token throughput (tok/s) | 11.84 |
| Peak output token throughput (tok/s) | 19.00 |
| Peak concurrent requests | 5.00 |
| Total token throughput (tok/s) | 16.65 |
| Mean TTFT (Time To First Token, 첫 토큰까지 시간) (ms) | 9254.64 |
| Median TTFT (ms) | 6679.67 |
| P99 TTFT (ms) | 20936.36 |
| Mean TPOT (Time Per Output Token) (ms) | 299.51 |
| Mean ITL (Inter-token Latency) (ms) | 303.01 |
| Mean E2EL (End-to-End Latency) (ms) | 105187.32 |
| Median E2EL (ms) | 93202.66 |
| P99 E2EL (ms) | 182983.59 |

## 확인된 사실

- 서버가 요청의 출력 길이(128)를 반영하지 않고, 고정값 MAX_NEW_TOKENS(512)까지 생성함. 요청당 평균 생성 토큰 수는 약 315개(6296/20)로 요청값보다 많음.
- 배칭 없이 순차 처리되어, Max Concurrency 4 기준 Mean TTFT 9254.64ms, Mean E2EL 105187.32ms로 대기(큐잉) 효과가 반영됨.
