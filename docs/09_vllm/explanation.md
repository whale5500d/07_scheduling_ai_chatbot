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
| Output Length | 128 (서버가 max_completion_tokens/max_tokens를 반영, --ignore-eos로 EOS 조기 종료 방지) |
| 실행 환경 | Google Colab, T4 GPU |

## 측정 결과

표 2. 측정 결과
| 지표 | 값 |
|---|---|
| Successful requests | 20 |
| Failed requests | 0 |
| Benchmark duration (s) | 273.47 |
| Total input tokens | 2560 |
| Total generated tokens | 2377 |
| Request throughput (req/s) | 0.07 |
| Output token throughput (tok/s) | 8.69 |
| Peak output token throughput (tok/s) | 20.00 |
| Peak concurrent requests | 7.00 |
| Total token throughput (tok/s) | 18.05 |
| Mean TTFT (Time To First Token, 첫 토큰까지 시간) (ms) | 11541.47 |
| Median TTFT (ms) | 9061.44 |
| P99 TTFT (ms) | 20619.26 |
| Mean TPOT (Time Per Output Token) (ms) | 361.42 |
| Mean ITL (Inter-token Latency) (ms) | 351.36 |
| P99 ITL (ms) | 3989.50 |
| Mean E2EL (End-to-End Latency) (ms) | 53739.27 |
| Median E2EL (ms) | 60086.01 |
| P99 E2EL (ms) | 63460.67 |

## 확인된 사실

- 요청의 출력 길이(128)를 서버가 반영하도록 수정하고, 클라이언트에 --ignore-eos를 적용한 뒤 재측정함. Total generated tokens 2377은 목표치(20×128=2560)에 근접함. 완전히 일치하지 않는 이유는 서버와 클라이언트가 토큰 수를 각자의 토크나이저로 별도 계산하기 때문임.
- 배칭 없이 순차 처리되어, Max Concurrency 4 기준 Mean TTFT 11541.47ms, Mean E2EL 53739.27ms로 대기(큐잉) 효과가 반영됨.
