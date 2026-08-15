# 1단계 베이스라인 서빙 측정 결과

## 측정 목적

1단계(Transformers model.generate() 직접 호출 방식) RAG 서빙의 Throughput (처리량), Latency (지연 시간) 측정값을 기록한다. 이후 vLLM 서빙 측정 결과와 비교할 기준선으로 삼는다.

## 실행 절차

### py (로컬)

```bash
# 1. 의존성 설치
uv add fastapi "uvicorn[standard]" pydantic torch transformers accelerate langchain-core langchain-huggingface sentence-transformers

# 2. 서버 실행
uv run uvicorn rag_pipeline_2.main:app --port 8000

# 3. 측정 (Linux 환경에서만 가능. macOS는 vllm 패키지 미지원)
uv add vllm

mkdir -p results

uv run vllm bench serve \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --dataset-name random \
    --num-prompts 20 \
    --max-concurrency 4 \
    --request-rate inf \
    --random-input-len 128 \
    --random-output-len 128 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result \
    --result-dir ./results \
    --result-filename baseline_result.json
```

로컬(macOS)에서 측정은 불가하고, Linux에서만 가능함. 실제 측정은 Colab에서 ipynb으로 수행함.

### ipynb (Colab)

1. GPU(T4)로 설정 후 런타임 실행.
2. retrospective.md 업로드.
3. results/baseline_result.json 확인.
4. 코드(스키마, 라우트 등)를 수정한 뒤에는 런타임을 재시작해야 함. 서버가 백그라운드 스레드로 이미 떠 있어서 해당 셀만 다시 실행해도 수정 사항이 반영되지 않기 때문.

## 측정 조건

표 1. 부하 조건
| 파라미터 | 값 |
|---|---|
| 모델 | Qwen/Qwen2.5-3B-Instruct (float16) |
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
| Benchmark duration (s) | 185.98 |
| Total input tokens | 2560 |
| Total generated tokens | 2380 |
| Request throughput (req/s) | 0.11 |
| Output token throughput (tok/s) | 12.80 |
| Peak output token throughput (tok/s) | 21.00 |
| Peak concurrent requests | 6.00 |
| Total token throughput (tok/s) | 26.56 |
| Mean TTFT (Time To First Token, 첫 토큰까지 시간) (ms) | 2370.07 |
| Median TTFT (ms) | 1689.12 |
| P99 TTFT (ms) | 5232.24 |
| Mean TPOT (Time Per Output Token) (ms) | 290.51 |
| Mean ITL (Inter-token Latency) (ms) | 283.82 |
| P99 ITL (ms) | 1164.54 |
| Mean E2EL (End-to-End Latency) (ms) | 36485.33 |
| Median E2EL (ms) | 38056.90 |
| P99 E2EL (ms) | 45133.91 |

## 확인된 사실

- 요청의 출력 길이(128)를 서버가 반영하도록 수정하고, 클라이언트에 --ignore-eos를 적용한 뒤 재측정함. Total generated tokens는 목표치(20×128=2560)에 근접함. 완전히 일치하지 않는 이유는 서버와 클라이언트가 토큰 수를 각자의 토크나이저로 별도 계산하기 때문임.
- 배칭 없이 순차 처리되어 대기(큐잉) 효과가 반영됨.
- Tesla T4 GPU(compute capability 7.5)는 bfloat16을 하드웨어 가속하지 않음. vLLM 엔진 서버 실행 시 이 사실이 에러로 확인됨(bfloat16은 compute capability 8.0 이상부터 지원). Transformers는 이 제약을 검사하지 않아 bfloat16으로도 에러 없이 동작했으나, 서빙 엔진 간 비교를 위해 정밀도를 float16으로 통일하고 재측정함.
- float16 재측정 결과, bfloat16 측정 대비 Mean TTFT(11541.47ms → 2370.07ms), Mean E2EL(53739.27ms → 36485.33ms) 모두 크게 개선됨. T4에서 bfloat16이 하드웨어 가속되지 않아 비효율적인 연산 경로를 탔던 것으로 추정됨.
