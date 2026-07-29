# uv가 관리하는 실제 인터프리터 버전(3.12.13)에 맞춤 — 시스템 python3(3.14.4)와 다름
FROM python:3.12-slim

# llama-cpp-python은 uv.lock 확인 결과 사전 빌드된 wheel이 없어(sdist만 존재)
# 설치 시점에 소스에서 컴파일됨 -> C++ 빌드 도구 필요
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# 의존성 정의 파일 먼저 복사 (레이어 캐싱: 소스 코드만 바뀌면 이 레이어는 재사용됨)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# find_project_root()가 pyproject.toml을 기준으로 PROJECT_ROOT를 찾으므로,
# src/, data/ 모두 pyproject.toml과 같은 계층 구조로 유지해야 함
COPY src ./src
COPY data ./data

RUN uv sync --frozen

# EC2에서 uv run uvicorn main:app을 src/ 안에서 실행하던 것과 동일한 조건으로 맞춤
WORKDIR /app/src

# HF Hub 캐시(GGUF 모델) 경로를 고정 -> 추후 3단계(Compose)에서 이 경로를 Volume으로 마운트 예정
ENV HF_HOME=/data/hf_cache

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
