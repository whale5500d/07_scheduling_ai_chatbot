# 11주차 챌린지 - Docker 기반 DevOps 적용

## 위클리 챌린지 로드맵

위클리 챌린지 로드맵
| 순서 | 무엇을 | 왜 | 산출물 |
| --- | --- | --- | --- |
| 1 | 기존 DaySync FastAPI 서버 기준 Dockerfile(도커파일) 작성 | 컨테이너 패키징의 최소 단위(단일 이미지)부터 검증해야 이후 Compose(컴포즈)·배포 단계가 안정적임 | Dockerfile |
| 2 | 로컬에서 `docker build`로 이미지 빌드 및 `docker run`으로 단독 실행 검증 | Compose 이전에 이미지 자체가 정상 동작하는지 격리된 상태로 먼저 확인 | 로컬 실행 가능한 Docker Image(도커 이미지) |
| 3 | Docker Compose(도커 컴포즈) 파일(`docker-compose.yml`) 작성 | 서버 실행 방식을 단일 명령어로 재현 가능하게 고정, 이후 서비스 확장(예: DB, 프록시 컨테이너 추가) 대비 | docker-compose.yml |
| 4 | 로컬에서 `docker compose up`으로 실행 검증 | 2단계(단독 실행)와 Compose 실행 결과가 동일한지 대조하여 Compose 설정 자체의 오류 여부 분리 | 로컬 Compose 실행 로그 |
| 5 | 이미지 배포 경로 결정 (Docker Registry(도커 레지스트리) 경유 vs 직접 전송) 및 Registry 계정/저장소 준비 | EC2로 이미지를 옮기는 방법이 여러 개(ECR, Docker Hub, `docker save`+`scp`)이므로 근거를 갖고 하나를 확정해야 이후 단계가 일관됨 | 결정 및 근거 기록 (RETROSPECTIVE.md 의사결정 레코드) |
| 6 | EC2 인스턴스에 Docker 설치 (기존 인스턴스 재사용 여부 확인 후 진행) | 컨테이너 실행 환경이 EC2에 있어야 이미지를 pull 및 실행 가능 | Docker 설치 완료된 EC2 |
| 7 | 5단계 방식으로 이미지를 EC2에 전달 후 `docker compose up`으로 컨테이너 실행 | 로컬에서 검증한 Compose 구성을 배포 대상 환경에서 동일하게 재현 | EC2 상에서 실행 중인 컨테이너 |
| 8 | 기존 nginx 리버스 프록시 설정과 컨테이너 포트 연결(연동) | 외부 접근 경로(80번 포트)를 컨테이너화된 서버로 그대로 유지 | 갱신된 nginx 설정 |
| 9 | 로컬 Mac에서 curl로 EC2 컨테이너 서버 응답 확인 | 컨테이너 배포가 실제 외부 접근 가능 상태인지 최종 검증 | 요청-응답 로그 |
| 10 | Github Actions(깃허브 액션즈) 워크플로우 파일(`.github/workflows/*.yml`) 작성 — CI(빌드/테스트) 단계 | 코드 푸시 시 자동 검증 단계부터 먼저 구축해야 CD(배포) 단계 실패 원인을 CI와 분리해서 진단 가능 | CI 워크플로우 파일 |
| 11 | 워크플로우에 이미지 빌드 및 Registry push 단계(CD 일부) 추가 | 5단계에서 확정한 Registry 경로를 자동화 파이프라인에 반영 | 이미지 자동 push 확인 로그 |
| 12 | EC2 자동 배포 단계 추가 (SSH 원격 명령 방식) 구성 | 사람이 수동으로 pull/재시작하던 7단계 과정을 파이프라인으로 대체 | 자동 배포 워크플로우 완성본 |
| 13 | 코드 푸시 → 전체 파이프라인(CI→이미지 push→EC2 자동 배포) 통합 테스트 | 각 단계가 개별로는 동작해도 전체 연결 시 실패할 수 있어 종단 간(end-to-end) 검증 필요 | 파이프라인 실행 로그 + 배포 확인 |
| 14 | 전체 회고 작성 | 과제 통합 마무리 및 트러블슈팅 기록 | RETROSPECTIVE.md 갱신 |

## 1단계 - Dockerfile 작성

### 사전 확인 사항

- `pyproject.toml`/`uv.lock` 확인 결과, `llama-cpp-python`은 사전 빌드된 wheel 없이 sdist(소스 배포판)만 등록되어 있어, 설치 시점에 소스에서 직접 컴파일됨을 확인. base image에 C++ 빌드 도구(build-essential, cmake) 필요.
- `src/paths.py`의 `find_project_root()`가 `pyproject.toml`을 마커로 상위 디렉터리를 탐색하는 구조임을 확인. 이미지 안에서도 `pyproject.toml`과 `src/`, `data/`가 동일한 상대 구조로 유지되어야 함.

## 2단계 - 로컬 빌드 및 단독 실행 검증

### 진행 과정

- EC2에 Docker Engine 미설치 상태 확인 → 공식 설치 스크립트(`get-docker.sh`)로 설치, `docker` 그룹에 사용자 추가
- `docker build -t scheduling-chatbot:test .` 빌드 성공 확인 (llama-cpp-python 컴파일 포함 약 14/14 단계 완료)
- `docker run` 실행 시 HF Hub 캐시(`~/.cache/huggingface`)를 `/data/hf_cache`로 마운트해, 기존에 받아둔 GGUF 모델을 재사용하도록 구성
- `curl http://localhost:8000/health` 응답으로 정상 기동 확인 (`indexed_chunks: 7`까지 확인되어 RAG 인덱싱까지 정상 동작 검증)

## 3단계 - Docker Compose 파일(docker-compose.yml) 작성

### 진행 과정

- `docker-compose.yml` 최초 작성 시 GGUF 모델을 Volume으로 마운트하는 계획을 세웠으나, 실제로는 `MODEL_PATH` 같은 단일 파일 경로가 아니라 HF Hub 캐시 디렉터리 전체(`HF_HOME`)를 마운트해야 하는 구조임을 1단계에서 먼저 확인함
- `.env` 파일 존재 및 키 구성(`GOOGLE_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`) 확인 — `RAG_BACKEND`는 미설정 상태로, 기본값(`rag_pipeline`)이 그대로 사용됨을 확인

## 4단계 - 로컬 docker compose up 실행 검증

### 진행 과정

- `docker compose up --build -d`로 기동 후 `curl http://localhost:8000/health` 응답으로 `indexed_chunks: 7` 확인 — 2단계 단독 실행 결과와 동일하게 정상 동작 확인

### 트러블슈팅

- `docker compose up --build -d` 최초 실행 시 포트 충돌(`port is already allocated`) 발생
  - 원인: 2단계에서 `docker run` 실행 시 `--rm` 옵션을 쓰지 않아, 검증 후에도 컨테이너가 백그라운드로 남아 8000번 포트를 계속 점유하고 있었음
  - 해결: `docker ps`로 잔존 컨테이너 확인 후 `docker stop`으로 정리, 재실행

## 5단계 - 이미지 배포 경로 결정 (Docker Registry)

- ECR(Elastic Container Registry) 선택. 상세 의사결정 근거(고려한 옵션, 결정 이유)는 [`05_ci_cd.md` - 의사결정: Docker Registry 선택](./retrospective/05_ci_cd.md#의사결정-docker-registry-선택) 참고

## 6단계 - EC2 인스턴스에 Docker 설치

### 진행 과정

- `docker` 명령 실행 시 미설치 상태 확인 (`docker: command not found`)
- Docker 공식 설치 스크립트 사용 (Ubuntu 저장소의 `docker.io`보다 Compose plugin까지 함께 설치되는 공식 스크립트 선택)

\`\`\`bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
\`\`\`

- `docker --version`(29.6.2), `docker compose version`(v5.3.1)으로 설치 확인

## 6단계 - EC2 인스턴스에 Docker 설치

### 진행 과정

- `docker` 명령 실행 시 미설치 상태 확인 (`docker: command not found`)
- Docker 공식 설치 스크립트 사용 (Ubuntu 저장소의 `docker.io`보다 Compose plugin까지 함께 설치되는 공식 스크립트 선택)

\`\`\`bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
\`\`\`

- `docker --version`(29.6.2), `docker compose version`(v5.3.1)으로 설치 확인

## 6단계 - EC2 인스턴스에 Docker 설치

### 진행 과정

- `docker` 명령 실행 시 미설치 상태 확인 (`docker: command not found`)
- Docker 공식 설치 스크립트 사용 (Ubuntu 저장소의 `docker.io`보다 Compose plugin까지 함께 설치되는 공식 스크립트 선택)

\`\`\`bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
\`\`\`

- `docker --version`(29.6.2), `docker compose version`(v5.3.1)으로 설치 확인

## 7단계 - 5단계 방식(ECR)으로 이미지를 EC2에 전달 후 컨테이너 실행

### 진행 과정

- ECR 인증을 위해 EC2 인스턴스에 IAM Role(`whale5500d-scheduling-ai-chatbot`) 연결 — Access Key를 인스턴스에 남기지 않기 위함
- ECR repository 생성, 로컬에서 빌드한 이미지를 태깅 후 push

\`\`\`bash
aws ecr create-repository --repository-name scheduling-chatbot --region ap-northeast-2
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 668425684373.dkr.ecr.ap-northeast-2.amazonaws.com
docker tag scheduling-chatbot:latest 668425684373.dkr.ecr.ap-northeast-2.amazonaws.com/scheduling-chatbot:latest
docker push 668425684373.dkr.ecr.ap-northeast-2.amazonaws.com/scheduling-chatbot:latest
\`\`\`

- `docker-compose.yml`을 `build: .`(로컬 빌드) 방식에서 `image:`(ECR 주소 직접 지정) 방식으로 전환

### 트러블슈팅

- IAM Role 연결 직후 `curl http://169.254.169.254/latest/meta-data/iam/security-credentials/`가 빈 값을 반환 — 원인은 이 인스턴스가 IMDSv2(토큰 기반)만 허용하는 설정이라, 토큰 없이 보낸 요청은 응답하지 않았기 때문. 토큰을 발급받아 헤더에 포함해 재요청하여 해결
- 첫 `docker compose up` 검증 시, `docker-compose.yml`에 `build: .`가 남아있어 실제로는 ECR pull이 아니라 로컬 재빌드 경로로 검증되고 있었음(로그의 `naming to docker.io/library/scheduling-chatbot:latest`로 발견). 로컬 이미지 태그를 강제로 삭제한 뒤 재실행해, 실제로 ECR에서 pull되는지(`Image ... Pulled` 로그) 다시 검증하여 바로잡음

## 8단계 - 기존 nginx 리버스 프록시 설정과 컨테이너 포트 연결

### 진행 과정

- 기존 nginx 설정(`/etc/nginx/sites-available/daysync`, `proxy_pass 127.0.0.1:8000`)을 확인한 결과, `docker-compose.yml`의 `ports: "8000:8000"`이 호스트 8000번을 컨테이너 8000번에 그대로 연결하므로, nginx 입장에서는 컨테이너 도입 전후로 대상(`127.0.0.1:8000`)이 동일함을 확인
- 별도 nginx 설정 변경 없이, `curl http://localhost:80/health`로 nginx 경유 정상 응답 확인

## 9단계 - 로컬 Mac에서 curl로 EC2 컨테이너 서버 응답 확인

### 진행 과정

- 인바운드 규칙에서 8000번 포트 제거 확인 (nginx가 대신 응답하는 80번만 외부에 개방하는 것이 맞는 구성)
- 로컬 Mac 터미널에서 `curl http://<EC2_퍼블릭_IP>/health` 정상 응답 확인

### 트러블슈팅

- 최초 검증 시, EC2 인스턴스 안에서 자기 자신의 퍼블릭 IP로 curl을 실행해 "외부 접근"을 확인했다고 판단했으나, 이는 실제 외부 클라이언트 경로와 다르게 동작할 수 있어(hairpin NAT 등) 유효한 검증이 아니었음 — 로컬 Mac에서 동일 명령을 재실행해 최종 검증

## 10단계 - Github Actions 워크플로우 작성 (CI: 빌드/테스트)

### 진행 과정

- `.github/workflows/ci.yml`의 `test` job 작성 — `uv python install 3.12`로 EC2 실제 인터프리터 버전과 동일하게 고정, `pytest -m "not slow"` 실행

### 트러블슈팅

- 최초 실행 시 테스트 수집(collection) 단계에서 에러 2건 발생. `--ignore` 옵션으로 우선 제외. 상세 원인은 [`05_ci_cd.md` - 에러 1](./retrospective/05_ci_cd.md#에러-1---test_full_pipelinepy-모듈-임포트-실패), [에러 2](./retrospective/05_ci_cd.md#에러-2---test_vector_storepy-파일명-충돌-import-file-mismatch) 참고
- CI 테스트 실행 시간 과다(5분 이상) → `tests/evaluate/` 하위 judge LLM 호출 테스트에 `@pytest.mark.slow` 적용. 개별 함수 단위로 마킹할 때마다 fixture scope(모듈/클래스)로 인해 다른 테스트가 비용을 대신 떠안는 현상이 반복되어, 클래스 단위 마킹으로 전환해 해결. 상세 내용은 [`05_ci_cd.md` - 의사결정: pytest slow 마커 적용 단위](./retrospective/05_ci_cd.md#의사결정-pytest-slow-마커-적용-단위) 참고
- `test_transformer_model.py`가 `torch.manual_seed` 재현성에 의존한 설계로 로컬/CI 간 다른 결과를 내며 실패 → 확률적 요소(샘플링)와 결정론적 로직(조기 종료)을 분리해 재작성하여 해결. 상세 내용은 [`05_ci_cd.md` - 에러 3](./retrospective/05_ci_cd.md#에러-3---test_transformer_modelpy-시드-재현성에-의존한-테스트-설계-문제) 참고
- `test_langsmith_eval.py`의 mock 함수 시그니처 불일치(TypeError)는 로드맵 범위 밖으로 판단해 수정하지 않고 `@pytest.mark.slow`로 CI에서 제외, 이슈로만 기록. 상세 내용은 [`05_ci_cd.md` - 이슈 2](./retrospective/05_ci_cd.md#이슈-2---test_langsmith_evalpy-mock-함수-시그니처-불일치) 참고

## 11단계 - 워크플로우에 이미지 빌드 및 Registry push 단계(CD 일부) 추가

### 진행 과정

- Github Actions에서 ECR push를 위한 AWS 인증 방식으로 Access Key 방식 선택 (OIDC 대비 간편함 우선, 개인 프로젝트 규모 고려)
- IAM 정책 비교: `AmazonEC2ContainerRegistryFullAccess`(저장소 삭제·정책 변경까지 포함)와 `AmazonEC2ContainerRegistryPowerUser`(push/pull만) 중 최소 권한 원칙에 따라 PowerUser 선택
- EC2 Role(`whale5500d-scheduling-ai-chatbot`)에 연결되어 있던 FullAccess를 PowerUser로 교체, 교체 후 `aws ecr describe-repositories`로 기존 동작 정상 재확인
- Github Actions 전용 신규 IAM 사용자 생성(PowerUser 정책), Access Key 발급 → Github 저장소 Secrets(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ECR_REPOSITORY_URI`) 등록
- `.github/workflows/ci.yml`에 `build-and-push` job 추가 (`needs: test`로 테스트 통과 후에만 실행, push 이벤트에서만 실행)
- 커밋·푸시 후 Actions에서 `test` job 성공(3분 52초) 확인

### 트러블슈팅

- `build-and-push` job 실행 결과 `docker build` 단계에서 `open Dockerfile: no such file or directory` 발생
  - 원인: 1~3단계에서 작성한 `Dockerfile`, `docker-compose.yml`, `.dockerignore`를 EC2 위에서 `vim`으로 직접 생성했을 뿐, 로컬 Mac(실제 git 저장소가 있는 곳)에는 한 번도 반영되지 않은 상태였음
  - 해결: `scp`로 EC2의 세 파일을 로컬 Mac으로 가져온 뒤 커밋·푸시

## 11단계 - 워크플로우에 이미지 빌드 및 Registry push 단계(CD 일부) 추가

### 진행 과정

- Github Actions에서 ECR push를 위한 AWS 인증 방식으로 Access Key 방식 선택 (OIDC 대비 간편함 우선, 개인 프로젝트 규모 고려)
- EC2 Role(`whale5500d-scheduling-ai-chatbot`)의 IAM 정책을 FullAccess에서 PowerUser로 교체
- Github Actions 전용 신규 IAM 사용자 생성(PowerUser 정책), Access Key 발급 → Github 저장소 Secrets(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ECR_REPOSITORY_URI`) 등록
- `.github/workflows/ci.yml`에 `build-and-push` job 추가 (`needs: test`로 테스트 통과 후에만 실행, push 이벤트에서만 실행)
- 커밋·푸시 후 Actions에서 `test` job 성공(3분 52초) 확인

### 트러블슈팅

- EC2 Role IAM 정책(FullAccess → PowerUser) 교체 배경 및 근거는 [`05_ci_cd.md` - 의사결정: EC2 Role IAM 정책 FullAccess → PowerUser 교체](./retrospective/05_ci_cd.md#의사결정-ec2-role-iam-정책-fullaccess--poweruser-교체) 참고
- `build-and-push` job 실행 결과 `docker build` 단계에서 `open Dockerfile: no such file or directory` 발생
  - 원인: 1~3단계에서 작성한 `Dockerfile`, `docker-compose.yml`, `.dockerignore`를 EC2 위에서 `vim`으로 직접 생성했을 뿐, 로컬 Mac(실제 git 저장소가 있는 곳)에는 한 번도 반영되지 않은 상태였음
  - 해결: `scp`로 EC2의 세 파일을 로컬 Mac으로 가져온 뒤 커밋·푸시

## 12단계 - EC2 자동 배포 단계 추가 (SSH 원격 명령 방식)

### 진행 과정

- Github Secrets에 `EC2_HOST`, `EC2_SSH_KEY` 추가 등록
- `.github/workflows/ci.yml`에 `deploy` job 추가 (`needs: build-and-push`, `appleboy/ssh-action` 사용)
- `docker-compose.yml` 동기화 방식은 방식 B(이미지 갱신만 자동화, compose 설정 변경은 수동 동기화) 채택

### 트러블슈팅

- 비용 절감을 위해 EC2를 필요할 때만 켜는 운영 방식을 유지하기로 하면서, Elastic IP 도입은 보류하고 `EC2_HOST` Secret을 인스턴스를 켤 때마다 수동 갱신하는 방식으로 결정
- `docker-compose.yml` 동기화 방식(방식 A vs B) 선택 배경은 [`05_ci_cd.md` - 의사결정: docker-compose.yml 동기화 방식](./retrospective/05_ci_cd.md#의사결정-docker-composeyml-동기화-방식-방식-b-채택) 참고
- 배포 시도 과정에서 파악된 세부 에러
  - 1차 시도: SSH 접속 자체 실패 (i/o timeout): [`05_ci_cd.md` - 에러 4](./retrospective/05_ci_cd.md#에러-4---deploy-job-ssh-접속-실패-io-timeout)
  - 2차 시도: ECR 로그인 누락으로 인한 배포 실패 (성공 표시로 오인): [`05_ci_cd.md` - 에러 5](./retrospective/05_ci_cd.md#에러-5---deploy-단계-ecr-로그인-누락으로-인한-배포-실패-성공으로-오인)
  - 3차 시도: 디스크 공간 부족 및 성공 표시 오탐 재발: [`05_ci_cd.md` - 에러 6](./retrospective/05_ci_cd.md#에러-6---deploy-job-디스크-공간-부족-및-성공-표시-오탐-재발)

### 추가로 다뤄볼 만한 항목

- Elastic IP 과금 정책(2024년 2월 이후 attach 여부 무관 과금)과 일반 퍼블릭 IP의 차이
- Docker `build`/`push`/`pull`/`up` 명령어 흐름과, EC2에서 ECR 인증 토큰이 세션 간 공유되는 방식(`~/.docker/config.json`, 12시간 만료)

## 13단계 - 코드 푸시 → 전체 파이프라인(CI→이미지 push→EC2 자동 배포) 통합 테스트

### 진행 과정

- 개별 단계(10~12단계)가 각각 성공했더라도, 실제로 코드 변경이 이미지에 반영되어 배포까지 이어지는지는 별도 검증이 필요하다고 판단
- `/health` 응답에 배포 검증용 `APP_VERSION`("v1-pipeline-test") 필드를 추가해, 이 값이 EC2까지 실제로 반영되는지를 종단 간(end-to-end) 확인 지표로 사용
- `main.py`에 상수 추가 및 `health()` 반환값 수정, `tests/test_main.py`의 관련 assert 3건도 함께 갱신 후 로컬 테스트 통과 확인
- 커밋·푸시 후 `test` → `build-and-push` → `deploy` 순서로 파이프라인 실행, 최종적으로 로컬 Mac에서 `curl http://<EC2_퍼블릭_IP>/health` 응답에 `"version":"v1-pipeline-test"` 확인

### 트러블슈팅

- 이 통합 테스트 과정에서 12단계의 에러 5(ECR 로그인 누락), 에러 6(디스크 공간 부족)이 실제로 발견되고 수정됨. 개별 단계 검증만으로는 드러나지 않았던 문제들이 종단 간 테스트에서 실제로 드러난 사례

## 14단계 - 전체 회고 작성

### 최종 인사이트

- 지난 주(10주차, Linux/OS 기초)가 EC2 인스턴스 위에 개인 프로젝트를 직접 배포·운영하는 것이었다면, 이번 주(11주차)는 그 위에 Docker 컨테이너화를 얹어 DevOps가 실제로 무엇을 다루는 영역인지 체득하는 과정이었음
- AWS 사용 범위가 EC2 단일 인스턴스 운영에서, ECR(이미지 저장소), IAM(Role/User/정책), Github Actions와의 인증 연동으로 확장됨. "Docker 이미지를 어떻게 안전하게 빌드·저장·배포할 것인가"라는 하나의 문제를 해결하면서 익힘.
- 실제로 겪은 에러들(포트 충돌, IAM 권한, SSH 접속 제한, ECR 인증 만료, 디스크 공간 부족, CI 성공 표시 오탐 등)이 각각 AWS/Docker/Linux의 서로 다른 계층에서 발생했고, 그 중 일부(Elastic IP 과금 정책, Docker 배포 파이프라인 전체 흐름과 인증 토큰 공유 방식, Secrets 중앙 관리 도구)는 이번 챌린지 범위를 벗어나는 별도 딥다이브 주제로 선별해 남겨둠
- 챌린지를 관통하는 하나의 인사이트: Docker는 새로운 격리 기술을 발명한 것이 아니라, 리눅스 커널의 namespace(격리)와 cgroups(자원 제한)를 union filesystem 기반의 계층형 이미지와 결합해, 실행 환경 자체를 코드와 함께 재현 가능한 단위로 고정한 도구였음
