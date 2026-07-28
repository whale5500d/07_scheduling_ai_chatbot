# 회고록

## 의사결정: Docker Registry 선택

- 문제 상황: 로컬에서 빌드한 Docker Image를 EC2 인스턴스에 배포하기 위한 경로 필요
- 고려한 옵션:
  1. Docker Hub — 관리형, 계정 생성만으로 사용 가능, AWS 인프라와 별도 인증 필요
  2. ECR (Elastic Container Registry) — AWS 관리형, IAM 통합, 동일 리전 내 전송
  3. Self-hosted Registry — 직접 인스턴스에 구축, 특별한 격리 요구사항 없이는 불필요한 관리 부담
- 결정 및 이유: ECR 선택. 이미 EC2/IAM 등 AWS 환경을 사용 중이므로 인증 통합 이점이 있고, 추후 Github Actions CI/CD 파이프라인에서 AWS 자격 증명 하나로 EC2 배포와 이미지 push를 모두 처리할 수 있어 일관성이 높음

## 에러 1 - test_full_pipeline.py 모듈 임포트 실패

- 문제 상황: Github Actions CI 도입 후 `pytest -m "not slow"` 실행 시 테스트 수집(collection) 단계에서 에러 발생
- 원인 분석: `tests/custom_transformer/test_full_pipeline.py`가 `from generate import generate`로 최상위 모듈을 import하는데, 해당 이름의 모듈이 `src/` 패키지 구조 어디에도 존재하지 않음. 특정 디렉터리를 작업 디렉터리로 두고 스크립트를 직접 실행하는 것을 전제로 작성된 것으로 추정되며, 프로젝트 루트 기준 pytest 수집 방식과 맞지 않음
- 재현 방법: 프로젝트 루트에서 `pytest -m "not slow"` 실행
- 해결 과정:
  - 근본 수정 대신 CI에서 해당 파일을 `--ignore` 옵션으로 우선 제외
  - 근본 원인(모듈 경로 문제)은 별도 이슈로 분리
- 배운 점: pytest 테스트 수집은 실행 위치(작업 디렉터리)에 의존적인 import를 허용하지 않으므로, 테스트 모듈은 항상 프로젝트 루트 기준 절대 경로로 import되도록 작성해야 함

## 에러 2 - test_vector_store.py 파일명 충돌 (import file mismatch)

- 문제 상황: 위와 동일한 pytest 실행에서 `tests/rag_pipeline/test_vector_store.py`와 `tests/langchain_pipeline/test_vector_store.py` 수집 시 충돌 에러 발생
- 원인 분석: 두 디렉터리에 동일한 파일명(`test_vector_store.py`)이 존재하고, 각 디렉터리에 `__init__.py`가 없어 pytest의 기본 import 방식(rootdir 기준 prepend)이 두 파일을 서로 다른 모듈로 구분하지 못함
- 재현 방법: 프로젝트 루트에서 `pytest -m "not slow"` 실행
- 해결 과정:
  - 근본 수정 대신 CI에서 두 파일 중 하나(`tests/rag_pipeline/test_vector_store.py`)를 `--ignore` 옵션으로 우선 제외
  - 근본 원인(패키지 구조에 `__init__.py` 부재)은 별도 이슈로 분리
- 배운 점: 여러 디렉터리에 동일한 테스트 파일명을 두려면, 각 디렉터리에 `__init__.py`를 추가해 패키지로 인식시키거나 `pyproject.toml`에서 `--import-mode=importlib`를 지정해야 이름 충돌을 피할 수 있음

## 에러 3 - test_transformer_model.py 시드 재현성에 의존한 테스트 설계 문제

- 문제 상황: `torch.manual_seed(0)`으로 고정한 결정적 재현 테스트가 로컬(Mac, torch 최신)과 CI(Github Actions, torch cpu 인덱스)에서 서로 다른 결과를 내며 실패
- 원인 분석: `torch.manual_seed`는 같은 torch 버전·같은 하드웨어(CPU 아키텍처) 내에서만 비트 단위 재현을 보장함. 플랫폼이 다르면 시드가 같아도 난수 생성 결과가 달라질 수 있어, "시드 고정으로 특정 토큰이 반드시 나온다"는 테스트의 전제 자체가 이식 가능(portable)하지 않았음
- 재현 방법: 로컬 Mac과 EC2/CI에서 각각 `torch.manual_seed(0)` 고정 테스트 실행 시 서로 다른 토큰 시퀀스 생성 확인
- 해결 과정:
  - 검증 대상을 "어떤 토큰이 샘플링되는가"(확률적, 플랫폼 의존적)와 "샘플링된 토큰이 eos일 때 즉시 멈추는가"(결정론적 로직)로 분리
  - `torch.multinomial`을 `monkeypatch`로 고정된 반환값을 주도록 대체하여, 시드나 플랫폼과 무관하게 조기 종료 메커니즘만 독립적으로 검증
  - 기존 하나의 통합 테스트 함수를 `TestTransformerLanguageModelForwardPass`, `TestTransformerLanguageModelGenerate` 클래스로 책임 분리
- 배운 점: 신경망 관련 테스트에서 `manual_seed`로 특정 출력값을 기대하는 방식은 플랫폼/버전 간 이식성이 없음. 확률적 요소와 결정론적 로직이 섞인 함수를 테스트할 때는, 확률적 요소를 monkeypatch로 고정해 결정론적 로직만 분리해서 검증해야 함

## 이슈 2 - test_langsmith_eval.py mock 함수 시그니처 불일치

- 문제 상황: `fake_evaluate_faithfulness()`가 인자 3개로 정의되어 있는데, 실제 `evaluate_faithfulness()` 호출은 인자 4개(judge 관련 인자 추가)로 이루어져 TypeError 발생
- 원인: 실제 함수 시그니처가 변경된 이후 테스트의 mock 함수가 갱신되지 않음
- 결정: 현재 로드맵 범위 밖이므로 수정하지 않고 이슈로 기록. `@pytest.mark.slow` 처리로 CI에서는 우선 제외됨

## 의사결정: pytest slow 마커 적용 단위

- 문제 상황: CI 테스트 실행 시간이 과도하게 길어(5분+), tests/evaluate/ 하위 judge LLM 호출 테스트에 slow 마커를 적용했으나, 개별 함수 단위로 마킹할 때마다 매번 다른 테스트가 대신 60초 이상 소요되는 현상이 반복됨
- 원인 분석: 해당 테스트들은 scope="module" 또는 scope="class" fixture로 judge 모델을 한 번만 초기화하는 구조. 개별 함수를 slow로 제외해도, 그 모듈/클래스에서 다음으로 실행되는 남은 테스트가 fixture 초기화 비용을 그대로 이어받음
- 결정 및 이유: 함수 단위가 아니라, 무거운 fixture를 공유하는 클래스 전체에 @pytest.mark.slow를 적용. fixture의 scope를 먼저 확인한 뒤 마킹 단위를 결정하는 것이 원칙
- 인사이트: pytest에서 슬로우 테스트를 분류할 때는 "어떤 테스트가 오래 걸리는가"가 아니라 "어떤 fixture가 무거운 리소스를 scope 단위로 공유하는가"를 먼저 확인해야 함

## 의사결정: docker-compose.yml 동기화 방식 (방식 B 채택)

- 문제 상황: Github Actions 자동 배포(deploy job)에서 EC2의 docker-compose.yml을 저장소 기준으로 항상 최신화할지(방식 A), 아니면 이미지 갱신만 자동화하고 compose 설정은 별도로 관리할지(방식 B) 결정 필요
- 고려한 옵션:
  1. 방식 A: deploy 스텝에서 `git pull` 후 `docker compose pull && up -d` 실행. docker-compose.yml 변경사항까지 자동 반영되지만, 배포 스텝의 책임 범위가 넓어짐
  2. 방식 B: `docker compose pull && up -d`만 실행. 이미지 갱신만 자동화, docker-compose.yml 변경은 반영되지 않음
- 결정 및 이유: 방식 B 채택. 현재 EC2를 프로젝트 진행 시에만 켜두는 방식으로 운영해(비용 절감 목적) 퍼블릭 IP가 고정되지 않고(Elastic IP 도입 보류), Github Secret의 EC2_HOST를 인스턴스를 켤 때마다 수동으로 갱신해야 하는 반자동 구조로 이미 운영 중임. 이 제약 위에서는 배포 스크립트의 책임 범위를 최소화하는 것이 더 일관됨
- 인사이트: 방식 B를 유지하는 한, docker-compose.yml을 수정할 때마다 로컬 저장소와 EC2 양쪽에 수동으로 동일하게 반영해야 함(자동화되지 않는 부분). 두 곳이 어긋나면 저장소 기준으로는 최신인데 실제 배포된 컨테이너는 이전 설정(포트, 볼륨 등)으로 남아있는 불일치가 발생할 수 있으므로, docker-compose.yml 변경 시 반드시 EC2 쪽도 함께 갱신했는지 체크리스트로 확인할 것

## 에러 4 - deploy job SSH 접속 실패 (i/o timeout)

- 문제 상황: Github Actions `deploy` job 최초 실행 시 EC2로의 SSH 접속 자체가 `i/o timeout`으로 실패
- 원인 분석: EC2 보안 그룹의 22번 포트 인바운드 규칙이 "내 IP"(집/오피스 두 곳)로만 제한되어 있었음. Github Actions runner는 매 실행마다 무작위 IP에서 접속을 시도하므로, 특정 IP로 좁혀진 인바운드 규칙으로는 애초에 접속 자체가 불가능한 구조였음
- 재현 방법: 보안 그룹 22번 포트가 특정 IP로 제한된 상태에서 Github Actions의 SSH 기반 job 실행
- 해결 과정:
  - 22번 포트 인바운드 소스를 `0.0.0.0/0`으로 개방하여 해결
- 배운 점: CI/CD 러너(Github Actions 등)에서 오는 접속은 고정된 소수 IP가 아니라 매번 바뀌는 광범위한 IP 대역에서 발생하므로, "내 IP" 제한 방식의 보안 그룹 규칙과 근본적으로 호환되지 않음. 자동화 파이프라인이 접근해야 하는 포트는 전체 개방하거나, 해당 서비스가 공개하는 IP 대역만 별도로 허용해야 함

## 에러 5 - deploy 단계 ECR 로그인 누락으로 인한 배포 실패 (성공으로 오인)

- 문제 상황: Github Actions deploy job이 "succeeded"로 표시되었으나, 실제로는 새 이미지가 EC2에 반영되지 않고 이전 컨테이너가 그대로 유지됨. /health 응답에 새로 추가한 version 필드가 나타나지 않아 발견함
- 원인 분석: deploy 스크립트가 `docker compose pull` 실행 전에 ECR 로그인을 스스로 수행하지 않음. 이전 실행(build-and-push job 또는 사람이 직접 실행한 세션)에서 남겨진 EC2의 ECR 인증 토큰(~/.docker/config.json, 유효기간 12시간)에 우연히 의존하고 있었음. 토큰이 만료된 상태에서 실행되어 `docker compose pull`이 403 Forbidden으로 실패했으나, 스크립트가 다음 줄(`docker compose up -d`)로 그대로 진행되었고, 이 명령 자체는 기존 컨테이너를 그대로 둔 채 에러 없이 끝나 job 전체가 성공으로 표시됨
- 재현 방법: EC2의 ECR 로그인 토큰이 만료된 상태에서 deploy job 실행 시 재현
- 해결 과정:
  - deploy 스크립트 맨 앞에 `aws ecr get-login-password | docker login` 단계를 추가해, 매 배포마다 독립적으로 로그인을 새로 수행하도록 수정 (EC2에 연결된 IAM Role 권한을 그대로 활용, Access Key 불필요)
- 배운 점:
  - 셸 스크립트 형태의 배포 단계에서는 앞 명령의 실패가 뒷 명령 실행을 자동으로 막지 않는다 (set -e 등이 없는 한). 여러 명령을 이어 실행할 때는 각 단계가 실패 시 전체를 중단시키는지 확인이 필요함
  - CI job의 "성공" 표시는 "마지막 명령이 에러 코드 없이 끝났는가"를 의미할 뿐, "의도한 결과가 실제로 발생했는가"를 보장하지 않음. 배포 자동화에는 배포 자체의 성공 여부와는 별개로, 결과를 확인하는 별도의 검증 단계(health check 등)가 필요함
  - 인증이 필요한 각 단계는 이전 단계나 이전 세션의 인증 상태에 암묵적으로 의존하지 않고, 그 단계 자체에서 독립적으로 인증을 수행하도록 설계해야 함

## 에러 6 - deploy job 디스크 공간 부족 및 성공 표시 오탐 재발

- 문제 상황: 에러 5(ECR 로그인 누락) 해결 이후 재실행한 `deploy` job에서, `docker compose pull`의 레이어 압축 해제 단계가 `no space left on device`로 실패. 이때도 `deploy` job 자체는 다시 "success"로 표시됨(에러 5와 동일한 오탐 패턴 재발)
- 원인 분석:
  - 디스크 부족: EC2의 EBS 볼륨(19GB)에서, 이미지 레이어(약 3GB) 압축 해제 시 순간적으로 필요한 여유 공간이 부족했음. `docker builder prune`으로 빌드 캐시를 정리해도, 다음 pull 시도에서 다시 재발할 만큼 여유 공간 자체가 구조적으로 빠듯한 상태였음
  - 성공 표시 오탐 재발: `deploy` 스크립트에 `docker compose pull` 실패 시 스크립트 실행을 중단시키는 장치(`set -e` 등)가 없어, `pull` 실패 후에도 다음 줄(`docker compose up -d`)로 그대로 진행되고, 그 명령이 기존 컨테이너를 그대로 둔 채 에러 없이 끝나 전체가 성공으로 보고됨
- 재현 방법: EBS 여유 공간이 이미지 레이어 크기 대비 부족한 상태에서 `docker compose pull` 실행
- 해결 과정:
  - EBS 볼륨을 19GB → 28GB로 확장 (`growpart /dev/nvme0n1 1`, `resize2fs /dev/root`로 온라인 확장, 인스턴스 재시작 불필요)
  - `deploy` 스크립트 최상단에 `set -e` 추가 — 이후 어떤 명령이든 실패 시 스크립트 전체가 즉시 중단되고 job이 정확히 실패로 반영됨
- 배운 점: 무거운 ML 의존성이 포함된 이미지는 압축 해제 시 이미지 크기의 1.5~2배 정도의 여유 디스크 공간이 필요할 수 있음. 셸 스크립트 기반 배포 단계에서는 각 명령의 실패가 다음 명령 실행을 자동으로 막지 않으므로(`set -e` 없이는), 여러 명령을 이어 실행하는 배포 스크립트에는 반드시 실패 전파 장치를 넣어야 함

## 의사결정: EC2 Role IAM 정책 FullAccess → PowerUser 교체

- 문제 상황: EC2 Role(`whale5500d-scheduling-ai-chatbot`)에 `AmazonEC2ContainerRegistryFullAccess`가 연결되어 있었고, Github Actions용 IAM 사용자 생성을 계기로 이 권한 범위가 필요 이상으로 넓은 것은 아닌지 확인 필요
- 고려한 옵션:
  1. `AmazonEC2ContainerRegistryFullAccess` 유지 — 실제 정책 문서 확인 결과 `"ecr:*"`로, push/pull뿐 아니라 저장소 삭제(`DeleteRepository`), 저장소 접근 정책 변경까지 포함하는 관리자급 권한
  2. `AmazonEC2ContainerRegistryPowerUser`로 교체 — push/pull 전용, 저장소 삭제·정책 변경 권한 없음
- 결정 및 이유: PowerUser로 교체. 이 EC2가 실제로 수행하는 동작은 이미지 pull(및 검증 단계의 push)뿐이므로 최소 권한 원칙에 부합. 교체 후 `aws ecr describe-repositories --region ap-northeast-2`로 기존 pull 동작이 정상 유지되는지 재검증하여 확인
- 인사이트: 탈취 시 피해 범위는 그 자격 증명에 연결된 IAM 정책이 허용하는 범위 그 자체이며, PowerUser로 좁히면 저장소 자체를 삭제당하거나 접근 정책을 임의로 변경당하는 위험을 원천 차단할 수 있다는 점
