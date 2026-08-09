# Server Configuration Setting

## 3단계 - 프로젝트 구동

### 3단계 트러블슈팅 1 - 메모리 및 디스크 부족 문제 개선 진행 과정

#### 문제 상황

- 생성 인스턴스 현황
  - AMI: Ubuntu 26.04 LTS (HVM), SSD Volume Type
  - 인스턴스 유형: t3.micro (1GiB)
  - 네트워크 설정: SSH 트래픽 허용 ("내 IP" 사용)
  - 그 외 변경 사항 없음. (초기 상태 유지)

- 인스턴스의 작은 용량 대비 프로젝트 용량이 커서 여러가지 세부 문제 직면
- 전체적인 원인과 해결 과정을 순차적으로 수행한 결과를 아래 내용으로 정리

#### 1차 시도 - torch 기본 빌드의 디스크 공간 부족

##### 문제 상황

- uv sync 실행 후, 에러 메시지 확인

  ```
  × Failed to download `torch==2.13.0`
  ├─▶ Failed to extract archive: torch-2.13.0-cp314-cp314-manylinux_2_28_x86_64.whl
  ├─▶ I/O operation failed during extraction
  ╰─▶ failed to write to file `.../torch/lib/libtorch_cuda.so`: No space left on device (os error 28)
  ```

##### 원인 분석

- 기본 PyPI 인덱스의 torch는 CUDA(GPU) 런타임 라이브러리(libtorch_cuda.so 등)를 포함한 빌드라 용량이 수 GB에 달함.
- GPU가 없는 t3.micro 인스턴스에서 불필요한 대용량 빌드이며, 8GiB 루트 볼륨(실사용 가능 6.7GiB)이 이를 받아 압축 해제할 여유가 없었음

##### 시도 과정

- (커밋 2f97e59) pyproject.toml에 [tool.uv.sources]로 torch = { index = "pytorch-cpu" } 및 pytorch-cpu 인덱스(https://download.pytorch.org/whl/cpu) 추가 (CUDA 미포함 CPU 전용 빌드로 전환)

#### 2차 시도 - torch 기본 인덱스 전환 후 지속된 디스크 공간 부족

##### 문제 상황

- `uv sync` 재실행 결과, 여전히 동일 성격의 에러 확인
  ```
  × Failed to download torch==2.13.0+cpu
  ├─▶ Failed to extract archive: torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl
  ├─▶ I/O operation failed during extraction
  ╰─▶ failed to write to file .../torch/lib/libtorch_python.so: No space left on device (os error 28)
  ```

##### 원인 분석

- `df -h`로 디스크 상태 확인 → 루트 파티션(`/dev/root`) 92% 사용, 585MiB 미만 남음
- torch를 CPU 전용 인덱스로 전환해 패키지 자체 용량은 줄였으나, 1차 시도에서 실패한 다운로드 시도들이 `~/.cache/uv`에 그대로 누적되어 있었음 (`du -sh ~/.cache/uv` 확인 결과 4.0GiB)
- 실패한 이전 시도들이 캐시에 남아 디스크 공간을 차지하고 있음.

##### 시도 과정

- `uv cache clean` 실행 — 캐시 23,187개 파일(3.9GiB) 삭제, 디스크 사용률 92%(560MiB 남음) → 33%(4.5GiB 남음)로 회복
- `uv sync` 재실행 → `torch==2.13.0+cpu`, `torchvision==0.28.0` 포함 107개 패키지 정상 설치 완료

#### 3차 시도 - Python 최신 버전(3.14.4)의 호환성 불안정 문제

##### 문제 상황

- `uv run uvicorn main:app` 및 `uv run python -c "from transformers import PreTrainedModel"` 실행 시 아래 에러 확인

  ```
  ModuleNotFoundError: Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?
  ```

##### 원인 분석

- 1, 2차 시도에서 확인되는 `cp314`(Python 3.14 전용 빌드) 태그를 근거로, 당시 설치된 Python 3.14의 생태계 성숙도 문제로 추정

##### 시도 과정

- `uv python install 3.12`로 Python 3.12 설치
- `uv venv --python 3.12`로 가상환경 재구성 (기존 `.venv` 교체)

#### 4차 시도 - torchvision CPU 인덱스 미지정으로 인한 ABI 불일치

##### 문제 상황

- `uv sync` 재실행 → `torch`/`torchvision` 포함 107개 패키지 정상 설치되었으나, `uv run python -c "from transformers import PreTrainedModel"` 실행 시 동일 에러 재발
  ```
  RuntimeError: operator torchvision::nms does not exist
  ModuleNotFoundError: Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?
  ```

##### 원인 분석

- `pyproject.toml`의 `[tool.uv.sources]`에 `torch`만 `pytorch-cpu` 인덱스로 지정되어 있고, `torchvision`은 지정이 없어 기본 인덱스(CUDA 지원 빌드)로 설치됨
- `torch`(CPU 빌드)와 `torchvision`(CUDA 빌드)의 바이너리 빌드 출처(ABI)가 서로 달라, `torchvision`이 등록하려는 연산자(`torchvision::nms`)를 `torch`가 인식하지 못해 발생

##### 시도 과정

- (커밋 `7cf4df1`) `pyproject.toml`의 `[tool.uv.sources]`에 `torchvision = { index = "pytorch-cpu" }` 추가
- `uv sync` 재실행 → `torchvision` 재설치 (`0.28.0` → `0.28.0+cpu`)
- `uv run python -c "from transformers import PreTrainedModel"` 실행 → 에러 없이 정상 완료, 문제 해결 확인

#### 5차 시도 - 스왑 파일 생성을 통한 메모리 확보

##### 문제 상황

- `uv run uvicorn main:app --host 0.0.0.0 --port 8000` 실행 중 서버 프로세스가 강제 종료됨
- `sudo journalctl -k | grep -i "out of memory|killed process"` 확인 결과

  ```
  Out of memory: Killed process 2620 (uvicorn) total-vm:2515768kB, anon-rss:584676kB, ...
  ```

##### 원인 분석

- t3.micro 인스턴스의 총 물리 메모리(약 1GiB) 대비, 임베딩 모델 및 관련 라이브러리(torch, transformers) 로딩만으로 메모리 사용량(`anon-rss` 584MB)이 한계에 근접해 OOM Killer가 프로세스를 강제 종료

##### 시도 과정

- `sudo fallocate -l 1G /swapfile` → `chmod 600` → `mkswap` → `swapon`으로 1GiB 스왑 생성
- `free -h`로 스왑 정상 적용 확인 (Swap: 1.0Gi)

#### 6차 시도 - 볼륨 확장(8GiB → 20GiB)을 통한 디스크 공간 확보

##### 문제 상황

- `uv run uvicorn main:app --host 0.0.0.0 --port 8000` 재실행 → OOM은 재발하지 않았으나, 이번엔 별개의 디스크 공간 부족 경고로 실패

  ```
  UserWarning: Not enough free disk space to download the file. The expected file size is: 10246.62 MB. ... only has 960.93 MB free disk space.
  ```

- 스왑 조치가 OOM을 실제로 해결했는지는 이 시도만으로 확정할 수 없음 — 디스크 공간 부족이라는 다른 문제로 먼저 막혔기 때문

##### 원인 분석

- gemma-4-E2B-it 모델(약 10GB)을 다운로드하기에 8GiB 루트 볼륨은 절대적으로 공간이 부족함
- 스왑으로 해결 가능한 메모리 문제와 달리, 이는 디스크 자체의 용량 한계이므로 볼륨 확장이 필요

##### 시도 과정

- AWS 콘솔에서 루트 볼륨 크기를 8GiB → 20GiB로 수정
- `lsblk`로 디스크(`nvme0n1`)는 20G로 인식되었으나 파티션(`nvme0n1p1`)은 6.9G로 그대로임을 확인
- `sudo growpart /dev/nvme0n1 1`로 파티션 확장
- `sudo resize2fs /dev/nvme0n1p1`로 파일시스템 확장
- `df -h` 확인 결과 루트 파티션 6.9G → 19G, 여유 공간 1.9G → 12G로 확장 완료

#### 7차 시도 - 인스턴스 유형 변경

##### 문제 상황

- 볼륨 확장(6차 시도) 후 `uv run uvicorn main:app` 재실행 → 디스크 부족 경고는 해소되었으나, 여전히 OOM 재발 확인 (`journalctl -k` 로그, PID 3171)

##### 원인 분석

- 스왑(1GiB)과 볼륨 확장(20GiB) 둘 다 메모리 부족 자체는 해결하지 못함 — t3.micro의 근본적인 메모리 사양(908Mi) 한계로 판단
- 인스턴스 자체의 메모리 사양을 높이기 위해 t3.medium으로 유형 변경 시도
- 현재 인스턴스가 속한 가용 영역(AZ, ap-northeast-2c)에서 t3.medium이 지원되지 않아 선택 자체가 불가능함을 확인
- 같은 AZ에서 지원되는 다른 유형(c7i-flex.large)으로 대체를 시도했으나, 이 역시 해당 AZ(ap-northeast-2c)에서 지원되지 않음을 확인
- AZ 자체를 옮겨야 한다는 결론에 도달, 기존 인스턴스의 AMI를 생성해 다른 AZ에서 재실행하는 방식으로 전환

##### 시도 과정

- 1차 시도
  - 인스턴스 자체의 메모리 사양을 높이기 위해 t3.medium으로 유형 변경 시도
  - 현재 인스턴스가 속한 가용 영역(AZ, ap-northeast-2c)에서 t3.medium이 지원되지 않아 선택 자체가 불가능함을 확인

- 2차 시도
  - 같은 AZ에서 지원되는 다른 유형(c7i-flex.large)으로 대체를 시도
  - 이 역시 해당 AZ(ap-northeast-2c)에서 지원되지 않음을 확인
  - AZ 자체를 옮겨야 한다는 결론에 도달

- 3차 시도
  - 기존 인스턴스의 AMI를 생성해 다른 AZ에서 재실행하는 방식으로 전환
    - AMI: 방금 생성한 AMI 사용
    - 인스턴스 유형: `c7i.flex.large` 선택(비용은 약 2배였으나, 프리 티어 조건과 AZ 제약을 동시에 만족하는 유일한 선택지)
    - 키 페어: 기존과 동일
    - 네트워크 설정: 신규 보안 그룹 생성, SSH만 "내 IP"로 허용
    - 스토리지: 20GiB

#### 8차 시도 - 모델 경량화를 통한 근본적 메모리 요구량 축소 및 서버 활성화

##### 문제 상황

- 7차 시도로 인스턴스를 향상했지만, c7i.flex.large는 4GiB의 물리 메모리를 갖고 있음.
- 반면 개인 프로젝트 모델은 자체 Transformer와 gemma-4-E2B-it를 사용하고 있음.

##### 원인 분석

- c7i-flex.large는 2 vCPU, 4GiB 메모리 사양인 반면, gemma-4-E2B-itd의 원본 가중치 모델이 10GB 이상이므로 메모리 문제는 해소 불가능
- 인스턴스 사양(하드웨어) 확장과 별개로 모델 자체의 메모리 요구량(소프트웨어) 축소로, 근본적인 해결 과정이 필요

##### 시도 과정

- 로컬(Mac)에서 백엔드 모델을 gemma-4-E2B-it에서 Qwen2.5-1.5B GGUF(양자화 적용)로 교체하는 작업 진행 (이전 LLM 최적화 챌린지에서 다뤘던 PTQ/GGUF 변환 기법 재적용)
- (커밋 `0903023`) `rag_pipeline`/`langchain` 백엔드를 Qwen2.5-1.5B GGUF로 교체

#### 인사이트

- 전반적인 인사이트
  - t3.micro에서 바로 c7i.flex.large로 넘어갈 수 있었으나, 다양한 에러 환경을 경험하고 싶었음. 실무에서는 어떤 디버깅 사례가 나올지 모르기 때문임.
  - 같은 문제라도 다른 에러 메시지로 확인 가능. 비록 다른 해결책이라도, 전체 맥락을 이해할 수 있어야 유연한 설계 및 수정이 가능
  - 이번 메모리 및 디스크 문제로 CS 지식이 에러 해결에 어떤 맥락으로 적용되는지 이해하기 위해 노력
- 상세 인사이트
  - 문제가 겹쳐서 발생할 때는 "지금 겪는 증상"과 "근본 원인"이 다를 수 있다는 걸 반복적으로 확인함 (1~2차 시도(디스크 부족)를 해결하는 과정에서 3차 시도(Python 버전 문제로 오판)처럼 다른 원인으로 잘못 짚는 경우가 있었고, 이는 에러 트레이스백에서 최종 표시되는 예외(ModuleNotFoundError)와 근본 원인 예외(RuntimeError)가 다를 수 있다는 걸 놓쳤기 때문이었음)
  - 가벼운 조치부터 시도하고 안 되면 점점 무거운 조치로 올라가는 순서(스왑 → 볼륨 확장 → 인스턴스 유형 변경 → 모델 경량화)가 합리적이었지만, 각 조치가 실제로 문제를 해결했는지 검증하지 않고 다음 문제로 넘어가면 이전 조치의 실효성이 뒤섞여 버림. (5차 시도(스왑)의 실제 효과는 6차 시도 이후에야 로그 재분석으로 뒤늦게 확인됨)
  - 인스턴스 사양을 올리는 것(하드웨어 확장)과 모델 자체를 경량화하는 것(소프트웨어 최적화)은 같은 문제(메모리 부족)에 대한 서로 다른 축의 해법이며, 한쪽만으로는 근본적 해결이 안 될 수 있다는 걸 실제로 겪으며 확인함. (c7i.flex.large(4GiB)로 올려도 gemma-4-E2B-it(10GB+) 같은 비양자화 대형 모델에는 여전히 부족했고, 결국 모델을 GGUF 양자화로 바꾸는 근본적 조치가 필요했음)
  - 클라우드 환경의 제약(AZ별 인스턴스 유형 가용성, 프리 티어 조건)이 기술적 해법의 선택지 자체를 제한할 수 있다는 것을 경험함. (비용/가용성 제약으로 인해 최적이라 생각한 선택(t3.medium)이 불가능했고, 차선책(c7i.flex.large)을 선택해야 했음)

### 3단계 트러블 슈팅 2 - llama-cpp-python 빌드를 위한 컴파일 도구 부재

#### 문제 상황

- Qwen2.5-1.5B GGUF 전환(8차 시도) 후 `uv sync` 실행 시 `llama-cpp-python` 빌드 실패

  ```
  × Failed to build llama-cpp-python==0.3.34
  ╰─▶ CMake Error: Could not find the compiler specified in the environment variable CC: cc.
  CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
  CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
  ```

#### 원인 분석

- `llama-cpp-python`은 순수 Python 패키지가 아니라 C/C++ 소스를 CMake로 빌드하는 네이티브 확장 패키지
- 인스턴스에 C/C++ 컴파일러(`gcc`/`g++`) 및 빌드 도구가 설치되어 있지 않아 CMake 설정 단계에서 실패

#### 해결 과정

- `sudo apt update` 후 `sudo apt install -y build-essential cmake`로 컴파일러 및 빌드 도구 일체 설치
- `uv sync` 재실행 → `llama-cpp-python` 빌드 성공(약 5분 소요), `diskcache`, `korean-chatbot` 포함 정상 설치 완료

#### 인사이트

- 이번 배포 과정에서 겪은 대부분의 문제(torch/torchvision, llama-cpp-python)가 "Python 패키지는 다 똑같이 설치된다"는 암묵적 전제가 깨짐. 실제로는 순수 Python 패키지(설치 즉시 사용 가능)와 네이티브 확장 패키지(설치 시점에 C/C++ 컴파일이 필요한 패키지)가 근본적으로 다른 설치 경로를 탄다는 걸 명확히 인식하게 됨
- `pip`/`uv` 같은 패키지 매니저가 "의존성 해결(resolve)"까지는 자동으로 처리해도, 그 의존성이 네이티브 빌드를 요구하는 순간부터는 OS 레벨의 도구(컴파일러, cmake)가 사전에 준비되어 있어야 한다는 전제 조건이 별도로 존재함
- 로컬(Mac) 개발 환경에서는 문제없이 설치되던 패키지가 EC2(Ubuntu, 최소 설치 이미지)에서 실패한 이유도 같은 맥락임. 로컬에는 이미 Xcode Command Line Tools 등으로 컴파일러가 갖춰져 있었지만, 최소 구성의 서버 이미지에는 기본적으로 이런 빌드 도구가 빠져 있음
- 앞으로 새로운 의존성을 추가할 때, "이 패키지가 네이티브 확장을 포함하는가"를 미리 확인하는 습관이 배포 환경에서의 예기치 못한 실패를 줄이는 데 도움이 됨

### 3단계 트러블 슈팅 3 - uvicorn 정상 실행 후 크롬 브라우저 접근 불가

#### 문제 상황

- `uv run uvicorn main:app --host 0.0.0.0 --port 8000` 실행 후 서버는 정상 기동되었으나, 로컬 Mac 크롬 브라우저에서 `퍼블릭 IP:8000` 접속 시 "사이트에 연결할 수 없음" 발생
- `0.0.0.0:8000`으로도 시도했으나 동일하게 실패 (`0.0.0.0`은 서버 자신을 가리키는 바인딩 주소일 뿐, 클라이언트가 접속할 수 있는 주소가 아니므로 애초에 잘못된 시도였음)

#### 원인 분석

- 인바운드 규칙에는 SSH(22번, "내 IP")만 등록되어 있었고, 8000번 포트에 대한 규칙이 없었음
- 보안 그룹은 명시적으로 허용한 포트 외에는 전부 차단(default deny)하므로, 애플리케이션이 정상 실행 중이어도 인바운드 규칙에 없는 포트는 인스턴스 외부에서 도달 자체가 불가능함
- 크롬 요청이 로컬 OS의 네트워크 스택에서 TCP/IP 4계층 기준으로 캡슐화되어 IP 헤더(출발지/목적지 주소)와 TCP 헤더(포트)가 채워지고, 보안 그룹은 이 겉면의 헤더 값만 확인해 통과 여부를 결정하며(HTTP header/body 내용은 확인하지 않음), 이 필터링과 NAT(퍼블릭 IP ↔ 사설 IP 변환)가 AWS 경계(인터넷 게이트웨이, IGW)에서 함께 처리됨

#### 시도 과정

- AWS 콘솔 → 보안 그룹 → 인바운드 규칙 편집 → "규칙 추가" → 유형: 사용자 지정 TCP, 포트: 8000, 소스: 내 IP
- 크롬에서 `퍼블릭 IP:8000` 재접속 → 정상 접근 확인

#### 인사이트

- SSH 개념 정리: SSH는 GitHub에서 쓰는 것과 EC2 접속에 쓰는 것이 동일한 프로토콜(공개키/개인키 기반 인증)이며, `.pem` 파일은 AWS가 발급한 키 형태일 뿐 핵심은 공개키/개인키 인증 원리라는 점. SSH(원격 제어)와 HTTP/HTTPS(콘텐츠 전달)는 목적 자체가 다른 별개의 프로토콜이라는 점
- Network 기반 개념 정리: 요청-응답 전체 경로에서 캡슐화/디캡슐화, 보안 그룹의 헤더 기반 필터링, NAT의 역할이 어떻게 맞물리는지 정리됨

### 3단계 딥다이브 주제 목록

- SSH 프로토콜 딥다이브 — pem 파일은 부수적 요소, 핵심은 공개키/개인키 인증 원리
- OOM Kill 딥다이브 — OOM Killer 개입 조건·과정과 관련 CS 키워드(가상 메모리, anon-rss, oom_score, cgroup 메모리 제한 등) 개념 및 인과관계 정리
- OSI 7계층과 TCP/IP 4계층의 차이 — 참조 모델 vs 실제 구현 모델의 대응 관계
- OS의 네트워크 스택 — 캡슐화/디캡슐화를 실제로 수행하는 커널 내부 구조와 동작 원리
- Python 네이티브 확장 패키지와 ABI 호환성 — CPU/CUDA 빌드 분리, 빌드 태그(cp314 등), manylinux 규격, 컴파일된 바이너리 패키지와 순수 Python 패키지의 근본적 차이
- 리눅스 블록 디바이스·파티션·파일시스템 계층 구조 — lsblk 구조, growpart/resize2fs가 별도 단계로 나뉘는 이유, EBS 볼륨 확장이 즉시 반영되지 않는 이유
- CMake 기반 네이티브 빌드 과정 — 소스→컴파일→링크 흐름, 컴파일 필요 패키지와 순수 Python 패키지의 설치 방식 차이, CC/CXX 환경변수 역할
- AWS 가용 영역(AZ)과 인스턴스 유형 가용성 — AZ마다 지원 유형이 다른 이유(물리 하드웨어 배치, 용량 관리 정책)
- 모델 양자화(Quantization)와 GGUF 포맷 — 비양자화 → 양자화 전환 시 메모리 요구량이 줄어드는 원리
