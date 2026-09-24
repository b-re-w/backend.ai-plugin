# E2E: WSL에서 Backend.AI를 띄워 labgpu를 확인하는 스크립트

WSL2(Ubuntu 24.04) + Docker Desktop에서 Backend.AI 안정판(기본 26.8.3)으로 manager·agent를 띄우고,
labgpu 플러그인과 스팟 컨트롤러를 실제로 돌려 보는 스크립트입니다. 결과는 [SPEC 3.2](../docs/SPEC.md#32-실기-검증표)에 있습니다.

모든 스크립트는 WSL 안에서 `bash e2e/<script>`로 실행합니다.

시험할 Backend.AI 버전은 환경변수 `BAI_REF`(git 태그, 기본 `26.8.3`)로 고릅니다(`env.sh`).
`backend.ai/` 폴더의 체크아웃은 건드리지 않고, 저장소에서 그 버전 파일만 꺼내(`git archive`)
`~/labgpu-e2e-<버전>`에 둡니다. 대용량 파일은 `fill_lfs.sh`가 로컬 LFS 저장소에서 채웁니다.

## 한 번만 하는 준비

- Docker Desktop 설정에서 WSL 통합에 해당 배포판(`Ubuntu-24.04`)을 켭니다.
- 파이썬 환경은 `03_install.sh`가 `~/.cache/labgpu-py313-<버전>`에 알아서 만듭니다.
- WSL을 다시 시작할 때마다, Backend.AI의 bind mount 전파를 위해 루트 마운트를 shared로 바꿉니다.

  ```bash
  wsl -d Ubuntu-24.04 -u root mount --make-rshared /
  ```

## 순서

| 스크립트 | 하는 일 |
|---|---|
| `01_stage.sh` | 소스를 WSL 파일시스템으로 복사(CRLF 정리), halfstack(Postgres·Valkey·etcd) 시작 |
| `03_install.sh` | 업스트림 BUILD 파일의 엔트리 포인트를 모은 shim 패키지와 labgpu 설치 |
| `04_config.sh` | manager·agent 설정, etcd, DB 스키마, fixture |
| `05_labgpu.sh` | 가짜 Primary 서버 GPU 구성, 종류별 슬롯 설정, 테스트용 커널 이미지 |
| `06_start.sh` | manager·agent 시작 (가짜 NVML) |
| `10_order.sh`, `11_ports.sh` | agent `allocation-order`, 컨테이너 포트 범위(Windows 예약 포트 회피) |
| `09_sessions.sh` | 종류별 세션 생성, 컨테이너 환경변수 확인 |
| `13_hami.sh` | HAMi-core 빌드 (CUDA 12.4 호환 커밋) |
| `14_real_prep.sh`, `08_agent_real.sh`, `15_real.sh` | 실제 GPU(NVML) 모드: 0.5 세션 두 개, HAMi-core 주입 확인 |
| `16_hami_debug.sh` | HAMi-core 단독 실행 진단 (WSL에서는 segfault. SPEC 3.2 참고) |
| `sync.sh` | 코드를 고친 뒤 플러그인을 다시 복사하고 테스트 실행 |

`api.py`는 Backend.AI 클라이언트 SDK로 manager API를 부르고, `fakegpu.py`는 가짜 NVML 파일을 실시간으로 고칩니다.

## 알려진 WSL 제약

- HAMi-core는 WSL2에서 동작하지 않습니다(`cuInit` 후킹 재귀). 메모리 강제는 네이티브 Linux 노드에서 확인하세요.
- WSL의 NVML은 프로세스별 정보를 주지 않고, Docker Desktop 컨테이너의 PID는 WSL 배포판에서 보이지 않습니다.
  그래서 소유자 활동 감지는 가짜 NVML 파일로 시험합니다.
- Windows가 예약한 TCP 포트 범위(30000번대 일부)와 Backend.AI 기본 컨테이너 포트 범위가 겹칩니다.
