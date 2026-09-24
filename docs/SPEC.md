# SPEC: labgpu 동작 명세

> 이 문서가 기준입니다. 코드와 이 문서가 다르면 코드가 버그입니다. 동작을 바꿀 때는 이 문서를 먼저
> 고칩니다. 배경과 목표는 [INTENT.md](INTENT.md)를 보세요.

## 0. 구성 요소

| 구성 요소 | 어디서 도나 | 하는 일 |
|---|---|---|
| `cuda_frac` 가속기 플러그인 (`labgpu.accelerator`) | 각 GPU 노드의 Backend.AI **agent** 프로세스 안 | GPU를 `cuda.shares`(소수) 단위로 할당하고, HAMi-core로 컨테이너별 GPU 메모리·SM 사용률을 제한합니다. |
| `labgpu-spot` 컨트롤러 (`labgpu.spot`) | 각 GPU 노드의 별도 systemd 서비스 (root) | 소유자가 안 쓰는 GPU를 감지해 스팟 컨테이너를 띄우고, 소유자가 돌아오면 회수합니다. |
| 공용 모듈 (`labgpu.fraction`, `labgpu.nvml`, `labgpu.procmap`, `labgpu.devalloc`) | 둘 다 | 할당량 계산, NVML 조회, PID→컨테이너 매핑. |

두 구성 요소는 **서로 독립적**입니다. 플러그인만 써도(fGPU만), 컨트롤러만 써도(오픈소스 `cuda`
플러그인 + 스팟) 동작해야 합니다.

### 의존성

- Python ≥ 3.12
- `nvidia-ml-py` (NVML 바인딩, `import pynvml`)
- 플러그인: Backend.AI agent 25.19가 이미 설치한 패키지(`ai.backend.agent`, `ai.backend.common`, `aiodocker`)
- 컨트롤러: `docker` CLI, 표준 라이브러리(`sqlite3`, `tomllib`)만
- 호스트에 설치된 HAMi-core `libvgpu.so` (vendoring하지 않음, 직접 빌드해 둠)

---

## 1. `cuda_frac` 가속기 플러그인

### 1.1 등록

- 엔트리 포인트 그룹 `backendai_accelerator_v21`에 두 종류를 등록합니다.
  - **`cuda_frac`** → `CUDAFracPlugin`. `key` 기본값 **`cuda`**. 노드의 GPU가 모두 같은 종류일 때 씁니다.
    WebUI가 `cuda.shares`를 fGPU로 인식합니다.
  - **`gpu_slot_1` ~ `gpu_slot_4`** → `GpuSlotPlugin1~4`. GPU 종류별 슬롯용입니다(1.11). `key` 설정이 없으면
    `init()`에서 예외를 내고, agent는 그 플러그인을 로그만 남기고 건너뜁니다.
- 같은 에이전트에서 오픈소스 `cuda` 플러그인과 함께 쓸 수 없습니다. `agent.toml`에서 막습니다.
  **주의:** `allow-`/`block-compute-plugins`는 엔트리 포인트 이름이 아니라 **모듈 경로 접두사**로 맞춥니다
  (업스트림 `match_plugin_list`). 그래서 `cuda_frac`은 `labgpu.accelerator.cuda_frac`, `gpu_slot_N`은
  `labgpu.accelerator.gpu_slot` 모듈에 따로 두어 각각 막을 수 있게 했습니다.

  ```toml
  [agent]
  block-compute-plugins = ["ai.backend.accelerator.cuda_open"]
  ```

- 플러그인 설정(`plugin_config`)은 etcd `config/plugins/accelerator/cuda_frac/` 아래에서 읽습니다
  (Backend.AI는 엔트리 포인트 이름으로 설정 경로를 정합니다).

### 1.2 설정 키

| 키 | 기본값 | 의미 |
|---|---|---|
| `allocation_mode` | `"fractional"` | `"fractional"`이면 `cuda.shares`, `"discrete"`이면 `cuda.device`(오픈소스와 동일) |
| `shares_per_device` | `"1"` | GPU 1장이 몇 share인지. 기본은 **1 share = GPU 1장**이라 `0.5`는 반 장입니다. |
| `quantum_size` | `"0.05"` | 할당 최소 단위 |
| `allocation_strategy` | `"fill"` | Backend.AI `FractionAllocMap`의 전략. `"fill"`은 요청 하나를 가장 여유 있는 GPU부터 채워 **최소한의 GPU에** 담습니다. `"evenly"`는 요청을 여러 GPU에 고르게 쪼갭니다(1.0 → 0.5+0.5). 주의: 두 전략 모두 **서로 다른 세션을 한 GPU에 모아 담지는 않습니다**(4.열린 질문 참고). |
| `hook_path` | `"/opt/labgpu/lib/libvgpu.so"` | 호스트의 HAMi-core 라이브러리 경로 |
| `reserved_memory` | `"0"` | 분할 할당 시 GPU별로 메모리 상한에서 빼 둘 바이트 수(CUDA 컨텍스트 여유분) |
| `sm_limit` | `"true"` | SM 사용률 제한(`CUDA_DEVICE_SM_LIMIT`)을 걸지 여부 |
| `device_mask` | 없음 | 쉼표로 구분한 GPU UUID 목록. 이 GPU들은 Backend.AI에 노출하지 않습니다. |
| `key` | `cuda_frac`: `"cuda"` / `gpu_slot_N`: **필수** | 플러그인 key. 슬롯은 `<key>.shares` 또는 `<key>.device`, 지표는 `<key>_mem`, `<key>_util`이 됩니다. 소문자·숫자·`-`만 씁니다. |
| `model_pattern` | `"*"` | 이 플러그인이 가져갈 GPU의 모델명 패턴(NVML 이름, 대소문자 무시, 쉼표로 여러 개, `*` 와일드카드) |
| `min_memory` / `max_memory` | 없음 | 가져갈 GPU의 총메모리 범위(`"60g"` 등). 같은 모델명의 메모리 변형(PRO 5000 48GB / 72GB)을 구분합니다. |
| `display_name` | 모델명 | WebUI에 보일 이름 (`human_readable_name`) |
| `display_unit` | `cuda`: `fGPU`/`GPU` · 그 밖의 key: 짧은 모델명(예: `PRO6000`) | WebUI가 숫자 뒤 단위이자 **세션 생성 화면의 가속기 종류 선택지 이름**으로 씁니다. 종류별 슬롯끼리 겹치지 않아야 구분됩니다. |

etcd 값은 모두 문자열로 들어오므로 문자열로 파싱합니다.

### 1.3 장치 식별

- 장치는 NVML로 조회합니다. `device_id`는 **NVML 인덱스 문자열**(`"0"`, `"1"`…)입니다.
- 컨테이너에 GPU를 붙일 때는 인덱스가 아니라 **UUID**(`GPU-xxxxxxxx-…`)로 요청합니다.
  CUDA 런타임 순서와 NVML 순서가 달라서 생기는 혼동을 막기 위해서입니다.
- `processing_units`는 100(퍼센트 단위)으로 둡니다. NVML은 SM 개수를 직접 주지 않고, 표시용으로만 쓰입니다.

### 1.4 강제(enforcement) 가능 여부 판정

`init()`에서 판정합니다.

1. NVML 초기화 실패, 또는 Docker에 `nvidia` 런타임이 없으면 → 플러그인 비활성(`enabled=False`).
2. `allocation_mode="fractional"`인데 `hook_path` 파일이 없으면 →
   **ERROR 로그를 남기고 `discrete` 모드로 강등**합니다. 강제할 수 없는 소수 할당은 내주지 않습니다.
3. `extra_info()`에 `fraction_enforced: "true"/"false"`를 넣어 관리자가 상태를 확인할 수 있게 합니다.

### 1.5 슬롯과 할당 맵

| 모드 | 슬롯 | 할당 맵 | 장치당 용량 |
|---|---|---|---|
| fractional | `cuda.shares` (COUNT) | `FractionAllocMap(quantum_size, allocation_strategy)` | `shares_per_device` |
| discrete | `cuda.device` (COUNT) | `DiscretePropertyAllocMap` | 1 |

`exclusive_slot_types = {"cuda.device", "cuda.shares"}`.

### 1.6 share → 제한값 계산 (`labgpu.fraction`)

장치 하나에 share `s`가 할당됐을 때, 비율 `f = s / shares_per_device`.

- `f ≥ 1` → 그 장치는 통째로 할당된 것이므로 **제한을 걸지 않습니다.**
- `f < 1` →
  - 메모리 상한(MiB) = `floor((f × 총메모리 − reserved_memory) / 2^20)`, 최소 1
  - SM 상한(%) = `ceil(f × 100)`, 1~100으로 자름

### 1.7 컨테이너 생성 인자 (`generate_docker_args`)

- `HostConfig.DeviceRequests = [{Driver: "nvidia", DeviceIDs: [<UUID>…], Capabilities: [["utility","compute","video","graphics","display"]]}]`
- 환경변수 (컨테이너 안에서는 붙은 GPU가 0부터 다시 번호가 매겨지므로 **로컬 인덱스**를 씁니다)
  - `LABGPU_DEVICE_UUIDS=<UUID>,<UUID>`: 컨트롤러가 소유자 GPU를 알아내는 데 씁니다.
  - fractional 모드에서 제한이 필요한 장치마다 `CUDA_DEVICE_MEMORY_LIMIT_<local_idx>=<MiB>m`
  - `sm_limit=true`이고 제한이 필요한 장치가 있으면 `CUDA_DEVICE_SM_LIMIT=<제한 장치들 중 최댓값>`
    (HAMi-core는 SM 제한을 전역 값 하나로 받습니다. 실제 노드에서 확인 필요)
  - `CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/labgpu-vgpu.cache`
- 할당이 비어 있으면 빈 dict를 돌려줍니다.

### 1.8 훅 라이브러리 (`get_hooks`)

- fractional 모드이고 강제 가능할 때만 `[hook_path]`를 돌려줍니다. Backend.AI agent가 이 파일을
  `/opt/kernel/libvgpu.so`로 마운트하고 `LD_PRELOAD`에 붙입니다.

### 1.9 기타 메서드

| 메서드 | 동작 |
|---|---|
| `generate_resource_data` | `CUDA_GLOBAL_DEVICE_IDS=<local>:<global>,…`, `CUDA_RESOURCE_VIRTUALIZED=1`(분할 장치가 있을 때) 또는 `0` |
| `get_attached_devices` | 장치별 `{"device_id", "model_name", "data": {"smp": SM%, "mem": 할당 메모리}}` |
| `restore_from_container` | 컨테이너의 resource spec에서 현재 모드의 슬롯 할당을 복원 |
| `gather_node_measures` | `cuda_mem`(bytes), `cuda_util`(%), 장치별·노드 합계 |
| `gather_container_measures` | **프로세스 단위로 귀속**합니다. NVML이 보고한 GPU 프로세스의 PID를 `/proc/<pid>/cgroup`으로 컨테이너 ID에 매핑해, 그 컨테이너 프로세스들의 메모리·SM 사용률만 합칩니다. 같은 GPU를 여러 컨테이너(분할 할당, 스팟)가 써도 서로의 사용량이 섞이지 않고, Backend.AI idle checker의 `cuda_util`도 정확해집니다. |
| `get_metadata` | fractional: `slot_name="cuda.shares"`, 표시 이름 `fGPU`, 소수 둘째 자리. discrete: `cuda.device`, `GPU` |

### 1.10 `device_alloc` 형태 호환 (`labgpu.devalloc`)

25.19는 `Mapping[SlotName, Mapping[DeviceId, Decimal]]`를, 최신 코드는 `DeviceAllocation`
(`units` 속성)을 넘깁니다. 모든 메서드는 입력을 먼저 `{slot: {device_id: Decimal}}`로 정규화합니다.

### 1.11 GPU 종류별 슬롯

한 노드에 여러 종류의 GPU가 있을 때, agent를 나누지 않고 **GPU 종류마다 슬롯 이름을 따로** 둡니다.
CPU·RAM은 agent 하나가 관리하므로 종류와 상관없이 유동적으로 쓰입니다.

- `gpu_slot_N`마다 `key`, `model_pattern`, `min_memory`/`max_memory`를 설정합니다. 설정은 etcd에 있어
  클러스터 전체가 공유합니다. 조건이 모델 기준이라 노드마다 따로 둘 필요가 없고, 맞는 GPU가 없는 노드에서는
  장치 0개인 플러그인이 됩니다(슬롯 용량 0).
- **중복 점유 방지.** 여러 플러그인이 같은 GPU를 가져가면 이중 할당이 됩니다. 각 플러그인은 `init()`에서
  가져갈 GPU의 UUID마다 점유 파일(`$LABGPU_CLAIM_DIR`, 기본 `/run/labgpu/claims`, 쓸 수 없으면 임시 디렉터리)을
  만듭니다. 같은 agent 프로세스(PID) 안에서 다른 플러그인이 이미 점유한 GPU가 있으면 **init을 실패**시킵니다.
  다른 PID가 남긴 파일(이전 실행)은 덮어씁니다.
- 같은 `key`를 두 플러그인에 주면 agent가 시작을 거부합니다(업스트림 동작).
- 이 방식을 쓸 때 `cuda_frac`은 막습니다: `block-compute-plugins = ["ai.backend.accelerator.cuda_open", "labgpu.accelerator.cuda_frac"]`.
  막지 않으면 모든 GPU를 가져가려다 점유 충돌로 한쪽이 init에 실패합니다(어느 쪽일지는 로딩 순서에 달림).
- 장치의 `device_name`은 플러그인 key입니다. agent의 affinity map이 이 이름으로 장치를 찾기 때문에,
  빠지면 할당 단계에서 "No suitable devices found"로 세션 생성이 실패합니다(E2E에서 발견).
- 운영자가 할 일 (예시 스크립트: `examples/per-model-slots.sh`):
  1. etcd `config/resource_slots`에 새 슬롯을 등록합니다(예: `pro6000.shares` → `count`). 매니저는 등록된
     슬롯만 WebUI에 넘깁니다. 26.x는 DB `resource_slot_types` 표에도 표시 정보를 둡니다.
  2. **agent.toml `[resource] allocation-order`에 새 key를 모두 넣습니다.** 기본값
     `["cuda", "rocm", "tpu", "cpu", "mem"]`에 없는 key가 요청되면 agent가 `ValueError: '<key>' is not in list`로
     커널 생성을 거부합니다(25.12부터 있는 설정이라 25.19에도 해당, E2E에서 발견).
  3. agent.toml `allow-compute-plugins`/`block-compute-plugins`를 모듈 경로로 적습니다
     (예: allow `["labgpu.accelerator"]`, block `["labgpu.accelerator.cuda_frac"]`).
  4. 이미지의 지원 가속기 목록(`ai.backend.accelerators` 라벨 또는 관리자 화면)에 새 key를 넣습니다.
  5. idle checker의 사용률 기준을 `<key>_util`, `<key>_mem`으로 적습니다.
  6. 자원 정책·프리셋에 종류별 슬롯(`pro6000.shares` 등)을 씁니다.

### 1.12 가짜 NVML 모드 (개발·시험 전용)

환경변수 `LABGPU_FAKE_NVML=<json 파일>`이 있으면 NVML 대신 이 파일을 읽습니다(매 호출마다 다시 읽음).
GPU가 없거나 다른 구성의 서버를 흉내 낼 때 씁니다.

```json
{"driver": "fake",
 "gpus": [{"uuid": "GPU-p6000-1", "name": "NVIDIA RTX PRO 6000", "memory": "96g",
           "used": "0", "util": 0,
           "processes": [{"pid": 1234, "mem": "4g", "sm": 30, "container": "<64자리 컨테이너 ID>"},
                         {"pid": 900, "mem": "200m", "name": "Xorg"}]}]}
```

- 플러그인: `nvidia` 런타임 확인을 건너뛰고, 컨테이너에 GPU를 붙이지 않습니다(`DeviceRequests` 없음).
  환경변수(`LABGPU_DEVICE_UUIDS`, HAMi-core 제한값)는 그대로 넣습니다. 시작할 때 WARNING을 남깁니다.
- 컨트롤러: `--gpus`를 붙이지 않습니다. PID→컨테이너 매핑은 파일의 `container` 필드를 씁니다.

---

## 2. `labgpu-spot` 컨트롤러

### 2.1 개념

- **소유자(owner)**: Backend.AI 커널 컨테이너(라벨 `ai.backend.kernel-id`가 있음)이면서 GPU가 붙어 있는 것.
- **점유(claimed) GPU**: 소유자 컨테이너가 하나라도 붙어 있는 GPU.
- **스팟(spot)**: 컨트롤러가 띄운 컨테이너(라벨 `labgpu.spot=1`). 스팟 작업 하나 = GPU 1장.
- **정체불명(unknown) 프로세스**: GPU를 쓰는데 소유자 컨테이너도 스팟도 아닌 프로세스(호스트 프로세스,
  Backend.AI 밖의 컨테이너). **소유자 활동으로 간주**합니다(안전 우선). 스팟 컨테이너의 프로세스라도
  그 스팟이 배정받은 GPU가 아닌 곳에 나타나면 정체불명으로 봅니다.
- **무시하는 프로세스(ignored)**: 컨테이너 밖(호스트)에서 돌면서 이름이 `ignored_processes`에 있는
  프로세스. 워크스테이션 GPU에 늘 떠 있는 화면 서버(`/usr/lib/xorg/Xorg`)처럼, 있어도 소유자 활동이
  아닌 것들입니다. 활동 판정에서 빼고 정체불명으로도 보지 않습니다. 이 프로세스가 쓰는 GPU 메모리는
  "사용 중"에 이미 들어 있어 빌려줄 양에서 자동으로 빠지고, 크게 늘면 "여유 메모리 부족" 조건으로
  회수됩니다. 이름은 `/proc/<pid>/comm`으로 비교하며, 컨테이너 안의 프로세스는 이름이 같아도 무시하지
  않습니다(컨테이너가 이름만 바꿔 빠져나가지 못하게).

### 2.2 설정 파일 `/etc/labgpu/spot.toml`

```toml
[controller]
poll_interval = 5            # 초
state_dir = "/var/lib/labgpu"
kill_switch_file = "/etc/labgpu/spot.disabled"   # 이 파일이 있으면 전부 회수하고 빌려주지 않음

[idle]
idle_minutes = 30            # 소유자가 이만큼 연속으로 쉬어야 빌려줌
owner_util_threshold = 5     # %, 소유자 프로세스 SM 사용률 합이 이 값을 넘으면 활동
owner_mem_delta_mib = 512    # 소유자 GPU 메모리가 기준보다 이만큼 늘면 활동
owner_cpu_threshold = 0.5    # 코어 수, 소유자 컨테이너 CPU 사용량이 넘으면 활동 (0이면 끔)
unclaimed_grace_seconds = 60 # 아무도 안 붙은 GPU를 빌려주기 전 대기
ignored_processes = ["Xorg"] # 호스트에서 늘 GPU를 쓰는, 소유자 활동이 아닌 프로세스 이름

[reclaim]
grace_seconds = 30           # SIGTERM 후 SIGKILL까지
mem_reserve_mib = 2048       # 스팟 메모리 상한 = 총 − 현재 사용 − 이 값

[spot]
hook_path = "/opt/labgpu/lib/libvgpu.so"
allow_unenforced = false     # true면 HAMi-core 없이도 빌려줌 (소유자 OOM 위험)
cpu_shares = 64              # Docker 기본 1024 대비 낮은 CPU 우선순위
default_ram = "16g"
host_ram_reserve = "32g"     # 호스트 MemAvailable에서 항상 남겨 둘 양
allowed_mount_roots = ["/vfroot"]
max_attempts = 20            # 회수로 인한 재시도 최대 횟수
```

### 2.3 관측 (매 틱)

GPU마다 다음을 모읍니다. 하나라도 실패하면 그 GPU는 **UNKNOWN**입니다.

- NVML: UUID, 인덱스, 총/사용 메모리, 실행 중인 compute 프로세스(PID, 사용 메모리),
  프로세스별 SM 사용률 샘플(`nvmlDeviceGetProcessUtilization`, 직전 틱 이후 샘플의 최댓값)
- PID → 컨테이너 ID: `/proc/<pid>/cgroup`에서 64자리 16진수 ID를 찾습니다.
- 프로세스 이름: `/proc/<pid>/comm` (무시 목록 비교용).
- Docker: 실행 중인 소유자 컨테이너와 그 GPU(UUID) 목록. 다음 순서로 찾습니다.
  1. 환경변수 `LABGPU_DEVICE_UUIDS` (cuda_frac 플러그인)
  2. `HostConfig.DeviceRequests[].DeviceIDs` (Driver `nvidia`; `GPU-`로 시작하면 UUID, 아니면 NVML 인덱스)
  3. 환경변수 `NVIDIA_VISIBLE_DEVICES` (값이 `all`이면 모든 GPU)
- 소유자 컨테이너 CPU 사용량: `State.Pid`의 cgroup v2 `cpu.stat`의 `usage_usec` 변화량 / 경과 시간.
  읽을 수 없으면 CPU 조건은 건너뜁니다.
- 호스트 `MemAvailable` (`/proc/meminfo`).

### 2.4 GPU 상태 기계 (`labgpu.spot.detector`)

```
            활동 감지                      idle_minutes 경과
  BUSY ◄───────────────── IDLE ──────────────────────────► LENDABLE
   ▲ ▲                     ▲                                  │ 스팟 시작
   │ │ 활동 감지           │ 스팟 종료(작업 완료)              ▼
   │ └──────────────── RECLAIMING ◄──── 회수 조건 ─────── LENT
   │                                                         
 UNKNOWN (관측 실패: 모든 상태에서 진입, 회수 수행, 복구 시 BUSY로)
```

틱마다 GPU의 **활동 여부**를 판정합니다. 다음 중 하나라도 참이면 활동입니다.

1. 소유자 프로세스 SM 사용률 합 > `owner_util_threshold`
2. 소유자 프로세스 메모리 합 > 기준값 + `owner_mem_delta_mib` (기준값은 마지막 활동 시점의 소유자 메모리.
   메모리가 줄면 기준값도 따라 내려갑니다)
3. 소유자 컨테이너 CPU 사용량 > `owner_cpu_threshold`
4. 정체불명 프로세스가 GPU를 쓰고 있음
5. 점유한 소유자 컨테이너 집합이 **늘어남**(새 세션이 이 GPU를 받음)

판정 규칙:

- 활동이면 `last_active = now`, 상태는 BUSY. 빌려준 상태였다면 **회수**합니다.
- 점유 GPU: `now − last_active ≥ idle_minutes`이면 LENDABLE.
- 비점유 GPU(소유자 없음): `now − last_active ≥ unclaimed_grace_seconds`이면 LENDABLE.
- 컨트롤러가 막 시작했을 때는 모든 GPU를 BUSY(`last_active = 시작 시각`)로 둡니다.
  즉 재시작 후 최소 `idle_minutes`(비점유는 grace) 동안은 새로 빌려주지 않습니다.

빌려준(LENT) 상태에서는 위 활동 조건에 더해 다음도 **회수 조건**입니다.

- GPU 여유 메모리(총 − 사용) < `mem_reserve_mib / 2` (스팟은 여유분을 남기고 시작하므로, 여유가
  절반 밑으로 줄었다면 누군가 예상 밖으로 메모리를 잡은 것입니다)
- UNKNOWN 진입, 킬 스위치 파일 존재, 해당 GPU나 노드가 `pause` 상태

### 2.5 계획 (`labgpu.spot.planner`)

순수 함수입니다. 입력: GPU별 판정 결과, 실행 중인 스팟 작업, 대기열, 호스트 여유 RAM. 출력: 동작 목록.

1. **회수 먼저.** 회수 조건에 걸린 LENT GPU마다 `Reclaim(job, gpu, reason)`.
2. **시작.** LENDABLE이면서 스팟이 없는 GPU마다, 대기열을 `priority 내림차순 → 제출 시각 오름차순`으로
   훑어 **처음 맞는 작업**을 고릅니다(backfill). 맞는다는 것은:
   - `job.gpu_mem ≤ 총 − 사용 − mem_reserve_mib` (작업이 gpu_mem을 안 적었으면 1GiB로 간주)
   - `job.ram ≤ MemAvailable − host_ram_reserve − (이번 틱에 이미 배정한 RAM)`
   - HAMi-core가 있거나 `allow_unenforced = true`
   - 작업에 `gpu_models`가 있으면 GPU 모델명이 패턴 중 하나와 맞음
3. 스팟 GPU 메모리 상한 = `총 − 사용 − mem_reserve_mib` (작업 요청값이 더 작아도 여유분 전체를 줍니다.
   소유자가 돌아오면 어차피 회수되기 때문입니다).

### 2.6 스팟 컨테이너 실행 (`labgpu.spot.docker`)

```
docker run -d --name labgpu-spot-<job_id>-<attempt>
  --label labgpu.spot=1 --label labgpu.job-id=<id> --label labgpu.gpu-uuid=<uuid>
  --gpus device=<uuid>
  --cpu-shares <cpu_shares>
  --memory <ram> --memory-swap <ram> --oom-score-adj 1000
  --user <uid>:<gid>
  -v <hook_path>:/opt/labgpu/libvgpu.so:ro -e LD_PRELOAD=/opt/labgpu/libvgpu.so
  -e CUDA_DEVICE_MEMORY_LIMIT_0=<MiB>m
  -e CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/labgpu-vgpu.cache
  -e LABGPU_SPOT=1 -e LABGPU_JOB_ID=<id> -e LABGPU_ATTEMPT=<n>
  -v <src>:<dst>[:ro] ...   (allowed_mount_roots 아래만)
  [-w <workdir>] [--entrypoint <entrypoint>]
  <image> <command...>
```

- 회수: `docker stop -t <grace_seconds> <container>` (SIGTERM → 유예 → SIGKILL). 컨트롤러 루프를
  막지 않도록 백그라운드 프로세스로 실행합니다.
- `docker run`은 `--pull never`로 실행합니다. 이미지를 받는 동안 컨트롤러 루프가 멈추면 회수가 늦어지기
  때문입니다. 이미지가 노드에 없으면 작업은 FAILED(`launch failed: …`)가 됩니다. `docker run` 자체가
  실패한 경우도 같습니다(자동 재시도 없음).
- 종료된 스팟 컨테이너는 로그를 `<state_dir>/logs/<job_id>.<attempt>.log`로 저장한 뒤 삭제합니다.

### 2.7 작업 대기열 (`labgpu.spot.jobs`, SQLite `<state_dir>/spot.db`)

작업 상태:

```
QUEUED ──시작──► RUNNING ──스스로 종료, exit 0──► SUCCEEDED
   ▲                │  └────스스로 종료, exit≠0──► FAILED
   │                ▼ 회수
   └──(attempts < max)── PREEMPTING ──(attempts ≥ max)──► FAILED
QUEUED/RUNNING ──cancel──► CANCELLED (실행 중이면 docker stop)
```

- 회수로 끝난 작업은 **실패가 아닙니다.** `attempts`와 `preemptions`를 올리고 다시 QUEUED가 됩니다.
- 기록 필드: id, name, 제출자 uid/gid, 스펙(JSON), priority, state, attempts, preemptions,
  gpu_uuid, container, 제출/시작/종료 시각, exit_code, 마지막 사유. 크레딧 제도에 쓸 수 있도록
  시도별 실행 기록(`runs` 테이블: job_id, attempt, gpu_uuid, 시작/종료, 종료 사유)도 남깁니다.
- `settings` 테이블: `paused_node`, `paused_gpus`(UUID 목록). CLI로 바꾸면 다음 틱에 반영됩니다.

### 2.8 작업 명세 파일 (TOML)

```toml
name = "resnet50-sweep"
image = "cr.backend.ai/stable/python-pytorch:2.3-py312-cuda12.4"
command = ["python", "train.py", "--resume-from", "/home/work/proj/ckpt"]
workdir = "/home/work/proj"      # 선택
entrypoint = ""                  # 선택, ""이면 이미지의 ENTRYPOINT를 비움
gpu_mem = "20g"                  # 선택, 필요한 최소 GPU 메모리
gpu_models = ["*PRO 6000*"]      # 선택, 이 모델명 패턴의 GPU에서만 실행 (대소문자 무시)
ram = "16g"                      # 선택, 기본 default_ram
priority = 0                     # 선택, 클수록 먼저
env = { WANDB_MODE = "offline" }
[[mounts]]
src = "/vfroot/local/user-xxxx/proj"
dst = "/home/work/proj"
readonly = false
```

- 제출할 때 모든 `src`는 `allowed_mount_roots` 중 하나 아래에 있어야 하고, `..`를 풀어낸 실제 경로로
  검사합니다.
- 작업은 제출자의 uid/gid로 실행합니다. root가 제출하면 `--as uid:gid`를 반드시 줘야 합니다
  (스팟을 root로 돌리지 않기 위해).

### 2.9 CLI `labgpu-spot`

| 명령 | 동작 |
|---|---|
| `labgpu-spot daemon [-c spot.toml]` | 컨트롤러 실행 (systemd용) |
| `labgpu-spot submit job.toml [--as uid:gid]` | 작업 제출, 작업 ID 출력 |
| `labgpu-spot ls [--all]` | 작업 목록 |
| `labgpu-spot cancel <job_id>` | 작업 취소 |
| `labgpu-spot status` | GPU별 상태(데몬이 매 틱 `<state_dir>/status.json`에 씀) |
| `labgpu-spot pause [--gpu UUID]` / `resume [--gpu UUID]` | 노드 또는 GPU 단위로 빌려주기 중지/재개. 중지하면 빌려준 것도 회수합니다. |

### 2.10 재시작 복구

데몬이 시작할 때:

1. 라벨 `labgpu.spot=1` 컨테이너를 모두 찾습니다.
2. DB에서 RUNNING/PREEMPTING인 작업마다: 컨테이너가 살아 있으면 그대로 LENT로 이어 갑니다.
   끝나 있으면 종료 처리(2.7), 컨테이너가 아예 없으면 회수된 것으로 보고 다시 QUEUED.
3. DB의 실행 중 작업과 연결되지 않은 스팟 컨테이너는 로그를 남기고 삭제합니다(`docker rm -f`).

데몬이 끝날 때:

- 정상 종료(SIGTERM/SIGINT)면 빌려준 GPU를 **모두 회수**하고 `docker stop`이 끝나기를 기다립니다.
  컨트롤러가 없으면 소유자를 지켜 줄 주체가 없기 때문입니다. 회수된 작업은 다음 시작 때 대기열로 돌아갑니다.
- 틱 처리 중 예외가 나면 모두 회수하고 다음 틱에서 계속합니다.
- 비정상 종료에 대비해 systemd 유닛의 `ExecStopPost`가 라벨 `labgpu.spot=1` 컨테이너를 멈춥니다.

### 2.11 로그

모든 lend/reclaim/launch/finish에 대해 한 줄 로그: GPU UUID, 작업 ID, 사유, 판정에 쓴 수치
(소유자 util, 소유자 mem, 기준값, 여유 메모리).

---

## 3. 테스트와 검증

### 3.1 단위 테스트 (GPU 없이, `pytest`)

- `fraction`: share→제한값, 환경변수 생성
- `devalloc`: 두 입력 형태 정규화
- `procmap`: cgroup v1/v2/systemd 형식 파싱
- `detector`: 상태 전이 전부(유휴 전환, 회수 조건 각각, UNKNOWN, 재시작 직후)
- `planner`: 회수 우선, backfill, RAM·GPU 메모리 적합성, 강제 불가 시 미대여
- `jobs`: 상태 전이, 재시도 한도, 재시작 복구
- `docker`: 생성되는 `docker run` 인자(마운트 검증 포함)

### 3.2 실기 검증표

개발 환경(Windows, GPU 없음)에서는 확인할 수 없는 항목입니다. 실제 노드에서 확인하면 상태를 바꿉니다.

2026-09-24에 WSL2(Ubuntu 24.04) + Docker Desktop에서 Backend.AI 26.9.0rc1 manager·agent를 소스로 띄우고
확인했습니다. GPU는 가짜 NVML(연구실 Primary 구성 흉내)과 실제 RTX 4050 Laptop 두 가지로 돌렸습니다.

| 항목 | 상태 |
|---|---|
| 플러그인이 실제 Backend.AI 클래스와 import·할당·docker 인자 생성까지 동작 (`tests/test_plugin_integration.py`) | 확인 (26.9 소스) |
| agent 하나가 GPU 종류별 슬롯(`pro6000.shares`, `pro5000l.shares`, `a6000.shares`)을 보고하고, 매니저 `/config/resource-slots/details`(WebUI가 읽는 API)에 이름·단위와 함께 나옴 | 확인 (26.9, 가짜 NVML) |
| 종류별 세션 생성: 각 세션이 맞는 GPU에 배정되고 CPU·RAM은 공용으로 집계됨, 꽉 찬 종류의 요청은 PENDING | 확인 (26.9, 가짜 NVML) |
| 분할 할당 시 HAMi-core 환경변수가 맞게 들어감 (96GB의 0.5 → `49152m`, SM 50) | 확인 |
| `get_hooks`로 넘긴 `libvgpu.so`가 agent에 의해 마운트되고 `LD_PRELOAD`에 붙음 | 확인 (`/opt/kernel/libbaihook.so:/opt/kernel/libvgpu.so`) |
| `block-compute-plugins`(모듈 경로)로 `cuda_frac`이 막히고, 설정 없는 `gpu_slot_4`는 건너뜀 | 확인 |
| 같은 etcd 설정에서 해당 GPU가 없는 노드는 슬롯 용량 0 | 확인 (실제 RTX 4050 노드에서 PRO 슬롯 0) |
| HAMi-core가 없으면 ERROR 후 장 단위(`<key>.device`)로 강등 | 확인 (실제 GPU) |
| 세션 컨테이너에 GPU가 UUID로 붙음 (`DeviceRequests`, `nvidia-smi -L`) | 확인 (실제 RTX 4050) |
| 실제 GPU 한 장을 0.5 세션 두 개가 나눠 받음 | 확인 (실제 RTX 4050, 스케줄링·환경변수까지) |
| **HAMi-core가 실제로 메모리 상한을 강제함** | **미확인.** WSL2에서는 HAMi-core 자체가 `cuInit` 후킹 재귀로 segfault합니다(Backend.AI 없이 단독 실행해도 동일). WSL의 `libcuda`가 Windows 드라이버로 넘기는 stub이라서입니다. **네이티브 Linux GPU 노드에서 확인해야 합니다.** |
| HAMi-core 빌드 | HEAD는 CUDA 12.5 이상 헤더가 필요합니다(`CUctxCreateParams`). CUDA 12.4로는 그 직전 커밋 `6b92be9`로 빌드됨. 노드의 CUDA 툴킷에 맞는 커밋을 고정해 쓰세요. |
| HAMi-core 공유 영역 파일은 같은 컨테이너 안의 다른 사용자(예: root `docker exec`)가 열면 `EACCES` | 확인. 세션 프로세스가 한 사용자로 돌면 문제없음 |
| HAMi-core `CUDA_DEVICE_SM_LIMIT`가 실제로 동작 | UNVERIFIED (네이티브 노드 필요) |
| 스팟: 유휴 시간 뒤 대여, `gpu_models`·`gpu_mem` 조건, 대여 메모리 = 총 − 사용 − 여유분 | 확인 (가짜 NVML, 실제 Backend.AI 소유자 세션) |
| 스팟: 소유자 사용률 상승 / 정체불명 프로세스 / 새 소유자 세션 → 회수(SIGTERM, exit 143) → 대기열 → 다른 GPU에 재대여 | 확인 (새 소유자 세션 회수는 실제 RTX 4050에서) |
| 스팟: pause/resume, 컨트롤러 SIGTERM 시 전부 회수, 허용 루트 밖 마운트 거부, 없는 이미지는 FAILED | 확인 |
| 스팟 컨테이너에 실제 GPU가 UUID로 붙음 (`--gpus device=<uuid>`) | 확인 (실제 RTX 4050) |
| Backend.AI 커널 이미지를 `docker run`으로 직접 실행 (`entrypoint = ""`) | 확인 (최소 커널 이미지) |
| `nvmlDeviceGetProcessUtilization`이 호스트 PID로 프로세스별 SM 사용률을 줌 | UNVERIFIED. WSL은 NVML 프로세스 정보를 주지 않음. 네이티브 노드 필요 |
| 소유자 CPU 조건(cgroup v2) | UNVERIFIED. Docker Desktop은 컨테이너 PID가 WSL 배포판에 보이지 않음 |
| **26.8.3(최신 안정판)**: 플러그인 로딩, 종류별 슬롯, 종류별 세션 배정과 HAMi-core 환경변수, 전체 테스트 61개 | 확인 (2026-09-24, WSL, 가짜 NVML) |
| **26.8.3 WebUI(26.8.1)**: 세션 생성 화면의 AI 가속기 종류 선택지가 PRO6000/PRO5000/A6000으로 나뉘고, 막대 최대값이 종류별로 2/1/1, CPU·메모리 최대값도 실제 서버 크기(21코어, 14.38GB) | 확인 (브라우저 자동 조작) |
| 26.9.0rc1에서는 세션 생성 화면의 자원 그룹 한도 요청(`accessible_scaling_groups`)이 업스트림 버그로 실패해 막대가 기본값(가속기 16, CPU 64)으로 나옴 | 업스트림 버그(2026-09-15 커밋 `700bc1c1fd`). 26.8.3에는 없음 |
| etcd `config/resource_slots`에 실제로 없는 가속기(`cuda.device` 등)가 등록되어 있으면 WebUI가 그것을 기본 선택지로 보여 줌 | 확인. 매니저가 시작할 때 넣는 것으로 보여, 운영 시 실제 종류만 남겨야 함(`e2e/prune_slots.sh`) |
| 호스트 `Xorg`가 모든 GPU에 떠 있어도 빌려주고, Xorg 메모리는 빌려줄 양에서 빠지며, Xorg 때문에 회수하지 않음. 컨테이너 안의 같은 이름 프로세스와 목록에 없는 호스트 프로세스는 여전히 회수 사유 | 확인 (26.8.3, 가짜 NVML, 2026-09-25) |
| 25.19에서 위 항목 전부 | UNVERIFIED (연구실은 최신 안정판으로 올릴 예정) |
| 스팟 회수 시 소유자 작업이 실패하지 않음 (S3) | UNVERIFIED (HAMi-core 강제가 전제) |

## 4. 열린 질문

- **세션 모아 담기(bin packing).** 업스트림 `fill`은 가장 여유 있는 GPU부터 고르므로, 0.25 세션 네 개는
  GPU 네 장에 흩어집니다. 스팟에 통째로 빌려줄 GPU를 남기려면 가장 덜 빈 GPU부터 고르는 할당 맵
  (`FractionAllocMap` 하위 클래스)이 필요합니다. affinity hint 처리와 충돌하지 않게 만드는 방법을 검토해야 합니다.

- 스팟을 Backend.AI 세션으로 편입(A안: `cuda.spot` 슬롯을 동적으로 노출)할 수 있는가? agent가 실행
  중에 슬롯 용량 변경을 매니저에 반영하는지 확인이 필요합니다.
- 소유자가 GPU 메모리를 크게 잡은 채 쉬는 경우(예: 70GB 모델 상주) 빌려줄 여유가 거의 없습니다.
  이런 GPU를 대시보드로 보여 주고 정책(세션 정리 권고)으로 풀지 결정해야 합니다.
- 크레딧: `runs` 테이블에 빌려준 GPU 시간이 쌓이지만, 소유자별 집계와 우선순위 반영은 아직 없습니다.
