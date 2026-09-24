# SPEC: labgpu 동작 명세

> 이 문서가 기준입니다. 코드와 이 문서가 다르면 코드가 버그입니다. 동작을 바꿀 때는 이 문서를 먼저
> 고칩니다. 배경과 목표는 [INTENT.md](INTENT.md)를 보세요.

## 0. 구성 요소

| 구성 요소 | 어디서 도나 | 하는 일 |
|---|---|---|
| `cuda_frac` 가속기 플러그인 (`labgpu.accelerator`) | 각 GPU 노드의 Backend.AI **agent** 프로세스 안 | GPU를 `cuda.shares`(소수) 단위로 할당하고, HAMi-core로 컨테이너별 GPU 메모리·SM 사용률을 제한합니다. |
| `labgpu-spot` 유휴 감시기 (`labgpu.spot`) | 각 GPU 노드의 별도 systemd 서비스 (root) | 소유자가 안 쓰는 GPU를 판정해 현황 파일에 씁니다. 스팟 실행 자체는 WebUI 실행 모드로 만들 예정이며 설계 승인 전입니다(2장). |
| 공용 모듈 (`labgpu.fraction`, `labgpu.nvml`, `labgpu.procmap`, `labgpu.devalloc`) | 둘 다 | 할당량 계산, NVML 조회, PID→컨테이너 매핑. |

두 구성 요소는 **서로 독립적**입니다. 플러그인만 써도(fGPU만), 감시기만 써도(오픈소스 `cuda`
플러그인) 동작해야 합니다.

### 의존성

- Python ≥ 3.12
- `nvidia-ml-py` (NVML 바인딩, `import pynvml`)
- 플러그인: Backend.AI agent(26.8.3 기준, 25.x도 고려)가 이미 설치한 패키지(`ai.backend.agent`, `ai.backend.common`, `aiodocker`)
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

25.x는 `Mapping[SlotName, Mapping[DeviceId, Decimal]]`를, 최신 코드는 `DeviceAllocation`
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
     커널 생성을 거부합니다(25.12부터 있는 설정, E2E에서 발견).
  3. agent.toml `allow-compute-plugins`/`block-compute-plugins`를 모듈 경로로 적습니다
     (예: allow `["labgpu.accelerator"]`, block `["labgpu.accelerator.cuda_frac"]`).
  4. 이미지의 지원 가속기 목록(`ai.backend.accelerators` 라벨 또는 관리자 화면)에 새 key를 넣습니다.
  5. idle checker의 사용률 기준을 `<key>_util`, `<key>_mem`으로 적습니다.
  6. 자원 정책·프리셋에 종류별 슬롯(`pro6000.shares` 등)을 씁니다.

### 1.13 세션별 GPU 대여 통계

WebUI 세션 세부 화면(5장)이 "내 GPU가 지금 스팟에 빌려 나가 있는가"를 보여 줄 수 있도록, 플러그인이
세션(컨테이너)별 통계 두 개를 함께 보고합니다. 매니저는 25.8부터 세션 통계(`live_stat`)를 Prometheus에서
읽으며, 통계 이름을 가리지 않으므로 이 값도 세션 주인의 권한으로 조회됩니다.

| 통계 | current | capacity |
|---|---|---|
| `<key>_lent` | 이 세션이 받은 이 종류 GPU 중 지금 스팟에 빌려준 수 | 이 세션이 받은 이 종류 GPU 수 |
| `<key>_lent_since` | 가장 먼저 빌려준 시각(유닉스 초), 없으면 0 | 항상 1 |

`<key>_lent_since`의 capacity를 늘 1로 두는 이유: 매니저는 최근 시간 창(`[metric] timewindow`, 기본 1시간)의
통계를 에이전트 서비스 ID 구분 없이 **더해서** 돌려줍니다. 에이전트가 재시작되면 새 서비스 ID가 생겨 같은
세션 값이 한동안 2배, 3배로 합쳐집니다(업스트림 동작, 26.8.3에서 확인, 기존 `*_util` 등도 같음). capacity 합이
곧 중복 배수 k가 되므로, 화면은 모든 값을 k로 나눠 원래 값을 되살립니다.

- 출처는 같은 노드 스팟 컨트롤러의 현황 파일입니다. 설정 키 `spot_status_path`(기본
  `/var/lib/labgpu/status.json`)로 위치를 바꿀 수 있습니다.
- 파일이 없거나 60초보다 오래됐으면(`spot_status_max_age`) 두 통계를 **보고하지 않습니다**. 모르는 상태를
  "빌려주지 않음"으로 보여 주지 않기 위해서입니다.
- 세션이 받은 GPU는 컨테이너 환경변수 `LABGPU_DEVICE_UUIDS`로 알아냅니다(컨테이너별로 한 번만 조회해 기억).

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

## 2. `labgpu-spot` 유휴 감시기

> 스팟은 Backend.AI WebUI 세션 런처의 실행 모드로 만듭니다(스팟 세션도 Backend.AI 세션).
> 그 설계는 아직 승인 전이라 이 절에 적지 않습니다. 지금 구현은 GPU가 소유자에게서 놀고 있는지
> 판정해 현황 파일에 쓰는 부분까지입니다. 아래 "빌려준(LENT)" 관련 규칙은 판정기에 남아 있는
> 회수 판정 규칙이며, 실제로 빌려주는 동작은 아직 없습니다.

### 2.1 개념

- **소유자(owner)**: Backend.AI 커널 컨테이너(라벨 `ai.backend.kernel-id`가 있음)이면서 GPU가 붙어 있는 것.
- **점유(claimed) GPU**: 소유자 컨테이너가 하나라도 붙어 있는 GPU.
- **정체불명(unknown) 프로세스**: GPU를 쓰는데 소유자 컨테이너가 아닌 프로세스(호스트 프로세스,
  Backend.AI 밖의 컨테이너). **소유자 활동으로 간주**합니다(안전 우선).
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

[idle]
idle_minutes = 30            # 소유자가 이만큼 연속으로 쉬어야 빌려줌
owner_util_threshold = 5     # %, 소유자 프로세스 SM 사용률 합이 이 값을 넘으면 활동
owner_mem_delta_mib = 512    # 소유자 GPU 메모리가 기준보다 이만큼 늘면 활동
owner_cpu_threshold = 0.5    # 코어 수, 소유자 컨테이너 CPU 사용량이 넘으면 활동 (0이면 끔)
unclaimed_grace_seconds = 60 # 아무도 안 붙은 GPU를 빌려주기 전 대기
ignored_processes = ["Xorg"] # 호스트에서 늘 GPU를 쓰는, 소유자 활동이 아닌 프로세스 이름

[reclaim]
mem_reserve_mib = 2048       # 빌려줄 수 있는 메모리 = 총 − 현재 사용 − 이 값

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
- UNKNOWN 진입

### 2.9 CLI `labgpu-spot`

| 명령 | 동작 |
|---|---|
| `labgpu-spot daemon [-c spot.toml]` | 컨트롤러 실행 (systemd용) |
| `labgpu-spot status` | GPU별 상태(데몬이 매 틱 `<state_dir>/status.json`에 씀) |


### 2.9.1 현황 파일 `<state_dir>/status.json`

매 틱 원자적으로 다시 씁니다. `labgpu-spot status`와 플러그인(1.13)이 읽습니다.

```json
{"updated_at": 1790000000.0,
 "gpus": [{"uuid": "GPU-...", "state": "LENDABLE", "lendable": true, "must_reclaim": false,
           "reasons": [], "lendable_memory": 0, "idle_for": 7800.0, "model": "NVIDIA RTX PRO 6000 ...",
           "lent_job": null, "lent_since": null}]}
```

`lent_job`, `lent_since`는 스팟을 빌려준 GPU에 쓸 자리입니다. 빌려주는 기능이 아직 없어 늘 `null`입니다.

### 2.10 재시작

재시작하면 모든 GPU를 방금 쓴 것으로 보고 유휴 시간을 처음부터 다시 잽니다(2.4). 저장하는 상태는 없습니다.

### 2.11 로그

관측 실패와 Docker 오류를 로그로 남깁니다. 판정 사유는 현황 파일의 `reasons`에 들어갑니다.

---

## 5. WebUI 세션 세부 화면 (사용자 포크 `backend.ai-webui`)

세션 세부 화면(`react/src/components/SessionDetailContent.tsx`)의 정보 표에 두 줄을 더합니다. WebUI 확장
페이지 방식으로는 이 화면에 끼워 넣을 수 없어 포크를 직접 고칩니다. 변경은 새 컴포넌트 파일 하나와
세부 화면의 몇 줄로 한정합니다.

- **포트**: 세션의 `service_ports`마다 `서비스명  호스트:호스트포트 → 컨테이너포트`. 호스트는 메인 커널의
  `agent_addr`에서 꺼낸 주소입니다(매니저 API에 포트가 열린 주소가 따로 없어서입니다. 에이전트의 RPC 주소와
  컨테이너 포트를 여는 주소가 다른 설치에서는 맞지 않을 수 있어 실기 확인 항목). 값을 누르면 복사됩니다.
  서비스가 없으면 줄을 숨깁니다.
- **GPU 대여**: 메인 커널 `live_stat`의 `*_lent`, `*_lent_since`(1.13)를 모아
  "빌려주는 중 (GPU n/m, 시작 시각부터 경과 시간)" 또는 "빌려주지 않음". 해당 통계가 하나도 없으면(스팟을
  쓰지 않는 노드, 현황이 오래됨, 매니저에 Prometheus가 없음) 줄을 숨깁니다.
- 문구는 i18n(`resources/i18n/ko.json`, `en.json`)의 `labgpu.*` 키로 둡니다.
- 권한: 세부 화면이 이미 쓰는 세션 조회 권한 그대로입니다(주인과 관리자).

## 3. 테스트와 검증

### 3.1 단위 테스트 (GPU 없이, `pytest`)

- `fraction`: share→제한값, 환경변수 생성
- `devalloc`: 두 입력 형태 정규화
- `procmap`: cgroup v1/v2/systemd 형식 파싱
- `detector`: 상태 전이 전부(유휴 전환, 회수 조건 각각, UNKNOWN, 재시작 직후)
- `docker`: 소유자 컨테이너의 GPU 참조 해석 순서

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
| `nvmlDeviceGetProcessUtilization`이 호스트 PID로 프로세스별 SM 사용률을 줌 | UNVERIFIED. WSL은 NVML 프로세스 정보를 주지 않음. 네이티브 노드 필요 |
| 소유자 CPU 조건(cgroup v2) | UNVERIFIED. Docker Desktop은 컨테이너 PID가 WSL 배포판에 보이지 않음 |
| **26.8.3(최신 안정판)**: 플러그인 로딩, 종류별 슬롯, 종류별 세션 배정과 HAMi-core 환경변수, 전체 테스트 61개 | 확인 (2026-09-24, WSL, 가짜 NVML) |
| **26.8.3 WebUI(26.8.1)**: 세션 생성 화면의 AI 가속기 종류 선택지가 PRO6000/PRO5000/A6000으로 나뉘고, 막대 최대값이 종류별로 2/1/1, CPU·메모리 최대값도 실제 서버 크기(21코어, 14.38GB) | 확인 (브라우저 자동 조작) |
| 26.9.0rc1에서는 세션 생성 화면의 자원 그룹 한도 요청(`accessible_scaling_groups`)이 업스트림 버그로 실패해 막대가 기본값(가속기 16, CPU 64)으로 나옴 | 업스트림 버그(2026-09-15 커밋 `700bc1c1fd`). 26.8.3에는 없음 |
| etcd `config/resource_slots`에 실제로 없는 가속기(`cuda.device` 등)가 등록되어 있으면 WebUI가 그것을 기본 선택지로 보여 줌 | 확인. 매니저가 시작할 때 넣는 것으로 보여, 운영 시 실제 종류만 남겨야 함(`e2e/prune_slots.sh`) |
| 호스트 `Xorg`가 모든 GPU에 떠 있어도 LENDABLE로 판정하고, Xorg 메모리는 빌려줄 양에서 빠지며, Xorg는 활동으로 보지 않음. 컨테이너 안의 같은 이름 프로세스와 목록에 없는 호스트 프로세스는 여전히 회수 사유 | 확인 (26.8.3, 가짜 NVML, 2026-09-25) |
| WebUI 세션 세부 화면(사용자 포크 v26.8.1 기반): "포트" 줄과 "GPU 대여" 줄. 빌려준 세션은 "빌려주는 중, GPU 1/1, 경과 시간", 아닌 세션은 "빌려주지 않음" | 확인 (26.8.3, WSL, Prometheus 포함, 브라우저 자동 조작, 2026-09-25) |
| 에이전트 재시작 뒤 매니저가 통계를 중복 합산해도 화면 값이 맞음(capacity 1 보정) | 단위 테스트로 확인 |
| 연구실 서버(26.8.3으로 올리는 중)에서 위 항목 전부 | UNVERIFIED |
| 스팟 회수 시 소유자 작업이 실패하지 않음 (S3) | UNVERIFIED (HAMi-core 강제가 전제) |

## 4. 열린 질문

- **세션 모아 담기(bin packing).** 업스트림 `fill`은 가장 여유 있는 GPU부터 고르므로, 0.25 세션 네 개는
  GPU 네 장에 흩어집니다. 스팟에 통째로 빌려줄 GPU를 남기려면 가장 덜 빈 GPU부터 고르는 할당 맵
  (`FractionAllocMap` 하위 클래스)이 필요합니다. affinity hint 처리와 충돌하지 않게 만드는 방법을 검토해야 합니다.

- 스팟 실행 모드(WebUI 세션 런처)의 설계. 26.8.3 agent는 플러그인의 `available_slots`를 30초마다 다시 읽어
  매니저에 보고하므로(`UpdateSlotsTask`) 슬롯 용량을 실행 중에 바꿀 수 있다는 점까지는 확인했습니다.
  나머지(슬롯 형태, 회수 방법, 같은 종류 GPU로의 자동 이동)는 설계 승인 뒤에 적습니다.
- 소유자가 GPU 메모리를 크게 잡은 채 쉬는 경우(예: 70GB 모델 상주) 빌려줄 여유가 거의 없습니다.
  이런 GPU를 대시보드로 보여 주고 정책(세션 정리 권고)으로 풀지 결정해야 합니다.
- 크레딧: 빌려준 GPU 시간의 소유자별 집계와 우선순위 반영은 아직 없습니다.
