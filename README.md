# labgpu: Backend.AI 부분 GPU + 스팟 GPU 대여

연구실 Backend.AI(오픈소스, 26.8.3 기준)에 두 가지 기능을 붙이는 패키지입니다.

| 기능 | 구성 요소 | 한 줄 설명 |
|---|---|---|
| **부분 GPU (fGPU)** | `cuda_frac` 가속기 플러그인 | `0.5` GPU처럼 소수로 할당하고, HAMi-core로 메모리·SM 사용률을 실제로 제한합니다. |
| **GPU 종류별 슬롯** | `gpu_slot_1~4` 가속기 플러그인 | 한 서버에 여러 종류 GPU가 있을 때 agent를 나누지 않고 `cuda-pro6000.shares`, `cuda-a6000.shares`처럼 종류마다 슬롯을 둡니다. CPU·RAM은 공용으로 유동적입니다. |
| **스팟 대여** | `gpu_spot_1~4` 플러그인 + `labgpu-spot` 감시기 | WebUI 세션 런처에서 `PRO6000-SPOT` 같은 스팟 종류를 고르면, 주인이 안 쓰는 GPU에서 세션이 돕니다. 주인이 돌아오면 같은 종류의 빈 GPU로 옮기고(cuda-checkpoint), 빈 곳이 없으면 멈춰 두었다가 이어 갑니다. |

두 기능은 따로 켤 수 있습니다. 왜 만드는지는 [docs/INTENT.md](docs/INTENT.md), 정확한 동작은
[docs/SPEC.md](docs/SPEC.md)를 보세요.

> **상태:** WSL2에서 Backend.AI 26.9 manager·agent를 띄워 종류별 슬롯, 세션 배정까지
> 확인했습니다. **HAMi-core의 실제 메모리 제한은 WSL에서 확인할 수 없어(HAMi-core가 WSL에서 동작하지 않음)
> 네이티브 Linux GPU 노드에서 먼저 확인해야 합니다.** 자세한 결과는 [SPEC 3.2](docs/SPEC.md#32-실기-검증표).
> 연구실 서버는 25.15.6에서 26.8.3으로 올리는 중이라, 실제 서버에서의 동작은 아직 확인 전입니다.

---

## 1. 준비물 (GPU 노드마다)

- Backend.AI agent 26.8.3, Docker + NVIDIA Container Toolkit
- Python 3.12 이상 (agent와 같은 가상환경)
- 이 저장소 체크아웃. 외부 도구는 체크아웃 안의 `.venv`에 root 없이 설치하고, 플러그인과 감시기가 기본으로
  거기서 찾습니다.
  - **HAMi-core** (`libvgpu.so`): GPU를 소수로 나눠 줄 때 세션마다 GPU 메모리와 SM 사용률을 실제로 제한하는
    라이브러리입니다. `scripts/install_hami_core.sh` → `.venv/lib/libvgpu.so` (Docker의 CUDA 이미지 안에서
    고정 커밋으로 빌드). 없으면 플러그인이 ERROR를 남기고 장 단위(`<key>.device`) 슬롯으로 돌아갑니다.
  - **cuda-checkpoint**: 스팟을 옮길 때 씁니다(4장). `scripts/install_cuda_checkpoint.sh` → `.venv/bin/cuda-checkpoint`.

## 2. 설치

agent가 쓰는 파이썬 환경에 이 체크아웃을 **editable(`-e`)** 로 설치합니다. 플러그인이 HAMi-core와
cuda-checkpoint를 체크아웃의 `.venv`에서 찾기 때문입니다(`-e` 없이 설치하면 `hook_path`와 `cuda_checkpoint`를 직접 적어야 합니다).

```bash
<agent-venv>/bin/pip install -e /path/to/backend.ai-plugin
cd /path/to/backend.ai-plugin && scripts/install_hami_core.sh && scripts/install_cuda_checkpoint.sh
```

## 3. 부분 GPU(fGPU) 켜기

1. **오픈소스 `cuda` 플러그인을 막습니다.** 둘 다 `cuda` 장치를 쓰기 때문에 함께 켤 수 없습니다.
   이 목록은 엔트리 포인트 이름이 아니라 **모듈 경로**로 맞춥니다. `agent.toml`:

   ```toml
   [agent]
   block-compute-plugins = ["ai.backend.accelerator.cuda_open"]
   ```

2. **플러그인 설정** (선택, 기본값이면 생략). 설정 경로는 엔트리 포인트 이름을 따라
   `config/plugins/accelerator/cuda_frac/`입니다.

   ```bash
   backend.ai mgr etcd put config/plugins/accelerator/cuda_frac/allocation_mode fractional
   backend.ai mgr etcd put config/plugins/accelerator/cuda_frac/quantum_size 0.05
   ```

   | 키 | 기본값 | 설명 |
   |---|---|---|
   | `allocation_mode` | `fractional` | `discrete`면 기존처럼 장 단위 |
   | `shares_per_device` | `1` | GPU 1장 = 1 share |
   | `quantum_size` | `0.05` | 최소 할당 단위 |
   | `allocation_strategy` | `fill` | 요청 하나를 최소한의 GPU에 담음 (`evenly`는 여러 GPU로 쪼갬) |
   | `hook_path` | `<체크아웃>/.venv/lib/libvgpu.so` | HAMi-core 위치 (`scripts/install_hami_core.sh`) |
   | `sm_limit` | `true` | SM 사용률 제한 여부 |

3. **agent 재시작.** 로그에 `mode=fractional enforced=True`가 보이면 성공입니다.
   HAMi-core 파일이 없으면 ERROR 로그를 남기고 **장 단위로 돌아갑니다**. 제한할 수 없는 소수 할당은 내주지 않습니다.

4. **자원 정책·프리셋 수정.** 기존 `cuda.device`로 잡은 키페어 정책, 자원 프리셋, 프로젝트 한도를
   `cuda.shares`로 바꿔야 합니다.

5. **확인.** WebUI에서 fGPU `0.5` 세션을 만들고 안에서:

   ```bash
   nvidia-smi                        # 메모리 총량이 약 절반으로 보여야 함
   python -c "import torch; x = torch.empty(int(30e9), dtype=torch.uint8, device='cuda')"   # 40GB GPU라면 OOM이 나야 함
   ```

## 3-1. GPU 종류별 슬롯 (한 서버에 여러 종류의 GPU)

Primary처럼 PRO 6000·PRO 5000 72GB·A6000이 섞인 서버를 agent 하나로 운영합니다.
[scripts/register_slots.sh](scripts/register_slots.sh)가 연구실 구성 그대로의 설정입니다. 매니저 호스트에서
실행하면 etcd 설정과 함께 DB의 슬롯 종류 표(`resource_slot_types`)까지 넣습니다. 26.x는 이 표에 없는 슬롯을
받아 주지 않으니(agent heartbeat 실패) 이 단계를 빼먹지 마세요.

1. etcd에 종류별 플러그인 설정과 슬롯을 등록합니다 (설정은 클러스터 공용, 서버마다 따로 둘 필요 없음).
2. **모든 GPU 노드의** `agent.toml`:

   ```toml
   [agent]
   allow-compute-plugins = ["labgpu.accelerator"]
   block-compute-plugins = ["labgpu.accelerator.cuda_frac"]

   [resource]
   # 새 key를 빠뜨리면 그 종류의 세션은 만들어지지 않습니다.
   allocation-order = ["cuda-pro6000", "cuda-pro5000-72", "cuda-pro5000", "cuda-a6000", "cpu", "mem"]
   ```

3. agent를 재시작하고 로그에서 `[gpu_slot_N] labgpu ...: key=... devices=[...]`를 확인합니다.
   Secondary처럼 해당 종류가 없는 노드에서는 그 슬롯이 0으로 보고됩니다.
4. 자원 정책·프리셋을 종류별 슬롯으로 바꿉니다. key를 `cuda-`로 시작하게 지으면 이미지 라벨을 고칠 필요가 없습니다(SPEC 1.11).

## 4. 스팟 대여

스팟은 WebUI 세션 런처에서 고르는 실행 모드입니다. 스팟 세션도 보통 Backend.AI 세션이라 세션 목록과
사용량 통계에 그대로 보입니다. 동작 규칙은 [SPEC 2.12](docs/SPEC.md)를 보세요.

**매니저에서 한 번:** [scripts/register_slots.sh](scripts/register_slots.sh)가 종류별 슬롯과 함께
스팟 플러그인(`gpu_spot_1~4`, key `cuda-pro6000-spot` 등)과 슬롯(`cuda-pro6000-spot.device`)을 등록합니다.
agent의 `allocation-order`에 스팟 key도 넣어야 합니다. key가 `cuda-`로 시작하므로 이미지는 고치지 않아도 됩니다.

**GPU 노드마다:**

1. cuda-checkpoint 설치(드라이버 580 이상): 플러그인 폴더에서 `scripts/install_cuda_checkpoint.sh` →
   `.venv/bin/cuda-checkpoint` (root 불필요, 감시기가 기본으로 여기서 찾음). 없으면 스팟은 옮기지 못하고 주인이 돌아올 때 내보내집니다.
2. agent 재시작. 감시기는 **agent 안에서** 돕니다. 처음 초기화되는 스팟 플러그인이 감시기를 백그라운드로 띄우므로
   따로 등록할 서비스가 없습니다. agent 로그에 `spot monitor started in this process`가 보이면 됩니다.
3. 감시기 설정은 바꾸고 싶을 때만 etcd에 넣습니다. 예:
   `backend.ai mgr etcd put config/plugins/accelerator/gpu_spot_1/monitor/idle/idle_minutes 60`
   (키 목록은 SPEC 2.2, [examples/spot.toml](examples/spot.toml)에 같은 구역이 있습니다).
4. 상태 확인: `<agent venv>/bin/labgpu-spot status --state-dir <agent var-base-path>/labgpu`

agent를 재시작하는 동안에는 감시도 멈춥니다. 멈춰 둔 스팟 목록은 파일(`<agent var-base-path>/labgpu/parked.json`)에 남아
재시작 뒤 이어 받습니다. 스팟 플러그인 설정 `monitor_enabled = "false"`로 감시기를 끄면 현황 파일이 갱신되지 않아
스팟 자리가 1분 안에 0이 됩니다.

### 스팟 세션을 쓰는 사람이 알아 둘 것

- 세션 런처의 AI 가속기 종류에서 `…-SPOT`을 고르고 1개를 요청합니다. 빈자리가 없으면 세션은 대기합니다.
- 같은 종류 GPU가 모두 보이지만 **하나만** 쓰세요(`cuda:0`). 둘 이상 쓰면 내보내집니다.
- 주인이 돌아오면 프로그램은 몇 초 멈췄다가 다른 GPU에서 **그대로 이어서** 돕니다. 오류는 없습니다.
- 옮길 GPU가 없으면 GPU에서 빠진 채 멈춰 기다립니다(기본 5분). 그래도 자리가 없으면 `KeyboardInterrupt`
  (SIGINT)를 받고, 30초 뒤 강제 종료됩니다. 이때 컨테이너 안에 `/tmp/labgpu-spot-evicted`가 생기므로,
  직접 멈춘 것과 구분해 체크포인트를 저장하고 끝내도록 짜 두세요. 세션은 남아 있고, 다시 실행하면 빈 GPU에서 돕니다.

## 5. 소유자가 알아 둘 것

- 세션을 켜 둔 채 GPU를 **30분(기본)** 동안 안 쓰면 그 GPU가 스팟에 빌려질 수 있습니다. 세션은 그대로 유지됩니다.
- GPU를 다시 쓰기 시작하면(커널 실행, 메모리 추가 할당, CPU 사용) 스팟이 회수됩니다. 회수가 끝날 때까지
  잠깐 GPU를 나눠 쓰므로 느릴 수 있습니다.
- 모델을 GPU 메모리에 올려 둔 채 쉬면, 그 메모리는 건드리지 않고 **남는 메모리만** 빌려줍니다.
- 서버에서 Backend.AI를 거치지 않고 GPU를 직접 쓰면(직접 `python` 실행, `docker run --gpus` 등) 그 GPU의
  스팟은 회수됩니다. 누구의 작업인지 알 수 없어 주인 쪽으로 판단하기 때문입니다. 화면 서버(`Xorg`)처럼
  늘 떠 있는 프로그램은 감시기 설정 `idle/ignored_processes`에 넣어 예외로 둡니다(기본값 `["Xorg"]`).
- 같은 GPU를 쓰는 프로세스끼리는 GPU 하드웨어 오류(Xid)가 번질 수 있습니다. 드물지만 알고 계세요.

## 6. 개발

```bash
pip install -e ".[test]"
pytest
```

`tests/test_plugin_integration.py`는 `ai.backend.agent`를 import할 수 있을 때만 돕니다(Linux 필요).
Backend.AI 소스로 돌리려면 Python 3.13 환경에 `backend.ai/requirements.txt`를 설치한 뒤
`PYTHONPATH=src:../backend.ai/src pytest`를 실행합니다.

GPU가 없어도 테스트는 돌아갑니다. 판단 로직(`labgpu.spot.detector`, `labgpu.spot.placement`, `labgpu.fraction`)은
순수 함수이고, NVML·Docker·Backend.AI는 얇은 어댑터(`labgpu.nvml`, `labgpu.spot.docker`,
`labgpu.accelerator.plugin`)에만 있습니다. 작업 원칙은 저장소 루트의 [AGENTS.md](../AGENTS.md)를 보세요.

```
src/labgpu/
  fraction.py        share → HAMi-core 제한값
  devalloc.py        Backend.AI 버전별 device_alloc 형태 정규화
  nvml.py            NVML 어댑터
  procmap.py         PID → 컨테이너 ID
  selection.py       GPU 종류 선택(모델명·메모리)과 중복 점유 방지
  accelerator/plugin.py        플러그인 구현 (cuda_frac.py, gpu_slot.py는 엔트리 포인트 모듈)
  accelerator/spot_plugin.py   스팟 플러그인 (gpu_spot.py가 엔트리 포인트 모듈)
  spotstatus.py      감시기 현황 파일 읽기 (대여 통계, 스팟 자리 수)
  spot/
    config.py  model.py  detector.py  observer.py  placement.py
    ckpt.py  docker.py  hostinfo.py  daemon.py  cli.py
```
