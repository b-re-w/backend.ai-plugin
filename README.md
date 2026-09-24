# labgpu: Backend.AI 부분 GPU + 스팟 GPU 대여

연구실 Backend.AI(오픈소스, 26.8.3 기준)에 두 가지 기능을 붙이는 패키지입니다.

| 기능 | 구성 요소 | 한 줄 설명 |
|---|---|---|
| **부분 GPU (fGPU)** | `cuda_frac` 가속기 플러그인 | `0.5` GPU처럼 소수로 할당하고, HAMi-core로 메모리·SM 사용률을 실제로 제한합니다. |
| **GPU 종류별 슬롯** | `gpu_slot_1~4` 가속기 플러그인 | 한 서버에 여러 종류 GPU가 있을 때 agent를 나누지 않고 `pro6000.shares`, `a6000.shares`처럼 종류마다 슬롯을 둡니다. CPU·RAM은 공용으로 유동적입니다. |
| **스팟 대여** (만드는 중) | WebUI 세션 런처의 스팟 실행 모드 + `labgpu-spot` 유휴 감시기 | 소유자가 GPU를 안 쓰는 동안 그 GPU를 스팟 세션에 빌려주고, 소유자가 돌아오면 회수합니다. 지금은 유휴 판정까지만 있고, 실행 모드는 설계 승인 전입니다. |

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
- **HAMi-core** `libvgpu.so`: 직접 빌드해서 `/opt/labgpu/lib/libvgpu.so`에 둡니다. 노드의 CUDA 툴킷에 맞는
  커밋을 고르세요. HAMi-core 최신판은 CUDA 12.5 이상 헤더가 필요하고, CUDA 12.4라면 `6b92be9`로 빌드됩니다.

  ```bash
  docker run --rm -v /opt/labgpu/lib:/out nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c '
    apt-get update && apt-get install -y git cmake &&
    git clone https://github.com/Project-HAMi/HAMi-core.git /src && cd /src &&
    git checkout 6b92be9 && make && cp build/libvgpu.so /out/'
  ```

## 2. 설치

agent가 쓰는 파이썬 환경에 설치합니다.

```bash
<agent-venv>/bin/pip install /path/to/backend.ai-plugin
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
   | `hook_path` | `/opt/labgpu/lib/libvgpu.so` | HAMi-core 위치 |
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
[examples/per-model-slots.sh](examples/per-model-slots.sh)가 연구실 구성 그대로의 설정입니다.

1. etcd에 종류별 플러그인 설정과 슬롯을 등록합니다 (설정은 클러스터 공용, 서버마다 따로 둘 필요 없음).
2. **모든 GPU 노드의** `agent.toml`:

   ```toml
   [agent]
   allow-compute-plugins = ["labgpu.accelerator"]
   block-compute-plugins = ["labgpu.accelerator.cuda_frac"]

   [resource]
   # 새 key를 빠뜨리면 그 종류의 세션은 만들어지지 않습니다.
   allocation-order = ["pro6000", "pro5000l", "pro5000", "a6000", "cpu", "mem"]
   ```

3. agent를 재시작하고 로그에서 `[gpu_slot_N] labgpu ...: key=... devices=[...]`를 확인합니다.
   Secondary처럼 해당 종류가 없는 노드에서는 그 슬롯이 0으로 보고됩니다.
4. 이미지의 지원 가속기 목록에 새 key(`pro6000` 등)를 넣고, 자원 정책·프리셋을 종류별 슬롯으로 바꿉니다.

## 4. 스팟 대여

스팟은 WebUI 세션 런처에서 고르는 실행 모드로 만듭니다(스팟 세션도 Backend.AI 세션). 설계 승인 전이라
아직 쓸 수 없습니다. 지금 들어 있는 것은 노드마다 GPU가 소유자에게서 놀고 있는지 판정하는 감시기입니다.

1. [examples/spot.toml](examples/spot.toml)을 `/etc/labgpu/spot.toml`로 복사하고 값을 조정합니다.
2. systemd 서비스를 등록합니다.

   ```bash
   sudo cp examples/labgpu-spot.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now labgpu-spot
   journalctl -u labgpu-spot -f
   ```

3. 상태 확인: `sudo labgpu-spot status`

감시기를 끄거나 지워도 Backend.AI와 소유자 세션에는 영향이 없습니다.

## 5. 소유자가 알아 둘 것 (스팟 실행 모드가 생긴 뒤)

- 세션을 켜 둔 채 GPU를 **30분(기본)** 동안 안 쓰면 그 GPU가 스팟에 빌려질 수 있습니다. 세션은 그대로 유지됩니다.
- GPU를 다시 쓰기 시작하면(커널 실행, 메모리 추가 할당, CPU 사용) 스팟이 회수됩니다. 회수가 끝날 때까지
  잠깐 GPU를 나눠 쓰므로 느릴 수 있습니다.
- 모델을 GPU 메모리에 올려 둔 채 쉬면, 그 메모리는 건드리지 않고 **남는 메모리만** 빌려줍니다.
- 서버에서 Backend.AI를 거치지 않고 GPU를 직접 쓰면(직접 `python` 실행, `docker run --gpus` 등) 그 GPU의
  스팟은 회수됩니다. 누구의 작업인지 알 수 없어 주인 쪽으로 판단하기 때문입니다. 화면 서버(`Xorg`)처럼
  늘 떠 있는 프로그램은 `/etc/labgpu/spot.toml`의 `ignored_processes`에 넣어 예외로 둡니다(기본값 `["Xorg"]`).
- 같은 GPU를 쓰는 프로세스끼리는 GPU 하드웨어 오류(Xid)가 번질 수 있습니다. 드물지만 알고 계세요.

## 6. 개발

```bash
pip install -e ".[test]"
pytest
```

`tests/test_plugin_integration.py`는 `ai.backend.agent`를 import할 수 있을 때만 돕니다(Linux 필요).
Backend.AI 소스로 돌리려면 Python 3.13 환경에 `backend.ai/requirements.txt`를 설치한 뒤
`PYTHONPATH=src:../backend.ai/src pytest`를 실행합니다.

GPU가 없어도 테스트는 돌아갑니다. 판단 로직(`labgpu.spot.detector`, `labgpu.fraction`)은
순수 함수이고, NVML·Docker·Backend.AI는 얇은 어댑터(`labgpu.nvml`, `labgpu.spot.docker`,
`labgpu.accelerator.plugin`)에만 있습니다. 작업 원칙은 저장소 루트의 [AGENTS.md](../AGENTS.md)를 보세요.

```
src/labgpu/
  fraction.py        share → HAMi-core 제한값
  devalloc.py        Backend.AI 버전별 device_alloc 형태 정규화
  nvml.py            NVML 어댑터
  procmap.py         PID → 컨테이너 ID
  selection.py       GPU 종류 선택(모델명·메모리)과 중복 점유 방지
  accelerator/plugin.py   플러그인 구현 (cuda_frac.py, gpu_slot.py는 엔트리 포인트 모듈)
  spot/
    config.py  model.py  detector.py  observer.py
    docker.py  hostinfo.py  daemon.py  cli.py
```
