D=$(dirname "$0"); . $D/env.sh; H=$W/run/labgpu/hami
echo "=== HAMi-core alone in a plain --gpus container (no Backend.AI), debug log"
docker run --rm --gpus all -v $H/libvgpu.so:/hami/libvgpu.so:ro -v $D/cuda_probe.py:/p.py:ro \
  -e LD_PRELOAD=/hami/libvgpu.so -e CUDA_DEVICE_MEMORY_LIMIT_0=3070m -e LIBCUDA_LOG_LEVEL=4 \
  stable/labgpu-test:1.0 python3 /p.py 2>&1 | tail -14
echo "=== NVML features HAMi-core relies on, inside WSL"
docker run --rm --gpus all stable/labgpu-test:1.0 sh -c 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 | head -3; ls /usr/lib/x86_64-linux-gnu/ | grep -E "libcuda|libnvidia-ml|dxcore" | head; ls /usr/lib/wsl/lib 2>/dev/null | head -5'
