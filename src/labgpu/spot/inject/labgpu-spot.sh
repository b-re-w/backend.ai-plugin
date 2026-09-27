# Mounted read-only into spot containers as /etc/profile.d/labgpu-spot.sh (and BASH_ENV) by the
# spot plugin (SPEC 2.12). After the monitor moved the session to another GPU it writes the GPU
# order and memory limits in force to /tmp/labgpu-spot-env; shells started now export them, so
# nvidia-smi and other programs see the GPU the session really has as the first one.
if [ -r /tmp/labgpu-spot-env ]; then
  while IFS='=' read -r _labgpu_key _labgpu_value; do
    case "$_labgpu_key" in
      CUDA_VISIBLE_DEVICES|LABGPU_SPOT_GPU|CUDA_DEVICE_MEMORY_LIMIT_*) export "$_labgpu_key=$_labgpu_value" ;;
    esac
  done < /tmp/labgpu-spot-env
  unset _labgpu_key _labgpu_value
fi
