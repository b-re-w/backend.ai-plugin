D=$(dirname "$0"); . $D/env.sh; H=$W/run/labgpu/hami; mkdir -p $H
docker run --rm -v $H:/out nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c '
  apt-get update -qq && apt-get install -y -qq git cmake >/dev/null 2>&1
  git clone -q https://github.com/Project-HAMi/HAMi-core.git /src && cd /src
  first=$(git log -S CUctxCreateParams --format=%h --reverse | head -1)
  git checkout -q ${first}^ && git log -1 --format="building HAMi-core %h (%cd), parent of $first"
  make >/tmp/b.log 2>&1 || { grep -iE "error" /tmp/b.log | head; exit 1; }
  cp build/libvgpu.so /out/ && ls -la /out/libvgpu.so && git log -1 --format=%H > /out/COMMIT' 2>&1 | grep -v "^==\|NVIDIA\|license\|Container\|^$\|docs.nvidia\|WARNING\|Use the\|CUDA Version\|governed\|By pulling\|A copy"
