#!/usr/bin/env bash
# Build HAMi-core (libvgpu.so) at a pinned commit and install it into this checkout's .venv/lib,
# where the labgpu plugins look for it by default (SPEC 1.2 `hook_path`).
#
#   scripts/install_hami_core.sh                    # -> <this checkout>/.venv/lib/libvgpu.so
#   scripts/install_hami_core.sh --prefix DIR       # somewhere else
#   CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 HAMI_REF=6b92be9 scripts/install_hami_core.sh
#
# Builds inside a CUDA devel container, so the node needs Docker (no root, no local CUDA toolkit).
# HAMi-core after 6b92be9 needs CUDA >= 12.5 headers; keep CUDA_IMAGE and HAMI_REF in step.
# Without the library the plugins log an ERROR and fall back to whole-GPU (`<key>.device`) slots.
set -euo pipefail

HAMI_REF=${HAMI_REF:-ec5d85a3d709e5ed138a1668ebfefd366c05ca1e}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}

prefix=$(cd "$(dirname "$0")/.." && pwd)/.venv/lib
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) prefix="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
mkdir -p "$prefix"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

docker run --rm -v "$out":/out -e REF="$HAMI_REF" -e OWNER="$(id -u):$(id -g)" "$CUDA_IMAGE" bash -c '
  set -e
  apt-get update -qq && apt-get install -y -qq git cmake >/dev/null 2>&1
  git clone -q https://github.com/Project-HAMi/HAMi-core.git /src && cd /src
  git checkout -q "$REF"
  make >/tmp/build.log 2>&1 || { grep -iE "error" /tmp/build.log | head -20; exit 1; }
  cp build/libvgpu.so /out/ && git log -1 --format=%H > /out/COMMIT
  chown -R "$OWNER" /out'

install -m 0644 "$out/libvgpu.so" "$prefix/libvgpu.so"
echo "installed $prefix/libvgpu.so (HAMi-core $(cut -c1-12 "$out/COMMIT"), built with $CUDA_IMAGE)"
