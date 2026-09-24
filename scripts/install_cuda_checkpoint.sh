#!/usr/bin/env bash
# Install NVIDIA cuda-checkpoint on a GPU node, pinned to a known commit and checksum.
#
#   sudo scripts/install_cuda_checkpoint.sh                 # -> /opt/labgpu/bin/cuda-checkpoint
#   scripts/install_cuda_checkpoint.sh --prefix ~/cc-test   # no root, for a one-off test
#
# The binary is NVIDIA's (see its LICENSE upstream); we never vendor it into this repo.
# To move to a newer upstream build, update REF and SHA256 together after checking the release notes.
set -euo pipefail

REF=00d5cce84c628088d6caa203fc4af40c1538b6f7
SHA256=707fa7f54136824d6c1d6dd724b9b1717610f831033c00d06da474de363a06db
URL="https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/${REF}/bin/x86_64_Linux/cuda-checkpoint"
MIN_DRIVER=550        # checkpoint / restore on the same GPU
MIGRATE_DRIVER=580    # restore onto a different GPU (--device-map)

prefix=/opt/labgpu/bin
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) prefix="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$(uname -m)" == x86_64 ]] || { echo "only x86_64 is supported by this script" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: is the NVIDIA driver installed?" >&2; exit 1; }

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
major=${driver%%.*}
echo "driver ${driver}"
if (( major < MIN_DRIVER )); then
  echo "driver ${driver} is older than ${MIN_DRIVER}: cuda-checkpoint does not work" >&2
  exit 1
fi
if (( major < MIGRATE_DRIVER )); then
  echo "WARNING: driver ${driver} < ${MIGRATE_DRIVER}: moving a process to another GPU is not supported" >&2
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl -fsSL -o "$tmp" "$URL"
echo "${SHA256}  ${tmp}" | sha256sum -c --quiet - || { echo "checksum mismatch, not installing" >&2; exit 1; }

mkdir -p "$prefix"
install -m 0755 "$tmp" "${prefix}/cuda-checkpoint"
echo "installed ${prefix}/cuda-checkpoint (upstream ${REF:0:12})"
"${prefix}/cuda-checkpoint" --help | head -3
