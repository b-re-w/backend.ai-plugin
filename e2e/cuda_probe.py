"""Inside a session: what GPU memory does CUDA report, and how much can we actually allocate?"""

import ctypes
import os

cuda = ctypes.CDLL("libcuda.so.1")


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: CUresult={rc}")


check(cuda.cuInit(0), "cuInit")
dev = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
name = ctypes.create_string_buffer(128)
cuda.cuDeviceGetName(name, 128, dev)
ctx = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
free, total = ctypes.c_size_t(), ctypes.c_size_t()
check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
MiB = 2**20
print(f"device={name.value.decode()} total={total.value // MiB}MiB free={free.value // MiB}MiB")
print("CUDA_DEVICE_MEMORY_LIMIT_0 =", os.environ.get("CUDA_DEVICE_MEMORY_LIMIT_0"))
print("LD_PRELOAD =", os.environ.get("LD_PRELOAD"))

# Allocate in 256 MiB chunks until the driver (or HAMi-core) refuses.
chunks, ptr = [], ctypes.c_uint64()
while True:
    rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), 256 * MiB)
    if rc != 0:
        print(f"allocation stopped after {len(chunks) * 256}MiB (CUresult={rc}; 2 = out of memory)")
        break
    chunks.append(ptr.value)
    if len(chunks) * 256 > 16 * 1024:
        print("allocated 16GiB without refusal")
        break
