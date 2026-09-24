"""
Small training loop for the cuda-checkpoint experiment (e2e/node/cc_migrate.sh).

Runs on the GPU with the given UUID, holds extra GPU memory, and prints the step, the loss and a
parameter checksum so a pause or a move to another GPU shows up as a gap, not a reset.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import torch


def device_by_uuid(uuid: str) -> torch.device:
    want = uuid.removeprefix("GPU-").lower()
    for i in range(torch.cuda.device_count()):
        if str(torch.cuda.get_device_properties(i).uuid).lower() == want:
            return torch.device(f"cuda:{i}")
    sys.exit(f"GPU {uuid} is not visible to this process")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--hold-gb", type=float, default=8.0)
    parser.add_argument("--every", type=int, default=50)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    torch.manual_seed(0)
    d = device_by_uuid(args.uuid)
    model = torch.nn.Sequential(
        torch.nn.Linear(4096, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 1)
    ).to(d)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    x = torch.randn(8192, 4096, device=d)
    y = torch.randn(8192, 1, device=d)
    hold = torch.ones(int(args.hold_gb * 2**30) // 4, device=d)
    print(f"start {time.time():.3f} device={d} {torch.cuda.get_device_name(d)}", flush=True)
    step = 0
    while True:
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if step % args.every == 0:
            checksum = float(sum(p.double().sum() for p in model.parameters()))
            print(
                f"step {step} t={time.time():.3f} loss={float(loss):.6f} "
                f"params={checksum:.6f} hold={float(hold[-1]):.1f}",
                flush=True,
            )
        step += 1


if __name__ == "__main__":
    main()
