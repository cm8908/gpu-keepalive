"""Synthetic workload for a single GPU. The supervisor runs this as a child process
and stops it with SIGTERM.

It is a separate process on purpose: SIGTERM -> process exit -> CUDA context teardown
releases every byte the worker held, with no fragmentation left behind. That is both
faster and more reliable than calling torch.cuda.empty_cache() in-process.
"""
from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time

GIB = 1024 ** 3
CANDIDATE_N = (1024, 2048, 4096, 8192, 16384)


def _die(signum, frame):  # noqa: ARG001
    # Exit straight from the signal handler. Any kernels still queued are torn down
    # with the context, so the GPU is handed back essentially immediately.
    os.write(2, b"[worker] signal received, exiting\n")
    os._exit(0)


def _pick_dtype(torch):
    """Widest matmul dtype this device actually supports."""
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) >= (5, 3):
        return torch.float16, "fp16"
    return torch.float32, "fp32"


def _time_matmul(torch, a, b, c, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def _pick_matmul_n(budget_bytes: int, itemsize: int, probe_n: int,
                   probe_seconds: float, max_kernel_seconds: float) -> int:
    """Largest side length that fits the memory budget and still keeps one kernel
    under max_kernel_seconds.

    Sizing off a measured timing rather than a fixed constant is what makes this
    portable: matmul cost grows as n**3, so the same rule lands on a sensible kernel
    length whether the device does 20 TFLOP/s or 2 PFLOP/s. Kernels that run too long
    make the duty cycle coarse -- with a 1 s slice, a 400 ms kernel cannot express a
    10 % target.
    """
    best = CANDIDATE_N[0]
    for n in CANDIDATE_N:
        if 3 * n * n * itemsize > budget_bytes:
            continue
        if probe_seconds * (n / probe_n) ** 3 > max_kernel_seconds:
            continue
        best = n
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, required=True, help="NVML index, for logging")
    ap.add_argument("--gpu-uuid", required=True,
                    help="NVML UUID; used to select the device so we never depend on "
                         "CUDA ordinal ordering matching NVML enumeration")
    ap.add_argument("--fill-bytes", type=int, default=0)
    ap.add_argument("--target-util", type=float, default=0.7)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--matmul-n", type=int, default=0, help="0 = size it automatically")
    ap.add_argument("--compute-reserve-bytes", type=int, default=0,
                    help="memory budget for the matmul buffers; 0 = 2 GiB")
    ap.add_argument("--max-kernel-ms", type=float, default=20.0,
                    help="upper bound on one matmul, so the duty cycle stays fine-grained")
    ap.add_argument("--slice-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = run forever")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _die)
    signal.signal(signal.SIGINT, _die)

    # Selecting by UUID sidesteps CUDA_DEVICE_ORDER: by default CUDA orders devices
    # FASTEST_FIRST while NVML enumerates by PCI bus, so index N is not always device N.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    if not torch.cuda.is_available():
        print(f"[worker gpu{args.gpu}] no CUDA device visible", file=sys.stderr)
        return 3

    dev = torch.device("cuda:0")
    dtype, dtype_name = _pick_dtype(torch)
    itemsize = torch.finfo(dtype).bits // 8
    free_bytes, _total = torch.cuda.mem_get_info(dev)

    def buffers(n):
        return (torch.randn(n, n, device=dev, dtype=dtype),
                torch.randn(n, n, device=dev, dtype=dtype),
                torch.empty(n, n, device=dev, dtype=dtype))

    probe_n = CANDIDATE_N[0]
    try:
        a, b, c = buffers(probe_n)
    except torch.cuda.OutOfMemoryError:
        print(f"[worker gpu{args.gpu}] cannot allocate even {probe_n}x{probe_n} buffers",
              file=sys.stderr)
        return 3

    _time_matmul(torch, a, b, c, 3)                       # warm up
    probe = max(_time_matmul(torch, a, b, c, 10), 1e-6)

    budget = min(int(free_bytes * 0.5), args.compute_reserve_bytes or (2 * GIB))
    n = args.matmul_n or _pick_matmul_n(budget, itemsize, probe_n, probe,
                                        args.max_kernel_ms / 1000.0)
    if n != probe_n:
        del a, b, c
        torch.cuda.empty_cache()
        try:
            a, b, c = buffers(n)
        except torch.cuda.OutOfMemoryError:
            n = probe_n
            a, b, c = buffers(n)

    _time_matmul(torch, a, b, c, 3)
    per_iter = max(_time_matmul(torch, a, b, c, 10), 1e-4)

    # Ballast. Shrink the chunk size rather than giving up on the first OOM.
    ballast = []
    filled = 0
    chunk = GIB
    while filled < args.fill_bytes and chunk >= 64 * 1024 ** 2:
        want = min(chunk, args.fill_bytes - filled)
        try:
            ballast.append(torch.empty(want, device=dev, dtype=torch.uint8))
            filled += want
        except torch.cuda.OutOfMemoryError:
            chunk //= 4

    print(f"[worker gpu{args.gpu}] pid={os.getpid()} ballast={filled / GIB:.1f}GiB "
          f"matmul={n} {dtype_name} per_iter={per_iter * 1e3:.2f}ms "
          f"target_util={args.target_util:.0%}", flush=True)

    started = time.time()
    rng = random.Random(os.getpid())

    while True:
        if args.max_seconds and time.time() - started >= args.max_seconds:
            return 0

        target = min(max(args.target_util + rng.uniform(-args.jitter, args.jitter), 0.01), 1.0)
        iters = max(1, round(args.slice_seconds * target / per_iter))

        t0 = time.perf_counter()
        for _ in range(iters):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        busy = time.perf_counter() - t0

        # Clocks, thermals and contention all move the cost of one matmul around.
        per_iter = 0.8 * per_iter + 0.2 * (busy / iters)

        if (rest := args.slice_seconds - busy) > 0:
            time.sleep(rest)


if __name__ == "__main__":
    sys.exit(main())
