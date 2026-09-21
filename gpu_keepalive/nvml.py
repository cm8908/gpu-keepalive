"""NVML sampling. The supervisor never creates a CUDA context; it only reads this."""
from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field

import pynvml

log = logging.getLogger(__name__)

_PROC_UTIL_SUPPORTED = True


@dataclass
class DeviceInfo:
    index: int
    uuid: str
    name: str
    mem_total: int
    mig_enabled: bool


@dataclass
class GpuSample:
    index: int
    mem_total: int
    mem_used: int
    mem_free: int
    util: int                                               # device-wide SM utilization, %
    procs: dict[int, int] = field(default_factory=dict)    # pid -> bytes of GPU memory
    proc_sm: dict[int, int] = field(default_factory=dict)  # pid -> SM utilization, %
    proc_util_ok: bool = True                               # per-process query succeeded

    def foreign(self, own_pids: set[int]) -> tuple[dict[int, int], dict[int, int]]:
        """Memory and SM utilization of everything that is not one of our workers."""
        mem = {p: m for p, m in self.procs.items() if p not in own_pids}
        sm = {p: u for p, u in self.proc_sm.items() if p not in own_pids}
        return mem, sm


def init() -> int:
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetCount()


def shutdown() -> None:
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlShutdown()


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def describe(index: int) -> DeviceInfo:
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    try:
        mig_current, _pending = pynvml.nvmlDeviceGetMigMode(handle)
        mig = bool(mig_current)
    except pynvml.NVMLError:
        mig = False            # not supported on this device == not in MIG mode
    return DeviceInfo(
        index=index,
        uuid=_decode(pynvml.nvmlDeviceGetUUID(handle)),
        name=_decode(pynvml.nvmlDeviceGetName(handle)),
        mem_total=int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
        mig_enabled=mig,
    )


def _running_procs(handle) -> dict[int, int]:
    for fn in ("nvmlDeviceGetComputeRunningProcesses_v3",
               "nvmlDeviceGetComputeRunningProcesses"):
        try:
            procs = getattr(pynvml, fn)(handle)
        except (AttributeError, pynvml.NVMLError):
            continue
        # usedGpuMemory is None when NVML cannot attribute it (permissions, MIG).
        # Record 0 bytes but keep the pid: the process existing is what matters.
        return {int(p.pid): int(p.usedGpuMemory or 0) for p in procs}
    return {}


def _proc_sm(handle, window_s: float) -> tuple[dict[int, int], bool]:
    """Per-process SM utilization over the last window_s. Multiple samples per pid
    collapse to their maximum, which errs toward 'busy'."""
    global _PROC_UTIL_SUPPORTED
    since = int((time.time() - window_s) * 1e6)
    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(handle, since)
    except pynvml.NVMLError as err:
        if getattr(err, "value", None) == pynvml.NVML_ERROR_NOT_FOUND:
            return {}, True            # no samples == no activity
        if _PROC_UTIL_SUPPORTED:
            log.warning("per-process utilization unavailable (%s); "
                        "falling back to conservative yielding", err)
            _PROC_UTIL_SUPPORTED = False
        return {}, False
    out: dict[int, int] = {}
    for s in samples:
        pid = int(s.pid)
        out[pid] = max(out.get(pid, 0), int(s.smUtil))
    return out, True


def sample(indices: list[int], window_s: float) -> dict[int, GpuSample]:
    out: dict[int, GpuSample] = {}
    for idx in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            util = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except pynvml.NVMLError:
            util = 0
        proc_sm, ok = _proc_sm(handle, window_s)
        out[idx] = GpuSample(
            index=idx,
            mem_total=int(mem.total),
            mem_used=int(mem.used),
            mem_free=int(mem.free),
            util=util,
            procs=_running_procs(handle),
            proc_sm=proc_sm,
            proc_util_ok=ok,
        )
    return out
