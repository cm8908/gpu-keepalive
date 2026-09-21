"""Configuration loading: TOML file, overridable by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

try:                          # Python 3.11+
    import tomllib
except ModuleNotFoundError:   # Python 3.9 / 3.10
    import tomli as tomllib  # type: ignore[no-redef]

GIB = 1024 ** 3

CONFIG_SEARCH = [
    os.environ.get("GPU_KEEPALIVE_CONFIG"),
    str(Path.home() / ".config" / "gpu-keepalive.toml"),
    "/etc/gpu-keepalive.toml",
    str(Path(__file__).resolve().parent.parent / "config" / "gpu-keepalive.toml"),
]


def _default_state_dir() -> str:
    env = os.environ.get("GPU_KEEPALIVE_STATE_DIR")
    if env:
        return env
    if os.access("/var/lib", os.W_OK):
        return "/var/lib/gpu-keepalive"
    return str(Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
               / "gpu-keepalive")


@dataclass
class Config:
    # --- Monitoring ---
    poll_interval: float = 5.0        # NVML sampling period, seconds
    idle_seconds: float = 1800.0      # required run of idle time before filling in
    idle_sm_threshold: int = 5        # foreign SM utilization below this % counts as idle

    # --- Memory ballast ---
    # Ballast = (free - headroom - compute_reserve) * mem_fraction.
    # Headroom scales with card size so the same config works on a 12 GiB laptop GPU
    # and a 180 GiB datacenter part.
    mem_fraction: float = 0.5
    headroom_fraction: float = 0.10   # fraction of TOTAL memory always left free
    headroom_min_gb: float = 1.0      # ...but never less than this
    min_fill_gb: float = 1.0          # below this, skip ballast and only burn compute
    compute_reserve_gb: float = 1.0   # set aside for the matmul buffers + CUDA context

    # --- Synthetic workload ---
    target_util: float = 0.70         # desired SM utilization, 0..1
    util_jitter: float = 0.05         # per-slice jitter so the graph is not a flat line
    matmul_n: int = 0                 # matmul side length; 0 = pick from free memory
    slice_seconds: float = 1.0        # duty-cycle period

    # --- Yielding ---
    foreign_mem_grow_mb: float = 256.0  # yield if foreign memory grows by this much
    kill_timeout: float = 5.0           # SIGTERM -> SIGKILL grace period

    # --- Scope ---
    gpus: list[int] | None = None     # None = every non-MIG device
    exclude_gpus: list[int] = field(default_factory=list)

    # --- Paths ---
    state_dir: str = field(default_factory=_default_state_dir)
    log_level: str = "INFO"

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / "state.json"

    @property
    def pause_path(self) -> Path:
        return Path(self.state_dir) / "pause.json"

    @property
    def holds_dir(self) -> Path:
        return Path(self.state_dir) / "holds"

    def headroom_bytes(self, mem_total: int) -> int:
        """Memory to leave untouched on a card of this size."""
        return int(max(mem_total * self.headroom_fraction, self.headroom_min_gb * GIB))


def load_config(path: str | None = None) -> Config:
    candidates = ([path] + CONFIG_SEARCH) if path else CONFIG_SEARCH
    raw: dict = {}
    used = None
    for cand in candidates:
        if cand and Path(cand).is_file():
            with open(cand, "rb") as fh:
                raw = tomllib.load(fh)
            used = cand
            break

    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg._source = used                               # type: ignore[attr-defined]
    cfg._unknown_keys = sorted(set(raw) - known)     # type: ignore[attr-defined]
    return cfg
