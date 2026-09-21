"""The daemon. Runs one state machine per GPU and never creates a CUDA context itself."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import control, nvml
from .config import GIB, Config, load_config

WORKER_PY = Path(__file__).resolve().parent / "worker.py"
log = logging.getLogger("gpu-keepalive")

PAUSED, BUSY, WATCH, ACTIVE, BLOCKED = "PAUSED", "BUSY", "WATCH", "ACTIVE", "BLOCKED"


@dataclass
class GpuState:
    index: int
    uuid: str
    name: str
    state: str = BUSY
    idle_since: float | None = None
    worker: subprocess.Popen | None = None
    worker_started: float = 0.0
    fill_bytes: int = 0
    baseline_mem: int = 0
    baseline_pids: set[int] = field(default_factory=set)
    failures: int = 0
    blocked_until: float = 0.0
    last_reason: str = ""

    @property
    def worker_pid(self) -> int | None:
        return self.worker.pid if self.worker else None


class Supervisor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stopping = False
        count = nvml.init()
        wanted = cfg.gpus if cfg.gpus is not None else list(range(count))
        excluded = set(cfg.exclude_gpus)

        self.gpus: dict[int, GpuState] = {}
        for idx in wanted:
            if not 0 <= idx < count or idx in excluded:
                continue
            info = nvml.describe(idx)
            if info.mig_enabled:
                log.warning("gpu%d (%s) is in MIG mode; skipping", idx, info.name)
                continue
            self.gpus[idx] = GpuState(index=idx, uuid=info.uuid, name=info.name)

        self.indices = list(self.gpus)
        Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
        cfg.holds_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ workers
    def _reap_orphans(self) -> None:
        """Kill workers left behind by a previous run."""
        me = os.getpid()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == me:
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().decode(errors="ignore")
            except OSError:
                continue
            if str(WORKER_PY) in cmdline:
                log.warning("killing orphaned worker pid=%s", entry.name)
                with contextlib.suppress(OSError):
                    os.kill(int(entry.name), signal.SIGKILL)

    def _start_worker(self, gs: GpuState, sample: nvml.GpuSample,
                      foreign_mem: dict[int, int]) -> None:
        cfg = self.cfg
        reserve = int(cfg.compute_reserve_gb * GIB)
        usable = sample.mem_free - cfg.headroom_bytes(sample.mem_total) - reserve
        fill = int(max(0, usable) * cfg.mem_fraction)
        if fill < cfg.min_fill_gb * GIB:
            # Not enough room to be worth it (a resident inference server holding most
            # of the card, say). Burn compute only.
            fill = 0

        cmd = [
            sys.executable, str(WORKER_PY),
            "--gpu", str(gs.index),
            "--gpu-uuid", gs.uuid,
            "--fill-bytes", str(fill),
            "--target-util", str(cfg.target_util),
            "--jitter", str(cfg.util_jitter),
            "--matmul-n", str(cfg.matmul_n),
            "--compute-reserve-bytes", str(reserve),
            "--slice-seconds", str(cfg.slice_seconds),
        ]
        try:
            proc = subprocess.Popen(cmd, start_new_session=True)
        except OSError as err:
            log.error("gpu%d: could not spawn worker: %s", gs.index, err)
            gs.failures += 1
            gs.blocked_until = time.time() + min(300 * gs.failures, 3600)
            return

        gs.worker = proc
        gs.worker_started = time.time()
        gs.fill_bytes = fill
        gs.baseline_mem = sum(foreign_mem.values())
        gs.baseline_pids = set(foreign_mem)
        gs.state = ACTIVE
        log.info("gpu%d: starting worker pid=%d ballast=%.1fGiB target_util=%.0f%%",
                 gs.index, proc.pid, fill / GIB, cfg.target_util * 100)

    def _stop_worker(self, gs: GpuState, reason: str) -> None:
        proc = gs.worker
        if proc is None:
            return
        gs.worker = None
        alive = time.time() - gs.worker_started

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=self.cfg.kill_timeout)
            except subprocess.TimeoutExpired:
                log.warning("gpu%d: worker ignored SIGTERM, sending SIGKILL", gs.index)
                proc.kill()
                proc.wait(timeout=5)
            except OSError:
                pass
        elif alive < 30 and proc.returncode not in (0, -signal.SIGTERM):
            # Died on its own almost immediately: back off before trying again.
            gs.failures += 1
            gs.blocked_until = time.time() + min(300 * gs.failures, 3600)
            log.error("gpu%d: worker exited after %.1fs with rc=%s; backing off %.0f min",
                      gs.index, alive, proc.returncode,
                      (gs.blocked_until - time.time()) / 60)

        if alive > 300:
            gs.failures = 0
        gs.last_reason = reason
        gs.idle_since = None          # yielding restarts the idle clock from zero
        gs.fill_bytes = 0
        log.info("gpu%d: stopped worker (%s, ran %.1f min)", gs.index, reason, alive / 60)

    # ------------------------------------------------------------------ decisions
    def _yield_reason(self, gs: GpuState, s: nvml.GpuSample, paused: bool,
                      foreign_mem: dict[int, int],
                      foreign_sm: dict[int, int]) -> str | None:
        cfg = self.cfg
        if paused:
            return "paused"
        if gs.worker and gs.worker.poll() is not None:
            return f"worker exited (rc={gs.worker.returncode})"

        if new_pids := set(foreign_mem) - gs.baseline_pids:
            return f"new process {sorted(new_pids)}"

        grown = sum(foreign_mem.values()) - gs.baseline_mem
        if grown > cfg.foreign_mem_grow_mb * 1024 ** 2:
            return f"foreign memory grew {grown / GIB:.1f}GiB"

        if hot := {p: u for p, u in foreign_sm.items() if u >= cfg.idle_sm_threshold}:
            return f"foreign compute {hot}"

        # Cannot attribute utilization but something else is resident: assume it is busy.
        if not s.proc_util_ok and foreign_mem:
            return "per-process utilization unavailable"
        return None

    @staticmethod
    def _foreign_activity(s: nvml.GpuSample, foreign_sm: dict[int, int]) -> int:
        """Foreign activity while no worker of ours is running. With nothing of ours on
        the device, device-wide utilization *is* the foreign utilization."""
        return max([s.util, *foreign_sm.values()])

    # ------------------------------------------------------------------ main loop
    def tick(self) -> None:
        cfg = self.cfg
        pstat = control.pause_status(cfg)
        own_pids = {gs.worker_pid for gs in self.gpus.values() if gs.worker_pid}
        samples = nvml.sample(self.indices, cfg.poll_interval + 1.0)
        now = time.time()

        for idx, gs in self.gpus.items():
            s = samples[idx]
            paused = control.is_paused(pstat, idx)
            foreign_mem, foreign_sm = s.foreign(own_pids)

            if gs.worker is not None:
                if reason := self._yield_reason(gs, s, paused, foreign_mem, foreign_sm):
                    self._stop_worker(gs, reason)
                    gs.state = PAUSED if paused else BUSY
                continue

            if paused:
                gs.state, gs.idle_since = PAUSED, None
                continue
            if now < gs.blocked_until:
                gs.state = BLOCKED
                continue

            if self._foreign_activity(s, foreign_sm) >= cfg.idle_sm_threshold:
                gs.state, gs.idle_since = BUSY, None
                continue

            if gs.idle_since is None:
                gs.idle_since = now
                log.info("gpu%d: went idle (util=%d%%, %d foreign process(es))",
                         idx, s.util, len(foreign_mem))
            gs.state = WATCH
            if now - gs.idle_since >= cfg.idle_seconds:
                self._start_worker(gs, s, foreign_mem)

        self._write_state(samples, pstat, own_pids)

    def _write_state(self, samples, pstat, own_pids) -> None:
        now = time.time()
        payload = {
            "updated": now,
            "pid": os.getpid(),
            "paused": {
                "all": pstat["all"],
                "gpus": sorted(pstat["gpus"]),
                "manual": pstat["manual"],
                "holds": pstat["holds"],
            },
            "config": {
                "idle_minutes": self.cfg.idle_seconds / 60,
                "mem_fraction": self.cfg.mem_fraction,
                "headroom_fraction": self.cfg.headroom_fraction,
                "target_util": self.cfg.target_util,
            },
            "gpus": [],
        }
        for idx, gs in self.gpus.items():
            s = samples[idx]
            fmem, fsm = s.foreign(own_pids)
            payload["gpus"].append({
                "index": idx,
                "name": gs.name,
                "state": gs.state,
                "util": s.util,
                "mem_used_gb": round(s.mem_used / GIB, 1),
                "mem_total_gb": round(s.mem_total / GIB, 1),
                "foreign_procs": len(fmem),
                "foreign_mem_gb": round(sum(fmem.values()) / GIB, 1),
                "foreign_sm": max(fsm.values(), default=0),
                "idle_for_s": round(now - gs.idle_since, 1) if gs.idle_since else 0,
                "worker_pid": gs.worker_pid,
                "fill_gb": round(gs.fill_bytes / GIB, 1),
                "uptime_s": round(now - gs.worker_started, 1) if gs.worker else 0,
                "last_reason": gs.last_reason,
                "blocked_for_s": round(max(0, gs.blocked_until - now), 1),
            })

        tmp = self.cfg.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.cfg.state_path)

    def run(self) -> int:
        self._reap_orphans()
        if not self.gpus:
            log.error("no usable GPUs; nothing to do")
            return 1
        log.info("watching gpus=%s idle_after=%.0fmin mem_fraction=%.0f%% "
                 "headroom=%.0f%% target_util=%.0f%%",
                 self.indices, self.cfg.idle_seconds / 60, self.cfg.mem_fraction * 100,
                 self.cfg.headroom_fraction * 100, self.cfg.target_util * 100)
        while not self.stopping:
            try:
                self.tick()
            except Exception:
                log.exception("tick failed; continuing")
            time.sleep(self.cfg.poll_interval)
        return 0

    def shutdown(self) -> None:
        self.stopping = True
        for gs in self.gpus.values():
            self._stop_worker(gs, "daemon shutting down")
        nvml.shutdown()


def main() -> int:
    cfg = load_config()
    # Under systemd the journal already stamps every line, and its clock is the
    # authoritative one.
    under_journal = bool(os.environ.get("JOURNAL_STREAM"))
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s" if under_journal
        else "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if unknown := getattr(cfg, "_unknown_keys", None):
        log.warning("ignoring unknown config keys: %s", unknown)
    log.info("config: %s", getattr(cfg, "_source", None) or "(defaults)")

    sup = Supervisor(cfg)

    def _sig(signum, frame):  # noqa: ARG001
        log.info("shutdown signal received")
        sup.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    try:
        return sup.run()
    finally:
        sup.shutdown()


if __name__ == "__main__":
    sys.exit(main())
