"""Pause and hold files, shared by the CLI and the supervisor.

Two ways to stop the synthetic workload:
  - pause.json        a person running `gpuidle pause`, with an optional expiry
  - holds/<pid>.json  `gpu-run` holding the GPUs for as long as a command runs.
                      Keyed by pid, so a crashed job releases its hold by itself.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by somebody else
    return True


def set_pause(cfg, seconds: float | None, reason: str = "") -> dict:
    payload = {
        "until": time.time() + seconds if seconds else None,
        "reason": reason,
        "by": os.environ.get("USER", "?"),
        "at": time.time(),
    }
    _atomic_write(cfg.pause_path, payload)
    return payload


def clear_pause(cfg) -> bool:
    try:
        cfg.pause_path.unlink()
        return True
    except FileNotFoundError:
        return False


def add_hold(cfg, pid: int, cmd: str, gpus: list[int] | None) -> Path:
    path = cfg.holds_dir / f"{pid}.json"
    _atomic_write(path, {"pid": pid, "cmd": cmd, "gpus": gpus, "at": time.time()})
    return path


def drop_hold(cfg, pid: int) -> None:
    with contextlib.suppress(FileNotFoundError):
        (cfg.holds_dir / f"{pid}.json").unlink()


def active_holds(cfg) -> list[dict]:
    """Live holds only; files left by dead processes are cleaned up on the way past."""
    out = []
    if not cfg.holds_dir.is_dir():
        return out
    for path in sorted(cfg.holds_dir.glob("*.json")):
        data = _read(path)
        if not data or not _alive(int(data.get("pid", -1))):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        out.append(data)
    return out


def pause_status(cfg) -> dict:
    """Global and per-GPU pause state.

    Returns {"manual": dict|None, "holds": [...], "all": bool, "gpus": set[int]}
    where "all" means every GPU is held and "gpus" lists individually held ones.
    """
    manual = _read(cfg.pause_path)
    if manual:
        until = manual.get("until")
        if until is not None and until <= time.time():
            clear_pause(cfg)
            manual = None

    holds = active_holds(cfg)
    paused_all = manual is not None
    paused_gpus: set[int] = set()
    for h in holds:
        if h.get("gpus") is None:
            paused_all = True
        else:
            paused_gpus.update(int(g) for g in h["gpus"])

    return {"manual": manual, "holds": holds, "all": paused_all, "gpus": paused_gpus}


def is_paused(status: dict, gpu: int) -> bool:
    return status["all"] or gpu in status["gpus"]
