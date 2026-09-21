"""gpu-run -- run a command with the synthetic workload held off the GPUs.

    gpu-run python train.py
    gpu-run --gpus 0,1 ./serve.sh

Takes a hold, waits for any running worker to actually go away, then runs the command.
The hold is keyed by this process's pid, so it is released even if we are killed.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
import time

from . import control
from .config import load_config

NUMERIC_DEVICES = re.compile(r"^\d+(,\d+)*$")


def _target_gpus(explicit: str | None) -> list[int] | None:
    """None means 'hold everything'."""
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # Only usable when it is a plain index list; UUID forms tell us nothing about
    # NVML indices without a lookup, so fall back to holding every GPU.
    if NUMERIC_DEVICES.match(visible):
        return [int(x) for x in visible.split(",")]
    return None


def _wait_for_yield(cfg, gpus: list[int] | None, timeout: float) -> None:
    import json
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = json.loads(cfg.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return                     # no daemon; nothing to wait for
        busy = [g["index"] for g in state["gpus"]
                if g["worker_pid"] and (gpus is None or g["index"] in gpus)]
        if not busy:
            return
        time.sleep(0.5)
    print(f"gpu-run: warning: workers still up after {timeout:.0f}s; continuing anyway",
          file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gpu-run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", help="comma-separated NVML indices; "
                                   "default: $CUDA_VISIBLE_DEVICES, else all")
    ap.add_argument("--wait-timeout", type=float, default=60.0)
    ap.add_argument("--config")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.print_usage(sys.stderr)
        print("gpu-run: no command given", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    gpus = _target_gpus(args.gpus)
    pid = os.getpid()

    control.add_hold(cfg, pid, " ".join(command), gpus)
    try:
        _wait_for_yield(cfg, gpus, args.wait_timeout)
        try:
            child = subprocess.Popen(command)
        except FileNotFoundError:
            print(f"gpu-run: command not found: {command[0]}", file=sys.stderr)
            return 127
        except OSError as err:
            print(f"gpu-run: {err}", file=sys.stderr)
            return 126

        def _forward(signum, frame):  # noqa: ARG001
            with contextlib.suppress(OSError):
                child.send_signal(signum)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _forward)

        rc = child.wait()
        # Mirror the shell convention for signal-terminated children.
        return 128 - rc if rc < 0 else rc
    finally:
        control.drop_hold(cfg, pid)


if __name__ == "__main__":
    sys.exit(main())
