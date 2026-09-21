"""gpuidle -- inspect state, pause and resume the synthetic workload."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from . import control
from .config import load_config

DURATION = re.compile(r"(\d+(?:\.\d+)?)([smhd])")

STATE_COLORS = {"ACTIVE": "32", "BUSY": "36", "WATCH": "33", "PAUSED": "35", "BLOCKED": "31"}


def _use_color(stream=sys.stdout) -> bool:
    return stream.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _use_color() else text


def parse_duration(text: str) -> float:
    matches = DURATION.findall(text.strip().lower())
    if not matches:
        raise argparse.ArgumentTypeError(
            f"not a duration: {text!r} (try 30m, 2h, 1h30m)")
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return sum(float(v) * unit[u] for v, u in matches)


def _age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _load_state(cfg):
    try:
        return json.loads(cfg.state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def cmd_status(cfg, args) -> int:
    state = _load_state(cfg)
    if state is None:
        print(f"cannot read {cfg.state_path}", file=sys.stderr)
        print("is the daemon running?  systemctl status gpu-keepalive", file=sys.stderr)
        return 1

    if args.json:
        json.dump(state, sys.stdout, indent=2)
        print()
        return 0

    age = time.time() - state["updated"]
    stale = "  " + _c("(stale -- daemon stopped?)", "33") if age > 30 else ""
    c = state["config"]
    print(f"gpu-keepalive  pid={state['pid']}  updated {_age(age)} ago{stale}")
    print(f"  idle after {c['idle_minutes']:.0f}min / ballast {c['mem_fraction']:.0%} "
          f"of free / headroom {c['headroom_fraction']:.0%} / target util {c['target_util']:.0%}")

    p = state["paused"]
    if p["all"] or p["gpus"]:
        scope = "all gpus" if p["all"] else f"gpus {p['gpus']}"
        bits = []
        if p["manual"]:
            until = p["manual"].get("until")
            left = f", {_age(until - time.time())} left" if until else ", no expiry"
            bits.append(f"manual by {p['manual'].get('by', '?')}{left}")
        bits += [f"gpu-run pid={h['pid']} [{h.get('cmd', '')[:40]}]" for h in p["holds"]]
        print("  " + _c(f"paused: {scope} -- {', '.join(bits) or '?'}", "33"))

    print()
    print(f"  {'GPU':<4} {'STATE':<8} {'UTIL':>5} {'MEMORY':>15} {'FOREIGN':>17} {'IDLE':>6}  WORKER")
    for g in state["gpus"]:
        mem = f"{g['mem_used_gb']:.0f}/{g['mem_total_gb']:.0f}GiB"
        foreign = f"{g['foreign_procs']}proc {g['foreign_mem_gb']:.0f}GiB sm{g['foreign_sm']}%"
        idle = _age(g["idle_for_s"]) if g["idle_for_s"] else "-"
        if g["worker_pid"]:
            worker = (f"pid={g['worker_pid']} ballast={g['fill_gb']:.0f}GiB "
                      f"up {_age(g['uptime_s'])}")
        elif g["blocked_for_s"]:
            worker = f"blocked {_age(g['blocked_for_s'])}"
        else:
            worker = g["last_reason"][:44]
        label = _c(format(g["state"], "<8"), STATE_COLORS.get(g["state"], "0"))
        print(f"  {g['index']:<4} {label} {g['util']:>4}% {mem:>15} "
              f"{foreign:>17} {idle:>6}  {worker}")
    return 0


def cmd_pause(cfg, args) -> int:
    seconds = parse_duration(args.duration) if args.duration else None
    control.set_pause(cfg, seconds, args.message or "")
    window = f"for {_age(seconds)}" if seconds else "with no expiry"
    print(f"paused {window}; any running worker stops within {cfg.poll_interval:.0f}s")
    if not seconds:
        print("resume with: gpuidle resume")
    return 0


def cmd_resume(cfg, args) -> int:
    if control.clear_pause(cfg):
        print("resumed; the idle clock restarts from zero")
    else:
        print("there was no manual pause")
    if holds := control.active_holds(cfg):
        print(f"note: still held by {len(holds)} live gpu-run hold(s): "
              f"{[h['pid'] for h in holds]}")
    return 0


def cmd_hold(cfg, args) -> int:
    """Used by gpu-run."""
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus else None
    if args.drop:
        control.drop_hold(cfg, args.pid)
    else:
        control.add_hold(cfg, args.pid, args.cmd or "", gpus)
    return 0


def cmd_wait_idle(cfg, args) -> int:
    """Block until no worker is running on the given GPUs. Used by gpu-run."""
    deadline = time.time() + args.timeout
    want = {int(x) for x in args.gpus.split(",")} if args.gpus else None
    while time.time() < deadline:
        state = _load_state(cfg)
        if state is None:
            return 0          # no daemon, nothing to wait for
        if not [g for g in state["gpus"]
                if g["worker_pid"] and (want is None or g["index"] in want)]:
            return 0
        time.sleep(0.5)
    print(f"warning: workers still running after {args.timeout:.0f}s", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gpuidle", description=__doc__)
    ap.add_argument("--config", help="path to a TOML config file")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("status", help="show per-GPU state")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("pause", help="stop the workload (e.g. gpuidle pause 2h)")
    s.add_argument("duration", nargs="?", help="omit for no expiry")
    s.add_argument("-m", "--message", help="why")
    s.set_defaults(func=cmd_pause)

    s = sub.add_parser("resume", help="lift a manual pause")
    s.set_defaults(func=cmd_resume)

    s = sub.add_parser("hold", help=argparse.SUPPRESS)
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--cmd")
    s.add_argument("--gpus")
    s.add_argument("--drop", action="store_true")
    s.set_defaults(func=cmd_hold)

    s = sub.add_parser("wait-idle", help=argparse.SUPPRESS)
    s.add_argument("--gpus")
    s.add_argument("--timeout", type=float, default=30.0)
    s.set_defaults(func=cmd_wait_idle)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        args.func, args.json = cmd_status, False
    return args.func(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
