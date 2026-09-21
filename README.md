# gpu-keepalive

Keeps idle NVIDIA GPUs occupied with a synthetic workload, and gets out of the way the
moment real work shows up.

A daemon watches every GPU independently. When a card has been idle for long enough it
starts a child process that holds a slice of memory and runs matmuls at a target
utilization. The instant anything else touches that card, the child is killed and every
byte it held is returned. Measured at 0.26 s on a B200.

```
BUSY ──(foreign SM util < 5%)──> WATCH ──(30 min unbroken)──> ACTIVE
  ^                                                             │
  └──────────────── any foreign activity, immediately ──────────┘
```

Turning on requires a sustained idle window; turning off takes a single sample. The
asymmetry is deliberate. It prevents flapping, and it errs toward the side that cannot
hurt anyone's job.

## Why you might want this

- Schedulers and allocation policies that reclaim, downsize, or deprioritize an
  allocation that looks idle.
- Utilization reporting on allocated or grant-funded compute, where a low average
  invites questions that a busy interactive machine does not deserve.
- Holding a memory reservation on a shared node so a neighbour does not take the card
  out from under a job you are about to start.
- Keeping clocks and thermals in a steady state between benchmark runs.

If your cluster's acceptable-use policy says something about synthetic load, read it
first. This tool makes an idle GPU look busy without doing useful work, and utilization
reporting cannot tell the difference.

## How "idle" is decided

Utilization alone is not enough, so the daemon judges on foreign activity: every
process on the card that is not one of its own workers. Per-process SM utilization comes
from `nvmlDeviceGetProcessUtilization`, which is why a card sitting at 100% because of
*our* worker is still correctly seen as free.

An inference server that is up but serving no requests counts as idle: memory pinned,
utilization at zero. There is nothing to ballast in that case, so the worker skips
memory entirely and only burns compute.

## When it yields

Any one of these ends the workload on the next sample:

| Trigger | The case it catches |
|---|---|
| A pid that was not there when the worker started | a training or serving job launching |
| Foreign memory grew by more than 256 MB | a resident process asking for more |
| Foreign SM utilization ≥ 5% | requests arriving at an already-resident server |
| Per-process utilization unreadable, something else resident | can't tell — so yield |
| `gpuidle pause`, or a `gpu-run` hold | a person said so |

Any pid it cannot account for is treated as somebody else's work. Processes in other
containers may not be visible in `/proc`; when that happens the answer is to yield, not
to assume the card is free.

## Why the worker is a separate process

`SIGTERM` → process exit → CUDA context teardown returns every byte with no
fragmentation left behind. That is stronger than calling `torch.cuda.empty_cache()`
in-process, and it cannot be blocked by a stuck Python thread. The worker exits straight
from its signal handler, so queued kernels are not waited on.

The supervisor itself never creates a CUDA context. It only reads NVML.

## Install

Requires Linux, an NVIDIA driver, Python 3.9+, and a CUDA-enabled PyTorch for the
worker.

```bash
pip install git+https://github.com/cm8908/gpu-keepalive.git
```

That installs the `gpuidle`, `gpu-run` and `gpu-keepalive` commands. PyTorch is
deliberately not a hard dependency, so install the build that matches your CUDA version.

To run it as a service, from a checkout:

```bash
sudo ./install.sh          # system-wide, runs as $SUDO_USER
./install.sh --user        # per-user, no root
sudo systemctl start gpu-keepalive
gpuidle status
```

A per-user service stops when your last session ends; `loginctl enable-linger "$USER"`
keeps it alive across logouts.

## Usage

```bash
gpuidle status             # per-GPU state, idle timers, ballast held
gpuidle status --json
gpuidle pause 2h           # for 2 hours; with no argument, until you resume
gpuidle resume

gpu-run python train.py            # holds the GPUs for as long as the command runs
gpu-run --gpus 0,1 ./serve.sh      # just those cards
```

`gpu-run` takes a hold, waits for the worker to be gone, and only then starts your
command. The hold is keyed by pid, so it is released even if the command crashes. Use it
for anything large: it closes the one race the polling loop cannot.

## Configuration

`/etc/gpu-keepalive.toml`, or `~/.config/gpu-keepalive.toml` for a user install.

| Key | Default | Meaning |
|---|---|---|
| `idle_seconds` | `1800` | idle time required before filling in |
| `idle_sm_threshold` | `5` | foreign SM utilization below this % is idle |
| `mem_fraction` | `0.5` | share of what's left over to hold as ballast |
| `headroom_fraction` | `0.10` | share of total memory always left free |
| `target_util` | `0.70` | desired SM utilization |
| `util_jitter` | `0.05` | per-slice jitter, so the graph is not a flat line |
| `matmul_n` | `0` | matmul size; `0` sizes it from a timing probe |
| `exclude_gpus` | `[]` | cards to leave alone |

Ballast is `(free - headroom - compute_reserve) * mem_fraction`, and headroom scales with
the card, so one config works everywhere:

| Card | Headroom | Ballast on an empty card |
|---|---|---|
| 12 GiB | 1.2 GiB | 4.9 GiB |
| 24 GiB | 2.4 GiB | 10.3 GiB |
| 80 GiB | 8.0 GiB | 35.5 GiB |
| 183 GiB | 18.3 GiB | 81.9 GiB |

More than half of free memory is always left alone. That is the defence against pushing
somebody else's job into an out-of-memory (OOM) error.

`target_util` defaults to 70% rather than 100% because running every card flat out costs
power and heat for nothing extra.

The worker sizes its own matmul from a timing probe: cost grows as `n³`, so it lands on
a kernel of roughly 20 ms whether the card does 20 TFLOP/s or 2 PFLOP/s. Kernels that run
too long make the duty cycle coarse: with a 1 s slice, a 400 ms kernel cannot express a
10% target. It also picks the widest dtype the device supports, falling back bf16 → fp16
→ fp32.

## Limitations

- Polling is every 5 seconds, so it cannot win a race against a job starting right now.
  `headroom_fraction` and `mem_fraction` are the cushion; `gpu-run` is the guarantee.
- A worker that dies within 30 seconds puts that card into exponential backoff, up to an
  hour. It shows as `BLOCKED` in `gpuidle status`.
- MIG-enabled devices are detected and skipped.
- NVIDIA only. Nothing here is portable to ROCm or XPU as written.

## Layout

```
gpu_keepalive/
  config.py       TOML configuration
  nvml.py         NVML sampling; no CUDA context is ever created here
  supervisor.py   per-GPU state machine, worker lifecycle, state.json
  worker.py       the synthetic workload; one child process per GPU
  control.py      pause and hold files
  cli.py          gpuidle
  gpu_run.py      gpu-run
```

Devices are selected by NVML UUID, not index. CUDA orders devices `FASTEST_FIRST` by
default while NVML enumerates by PCI bus, so index *n* is not reliably device *n*.

## License

MIT. See [LICENSE](LICENSE).
