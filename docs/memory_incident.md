# Out-of-memory shutdowns: diagnosis and fix

**Summary.** Twice, the development machine ran out of memory and had to be
hard-shut while orbit-hunting jobs were running — the second time the IDE was
reporting *>50 GB* in use on a **16 GB** machine. The cause was not a memory
leak in any script. It was **process oversubscription**: parallel scans sized
their worker pool to CPU cores (`mp.cpu_count()` = 8), and running two heavy
jobs at once put ~16 scipy/numba processes on a 16 GB box, exhausting RAM and
driving macOS deep into swap until the system became unusable. The fix sizes
pools by *RAM headroom* instead of core count and adds a process-tree memory
watchdog that aborts a runaway before it can swap the machine.

## What happened

During a session that was concurrently (a) running the period-doubling /
continuation jobs and (b) packaging the code as `threebody_physics` for a
downstream ML project, the laptop's memory filled, the machine became
unresponsive, and it had to be powered off. The same had happened once before.
The "50 GB" figure came from macOS counting compressed + swapped memory; it was
not a single 50 GB allocation.

## Diagnosis

The first hypothesis — a leak in a long integration or continuation loop — did
not survive inspection:

- **Every `solve_ivp` call is bounded.** `integrate_orbit` and the variational
  integrator use `max_step=0.01`, so the number of stored steps per call is
  capped. On a genuine near-collision, SciPy returns `success=False` rather than
  looping forever; it does not silently allocate without limit.
- **The continuation/PD loops retain only small data.** `continuation.trace_family`
  and the `pd_branch_v2` daughter loop append plain float dicts (a, c, T, L,
  multiplier magnitudes) per step — never the `sol` objects, which are the only
  heavyweight thing (each carries a full dense-output interpolant).
- **Plotting is headless and one-shot.** All plotters use the `Agg` backend and
  draw a single figure at the end of a run; no figure accumulates in a loop.
- **On-disk data is small** (~400 MB total), so IDE indexing alone cannot
  explain 50 GB.

A single script's resident memory therefore stays in the hundreds of MB. The
blow-up was only ever observed when **jobs overlapped**, which pointed at the
real mechanism.

### Root cause

The machine is **16 GB / 8 cores**. The parallel scanner defaulted to one worker
per core:

```python
if n_workers is None:
    n_workers = mp.cpu_count()          # = 8 on this machine
```

Each worker is a separate Python process holding numpy + scipy + numba, ~0.5–1 GB
resident. One scan ≈ 4 GB. Launch a **second** heavy job concurrently — as the
session did — and ~16 such processes coexist, RAM is exhausted, and the OS falls
back on swap/compressed memory. Throughput collapses, "memory used" climbs into
the tens of GB, and the machine has to be force-shut. On a 16 GB box the binding
constraint is **RAM headroom, not core count** — and `cpu_count()` is blind to it.

## The fix

A new dependency-free module, `memguard.py` (psutil is not installed; it reads
RSS via `ps`), with two layers.

**1. Size pools by memory, not cores.** `safe_worker_count()` budgets resident
memory per worker and leaves room for the OS, the IDE, and the parent process,
returning ~4 workers on this machine instead of 8:

```python
n_workers = memguard.safe_worker_count()   # RAM-aware; was mp.cpu_count()
```

Wired into `scanner.scan_parallel`, `bk_bias.run_census`, and `reproduce.py`.

**2. A process-tree watchdog.** `memguard.install()` starts a daemon thread that
every few seconds sums the resident memory of this process *and all its
descendant workers* and, if the total crosses a ceiling (default 60% of RAM
≈ 9.6 GB), SIGTERM/SIGKILLs the workers and aborts — well before the OS starts
thrashing. `RLIMIT_AS` was rejected as the mechanism because numba/LLVM reserve
large *virtual* ranges without using physical RAM, which trips spurious
`MemoryError`s inside JIT'd code; watching *resident* memory avoids that.

Both layers are tunable by environment variable:

| variable | effect |
| --- | --- |
| `THREEBODY_MAX_WORKERS` | hard cap on pool size |
| `THREEBODY_MEM_LIMIT_GB` | watchdog ceiling (combined RSS) |

The scanner's argument grid was also changed to stream lazily rather than
materialise all `n_rows × n_cols` tuples in the parent.

### Verification

A deliberate "memory bomb" (four workers each growing a buffer) under a 0.5 GB
test ceiling triggered the watchdog at 2.1 GB across 7 processes, killed the
tree, exited with code 137, and left **no orphaned processes**. With the default
ceiling, normal scans run untouched.

## The remaining safeguard is operational

Each job's watchdog only sees *its own* process tree, so two concurrent jobs can
still over-commit RAM without either one individually breaching its ceiling. The
code makes a single job safe; the standing rule that closes the gap is:

> **Do not run two heavy multiprocessing jobs at once. Serialise them.**

This is recorded in `CLAUDE.md` (Running jobs safely) and in project memory so
it survives across sessions, since these jobs are launched programmatically.

One manual IDE step remains outside the codebase: mark the generated-data
directories (`mini_results/`, `*.npz`, `*.egg-info`) as *Excluded* in PyCharm so
the indexer stops churning over them on every run.
