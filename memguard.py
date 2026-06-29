"""Process-tree memory watchdog — keeps a runaway job from taking the box down.

Why this exists
---------------
The hunting jobs are launched as plain `python script.py` subprocesses (often
several at once). On a 16 GB machine the failure mode is not a single huge
allocation — it is *many* processes each holding numpy/scipy/numba (~0.5 GB
resident apiece). A parallel scan spawns one worker per core; run two scans
concurrently and physical RAM is gone, macOS falls into swap/compressed memory,
and the laptop becomes unresponsive and has to be force-shut. Twice now.

`resource.setrlimit(RLIMIT_AS, ...)` is the textbook fix but is unreliable
here: numba/LLVM and macOS frameworks reserve large *virtual* address ranges
without touching physical RAM, so a tight RLIMIT_AS trips spurious MemoryErrors
or segfaults inside JIT'd code. Instead we watch *resident* memory (RSS) of this
process **and all its descendants** (the multiprocessing workers) and abort the
whole tree if the total crosses a ceiling — well before the OS starts thrashing.

No third-party deps (psutil is not installed); RSS comes from `ps`.

Usage
-----
    import memguard
    memguard.install()              # ceiling from THREEBODY_MEM_LIMIT_GB or auto

Call once, early, in the *parent* process. Workers are descendants, so the
parent watchdog already accounts for their RSS — no need to instrument workers.
Override the ceiling with the THREEBODY_MEM_LIMIT_GB env var.
"""

import os
import signal
import subprocess
import sys
import threading
import time

_installed = False


def _total_ram_gb():
    """Physical RAM in GB (macOS sysctl, with a POSIX fallback)."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        return int(out) / 2**30
    except Exception:
        try:
            return (os.sysconf("SC_PHYS_PAGES") *
                    os.sysconf("SC_PAGE_SIZE")) / 2**30
        except Exception:
            return 16.0  # conservative default for this machine


def _descendant_rss_kb(root_pid):
    """Total RSS (KB) of root_pid and every process descended from it.

    One `ps` call builds the {pid: (ppid, rss)} table; we BFS the child tree
    from root_pid. Returns (total_kb, n_procs). Best-effort: any parse hiccup
    just yields what we could read, so the watchdog never crashes the job.
    """
    out = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="])
    info = {}                      # pid -> (ppid, rss_kb)
    children = {}                  # ppid -> [pid, ...]
    for line in out.split(b"\n"):
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss = (int(parts[0]), int(parts[1]), int(parts[2]))
        info[pid] = (ppid, rss)
        children.setdefault(ppid, []).append(pid)

    total, n, stack = 0, 0, [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in info:
            continue
        seen.add(pid)
        total += info[pid][1]
        n += 1
        stack.extend(children.get(pid, ()))
    return total, n


def _kill_descendants(root_pid):
    """SIGTERM, then SIGKILL, every descendant of root_pid (not root itself)."""
    try:
        _, _ = _descendant_rss_kb(root_pid)  # warm the table via the same path
    except Exception:
        pass
    # Recompute the descendant pid set directly.
    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,ppid="])
    except Exception:
        return
    children = {}
    for line in out.split(b"\n"):
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    descendants, stack = [], list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(children.get(pid, ()))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in descendants:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig is signal.SIGTERM:
            time.sleep(1.0)


def install(limit_gb=None, interval=5.0, verbose=True):
    """Start a daemon watchdog that aborts this process tree above `limit_gb`.

    limit_gb : ceiling on combined RSS of this process + descendants.
               Default: THREEBODY_MEM_LIMIT_GB env var, else 60% of physical RAM
               (≈9.6 GB on a 16 GB box) — leaves headroom for the OS + an open IDE.
    interval : seconds between RSS samples.
    """
    global _installed
    if _installed:
        return
    _installed = True

    if limit_gb is None:
        env = os.environ.get("THREEBODY_MEM_LIMIT_GB")
        limit_gb = float(env) if env else max(4.0, 0.60 * _total_ram_gb())
    limit_kb = limit_gb * 2**20
    root_pid = os.getpid()

    if verbose:
        print(f"[memguard] ceiling {limit_gb:.1f} GB on pid {root_pid} "
              f"+ workers (set THREEBODY_MEM_LIMIT_GB to change)", flush=True)

    def watch():
        breaches = 0
        while True:
            time.sleep(interval)
            try:
                rss_kb, n = _descendant_rss_kb(root_pid)
            except Exception:
                continue
            if rss_kb > limit_kb:
                breaches += 1
                # Require two consecutive samples over the line to avoid
                # killing on a brief transient spike.
                if breaches < 2:
                    continue
                sys.stderr.write(
                    f"\n[memguard] ABORT: process tree at "
                    f"{rss_kb / 2**20:.1f} GB across {n} procs exceeds the "
                    f"{limit_gb:.1f} GB ceiling. Killing workers to protect "
                    f"the machine.\n")
                sys.stderr.flush()
                _kill_descendants(root_pid)
                os._exit(137)
            else:
                breaches = 0

    threading.Thread(target=watch, name="memguard", daemon=True).start()


def safe_worker_count(reserve_gb_per_worker=2.0, leave_free_gb=7.0,
                      hard_cap=None):
    """Workers that fit in RAM, not just cores.

    Sizes a pool by *memory headroom* rather than `cpu_count()`. Each scipy/numba
    worker resides at ~0.5–1 GB; on a 16 GB box with an IDE open we must leave a
    few GB free. Returns at least 1.

    reserve_gb_per_worker : budgeted resident memory per worker.
    leave_free_gb         : RAM to keep for the OS, IDE, and the parent process.
    hard_cap              : optional absolute ceiling (also via
                            THREEBODY_MAX_WORKERS env var).
    """
    import multiprocessing as mp
    by_ram = int((_total_ram_gb() - leave_free_gb) / reserve_gb_per_worker)
    n = max(1, min(mp.cpu_count() - 1, by_ram))
    env = os.environ.get("THREEBODY_MAX_WORKERS")
    if env:
        n = min(n, int(env))
    if hard_cap:
        n = min(n, hard_cap)
    return max(1, n)