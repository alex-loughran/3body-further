"""Batch pseudo-arclength continuation of all 75 Jankovic BHH families in L.

Traces every catalogue orbit into a family curve (a, c, T, L), then aggregates:

  - the fold map: turning points in L where families are born/annihilate
  - stability changes along each family (unit-circle crossings)
  - stability windows: any L interval where a family is linearly stable
    (the L=0.83 b^3 window was the first; others may exist)

ROBUSTNESS (this replaces an earlier multiprocessing.Pool version that
deadlocked): each family is traced in its OWN subprocess with a hard
wall-clock timeout. This is the only reliable way to bound runtime, because
continuation marches families through chaotic regions where DOP853 grinds on
near-collisions and signal-based timeouts can't interrupt scipy's C code.

  - one subprocess per family, killed if it exceeds --timeout seconds
  - each family's result is saved to batch_families/fam_<nr>.json on completion
    => incremental (a crash loses only the in-flight family) and resumable
    (re-running skips families already on disk; use --force to redo)
  - up to --workers subprocesses run concurrently

Usage:
    python batch_continuation.py [--workers N] [--timeout 300]
                                 [--L-min 0.0] [--L-max 1.3] [--ds 0.02]
                                 [--orbits 1 2 3] [--force]
    python batch_continuation.py --single NR   # trace one family (worker mode)

Outputs: batch_continuation.json (+ batch_continuation.png overview)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from three_body import ALL_ORBITS
from continuation import trace_family, find_folds, find_bifurcations

OUTDIR = "batch_families"
MAX_STEPS = 400


def _family_path(nr):
    return os.path.join(OUTDIR, f"fam_{nr}.json")


def trace_single(nr, L_min, L_max, ds):
    """Worker mode: trace one family and write its result file."""
    o = next(x for x in ALL_ORBITS if x[0] == nr)
    _, L, a, c, T, k = o
    fam = trace_family(a, c, T, L, L_min=L_min, L_max=L_max,
                       ds0=ds, max_steps=MAX_STEPS, verbose=False)
    Ls = [p["L"] for p in fam]
    stable = [p for p in fam if p["n_unstable"] == 0]
    result = {
        "nr": nr, "k": k, "L0": L,
        "n_points": len(fam),
        "L_range": [min(Ls), max(Ls)] if Ls else None,
        "folds": find_folds(fam),
        "bifurcations": find_bifurcations(fam),
        "stable_points": [{"L": p["L"], "a": p["a"], "c": p["c"],
                           "T": p["T"], "lambda_max": p["lambda_max"]}
                          for p in stable],
        "points": [{"L": p["L"], "a": p["a"], "c": p["c"], "T": p["T"],
                    "lambda_max": p["lambda_max"],
                    "n_unstable": p["n_unstable"]} for p in fam],
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(_family_path(nr), "w") as f:
        json.dump(result, f)
    return result


def _run_subprocess(nr, L_min, L_max, ds, timeout):
    """Driver-side: trace family `nr` in a killable subprocess with timeout."""
    cmd = [sys.executable, os.path.abspath(__file__), "--single", str(nr),
           "--L-min", str(L_min), "--L-max", str(L_max), "--ds", str(ds)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True,
                              text=True)
    except subprocess.TimeoutExpired:
        return nr, "timeout", time.time() - t0
    if proc.returncode != 0 or not os.path.exists(_family_path(nr)):
        return nr, f"error(rc={proc.returncode})", time.time() - t0
    return nr, "ok", time.time() - t0


def aggregate_and_report():
    """Load all per-family files, aggregate, save summary + plot."""
    results = []
    for fn in sorted(os.listdir(OUTDIR)):
        if fn.startswith("fam_") and fn.endswith(".json"):
            with open(os.path.join(OUTDIR, fn)) as f:
                results.append(json.load(f))

    n_folds = sum(len(r["folds"]) for r in results)
    with_stable = [r for r in results if r["stable_points"]]
    print(f"\n  {len(results)} families on disk")
    print(f"  total folds found: {n_folds}")
    print(f"  families with a stable window: {len(with_stable)}")

    if with_stable:
        print(f"\n  Stable windows (by word length k):")
        for r in sorted(with_stable, key=lambda r: r["k"]):
            Ls = [p["L"] for p in r["stable_points"]]
            lam = min(p["lambda_max"] for p in r["stable_points"])
            print(f"    #{r['nr']:>2} (b^{r['k']}): L in "
                  f"[{min(Ls):.4f}, {max(Ls):.4f}], "
                  f"{len(Ls)} pts, lambda_max_min={lam:.5f}")

    with open("batch_continuation.json", "w") as f:
        json.dump({"families": results,
                   "summary": {"n_families": len(results), "n_folds": n_folds,
                               "n_stable": len(with_stable)}}, f, indent=1)
    print("\nSaved: batch_continuation.json")
    _plot(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=int, help="worker mode: trace one nr")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--timeout", type=float, default=200.0,
                        help="per-family wall-clock cap (s)")
    parser.add_argument("--L-min", type=float, dest="L_min", default=0.0)
    parser.add_argument("--L-max", type=float, dest="L_max", default=1.3)
    parser.add_argument("--band", type=float, default=0.45,
                        help="trace each family only over [L0-band, L0+band] "
                             "(clamped to [L-min, L-max]); families don't "
                             "exist far from their catalogue L, and forcing "
                             "the trace into far chaotic regions just grinds")
    parser.add_argument("--ds", type=float, default=0.03)
    parser.add_argument("--orbits", type=int, nargs="*")
    parser.add_argument("--force", action="store_true",
                        help="re-trace families already on disk")
    args = parser.parse_args()

    if args.single is not None:
        trace_single(args.single, args.L_min, args.L_max, args.ds)
        return

    os.makedirs(OUTDIR, exist_ok=True)
    orbits = ALL_ORBITS
    if args.orbits:
        orbits = [o for o in ALL_ORBITS if o[0] in args.orbits]
    todo = [o[0] for o in orbits
            if args.force or not os.path.exists(_family_path(o[0]))]
    skipped = len(orbits) - len(todo)

    L0_of = {o[0]: o[1] for o in ALL_ORBITS}
    print(f"=== Batch continuation: {len(todo)} families to trace "
          f"({skipped} already on disk), {args.workers} workers, "
          f"{args.timeout:.0f}s/family cap, band=+/-{args.band} ===", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for nr in todo:
            lo = max(args.L_min, L0_of[nr] - args.band)
            hi = min(args.L_max, L0_of[nr] + args.band)
            futs[ex.submit(_run_subprocess, nr, lo, hi, args.ds,
                           args.timeout)] = nr
        for fut in as_completed(futs):
            nr, status, dt = fut.result()
            done += 1
            print(f"  [{done}/{len(todo)}] #{nr}: {status} ({dt:.0f}s)",
                  flush=True)
    print(f"\n  traced in {time.time() - t0:.0f}s")
    aggregate_and_report()


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    for r in results:
        P = r["points"]
        if not P:
            continue
        ax.plot([p["L"] for p in P], [p["a"] for p in P], "-", lw=0.5,
                alpha=0.6)
        for fold in r["folds"]:
            ax.plot(fold["L"], fold["a"], "k.", ms=3)
    ax.set_xlabel("L"); ax.set_ylabel("a")
    ax.set_title("All BHH families in (L, a) — black dots = folds")

    ax = axes[1]
    for r in results:
        P = r["points"]
        if not P:
            continue
        ax.plot([p["L"] for p in P], [max(p["lambda_max"], 1.0) for p in P],
                "-", lw=0.5, alpha=0.5)
        if r["stable_points"]:
            sp = r["stable_points"]
            ax.plot([p["L"] for p in sp], [1.0] * len(sp), "g.", ms=6)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("L"); ax.set_ylabel("lambda_max")
    ax.set_title("Stability along families (green = stable windows)")
    plt.tight_layout()
    plt.savefig("batch_continuation.png", dpi=130)
    print("Plot: batch_continuation.png")


if __name__ == "__main__":
    main()
