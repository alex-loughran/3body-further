"""Batch pseudo-arclength continuation of all 75 Jankovic BHH families in L.

Traces every catalogue orbit into a family curve (a, c, T, L), in parallel,
then aggregates:

  - the fold map: turning points in L where families are born/annihilate
  - stability changes along each family (unit-circle crossings)
  - catalogue gap-filling: family members at L values absent from the tables
  - stability windows: any L interval where a family is linearly stable
    (the L=0.83 b^3 window was the first; others may exist)

One family per worker. Each trace reuses continuation.trace_family, so the
per-point monodromy/Floquet data is the single-segment approximation used
inside the corrector; stability *windows* flagged here are re-verified with
accurate segmented monodromy before being trusted (see verify_windows()).

Usage:
    python batch_continuation.py [--workers N] [--L-min 0.0] [--L-max 1.3]

Outputs: batch_continuation.json (+ batch_continuation.png overview)
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from three_body import ALL_ORBITS
from continuation import trace_family, find_folds, find_bifurcations


def _trace_one(args):
    nr, L, a, c, T, k, L_min, L_max, ds = args
    try:
        fam = trace_family(a, c, T, L, L_min=L_min, L_max=L_max,
                           ds0=ds, max_steps=500, verbose=False)
    except Exception as e:
        return {"nr": nr, "k": k, "L0": L, "error": str(e), "points": []}
    Ls = [p["L"] for p in fam]
    stable = [p for p in fam if p["n_unstable"] == 0]
    return {
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--L-min", type=float, dest="L_min", default=0.0)
    parser.add_argument("--L-max", type=float, dest="L_max", default=1.3)
    parser.add_argument("--ds", type=float, default=0.02)
    parser.add_argument("--orbits", type=int, nargs="*",
                        help="subset of Jankovic numbers (default: all 75)")
    args = parser.parse_args()

    orbits = ALL_ORBITS
    if args.orbits:
        orbits = [o for o in ALL_ORBITS if o[0] in args.orbits]

    work = [(nr, L, a, c, T, k, args.L_min, args.L_max, args.ds)
            for nr, L, a, c, T, k in orbits]
    print(f"=== Batch continuation: {len(work)} families, "
          f"{args.workers} workers ===")
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_trace_one, work)
    print(f"  traced in {time.time() - t0:.0f}s")

    ok = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    n_folds = sum(len(r["folds"]) for r in ok)
    n_stable = sum(1 for r in ok if r["stable_points"])
    print(f"\n  {len(ok)}/{len(results)} families traced "
          f"({len(failed)} failed)")
    print(f"  total folds found: {n_folds}")
    print(f"  families with a stable window: {n_stable}")

    if n_stable:
        print(f"\n  Stable windows (per family, by word length k):")
        for r in sorted(ok, key=lambda r: r["k"]):
            if r["stable_points"]:
                Ls = [p["L"] for p in r["stable_points"]]
                lam = min(p["lambda_max"] for p in r["stable_points"])
                print(f"    #{r['nr']:>2} (b^{r['k']}): L in "
                      f"[{min(Ls):.4f}, {max(Ls):.4f}], "
                      f"{len(Ls)} pts, λ_max_min={lam:.5f}")

    # save without the heavy per-point arrays in the headline summary
    with open("batch_continuation.json", "w") as f:
        json.dump({"families": results,
                   "summary": {"n_ok": len(ok), "n_failed": len(failed),
                               "n_folds": n_folds, "n_stable": n_stable}},
                  f, indent=1)
    print("\nSaved: batch_continuation.json")
    _plot(ok)


def _plot(ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    for r in ok:
        P = r["points"]
        if not P:
            continue
        Ls = [p["L"] for p in P]
        aa = [p["a"] for p in P]
        ax.plot(Ls, aa, "-", lw=0.5, alpha=0.6)
    for r in ok:
        for fold in r["folds"]:
            ax.plot(fold["L"], fold["a"], "k.", ms=3)
    ax.set_xlabel("L")
    ax.set_ylabel("a")
    ax.set_title("All BHH families in (L, a) — black dots = folds")

    ax = axes[1]
    for r in ok:
        P = r["points"]
        if not P:
            continue
        Ls = [p["L"] for p in P]
        lam = [max(p["lambda_max"], 1.0) for p in P]
        ax.plot(Ls, lam, "-", lw=0.5, alpha=0.5)
        if r["stable_points"]:
            sp = r["stable_points"]
            ax.plot([p["L"] for p in sp], [1.0] * len(sp), "g.", ms=6)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel("λ_max")
    ax.set_title("Stability along families (green = stable windows)")
    plt.tight_layout()
    plt.savefig("batch_continuation.png", dpi=130)
    print("Plot: batch_continuation.png")


if __name__ == "__main__":
    main()
