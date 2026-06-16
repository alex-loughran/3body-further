"""Hunt for linear-stability windows in short-word BHH families.

The first stable orbit at L != 0 was found in the Jankovic #2 (b^3) family, where
lambda_max dips to 1 over a narrow L window. This asks whether that is isolated
or part of a pattern: continue a selected set of OTHER short-word families
(low k = short period, fast) over a band in L and look for the same signature.

Two-stage per family (the dip_trace recipe):
  1. trace_family -> curve points in (a,c,T,L) with single-segment lambda_max
  2. ACCURATE auto-segmented Floquet (analyse_orbit) on every point, in
     parallel -- single-segment lambda_max is off by ~0.02 near a window, so the
     accurate pass is what actually confirms/denies a stable interval.

Families are traced one at a time (sequential); max_steps is kept low so no
single family can grind indefinitely (the all-75 sweep was infeasible for
exactly that reason -- here we run a hand-picked handful).

Selection: short-word families other than #2, biased to higher L (mean
instability falls with L), spanning b^3..b^6.

Usage: python window_hunt.py [--band 0.3] [--ds 0.02] [--max-steps 60]
                             [--orbits 1 3 4 ...]
Outputs: window_hunt.json
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from three_body import ALL_ORBITS
from continuation import trace_family
from dip_trace import _accurate_floquet

SELECTED = [1, 3, 4, 5, 7, 8, 10, 11]   # short-word families, not #2


def hunt_one(nr, band, ds, max_steps, workers):
    o = next(x for x in ALL_ORBITS if x[0] == nr)
    _, L, a, c, T, k = o
    t0 = time.time()
    try:
        fam = trace_family(a, c, T, L, L_min=L - band, L_max=L + band,
                           ds0=ds, ds_max=ds, max_steps=max_steps,
                           verbose=False)
    except Exception as e:
        return {"nr": nr, "k": k, "L0": L, "error": str(e)}
    if not fam:
        return {"nr": nr, "k": k, "L0": L, "error": "no points"}

    with mp.Pool(workers) as pool:
        pts = [p for p in pool.map(_accurate_floquet, fam) if p.get("accurate")]
    pts.sort(key=lambda p: p["L"])
    if not pts:
        return {"nr": nr, "k": k, "L0": L, "error": "no accurate points"}

    best = min(pts, key=lambda p: p["lambda_max_accurate"])
    stable = [p for p in pts if p["is_stable"]]
    Ls = [p["L"] for p in pts]
    out = {
        "nr": nr, "k": k, "L0": L,
        "n_points": len(pts),
        "L_range": [min(Ls), max(Ls)],
        "min_lambda_max": best["lambda_max_accurate"],
        "L_at_min": best["L"],
        "n_stable_points": len(stable),
        "stable_L": [p["L"] for p in stable],
        "best_orbit": {kk: best[kk] for kk in ("a", "c", "T", "L")},
        "seconds": time.time() - t0,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", type=float, default=0.3)
    parser.add_argument("--ds", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, dest="max_steps", default=60)
    parser.add_argument("--orbits", type=int, nargs="*", default=SELECTED)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--out", type=str, default="window_hunt.json")
    args = parser.parse_args()

    print(f"=== Stable-window hunt: families {args.orbits}, "
          f"band +/-{args.band}, ds={args.ds}, max_steps={args.max_steps} ===")
    results = []
    for nr in args.orbits:
        r = hunt_one(nr, args.band, args.ds, args.max_steps, args.workers)
        results.append(r)
        if r.get("error"):
            print(f"  #{nr:>2}: ERROR {r['error']}")
            continue
        flag = ("  <== STABLE WINDOW" if r["n_stable_points"] else "")
        print(f"  #{r['nr']:>2} (b^{r['k']}): min lambda_max="
              f"{r['min_lambda_max']:.5f} at L={r['L_at_min']:.4f}  "
              f"[{r['L_range'][0]:.3f},{r['L_range'][1]:.3f}], "
              f"{r['n_stable_points']} stable pts ({r['seconds']:.0f}s){flag}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved: {args.out}")

    winners = [r for r in results if not r.get("error") and r["n_stable_points"]]
    near = [r for r in results if not r.get("error")
            and not r["n_stable_points"] and r["min_lambda_max"] < 1.05]
    print("\n=== Summary ===")
    if winners:
        print("  NEW stable windows found:")
        for r in winners:
            print(f"    #{r['nr']} (b^{r['k']}): L in "
                  f"[{min(r['stable_L']):.4f}, {max(r['stable_L']):.4f}], "
                  f"min lambda_max={r['min_lambda_max']:.6f}")
    else:
        print("  No new stable windows in this set.")
    if near:
        print("  Near-misses (min lambda_max < 1.05, worth a finer trace):")
        for r in near:
            print(f"    #{r['nr']} (b^{r['k']}): "
                  f"min lambda_max={r['min_lambda_max']:.4f} at "
                  f"L={r['L_at_min']:.4f}")


if __name__ == "__main__":
    main()
