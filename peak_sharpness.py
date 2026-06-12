"""RPF peak sharpness analysis across the known Jankovic BHH orbits.

For every catalogue orbit, measures the width of its RPF peak in the
(a, c) scan plane by sampling the RPF along 1D slices through the orbit
— using the same integration settings as the scan campaigns, so the
measured widths are exactly what the scanner sees.

From the widths, models grid detection as a geometric lottery: a peak of
full width w is hit by a uniform grid of spacing d with probability
min(1, w/d) per axis, independently in a and c. This yields:

  - predicted recovery rate per L at the 500x500 campaign resolution
    (testable against the observed 2/9 at L=0.8)
  - the grid resolution required for a target recovery rate
  - peak width scaling with word length k and angular momentum L
    (the quantitative case for ML-guided search over brute-force grids)

Usage:
    python peak_sharpness.py [--L 0.8] [--threshold 3.5] [--workers N]

Outputs: peak_sharpness.json, peak_sharpness.png
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from three_body import (
    ALL_ORBITS,
    initial_conditions_from_params,
    is_negative_energy,
    return_proximity,
)

# Scan campaign settings (configs/bhh_500x500_jankovic.toml) — widths are
# only meaningful at the settings the scanner actually used.
T_MAX = 16.0
N_SAMPLES = 800
T_MIN_FRAC = 0.15

# Campaign grid geometry
A_RANGE = (0.05, 0.6)
C_RANGE = (-3.5, 4.0)


def offset_grid(n_per_side=12, lo=1e-4, hi=0.05):
    """Symmetric log-spaced offsets around 0 (resolves widths 1e-4..1e-1)."""
    pos = np.logspace(np.log10(lo), np.log10(hi), n_per_side)
    return np.concatenate([-pos[::-1], [0.0], pos])


def _evaluate_point(args):
    """RPF at one offset point. Mirrors scanner._evaluate_point."""
    key, a, c, L = args
    if a <= 0.001 or not is_negative_energy(a, c, L):
        return key, np.nan
    try:
        state0 = initial_conditions_from_params(a, c, L)
        d_min, _, _ = return_proximity(state0, T_MAX, t_min_frac=T_MIN_FRAC,
                                       n_samples=N_SAMPLES)
        return key, -np.log10(max(d_min, 1e-15))
    except (RuntimeError, FloatingPointError):
        return key, np.nan


def crossing_width(deltas, rpf, threshold):
    """Full peak width at the threshold, from a sampled 1D RPF slice.

    Walks outward from delta=0 on each side to the first sample below the
    threshold, then linearly interpolates the crossing. NaN samples count
    as below threshold. Returns 0.0 if the centre itself is below.
    """
    i0 = int(np.argmin(np.abs(deltas)))
    if not np.isfinite(rpf[i0]) or rpf[i0] < threshold:
        return 0.0

    def side(indices):
        prev_i = i0
        for i in indices:
            v = rpf[i] if np.isfinite(rpf[i]) else -np.inf
            if v < threshold:
                v_prev = rpf[prev_i]
                frac = (v_prev - threshold) / (v_prev - v)
                return abs(deltas[prev_i] +
                           frac * (deltas[i] - deltas[prev_i]))
            prev_i = i
        # Peak wider than the sampled range — report the range edge
        return abs(deltas[indices[-1]])

    left = side(range(i0 - 1, -1, -1))
    right = side(range(i0 + 1, len(deltas)))
    return left + right


def measure_all(orbits, deltas, n_workers, verbose=True):
    """RPF slices along a and c for every orbit, in one flat parallel pass.

    Returns {nr: {"a": rpf_array, "c": rpf_array}}.
    """
    tasks = []
    for nr, L, a, c, T, k in orbits:
        for di, d in enumerate(deltas):
            tasks.append(((nr, "a", di), a + d, c, L))
            tasks.append(((nr, "c", di), a, c + d, L))

    if verbose:
        print(f"  {len(tasks)} RPF evaluations across {len(orbits)} orbits "
              f"({n_workers} workers)")

    curves = {nr: {"a": np.full(len(deltas), np.nan),
                   "c": np.full(len(deltas), np.nan)}
              for nr, *_ in orbits}
    t0 = time.time()
    done = 0
    with mp.Pool(n_workers) as pool:
        for (nr, axis, di), val in pool.imap_unordered(_evaluate_point, tasks,
                                                       chunksize=8):
            curves[nr][axis][di] = val
            done += 1
            if verbose and done % max(1, len(tasks) // 20) == 0:
                el = time.time() - t0
                print(f"\r  {done}/{len(tasks)} "
                      f"[{el:.0f}s, ~{el * (len(tasks) - done) / done:.0f}s left]",
                      end="", flush=True)
    if verbose:
        print(f"\r  done in {time.time() - t0:.0f}s" + " " * 30)
    return curves


def detection_probability(w_a, w_c, n_grid):
    """P(grid point lands inside the peak) for an n_grid x n_grid scan."""
    da = (A_RANGE[1] - A_RANGE[0]) / (n_grid - 1)
    dc = (C_RANGE[1] - C_RANGE[0]) / (n_grid - 1)
    return min(1.0, w_a / da) * min(1.0, w_c / dc)


def analyse(results, n_grid=500, target=0.8):
    """Print the per-L recovery table, width-vs-k fit, and the resolution
    needed for the target recovery rate."""
    print(f"\n=== Peak widths (threshold crossing, full width) ===")
    print(f"{'nr':>4} {'L':>7} {'k':>4} {'rpf(0)':>7} {'w_a':>10} {'w_c':>10} "
          f"{'P_detect@' + str(n_grid):>13}")
    print("-" * 62)
    for r in results:
        print(f"{r['nr']:>4} {r['L']:>7.3f} {r['k']:>4} {r['rpf_centre']:>7.2f} "
              f"{r['w_a']:>10.2e} {r['w_c']:>10.2e} {r['P_detect']:>13.3f}")

    print(f"\n=== Predicted recovery per L at {n_grid}x{n_grid} ===")
    by_L = {}
    for r in results:
        by_L.setdefault(round(r["L"], 4), []).append(r)
    print(f"{'L':>8} {'orbits':>7} {'expected':>9} {'undetectable':>13}")
    print("-" * 42)
    for L in sorted(by_L):
        rs = by_L[L]
        exp = sum(r["P_detect"] for r in rs)
        n_dead = sum(1 for r in rs if r["w_a"] == 0 or r["w_c"] == 0)
        print(f"{L:>8.3f} {len(rs):>7} {exp:>9.2f} {n_dead:>13}")

    # Width scaling with word length (log-linear fit over detectable peaks)
    det = [r for r in results if r["w_a"] > 0 and r["w_c"] > 0]
    if len(det) >= 3:
        ks = np.array([r["k"] for r in det])
        for axis in ("w_a", "w_c"):
            ws = np.array([r[axis] for r in det])
            slope, intercept = np.polyfit(ks, np.log10(ws), 1)
            print(f"\n  log10({axis}) ≈ {slope:.3f}·k + {intercept:.2f}  "
                  f"(width shrinks 10x every Δk ≈ {abs(1 / slope):.1f})")

    # Resolution required for the target mean recovery
    print(f"\n=== Resolution for {target:.0%} mean recovery ===")
    n_required = None
    for n in np.unique(np.logspace(np.log10(200), 6, 200).astype(int)):
        mean_p = np.mean([detection_probability(r["w_a"], r["w_c"], n)
                          for r in results])
        if mean_p >= target:
            n_required = n
            break
    if n_required:
        print(f"  ~{n_required}x{n_required} "
              f"({n_required**2 / 1e6:.0f}M points; "
              f"{n_required**2 / n_grid**2:.0f}x the {n_grid}x{n_grid} cost)")
    else:
        ceiling = np.mean([1.0 if (r["w_a"] > 0 and r["w_c"] > 0) else 0.0
                           for r in results])
        print(f"  unreachable by grid refinement: {1 - ceiling:.0%} of "
              f"orbits have no detectable peak at these scan settings")
    return n_required


def plot(results, deltas, curves, n_grid, path="peak_sharpness.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det = [r for r in results if r["w_a"] > 0 and r["w_c"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    Ls = [r["L"] for r in det]
    sc = ax.scatter([r["k"] for r in det], [r["w_c"] for r in det],
                    c=Ls, cmap="viridis", s=30)
    ax.scatter([r["k"] for r in det], [r["w_a"] for r in det],
               c=Ls, cmap="viridis", s=30, marker="^")
    da = (A_RANGE[1] - A_RANGE[0]) / (n_grid - 1)
    dc = (C_RANGE[1] - C_RANGE[0]) / (n_grid - 1)
    ax.axhline(dc, color="r", ls="--", lw=1, label=f"dc @ {n_grid}x{n_grid}")
    ax.axhline(da, color="r", ls=":", lw=1, label=f"da @ {n_grid}x{n_grid}")
    ax.set_yscale("log")
    ax.set_xlabel("word length k")
    ax.set_ylabel("peak full width (o = c-axis, ^ = a-axis)")
    ax.legend(fontsize=8)
    ax.set_title("Peak width vs word length")
    fig.colorbar(sc, ax=ax, label="L")

    ax = axes[1]
    for r in sorted(results, key=lambda r: r["k"])[:6]:
        ax.plot(deltas, curves[r["nr"]]["c"], "-o", ms=2.5, lw=0.8,
                label=f"#{r['nr']} k={r['k']} L={r['L']:.2f}")
    ax.axhline(3.5, color="k", ls="--", lw=1)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Δc from orbit")
    ax.set_ylabel("RPF  (-log10 d_min)")
    ax.set_title("RPF slices through 6 lowest-k peaks (c-axis)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ns = np.unique(np.logspace(np.log10(200), 5, 80).astype(int))
    mean_p = [np.mean([detection_probability(r["w_a"], r["w_c"], n)
                       for r in results]) for n in ns]
    ax.plot(ns, mean_p, "b-")
    ax.axhline(0.8, color="k", ls="--", lw=1, label="80% target")
    ax.axvline(n_grid, color="r", ls="--", lw=1, label=f"{n_grid} (campaign)")
    ax.set_xscale("log")
    ax.set_xlabel("grid resolution N (NxN)")
    ax.set_ylabel("expected recovery rate")
    ax.set_title("Recovery vs resolution (uniform grid)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"\nPlot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, help="restrict to one L value")
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--n-grid", type=int, default=500,
                        help="campaign resolution for detection prediction")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--out", default="peak_sharpness.json")
    args = parser.parse_args()

    orbits = ALL_ORBITS
    if args.L is not None:
        orbits = [o for o in orbits if abs(o[1] - args.L) < 0.01]
    print(f"=== Peak sharpness: {len(orbits)} Jankovic orbits, "
          f"threshold={args.threshold} ===")

    deltas = offset_grid()
    curves = measure_all(orbits, deltas, args.workers)

    results = []
    for nr, L, a, c, T, k in orbits:
        i0 = int(np.argmin(np.abs(deltas)))
        w_a = crossing_width(deltas, curves[nr]["a"], args.threshold)
        w_c = crossing_width(deltas, curves[nr]["c"], args.threshold)
        results.append({
            "nr": nr, "L": L, "a": a, "c": c, "T": T, "k": k,
            "rpf_centre": float(curves[nr]["a"][i0]),
            "w_a": w_a, "w_c": w_c,
            "P_detect": detection_probability(w_a, w_c, args.n_grid),
        })

    analyse(results, n_grid=args.n_grid)

    with open(args.out, "w") as f:
        json.dump({
            "scan_settings": {"T_max": T_MAX, "n_samples": N_SAMPLES,
                              "t_min_frac": T_MIN_FRAC,
                              "threshold": args.threshold},
            "deltas": deltas.tolist(),
            "curves": {str(nr): {ax: curves[nr][ax].tolist()
                                 for ax in ("a", "c")} for nr in curves},
            "results": results,
        }, f, indent=1)
    print(f"Data saved: {args.out}")

    plot(results, deltas, curves, args.n_grid)


if __name__ == "__main__":
    main()
