"""Geometric analysis of the pure-b^k topological bias in BHH scans.

Every periodic orbit found in BHH (a, c) scans carries a pure single-letter
free group word (b^k, or a^k for the five Jankovic L=0.65 orbits with c>0),
while the symmetric (vx, vy) plane yields mixed words freely. This script
investigates why, in three parts:

Letter geometry. Tracing the word-reading tables: letter b = one winding
around the 1-2 collision puncture on the shape sphere (syzygy middle-body
sequence ...2,1,2...), letter a = one winding around the 2-3 puncture
(...2,3,2...). Mixed words require the trajectory to alternate punctures.

Initial-condition geometry (analytic). BHH states start ON the shape-sphere
equator: rho0=(a,0), lam0=(b,0) is collinear, with body 2 in the middle.
The equatorial longitude runs from the 1-2 puncture (a=0) to the 2-3
puncture (a=sqrt(3)b). The parameters split the angular momentum between
the Jacobi oscillators: L_rho = a*c (binary winding), L_lam = L - a*c
(outer winding). The first off-equator excursion enters the hemisphere
given by sign(zdot(0)) = sign(a*L - c*(1 + a^2)) (for b=1).

Syzygy census (numerical). Integrates every grid point of the (a, c) plane
at the campaign scan settings and records which puncture transitions occur
(1-2, 2-3, 1-3) — for ALL trajectories, not just periodic ones. If mixed
transitions never occur, the bias is kinematic (the slice cannot reach
mixed topologies); if they occur but no periodic orbits live there, the
bias is dynamical selection. The symmetric plane is run as a control.

Usage:
    python bk_bias.py analytic              Print + plot the IC geometry
    python bk_bias.py census [--n 100]      Run the syzygy census (slow)
    python bk_bias.py report                Summarise a saved census

Outputs: bk_bias_geometry.png, bk_bias_census.npz, bk_bias_census.png
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from three_body import (
    ALL_ORBITS,
    integrate_orbit,
    to_jacobi,
)
from parametrisations import SymmetricBuilder, BHHBuilder

# Campaign scan settings — the census sees what the scanner saw
T_MAX = 16.0
A_RANGE = (0.05, 0.6)
C_RANGE = (-3.5, 4.0)
VX_RANGE = (0.01, 0.6)
VY_RANGE = (0.01, 0.6)

# Transition bitmask values: which equator arcs the trajectory connects.
# Arc label = middle body. b-winding alternates arcs 2,1; a-winding 2,3.
T_12, T_23, T_13 = 1, 2, 4
SKIPPED = -1

CLASS_NAMES = {
    0: "no transitions",
    T_12: "pure 1-2 (b-type)",
    T_23: "pure 2-3 (a-type)",
    T_13: "pure 1-3",
    T_12 | T_23: "mixed 12+23",
    T_12 | T_13: "mixed 12+13",
    T_23 | T_13: "mixed 23+13",
    T_12 | T_23 | T_13: "mixed all",
    SKIPPED: "skipped (E>=0 / failed)",
}


# ---------------------------------------------------------------------------
# Analytic IC geometry
# ---------------------------------------------------------------------------

def equator_longitude(a, b=1.0):
    """Shape-sphere longitude (degrees) of the BHH initial condition.

    Collision punctures sit at: 1-2 at 90 deg (a=0), 2-3 at -30 deg
    (a=sqrt(3)b), 1-3 at 210 deg (a=-sqrt(3)b).
    """
    R2 = a**2 + b**2
    x = 2 * a * b / R2
    y = (b**2 - a**2) / R2
    return np.degrees(np.arctan2(y, x))


def zdot0_sign_boundary(a, L, b=1.0):
    """The c value where zdot(0) = 0: c = a*L / (b^2 + a^2) (per unit b).

    zdot(0) is proportional to a*L - c*(b^2 + a^2); above this curve the
    trajectory first dips into the southern hemisphere, below it the
    northern.
    """
    return a * L / (b**2 + a**2)


def analytic_report(L_show=0.8, path="bk_bias_geometry.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== BHH initial-condition geometry ===\n")
    print("Equatorial longitude of ICs (punctures: 1-2 at 90deg, "
          "2-3 at -30deg):")
    for a in (0.01, 0.05, 0.2, 0.6, 1.0, 1.7):
        print(f"  a={a:>5.2f}  ->  {equator_longitude(a):>6.1f} deg")
    print(f"\nScan range a in [{A_RANGE[0]}, {A_RANGE[1]}] covers "
          f"{equator_longitude(A_RANGE[1]):.0f}..{equator_longitude(A_RANGE[0]):.0f} deg "
          f"of the 120 deg arc — the third nearest the 1-2 puncture.")

    print("\nAngular momentum split: L_rho = a*c (binary), "
          "L_lam = L - a*c (outer).")
    print("Jankovic orbits: c<0 -> L_rho in [-0.65,-0.22], pure b (70/75);")
    print("                 c>0 -> L_lam ~ 0,  pure a  (5/75, all L=0.65).")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    a_vals = np.linspace(0.001, np.sqrt(3) - 0.001, 400)
    ax.plot(a_vals, [equator_longitude(a) for a in a_vals], "k-")
    ax.axhline(90, color="b", ls="--", lw=1)
    ax.axhline(-30, color="r", ls="--", lw=1)
    ax.text(1.0, 92, "1-2 collision puncture (b-winding)", color="b", fontsize=9)
    ax.text(0.05, -27, "2-3 collision puncture (a-winding)", color="r", fontsize=9)
    ax.axvspan(*A_RANGE, alpha=0.15, color="g", label="scan range")
    ax.set_xlabel("a")
    ax.set_ylabel("initial equatorial longitude (deg)")
    ax.set_title("BHH ICs live on the equator arc between\nthe 1-2 and 2-3 punctures")
    ax.legend()

    ax = axes[1]
    a_grid = np.linspace(*A_RANGE, 200)
    ax.plot(a_grid, zdot0_sign_boundary(a_grid, L_show), "k-",
            label=f"zdot(0)=0 (L={L_show})")
    for L_rho in (-0.3, 0.3):
        ax.plot(a_grid, L_rho / a_grid, ls="--", lw=1,
                label=f"L_rho = a*c = {L_rho}")
    ax.plot(a_grid, L_show / a_grid, ls=":", lw=1.5, color="purple",
            label=f"L_lam = 0 (a*c = L = {L_show})")
    for nr, L, a, c, T, k in ALL_ORBITS:
        color = "r" if c > 0 else "b"
        ax.plot(a, c, "o", ms=3, color=color)
    ax.set_xlim(*A_RANGE)
    ax.set_ylim(*C_RANGE)
    ax.set_xlabel("a")
    ax.set_ylabel("c")
    ax.set_title("(a, c) plane: angular momentum split.\n"
                 "blue = Jankovic pure-b, red = pure-a")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"\nPlot saved: {path}")


# ---------------------------------------------------------------------------
# Syzygy census
# ---------------------------------------------------------------------------

def syzygy_transitions(state0, T_max=T_MAX, n_samples=12000):
    """Integrate a trajectory and return (bitmask, n_syzygies).

    Bitmask records which arc transitions occur (T_12 | T_23 | T_13),
    where an arc is labelled by the body in the middle at each equator
    crossing. Consecutive duplicate arcs (grazing) are filtered, matching
    read_free_group_word.
    """
    sol = integrate_orbit(state0, T_max)
    t_eval = np.linspace(0, min(T_max, sol.t[-1]), n_samples)
    states = sol.sol(t_eval)  # (12, n)

    r = states[:6].reshape(3, 2, -1)
    rho = (r[0] - r[1]) / np.sqrt(2)
    lam = (r[0] + r[1] - 2 * r[2]) / np.sqrt(6)
    z_num = rho[0] * lam[1] - rho[1] * lam[0]

    crossings = np.nonzero(z_num[:-1] * z_num[1:] < 0)[0]
    if len(crossings) == 0:
        return 0, 0

    # Middle body at each crossing: project onto the longest pair axis
    seq = []
    for i in crossings:
        # Use whichever bracketing sample is closer to the equator
        j = i if abs(z_num[i]) < abs(z_num[i + 1]) else i + 1
        pos = states[:6, j].reshape(3, 2)
        d01 = np.sum((pos[0] - pos[1]) ** 2)
        d02 = np.sum((pos[0] - pos[2]) ** 2)
        d12 = np.sum((pos[1] - pos[2]) ** 2)
        if d01 >= d02 and d01 >= d12:
            direction = pos[1] - pos[0]
        elif d02 >= d12:
            direction = pos[2] - pos[0]
        else:
            direction = pos[2] - pos[1]
        projs = pos @ direction
        middle = int(np.argsort(projs)[1]) + 1
        if not seq or middle != seq[-1]:
            seq.append(middle)

    mask = 0
    for s1, s2 in zip(seq, seq[1:]):
        pair = {s1, s2}
        if pair == {1, 2}:
            mask |= T_12
        elif pair == {2, 3}:
            mask |= T_23
        elif pair == {1, 3}:
            mask |= T_13
    return mask, len(seq)


def _census_point(args):
    i, j, p1, p2, builder = args
    state0 = builder((p1, p2))
    if state0 is None:
        return i, j, SKIPPED, 0
    try:
        mask, n_syz = syzygy_transitions(state0)
        return i, j, mask, n_syz
    except (RuntimeError, FloatingPointError):
        return i, j, SKIPPED, 0


def run_census(planes, n_grid=100, n_workers=None, save_path="bk_bias_census.npz"):
    """Census every plane in `planes`: list of (label, builder, row_range,
    col_range). Saves all classification maps to one npz."""
    if n_workers is None:
        n_workers = mp.cpu_count()

    out = {}
    for label, builder, row_range, col_range in planes:
        rows = np.linspace(*row_range, n_grid)
        cols = np.linspace(*col_range, n_grid)
        tasks = [(i, j, rv, cv, builder)
                 for i, rv in enumerate(rows) for j, cv in enumerate(cols)]
        class_map = np.full((n_grid, n_grid), SKIPPED, dtype=np.int8)
        syz_map = np.zeros((n_grid, n_grid), dtype=np.int16)

        print(f"\n=== Census: {label} ({n_grid}x{n_grid}, "
              f"{n_workers} workers) ===")
        t0 = time.time()
        done = 0
        with mp.Pool(n_workers) as pool:
            for i, j, mask, n_syz in pool.imap_unordered(
                    _census_point, tasks, chunksize=16):
                class_map[i, j] = mask
                syz_map[i, j] = n_syz
                done += 1
                if done % max(1, len(tasks) // 20) == 0:
                    el = time.time() - t0
                    print(f"\r  {done}/{len(tasks)} "
                          f"[{el:.0f}s, ~{el * (len(tasks) - done) / done:.0f}s left]",
                          end="", flush=True)
        print(f"\r  done in {time.time() - t0:.0f}s" + " " * 30)

        out[f"{label}_class"] = class_map
        out[f"{label}_syz"] = syz_map
        out[f"{label}_rows"] = rows
        out[f"{label}_cols"] = cols
        _print_fractions(label, class_map)
        np.savez(save_path, **out)

    print(f"\nSaved: {save_path}")
    return out


def _print_fractions(label, class_map):
    total = class_map.size
    valid = int(np.sum(class_map >= 0))
    print(f"  {label}: {valid}/{total} integrated")
    for val in sorted(CLASS_NAMES):
        n = int(np.sum(class_map == val))
        if n and val != SKIPPED:
            print(f"    {CLASS_NAMES[val]:<24} {n:>7} ({100 * n / valid:.1f}% of valid)")


def census_plot(npz_path="bk_bias_census.npz", path="bk_bias_census.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    data = np.load(npz_path)
    labels = sorted({k.rsplit("_", 1)[0] for k in data.files})

    # Class colour scheme: skip grey, none white, pure-12 blue, pure-23 red,
    # pure-13 orange, mixed shades of green/purple
    colors = {SKIPPED: "#cccccc", 0: "#ffffff", T_12: "#2060c0",
              T_23: "#c03030", T_13: "#e08020", T_12 | T_23: "#20a050",
              T_12 | T_13: "#8040c0", T_23 | T_13: "#a0a020",
              T_12 | T_23 | T_13: "#103010"}
    bounds = sorted(colors)
    cmap = ListedColormap([colors[b] for b in bounds])
    norm = BoundaryNorm([b - 0.5 for b in bounds] + [bounds[-1] + 0.5], cmap.N)

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        rows = data[f"{label}_rows"]
        cols = data[f"{label}_cols"]
        cm = data[f"{label}_class"]
        ax.pcolormesh(rows, cols, cm.T, cmap=cmap, norm=norm, shading="auto")
        if label.startswith("bhh"):
            L_val = float(label.split("L")[1])
            sel = [(a, c) for nr, L, a, c, T, k in ALL_ORBITS
                   if abs(L - L_val) < 0.01]
            if sel:
                ax.plot(*zip(*sel), "k*", ms=9, mew=0.5, mec="w",
                        label="Jankovic orbits")
                ax.legend(fontsize=8)
            ax.set_xlabel("a")
            ax.set_ylabel("c")
        else:
            ax.set_xlabel("vx")
            ax.set_ylabel("vy")
        ax.set_title(label)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[v])
               for v in bounds if v != SKIPPED]
    names = [CLASS_NAMES[v] for v in bounds if v != SKIPPED]
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=8)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(path, dpi=130)
    print(f"Plot saved: {path}")


def report(npz_path="bk_bias_census.npz"):
    data = np.load(npz_path)
    labels = sorted({k.rsplit("_", 1)[0] for k in data.files})
    for label in labels:
        _print_fractions(label, data[f"{label}_class"])
    census_plot(npz_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analytic")
    p_an.set_defaults(func=lambda args: analytic_report())

    p_cen = sub.add_parser("census")
    p_cen.add_argument("--n", type=int, default=100)
    p_cen.add_argument("--workers", type=int, default=None)
    p_cen.add_argument("--L", type=float, nargs="+",
                       default=[0.65, 0.8, 1.0])

    def _run(args):
        planes = [(f"bhh_L{L}", BHHBuilder(L=L), A_RANGE, C_RANGE)
                  for L in args.L]
        planes.append(("symmetric", SymmetricBuilder(), VX_RANGE, VY_RANGE))
        run_census(planes, n_grid=args.n, n_workers=args.workers)
        census_plot()
    p_cen.set_defaults(func=_run)

    p_rep = sub.add_parser("report")
    p_rep.set_defaults(func=lambda args: report())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
