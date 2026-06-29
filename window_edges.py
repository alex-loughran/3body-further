"""Map BOTH edges of the L=0.83 stable b^4 window and classify each bifurcation.

The window's UPPER edge is a Krein collision at L~0.83097 (period_double.py).
The LOWER edge has not been classified. This marches the base orbit down in L
from the stable point until stability is lost, records the multiplier nearest
the unit circle and how it exits, and reports the precise [L_lo, L_hi] window
plus the bifurcation type at each end:

  - real multiplier -> +1   : tangent / fold (saddle-node of orbits)
  - real multiplier -> -1   : period-doubling
  - complex pair leaving |z|=1 (angle != 0, pi) : Krein (Neimark-Sacker/torus)

This is the precise 1D characterisation of the window. Combined with the 2D
KAM-confinement result (shadowing.py), it answers "fragile 1D window vs robust
region": the periodic-orbit stable SET is the 1D arc found here; the bounded
QUASI-periodic neighbourhood around it is the 2D region shadowing.py measured.

Usage: python window_edges.py
Outputs: window_edges.json
"""

import json
import numpy as np

from period_double import _refine_analyse, STABLE


def exit_type(mults):
    """Classify how stability is first lost: inspect the multiplier(s) that
    have just left the unit circle (|z|>1.001)."""
    out = [z for z in mults if abs(z) > 1.001]
    if not out:
        return "stable", None
    z = max(out, key=abs)
    ang = abs(np.angle(z))
    if ang < 0.15:
        return "fold (+1)", float(ang)
    if ang > np.pi - 0.15:
        return "period-doubling (-1)", float(ang)
    return "Krein (complex pair)", float(ang)


def march(direction, dL, max_steps):
    """March from STABLE in L by sign(direction)*dL until stability is lost.
    Returns (rows, edge_row) where edge_row is the first unstable orbit."""
    a, c, T = STABLE["a"], STABLE["c"], STABLE["T"]
    L = STABLE["L"]
    rows, edge = [], None
    sgn = 1.0 if direction > 0 else -1.0
    for _ in range(max_steps):
        m = _refine_analyse(a, c, L, T)
        if m is None:
            print(f"  L={L:.6f}: refine failed (family may end here)")
            break
        a, c, T = m["a"], m["c"], m["T"]
        et, ang = exit_type(m["mults"])
        row = {k: m[k] for k in ("L", "a", "c", "T", "lambda_max",
                                 "n_unstable")}
        row["exit"] = et
        row["exit_angle"] = ang
        rows.append(row)
        tag = "" if m["n_unstable"] == 0 else f"  <-- UNSTABLE ({et})"
        print(f"  L={L:.6f} lam_max={m['lambda_max']:.5f} "
              f"n_unst={m['n_unstable']}{tag}")
        if m["n_unstable"] > 0:
            edge = row
            break
        L += sgn * dL
    return rows, edge


def refine_edge(lo_row, hi_row, dL_fine=2e-6, max_iter=40):
    """Bisect between a stable row and the first unstable row to pin the edge L
    to ~dL_fine. Returns (L_edge, exit_type)."""
    a, c, T = lo_row["a"], lo_row["c"], lo_row["T"]
    Llo, Lhi = lo_row["L"], hi_row["L"]
    et = hi_row["exit"]
    for _ in range(max_iter):
        if abs(Lhi - Llo) < dL_fine:
            break
        Lm = 0.5 * (Llo + Lhi)
        m = _refine_analyse(a, c, Lm, T)
        if m is None:
            break
        if m["n_unstable"] > 0:
            Lhi = Lm
            et, _ = exit_type(m["mults"])
        else:
            Llo, a, c, T = Lm, m["a"], m["c"], m["T"]
    return 0.5 * (Llo + Lhi), et


def main():
    out = {}
    print(f"=== marching UP in L from stable point L={STABLE['L']} ===")
    up_rows, up_edge = march(+1, 2e-5, 60)
    print(f"\n=== marching DOWN in L from stable point L={STABLE['L']} ===")
    dn_rows, dn_edge = march(-1, 2e-5, 60)

    summary = {}
    if up_edge is not None:
        stable_before = [r for r in up_rows if r["n_unstable"] == 0]
        L_up, et_up = refine_edge(stable_before[-1], up_edge)
        summary["upper"] = {"L": float(L_up), "type": et_up}
        print(f"\nUPPER edge: L={L_up:.7f}  ({et_up})")
    if dn_edge is not None:
        stable_before = [r for r in dn_rows if r["n_unstable"] == 0]
        L_dn, et_dn = refine_edge(stable_before[-1], dn_edge)
        summary["lower"] = {"L": float(L_dn), "type": et_dn}
        print(f"LOWER edge: L={L_dn:.7f}  ({et_dn})")

    if "lower" in summary and "upper" in summary:
        w = summary["upper"]["L"] - summary["lower"]["L"]
        summary["width"] = float(w)
        print(f"\nSTABLE WINDOW: L in [{summary['lower']['L']:.7f}, "
              f"{summary['upper']['L']:.7f}]  width={w:.2e}")

    out["up_march"] = up_rows
    out["down_march"] = dn_rows
    out["summary"] = summary
    json.dump(out, open("window_edges.json", "w"), indent=1)
    print("\nSaved: window_edges.json")


if __name__ == "__main__":
    main()
