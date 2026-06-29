"""Diagnose + fix the period-doubled branch search off the L=0.83 window.

period_double_branch.py validated its full-state rotation-reduced Newton (it
re-finds the base orbit) but never landed a period-2T orbit. Two hypotheses:

  (H1) SEED AMPLITUDE TOO SMALL.  At a period-doubling the new branch grows as
       amplitude ~ K*sqrt(L - L_PD).  At dL=1.5e-3 the true amplitude is
       ~sqrt(1.5e-3) ~ 0.04, but the old search only seeded eps <= 3e-3 -- an
       order of magnitude short, so Newton fell back into the (still-existing)
       base-orbit basin and got rejected by the half-period check.

  (H2) SEED DIRECTION CONTAMINATED.  The bare Cartesian -1 eigenvector v has
       components along the gauge directions (flow f, rotation g) and the dL
       direction.  Projecting v onto the complement of span{f, g, dL} gives the
       genuine transverse period-doubling direction.

This script: (1) confirms the -1 crossing is a real reduced-dynamics PD by
checking the cleaned eigenvector is transverse to the trivial directions, then
(2) does a WIDE amplitude scan with the cleaned seed at several L past L_PD.

Usage: python pd_branch_fix.py
"""

import json
import numpy as np

from three_body import (
    initial_conditions_from_params, integrate_orbit, to_Z_vector,
    compute_angular_momentum, compute_energy, _eom_core,
)
from floquet import compute_monodromy, newton_refine_bhh, floquet_multipliers
from period_double_branch import solve, project, rot_gen, dL_dx, lam_max, L_PD

SEED = {"a": 0.246486, "c": -2.035290}
T_BASE = 4.8779


def split_eigvec(M):
    """Return (lambda, real eigenvector) of the real negative multiplier with
    |lambda|>1 -- the period-doubling mode -- or (None, None) if not split."""
    ev, evec = np.linalg.eig(M)
    cand = [(abs(ev[i]), i) for i in range(12)
            if ev[i].real < 0 and abs(ev[i].imag) < 1e-3 and abs(ev[i]) > 1.001]
    if not cand:
        return None, None
    _, ki = max(cand)
    v = np.real(evec[:, ki])
    return ev[ki], v / np.linalg.norm(v)


def clean_direction(v, x0):
    """Project v onto the complement of span{flow f, rotation g, dL gradient}:
    the genuine transverse perturbation direction (also COM/momentum free)."""
    f = _eom_core(x0)
    g = rot_gen(x0)
    dl = dL_dx(x0)
    basis = [f, g, dl]
    w = project(v.copy())            # strip COM + total-momentum components
    for u in basis:
        u = project(u.copy())
        nu = u @ u
        if nu > 1e-30:
            w = w - (w @ u) / nu * u
    overlap = 1.0 - np.linalg.norm(w) / np.linalg.norm(project(v.copy()))
    return w / np.linalg.norm(w), float(overlap)


def main():
    out = {}
    print("=== refine base orbit at L_PD ===")
    a, c, T, ok, _ = newton_refine_bhh(SEED["a"], SEED["c"], L_PD, T_BASE,
                                       tol=1e-12)
    print(f"  base @ L_PD={L_PD}: a={a:.8f} c={c:.8f} T={T:.8f} ok={ok}")

    # H2 check at a representative L past L_PD: is the cleaned -1 mode transverse?
    L_chk = L_PD + 0.001
    a1, c1, T1, ok1, _ = newton_refine_bhh(a, c, L_chk, T, tol=1e-12)
    xb1 = initial_conditions_from_params(a1, c1, L_chk)
    M1, _ = compute_monodromy(xb1, T1)
    lam, v = split_eigvec(M1)
    print(f"\n=== H2: eigenvector cleanliness at L={L_chk:.6f} ===")
    if lam is None:
        print("  NOT split here -- unexpected; aborting.")
        json.dump({"error": "not split at L_chk"},
                  open("pd_branch_fix.json", "w"), indent=1)
        return
    v_clean, overlap = clean_direction(v, xb1)
    print(f"  split eigenvalue   = {lam.real:+.5f}")
    print(f"  gauge/dL overlap   = {overlap:.3f}  "
          f"({'mostly transverse' if overlap < 0.3 else 'contaminated'})")
    out["eigenvalue"] = float(lam.real)
    out["gauge_overlap"] = overlap

    # H1: wide amplitude scan with the CLEANED seed, at several L past L_PD.
    print("\n=== H1: wide-amplitude branch search (cleaned seed) ===")
    print(f"  {'L':>10} {'dL':>8} {'~sqrt(dL)':>10} {'eps':>8} "
          f"{'|F|':>10} {'P/2T':>7} {'half-close':>11} {'verdict':>10}")
    found = None
    scan_rows = []
    for dL in [0.0006, 0.001, 0.002, 0.004]:
        L_t = L_PD + dL
        a2, c2, T2, ok2, _ = newton_refine_bhh(a, c, L_t, T, tol=1e-12)
        if not ok2:
            print(f"  L={L_t:.6f}: base refine failed; skip")
            continue
        xb2 = initial_conditions_from_params(a2, c2, L_t)
        M2, _ = compute_monodromy(xb2, T2)
        lam2, v2 = split_eigvec(M2)
        if lam2 is None:
            print(f"  L={L_t:.6f}: not split; skip")
            continue
        v2c, _ = clean_direction(v2, xb2)
        amp = np.sqrt(dL)
        # seed amplitudes bracketing sqrt(dL), both signs
        for eps in [0.5 * amp, amp, 1.5 * amp, 2.0 * amp,
                    -0.5 * amp, -amp, -1.5 * amp, -2.0 * amp]:
            x0g = project(xb2 + eps * v2c)
            x0d, Pd, okd, rd = solve(x0g, 2.0 * T2, L_t)
            half = np.nan
            verdict = "diverge"
            if okd and rd < 1e-8:
                half = float(np.linalg.norm(
                    to_Z_vector(integrate_orbit(x0d, Pd / 2).sol(Pd / 2))
                    - to_Z_vector(x0d)))
                if half > 1e-3 and abs(Pd - 2 * T2) < 0.5 * T2:
                    verdict = "DOUBLED"
                elif half < 1e-4:
                    verdict = "->base"
                else:
                    verdict = "other"
            scan_rows.append({"L": L_t, "dL": dL, "eps": float(eps),
                              "F": float(rd), "P": float(Pd),
                              "half": half, "verdict": verdict})
            mark = "  <==" if verdict == "DOUBLED" else ""
            print(f"  {L_t:>10.6f} {dL:>8.4f} {amp:>10.4f} {eps:>8.4f} "
                  f"{rd:>10.2e} {Pd/(2*T2):>7.3f} {half:>11.2e} "
                  f"{verdict:>10}{mark}")
            if verdict == "DOUBLED" and found is None:
                found = {"L": L_t, "eps": float(eps), "x0": x0d.tolist(),
                         "P": float(Pd), "half_close": half,
                         "lambda_max": lam_max(x0d, Pd),
                         "E": float(compute_energy(x0d))}
        if found:
            break

    out["scan"] = scan_rows
    out["doubled"] = found
    if found:
        print(f"\nDOUBLED ORBIT FOUND: L={found['L']:.6f} eps={found['eps']:+.4f} "
              f"P={found['P']:.5f} lambda_max={found['lambda_max']:.3f}")
    else:
        print("\nNo doubled orbit in this scan. Inspect verdicts: all '->base' "
              "means seed still inside base basin (widen eps); 'diverge' means "
              "seed flew off (try finer eps / closer L).")
    json.dump(out, open("pd_branch_fix.json", "w"), indent=1)
    print("Saved: pd_branch_fix.json")


if __name__ == "__main__":
    main()
