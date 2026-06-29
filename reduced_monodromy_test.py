"""Decisive test: is the L_PD "-1 multiplier crossing" a real period-doubling
or an artifact of the per-period rotation of an L!=0 relative periodic orbit?

A BHH orbit with L!=0 is a RELATIVE periodic orbit: x(T) = R(theta) x(0), it
closes only up to a rigid rotation by theta. floquet.compute_monodromy returns
the RAW M = dx(T)/dx(0) and floquet_multipliers eigen-decomposes it directly --
no rotation reduction. The physically correct Floquet multipliers are those of
the REDUCED monodromy M_r = R(-theta) M, which undoes the bulk rotation in the
tangent space.

If theta ~ pi near L_PD, a trivial (+1) symmetry direction of M_r appears at
e^{i theta} ~ -1 in the raw M -- a FAKE period-doubling. This script computes
theta and compares spec(M) vs spec(M_r) at L just past L_PD.

VERDICT logic:
  - raw M has an eigenvalue near -1 AND reduced M_r does NOT  -> ARTIFACT
  - both have it                                              -> real PD
"""

import numpy as np
from three_body import (
    initial_conditions_from_params, integrate_orbit, to_Z_vector,
)
from floquet import compute_monodromy, newton_refine_bhh

SEED = {"a": 0.246486, "c": -2.035290}
L_PD = 0.831064
T_BASE = 4.8779


def rot_block(theta):
    """12x12 phase-space rotation: rotate each (x,y) and (vx,vy) pair by theta."""
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]])
    R = np.zeros((12, 12))
    for i in range(6):
        R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R2
    return R


def per_period_rotation(x0, xT):
    """theta s.t. xT ~ R(theta) x0, from the complex (x+iy) ratio of positions,
    averaged over the three bodies (weighted by radius for robustness)."""
    angs, wts = [], []
    for i in range(3):
        z0 = x0[2 * i] + 1j * x0[2 * i + 1]
        zT = xT[2 * i] + 1j * xT[2 * i + 1]
        if abs(z0) > 1e-6:
            angs.append(np.angle(zT / z0))
            wts.append(abs(z0))
    # circular weighted mean
    angs = np.array(angs); wts = np.array(wts)
    m = np.angle(np.sum(wts * np.exp(1j * angs)) / np.sum(wts))
    return m, angs


def near_minus1(eigs, tol=0.06):
    """eigenvalues within tol of -1 (real, |Im| small)."""
    return [e for e in eigs if abs(e.real + 1.0) < tol and abs(e.imag) < tol]


def analyse(a, c, L, T):
    a, c, T, ok, _ = newton_refine_bhh(a, c, L, T, tol=1e-12)
    if not ok:
        return None
    x0 = initial_conditions_from_params(a, c, L)
    M, xT = compute_monodromy(x0, T)
    theta, per_body = per_period_rotation(x0, xT)
    # confirm it really is a pure rotation: residual after undoing it
    R = rot_block(theta)
    rot_resid = np.linalg.norm(R @ x0 - xT) / np.linalg.norm(xT)
    Mr = rot_block(-theta) @ M
    eig_raw = np.linalg.eigvals(M)
    eig_red = np.linalg.eigvals(Mr)
    return {
        "a": a, "c": c, "L": L, "T": T, "theta": theta,
        "theta_over_pi": theta / np.pi, "per_body": per_body.tolist(),
        "rot_resid": float(rot_resid),
        "eig_raw": eig_raw, "eig_red": eig_red,
        "raw_m1": near_minus1(eig_raw), "red_m1": near_minus1(eig_red),
    }


def show(tag, r):
    print(f"\n--- {tag}: L={r['L']:.6f} ---")
    print(f"  per-period rotation theta = {r['theta']:+.5f} rad "
          f"= {r['theta_over_pi']:+.4f} * pi")
    print(f"  (rotation check: ||R x0 - xT||/||xT|| = {r['rot_resid']:.2e})")
    print(f"  RAW monodromy eigenvalues near -1 : "
          f"{[f'{e.real:+.4f}{e.imag:+.4f}j' for e in r['raw_m1']] or 'NONE'}")
    print(f"  REDUCED (R(-theta)M) near -1      : "
          f"{[f'{e.real:+.4f}{e.imag:+.4f}j' for e in r['red_m1']] or 'NONE'}")


def main():
    a, c, T = SEED["a"], SEED["c"], T_BASE
    # refine base at L_PD then probe just above (where raw M shows the split)
    r0 = analyse(a, c, L_PD, T)
    show("at L_PD", r0)
    a, c, T = r0["a"], r0["c"], r0["T"]
    results = [r0]
    for L in [L_PD + 0.0004, L_PD + 0.001, L_PD + 0.002]:
        r = analyse(a, c, L, T)
        if r is None:
            print(f"\n  L={L:.6f}: refine failed")
            continue
        a, c, T = r["a"], r["c"], r["T"]
        show(f"L_PD+{L-L_PD:.4f}", r)
        results.append(r)

    raw_has = any(r["raw_m1"] for r in results)
    red_has = any(r["red_m1"] for r in results)
    print("\n" + "=" * 64)
    if raw_has and not red_has:
        print("VERDICT: ARTIFACT. The -1 crossing exists ONLY in the raw "
              "monodromy, not in the rotation-reduced monodromy. The per-period "
              f"rotation theta ~ {r0['theta_over_pi']:.3f}*pi maps a trivial "
              "symmetry direction to -1. There is NO genuine period-doubling "
              "bifurcation and NO 2T orbit -- consistent with all Newton seeds "
              "diverging.")
    elif raw_has and red_has:
        print("VERDICT: REAL period-doubling -- the -1 survives rotation "
              "reduction. The branch search must be failing for another reason "
              "(seeding / continuation), not because the bifurcation is fake.")
    else:
        print("VERDICT: inconclusive -- no clean -1 in raw M at these L; "
              "re-bracket L_PD.")
    print("=" * 64)

    # also report the FULL reduced spectrum at the last good L for the record
    last = results[-1]
    mags_raw = sorted(abs(last["eig_raw"]), reverse=True)
    mags_red = sorted(abs(last["eig_red"]), reverse=True)
    print(f"\n|lambda| spectrum at L={last['L']:.6f}:")
    print("  raw:    " + "  ".join(f"{m:.4f}" for m in mags_raw))
    print("  reduced:" + "  ".join(f"{m:.4f}" for m in mags_red))


if __name__ == "__main__":
    main()
