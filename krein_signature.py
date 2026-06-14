"""Krein signatures of the multipliers colliding at the L=0.83097 window edge.

A Krein collision DESTABILISES (multipliers leave the unit circle) only when
the two colliding on-circle multipliers have OPPOSITE Krein signature; equal
signatures pass through harmlessly. This script computes the signatures
directly for the #2 b^3 family as L approaches the upper window edge, and
identifies the converging pair.

Krein signature of an on-circle multiplier lambda = e^{i theta} (theta != 0,pi)
with eigenvector v:  kappa = sign( i * v^H J v ),  which is real and nonzero,
where J = [[0, I6], [-I6, 0]] is the symplectic form in (positions,velocities)
coordinates (unit masses => velocity = momentum, so the monodromy is symplectic
in these coordinates).

Usage: python krein_signature.py
"""

import numpy as np

from three_body import initial_conditions_from_params
from floquet import newton_refine_bhh, analyse_orbit

STABLE = {"a": 0.246486, "c": -2.035290, "L": 0.830800, "T": 4.880107}

# Symplectic form in (q, v) coordinates (q = state[:6], v = state[6:]).
J = np.zeros((12, 12))
J[:6, 6:] = np.eye(6)
J[6:, :6] = -np.eye(6)


def krein_sign(v):
    """Krein signature of a unit-circle eigenvector v (complex)."""
    q = 1j * (np.conjugate(v) @ J @ v)
    return q.real  # real by construction; sign = signature, |.| = strength


def on_circle_modes(M, tol=1e-3):
    """Return upper-half-plane on-circle multipliers with angle + Krein sign."""
    evals, evecs = np.linalg.eig(M)
    modes = []
    for lam, v in zip(evals, evecs.T):
        if abs(abs(lam) - 1.0) > tol:
            continue
        theta = float(np.angle(lam))
        if theta <= 1e-6 or theta >= np.pi - 1e-6:
            continue  # skip trivial +1 pair and the exactly--1 case
        if theta < 0:
            continue  # keep only upper half; conjugate carries opposite sign
        k = krein_sign(v / np.linalg.norm(v))
        modes.append({"theta": theta, "kappa_val": float(k),
                      "kappa": int(np.sign(k)), "lam": lam})
    modes.sort(key=lambda m: m["theta"])
    return modes


def main():
    a, c, T = STABLE["a"], STABLE["c"], STABLE["T"]
    # March up to just below the collision (n_unstable 0 -> 2 at ~0.83097).
    L_vals = [0.83088, 0.83092, 0.83094, 0.83096]
    print("Symplecticity check (||M^T J M - J||) and on-circle Krein modes:\n")
    history = []
    for L in L_vals:
        a, c, T, ok, _ = newton_refine_bhh(a, c, L, T, tol=1e-11)
        res = analyse_orbit(initial_conditions_from_params(a, c, L), T,
                            verbose=False)
        M = res["monodromy"]
        symp_err = np.linalg.norm(M.T @ J @ M - J)
        modes = on_circle_modes(M)
        n_unstable = sum(1 for m in res["multipliers"] if abs(m) > 1.001)
        print(f"L={L:.5f}  ||M^TJM-J||={symp_err:.2e}  n_unstable={n_unstable}")
        for m in modes:
            print(f"    theta={m['theta']:.4f}  kappa={m['kappa']:+d}  "
                  f"(i v^H J v = {m['kappa_val']:+.4e})")
        history.append((L, modes))
        print()

    # Identify the converging pair: the two upper-half modes whose angle gap
    # shrinks fastest as L rises.
    L_last, modes_last = history[-1]
    print("=== Converging pair at the last L below the collision ===")
    if len(modes_last) >= 2:
        gaps = [(modes_last[i + 1]["theta"] - modes_last[i]["theta"], i)
                for i in range(len(modes_last) - 1)]
        gap, i = min(gaps)
        m1, m2 = modes_last[i], modes_last[i + 1]
        print(f"  closest pair: theta={m1['theta']:.4f} (kappa={m1['kappa']:+d})"
              f"  and  theta={m2['theta']:.4f} (kappa={m2['kappa']:+d})"
              f"  gap={gap:.4f}")
        verdict = ("OPPOSITE -> destabilising Krein collision (multipliers "
                   "leave the circle)" if m1["kappa"] != m2["kappa"]
                   else "EQUAL -> would pass through harmlessly")
        print(f"  signatures are {verdict}")


if __name__ == "__main__":
    main()
