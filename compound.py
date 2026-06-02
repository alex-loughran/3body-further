"""2nd compound matrix method for Floquet stability analysis.

An alternative to the standard variational method in floquet.py.
Instead of integrating the 12×12 state transition matrix (STM),
integrates its 2nd exterior power — a 66×66 matrix whose eigenvalues
are products λ_i·λ_j of pairs of Floquet multipliers.

Implemented as a comparison/validation tool. See METHOD.md §14 for
the empirical comparison showing the standard method + segmentation
is faster and more accurate for this problem.

Usage:
    from compound import compare_floquet_methods
    result = compare_floquet_methods(state0, T_period, verbose=True)
"""

import time

import numpy as np
from scipy.integrate import solve_ivp

from three_body import _eom_core
from floquet import (
    _gravity_jacobian,
    compute_monodromy,
    floquet_multipliers,
    classify_stability,
)


# ---------------------------------------------------------------------------
# Precomputed index structure for the 2nd additive compound
# ---------------------------------------------------------------------------

# The compound acts on the space of 2-vectors: basis elements e_i ∧ e_j
# with 0 <= i < j < 12, giving C(12,2) = 66 basis elements.
#
# For J^[2]_{(p,q),(i,j)} with p<q, i<j:
#   diagonal (p,q)==(i,j):  J[i,i] + J[j,j]
#   q==j, p!=i:             J[p,i]
#   p==j:                  -J[q,i]
#   p==i, q!=j:             J[q,j]
#   q==i:                  -J[p,j]
#   else:                   0
#
# The sparsity pattern is fixed, so we precompute which entries are nonzero
# and where they read from in J. This turns the compound construction into
# two vectorized array operations per call (no Python loops in the hot path).

_N_PHASE = 12
_N_COMPOUND = 66  # C(12, 2)

# Build pair list and lookup
_PAIRS = []
_PAIR_IDX = {}
for _i in range(_N_PHASE):
    for _j in range(_i + 1, _N_PHASE):
        _PAIR_IDX[(_i, _j)] = len(_PAIRS)
        _PAIRS.append((_i, _j))

# Precompute diagonal entries: compound[k,k] = J[p,p] + J[q,q]
_DIAG_IDX = np.arange(_N_COMPOUND)
_DIAG_P = np.array([p for p, q in _PAIRS])
_DIAG_Q = np.array([q for p, q in _PAIRS])

# Precompute off-diagonal entries
_offdiag_rows = []
_offdiag_cols = []
_offdiag_jr = []
_offdiag_jc = []
_offdiag_signs = []

for _col, (_i, _j) in enumerate(_PAIRS):
    for _row, (_p, _q) in enumerate(_PAIRS):
        if _p == _i and _q == _j:
            continue  # diagonal, handled separately
        elif _q == _j and _p != _i:
            _offdiag_rows.append(_row)
            _offdiag_cols.append(_col)
            _offdiag_jr.append(_p)
            _offdiag_jc.append(_i)
            _offdiag_signs.append(1.0)
        elif _p == _j:
            _offdiag_rows.append(_row)
            _offdiag_cols.append(_col)
            _offdiag_jr.append(_q)
            _offdiag_jc.append(_i)
            _offdiag_signs.append(-1.0)
        elif _p == _i and _q != _j:
            _offdiag_rows.append(_row)
            _offdiag_cols.append(_col)
            _offdiag_jr.append(_q)
            _offdiag_jc.append(_j)
            _offdiag_signs.append(1.0)
        elif _q == _i:
            _offdiag_rows.append(_row)
            _offdiag_cols.append(_col)
            _offdiag_jr.append(_p)
            _offdiag_jc.append(_j)
            _offdiag_signs.append(-1.0)

_OFFDIAG_ROWS = np.array(_offdiag_rows)
_OFFDIAG_COLS = np.array(_offdiag_cols)
_OFFDIAG_JR = np.array(_offdiag_jr)
_OFFDIAG_JC = np.array(_offdiag_jc)
_OFFDIAG_SIGNS = np.array(_offdiag_signs)

# Clean up module namespace
del _offdiag_rows, _offdiag_cols, _offdiag_jr, _offdiag_jc, _offdiag_signs
del _i, _j, _col, _row, _p, _q


# ---------------------------------------------------------------------------
# Compound matrix construction and integration
# ---------------------------------------------------------------------------

def _additive_compound_2(J):
    """Compute the 2nd additive compound of a 12×12 matrix J.

    Returns the 66×66 matrix J^[2] such that if dΦ/dt = J·Φ, then
    dΦ^(2)/dt = J^[2]·Φ^(2), where Φ^(2) is the 2nd multiplicative
    compound (matrix of all 2×2 minors of Φ).
    """
    C = np.zeros((_N_COMPOUND, _N_COMPOUND))
    # Diagonal: C[k,k] = J[p_k, p_k] + J[q_k, q_k]
    C[_DIAG_IDX, _DIAG_IDX] = J[_DIAG_P, _DIAG_P] + J[_DIAG_Q, _DIAG_Q]
    # Off-diagonal: each entry maps to one element of J with a sign
    C[_OFFDIAG_ROWS, _OFFDIAG_COLS] = _OFFDIAG_SIGNS * J[_OFFDIAG_JR, _OFFDIAG_JC]
    return C


def _eom_with_compound(t, state_ext):
    """Equations of motion for orbit + 2nd compound variational system.

    state_ext = [orbit (12), C_compound flattened (66×66 = 4356)]
    Total: 4368 components.

    The orbit evolves under the standard three-body EOM. The compound
    matrix C evolves under dC/dt = J^[2](t) @ C, where J^[2] is the
    2nd additive compound of the 12×12 Jacobian.
    """
    orbit = state_ext[:12]
    C = state_ext[12:].reshape(_N_COMPOUND, _N_COMPOUND)

    orbit_dot = _eom_core(orbit)

    G = _gravity_jacobian(orbit[:6])
    J = np.zeros((12, 12))
    J[:6, 6:] = np.eye(6)
    J[6:, :6] = G

    J2 = _additive_compound_2(J)
    C_dot = J2 @ C

    deriv = np.empty(12 + _N_COMPOUND * _N_COMPOUND)
    deriv[:12] = orbit_dot
    deriv[12:] = C_dot.ravel()
    return deriv


def compute_compound_monodromy(state0, T_period, rtol=1e-12, atol=1e-14,
                                max_step=0.01):
    """Integrate the 2nd compound variational equations for one period.

    Instead of integrating the 12×12 STM (144 components), integrates the
    66×66 2nd compound STM (4356 components). The compound STM's eigenvalues
    are products λ_i·λ_j of pairs of Floquet multipliers.

    Parameters
    ----------
    state0 : (12,) array — initial conditions
    T_period : float — orbital period
    rtol, atol : float — integrator tolerances
    max_step : float — maximum step size

    Returns
    -------
    C_monodromy : (66, 66) array — 2nd compound of the monodromy matrix
    final_state : (12,) array — orbit state at t=T_period
    """
    n_ext = 12 + _N_COMPOUND * _N_COMPOUND
    state_ext0 = np.zeros(n_ext)
    state_ext0[:12] = state0
    state_ext0[12:] = np.eye(_N_COMPOUND).ravel()

    sol = solve_ivp(
        _eom_with_compound, [0, T_period], state_ext0,
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(
            f"Compound variational integration failed: {sol.message}")

    final = sol.sol(T_period)
    C_monodromy = final[12:].reshape(_N_COMPOUND, _N_COMPOUND)
    final_state = final[:12]
    return C_monodromy, final_state


# ---------------------------------------------------------------------------
# Comparison tool
# ---------------------------------------------------------------------------

def compare_floquet_methods(state0, T_period, verbose=True):
    """Run both standard and compound Floquet analysis and compare.

    Computes Floquet multipliers via the standard variational method and
    the 2nd compound matrix method, then compares:
    1. Pairwise products of standard multipliers vs compound eigenvalues
    2. Determinant checks for both methods
    3. Stability classification agreement
    4. Timing

    Returns
    -------
    dict with keys: standard_multipliers, compound_eigenvalues,
        standard_products (sorted), compound_sorted, max_relative_error,
        standard_time, compound_time, standard_det, compound_det
    """
    # --- Standard method ---
    t0 = time.time()
    M, final_std = compute_monodromy(state0, T_period)
    std_mults = floquet_multipliers(M)
    std_time = time.time() - t0

    std_det = np.prod(std_mults).real

    # Compute all C(12,2)=66 pairwise products from standard multipliers
    std_products = []
    for i in range(12):
        for j in range(i + 1, 12):
            std_products.append(std_mults[i] * std_mults[j])
    std_products = np.array(std_products)
    std_products_sorted = std_products[np.argsort(-np.abs(std_products))]

    # --- Compound method ---
    t0 = time.time()
    C_mono, final_cmp = compute_compound_monodromy(state0, T_period)
    cmp_time = time.time() - t0

    cmp_eigs = np.linalg.eigvals(C_mono)
    cmp_eigs_sorted = cmp_eigs[np.argsort(-np.abs(cmp_eigs))]
    cmp_det = np.prod(cmp_eigs).real

    # --- Compare ---
    # Match by magnitude-sorted order and compute relative errors
    errors = np.abs(np.abs(std_products_sorted) - np.abs(cmp_eigs_sorted))
    denom = np.maximum(np.abs(std_products_sorted), 1e-15)
    rel_errors = errors / denom
    max_rel_err = np.max(rel_errors)

    # Stability from compound: if all |μ| ≈ 1, orbit is stable
    std_stab = classify_stability(std_mults)
    cmp_max_mag = np.max(np.abs(cmp_eigs))

    if verbose:
        print("=" * 70)
        print("FLOQUET METHOD COMPARISON")
        print("=" * 70)

        print(f"\nTiming:")
        print(f"  Standard method:  {std_time:.2f}s  (156 components)")
        print(f"  Compound method:  {cmp_time:.2f}s  (4368 components)")
        print(f"  Slowdown factor:  {cmp_time / max(std_time, 1e-6):.1f}×")

        print(f"\nDeterminant checks:")
        print(f"  Standard det(M) = {std_det:.10f}  "
              f"(err = {abs(std_det - 1):.2e})")
        print(f"  Compound det(C) = {cmp_det:.10f}  "
              f"(should be det(M)^11 = {std_det**11:.6f}, "
              f"err = {abs(cmp_det - 1):.2e})")

        print(f"\nStandard Floquet multipliers:")
        for i, m in enumerate(std_mults):
            print(f"  λ_{i+1:2d} = {m.real:+.10f} {m.imag:+.10f}i   "
                  f"|λ| = {abs(m):.10f}")

        print(f"\nStability (standard): "
              f"{'STABLE' if std_stab['is_stable'] else 'UNSTABLE'}, "
              f"λ_max = {std_stab['max_instability']:.6f}")
        print(f"Stability (compound): max |μ| = {cmp_max_mag:.6f}  "
              f"(> 1 ⟹ unstable)")

        print(f"\nPairwise product comparison (top 20 by magnitude):")
        print(f"  {'#':>3}  {'|std product|':>14}  {'|cmp eigenval|':>14}  "
              f"{'rel error':>12}")
        print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*12}")
        for k in range(min(20, len(std_products_sorted))):
            print(f"  {k+1:3d}  {abs(std_products_sorted[k]):14.8f}  "
                  f"{abs(cmp_eigs_sorted[k]):14.8f}  "
                  f"{rel_errors[k]:12.2e}")

        print(f"\nMax relative error across all 66 products: {max_rel_err:.2e}")
        print("=" * 70)

    return {
        "standard_multipliers": std_mults,
        "compound_eigenvalues": cmp_eigs_sorted,
        "standard_products": std_products_sorted,
        "max_relative_error": max_rel_err,
        "standard_time": std_time,
        "compound_time": cmp_time,
        "standard_det": std_det,
        "compound_det": cmp_det,
        "standard_stability": std_stab,
        "compound_max_eigenvalue": cmp_max_mag,
    }
