"""Candidate processing pipeline: scan → extract → refine → classify → cross-reference.

Takes .npz scan files and produces a classified orbit catalogue as JSON.
"""

import json
import numpy as np

from three_body import (
    find_candidates,
    build_state_symmetric,
    initial_conditions_from_params,
    return_proximity,
    compute_energy,
    compute_angular_momentum,
    rescale_to_energy,
    integrate_orbit,
    read_free_group_word,
    SUVAKOV_TABLE1,
    SUVAKOV_TABLE2,
    ALL_ORBITS,
)
from floquet import (
    newton_refine_symmetric,
    newton_refine_bhh,
    analyse_orbit,
)


# ---------------------------------------------------------------------------
# Scan loading
# ---------------------------------------------------------------------------

def load_scan(path):
    """Load a scan .npz and return a normalised dict.

    Handles both scanner.py format (row_vals, col_vals) and legacy formats
    (vx_vals/vy_vals or a_vals/c_vals).

    Returns dict with keys: row_vals, col_vals, rpf_map.
    """
    data = np.load(path)
    keys = set(data.files)

    if "row_vals" in keys and "col_vals" in keys:
        # scanner.py stores rpf_map[row_idx, col_idx], but find_candidates
        # expects rpf_map[col_idx, row_idx] (the old scan_rpf convention
        # where first arg indexes columns and second indexes rows).
        # Transpose so find_candidates extracts correct (row, col) pairs.
        return {
            "row_vals": data["row_vals"],
            "col_vals": data["col_vals"],
            "rpf_map": data["rpf_map"].T,
        }
    elif "vx_vals" in keys and "vy_vals" in keys:
        return {
            "row_vals": data["vx_vals"],
            "col_vals": data["vy_vals"],
            "rpf_map": data["rpf_map"],
        }
    elif "a_vals" in keys and "c_vals" in keys:
        return {
            "row_vals": data["a_vals"],
            "col_vals": data["c_vals"],
            "rpf_map": data["rpf_map"],
        }
    else:
        raise ValueError(f"Unrecognised scan format. Keys: {keys}")


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_candidates(scan_data, threshold=3.5, min_separation=0.02):
    """Extract candidate peaks from a scan.

    Returns list of dicts: [{"params": (p1, p2), "rpf": val}, ...]
    """
    raw = find_candidates(
        scan_data["row_vals"], scan_data["col_vals"], scan_data["rpf_map"],
        threshold=threshold, min_separation=min_separation,
    )
    return [{"params": (a, c), "rpf": val} for a, c, val in raw]


# ---------------------------------------------------------------------------
# Period estimation
# ---------------------------------------------------------------------------

def estimate_period(params, parametrisation, L=None, T_max=16.0,
                    n_samples=2000):
    """Re-run return_proximity on a candidate to get T_guess.

    Returns (d_min, T_guess).
    """
    state0 = _build_state(params, parametrisation, L)
    d_min, T_guess, _ = return_proximity(state0, T_max, n_samples=n_samples)
    return d_min, T_guess


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def refine_candidate(params, T_guess, parametrisation, L=None, verbose=False,
                     n_samples=2000):
    """Newton-refine a candidate orbit.

    Returns dict with params_refined, T, converged, d_min, state0.
    """
    if parametrisation == "symmetric":
        vx_r, vy_r, T_r, converged, info = newton_refine_symmetric(
            params[0], params[1], T_guess, verbose=verbose)
        params_r = (vx_r, vy_r)
        state0 = build_state_symmetric(vx_r, vy_r)
    elif parametrisation == "bhh":
        a_r, c_r, T_r, converged, info = newton_refine_bhh(
            params[0], params[1], L, T_guess, verbose=verbose)
        params_r = (a_r, c_r)
        state0 = initial_conditions_from_params(a_r, c_r, L)
    else:
        raise ValueError(f"Unknown parametrisation: {parametrisation}")

    # Compute residual d_min at refined parameters
    d_min, _, _ = return_proximity(state0, T_r * 1.2, n_samples=n_samples)

    return {
        "params_refined": params_r,
        "T": T_r,
        "converged": converged,
        "d_min": d_min,
        "state0": state0,
        "monodromy": info.get("monodromy"),
    }


# ---------------------------------------------------------------------------
# Classification (Floquet + topology)
# ---------------------------------------------------------------------------

def classify_candidate(state0, T):
    """Run Floquet analysis and read free group word.

    Returns dict with stability info, multipliers, word, and validation.
    """
    result = analyse_orbit(state0, T, verbose=False)

    sol = integrate_orbit(state0, T)
    word = read_free_group_word(sol, T)

    return {
        "stability": result["stability"],
        "multipliers": result["multipliers"],
        "sol": sol,
        "word": word,
        "valid": result["valid"],
    }


# ---------------------------------------------------------------------------
# Orbit visualization
# ---------------------------------------------------------------------------

def save_orbit_plot(state0, T, sol, candidate_id, output_dir="orbit_plots",
                    word="", stable=False):
    """Save trajectory + shape sphere plot for a candidate orbit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from three_body import to_jacobi, to_shape_sphere
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True)
    n_points = min(5000, max(2000, int(T * 500)))
    t_eval = np.linspace(0, T, n_points)
    states = np.array([sol.sol(t) for t in t_eval])

    r1 = states[:, 0:2]
    r2 = states[:, 2:4]
    r3 = states[:, 4:6]

    shape_pts = []
    for st in states:
        r = st[:6].reshape(3, 2)
        v = st[6:].reshape(3, 2)
        rho, lam, _, _ = to_jacobi(r, v)
        shape_pts.append(to_shape_sphere(rho, lam))
    shape_pts = np.array(shape_pts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(r1[:, 0], r1[:, 1], "r-", lw=0.5, label="Body 1")
    ax.plot(r2[:, 0], r2[:, 1], "g-", lw=0.5, label="Body 2")
    ax.plot(r3[:, 0], r3[:, 1], "b-", lw=0.5, label="Body 3")
    ax.plot(*r1[0], "ro", ms=5)
    ax.plot(*r2[0], "go", ms=5)
    ax.plot(*r3[0], "bo", ms=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("Real space")

    ax = axes[1]
    sp_x = shape_pts[:, 0] / (1 + shape_pts[:, 2] + 1e-15)
    sp_y = shape_pts[:, 1] / (1 + shape_pts[:, 2] + 1e-15)
    ax.plot(sp_x, sp_y, "k-", lw=0.4)
    for angle in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        cx, cy = np.cos(angle), np.sin(angle)
        ax.plot(cx / 2, cy / 2, "r*", ms=10)
    ax.set_xlabel("X (stereo)")
    ax.set_ylabel("Y (stereo)")
    ax.set_aspect("equal")
    ax.set_title("Shape sphere (stereographic)")

    stab_str = "stable" if stable else "unstable"
    fig.suptitle(f"{candidate_id} | word: {word} | {stab_str} | T={T:.4f}",
                 fontsize=12)
    plt.tight_layout()

    path = str(Path(output_dir) / f"{candidate_id}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Free group word utilities
# ---------------------------------------------------------------------------

def canonical_word(word):
    """Canonical cyclic form of a free group word.

    Cyclic permutations of a word represent the same orbit (different
    starting point). Returns the lexicographically smallest rotation.
    """
    if not word or "?" in word:
        return word
    doubled = word + word
    n = len(word)
    rotations = [doubled[i:i + n] for i in range(n)]
    return min(rotations)


def detect_period_multiple(word_candidate, word_known):
    """Check if word_candidate is word_known repeated k times (up to cyclic perm).

    Returns k if word_candidate is a k-fold repeat of word_known, else None.
    """
    lc = len(word_candidate)
    lk = len(word_known)

    if lk == 0 or lc == 0 or lc % lk != 0:
        return None

    k = lc // lk
    canon_known = canonical_word(word_known)

    # Check if word_candidate is any cyclic permutation of word_known repeated k times
    for i in range(lk):
        rotated = word_known[i:] + word_known[:i]
        repeated = rotated * k
        if canonical_word(repeated) == canonical_word(word_candidate):
            return k

    return None


# ---------------------------------------------------------------------------
# Cross-reference against known orbits
# ---------------------------------------------------------------------------

# Cache: maps (source, index) -> free group word
# Persisted to _known_words.json so it survives across runs.
_known_words_cache = {}
_WORD_CACHE_PATH = "known_words_cache.json"


def _load_word_cache():
    """Load cached known orbit words from disk."""
    if _known_words_cache:
        return
    try:
        with open(_WORD_CACHE_PATH) as f:
            raw = json.load(f)
        # Mutate in-place (not reassign) so imported references stay valid
        for k, v in raw.items():
            parts = k.split("|")
            _known_words_cache[(parts[0], int(parts[1]))] = v
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def _save_word_cache():
    """Persist known orbit word cache to disk."""
    # Convert tuple keys to strings for JSON
    raw = {f"{src}|{idx}": word for (src, idx), word in _known_words_cache.items()}
    with open(_WORD_CACHE_PATH, "w") as f:
        json.dump(raw, f, indent=2)


def _compute_known_word(state0, T):
    """Integrate and read the free group word for a known orbit."""
    sol = integrate_orbit(state0, T)
    return read_free_group_word(sol, T)


def _get_known_symmetric_words(verbose=False):
    """Get canonical words for all known symmetric orbits (lazy, cached).

    Includes the figure-eight, Suvakov Table 1 (15 named), and
    Suvakov Table 2 (19 figure-eight satellites).
    """
    _load_word_cache()
    computed_new = False
    results = []

    # The figure-eight itself (not in Suvakov tables but is the baseline orbit)
    key = ("figure-eight", 0)
    if key not in _known_words_cache:
        if verbose:
            print("  Computing word for figure-eight...")
        state0 = build_state_symmetric(0.3471128135672417, 0.532726851767674)
        _known_words_cache[key] = _compute_known_word(state0, 6.3250)
        computed_new = True
    results.append(("figure-eight", _known_words_cache[key], 6.3250))

    # Suvakov Table 1: 15 named families
    for idx, (name, vx, vy, T) in enumerate(SUVAKOV_TABLE1):
        key = ("suvakov1", idx)
        if key not in _known_words_cache:
            if verbose:
                print(f"  Computing word for {name}...")
            state0 = build_state_symmetric(vx, vy)
            _known_words_cache[key] = _compute_known_word(state0, T)
            computed_new = True
        results.append((name, _known_words_cache[key], T))

    # Suvakov Table 2: 19 figure-eight satellites
    for idx, (name, vx, vy, T, k) in enumerate(SUVAKOV_TABLE2):
        key = ("suvakov2", idx)
        if key not in _known_words_cache:
            if verbose:
                print(f"  Computing word for {name}...")
            state0 = build_state_symmetric(vx, vy)
            _known_words_cache[key] = _compute_known_word(state0, T)
            computed_new = True
        results.append((f"satellite {name}", _known_words_cache[key], T))

    if computed_new:
        _save_word_cache()
    return results


def _get_known_bhh_words(L_target, verbose=False):
    """Get canonical words for Jankovic BHH orbits at a given L (lazy, cached)."""
    _load_word_cache()
    computed_new = False
    results = []
    for idx, (nr, L, a, c, T, k) in enumerate(ALL_ORBITS):
        if abs(L - L_target) > 0.01:
            continue
        key = ("bhh", idx)
        if key not in _known_words_cache:
            if verbose:
                print(f"  Computing word for orbit #{nr} (L={L})...")
            state0 = initial_conditions_from_params(a, c, L)
            _known_words_cache[key] = _compute_known_word(state0, T)
            computed_new = True
        results.append((f"Jankovic #{nr}", _known_words_cache[key], T))
    if computed_new:
        _save_word_cache()
    return results


def _get_known_ll_words(verbose=False):
    """Get canonical words for Li & Liao 695 orbits (lazy, cached).

    These are all symmetric L=0 orbits. Computing words for all 695
    is expensive (~1 hour), so results are persisted to the word cache.
    """
    from ll_data import load_ll_orbits

    _load_word_cache()
    computed_new = False
    results = []

    ll_orbits = load_ll_orbits()
    for idx, (name, v1, v2, T, T_star, Lf) in enumerate(ll_orbits):
        key = ("ll", idx)
        if key not in _known_words_cache:
            if verbose:
                print(f"  Computing word for LL {name}...")
            state0 = build_state_symmetric(v1, v2)
            _known_words_cache[key] = _compute_known_word(state0, T)
            computed_new = True
            # Save periodically (every 50 orbits) since this is slow
            if computed_new and idx % 50 == 0:
                _save_word_cache()
        results.append((f"LL {name}", _known_words_cache[key], T))

    if computed_new:
        _save_word_cache()
    return results


def cross_reference(word, T, parametrisation, L=None, verbose=False):
    """Match a candidate against known orbits by free group word.

    For symmetric orbits, checks against Suvakov, figure-eight satellites,
    AND Li & Liao 695 families.

    Returns dict with match info, or None if no match found.
    """
    if "?" in word:
        return None

    canon = canonical_word(word)

    if parametrisation == "symmetric":
        known = _get_known_symmetric_words(verbose=verbose)
        # Also check against Li & Liao 695 families
        known += _get_known_ll_words(verbose=verbose)
    elif parametrisation == "bhh":
        known = _get_known_bhh_words(L, verbose=verbose)
    else:
        return None

    for name, known_word, known_T in known:
        if "?" in known_word:
            continue

        canon_known = canonical_word(known_word)

        # Exact match
        if canon == canon_known:
            return {
                "matched_name": name,
                "k_multiple": 1,
                "source": parametrisation,
            }

        # Period multiple: candidate is k repeats of known
        k = detect_period_multiple(word, known_word)
        if k is not None and k > 1:
            # Confirm with period ratio
            ratio = T / known_T
            if abs(ratio - k) < 0.5:
                return {
                    "matched_name": name,
                    "k_multiple": k,
                    "source": parametrisation,
                }

        # Reverse: known is k repeats of candidate
        k_rev = detect_period_multiple(known_word, word)
        if k_rev is not None and k_rev > 1:
            ratio = known_T / T
            if abs(ratio - k_rev) < 0.5:
                return {
                    "matched_name": name,
                    "k_multiple": -k_rev,  # negative = candidate is shorter
                    "source": parametrisation,
                }

    return None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_scan(scan_path, parametrisation, L=None, threshold=3.5,
                 T_max=16.0, output_path=None, verbose=True):
    """Full pipeline: load → extract → refine → classify → cross-reference.

    Returns list of candidate dicts. Saves to JSON if output_path given.
    """
    if verbose:
        print(f"Loading scan from {scan_path}...")
    scan_data = load_scan(scan_path)
    n_rows, n_cols = scan_data["rpf_map"].shape
    if verbose:
        print(f"  Grid: {n_rows}x{n_cols}")

    if verbose:
        print(f"Extracting candidates (threshold={threshold})...")
    candidates = extract_candidates(scan_data, threshold=threshold)
    if verbose:
        print(f"  Found {len(candidates)} candidates")

    if len(candidates) == 0:
        if verbose:
            print("  No candidates found. Try lowering threshold.")
        return []

    results = []
    for idx, cand in enumerate(candidates):
        params = cand["params"]
        if verbose:
            print(f"\n--- Candidate {idx + 1}/{len(candidates)}: "
                  f"params=({params[0]:.6f}, {params[1]:.6f}), rpf={cand['rpf']:.2f} ---")

        # Estimate period
        try:
            d_min_raw, T_guess = estimate_period(
                params, parametrisation, L, T_max)
            if verbose:
                print(f"  Period estimate: T={T_guess:.4f}, d_min={d_min_raw:.2e}")
        except (RuntimeError, FloatingPointError) as e:
            if verbose:
                print(f"  Period estimation failed: {e}")
            continue

        # Refine
        try:
            ref = refine_candidate(params, T_guess, parametrisation, L,
                                   verbose=verbose)
            if verbose:
                pr = ref["params_refined"]
                print(f"  Refined: ({pr[0]:.10f}, {pr[1]:.10f}), "
                      f"T={ref['T']:.8f}, d_min={ref['d_min']:.2e}, "
                      f"converged={ref['converged']}")
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"  Refinement failed: {e}")
            continue

        if not ref["converged"]:
            if verbose:
                print("  Skipping (did not converge)")
            continue

        # Classify
        try:
            cls = classify_candidate(ref["state0"], ref["T"])
            if verbose:
                stab = cls["stability"]
                print(f"  Word: {cls['word']}, stable={stab['is_stable']}, "
                      f"det(M)={stab['determinant']:.6f}")
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"  Classification failed: {e}")
            continue

        # Cross-reference
        xref = cross_reference(cls["word"], ref["T"], parametrisation, L,
                               verbose=verbose)
        if verbose:
            if xref:
                k_str = f" (period x{xref['k_multiple']})" if xref["k_multiple"] != 1 else ""
                print(f"  Match: {xref['matched_name']}{k_str}")
            else:
                print(f"  No match in known catalogue — possible new orbit")

        # Deduplication: skip if refined params match an existing result
        # Check both raw parameter match AND energy-normalised match
        # (same orbit family at different energies)
        pr = ref["params_refined"]
        is_dup = False
        for prev in results:
            pp = prev["params_refined"]
            # Direct parameter match
            if (abs(pr[0] - pp[0]) < 1e-6 and abs(pr[1] - pp[1]) < 1e-6
                    and abs(ref["T"] - prev["T"]) < 1e-4):
                if verbose:
                    print(f"  Duplicate of {prev['id']} — skipping")
                is_dup = True
                break
            # Energy-normalised match: same family at different energy
            # Rescale both to E=-0.5, compare periods and words
            # Use canonical form so cyclic permutations are caught
            prev_word = prev.get("free_group_word", "")
            if (canonical_word(cls["word"]) == canonical_word(prev_word)
                    and cls["word"] != "?" and prev_word != "?"):
                try:
                    _, tf_new = rescale_to_energy(ref["state0"])
                    T_norm_new = ref["T"] * tf_new
                    state_prev = _build_state(prev["params_refined"], parametrisation, L)
                    _, tf_prev = rescale_to_energy(state_prev)
                    T_norm_prev = prev["T"] * tf_prev
                    if abs(T_norm_new - T_norm_prev) / max(T_norm_new, T_norm_prev) < 1e-3:
                        if verbose:
                            print(f"  Same family as {prev['id']} (energy-normalised) — skipping")
                        is_dup = True
                        break
                except (ValueError, ZeroDivisionError):
                    pass
        if is_dup:
            continue

        # Save orbit plot
        cand_id = f"candidate_{idx + 1:03d}"
        if output_path:
            plot_dir = output_path.replace(".json", "_plots")
            try:
                plot_path = save_orbit_plot(
                    ref["state0"], ref["T"], cls["sol"], cand_id,
                    output_dir=plot_dir, word=cls["word"],
                    stable=cls["stability"]["is_stable"])
                if verbose:
                    print(f"  Plot saved: {plot_path}")
            except Exception as e:
                if verbose:
                    print(f"  Plot failed: {e}")

        # Build result
        E = compute_energy(ref["state0"])
        entry = {
            "id": cand_id,
            "parametrisation": parametrisation,
            "params_raw": list(params),
            "params_refined": list(ref["params_refined"]),
            "L": L if L is not None else 0.0,
            "T": ref["T"],
            "d_min": ref["d_min"],
            "E": E,
            "converged": ref["converged"],
            "free_group_word": cls["word"],
            "is_stable": cls["stability"]["is_stable"],
            "max_instability": cls["stability"]["max_instability"],
            "determinant": cls["stability"]["determinant"],
            "n_unit_eigenvalues": cls["stability"]["n_unit"],
            "monodromy_valid": cls["valid"],
            "match": xref,
            "is_new": xref is None,
        }
        results.append(entry)

        # Incremental checkpoint
        if output_path and len(results) % 5 == 0:
            _save_results(results, output_path + ".partial")
            if verbose:
                print(f"  [Checkpoint: {len(results)} results saved]")

    if verbose:
        n_new = sum(1 for r in results if r["is_new"])
        n_matched = len(results) - n_new
        print(f"\n=== Summary ===")
        print(f"  Processed: {len(candidates)} candidates")
        print(f"  Refined:   {len(results)}")
        print(f"  Matched:   {n_matched}")
        print(f"  New:       {n_new}")

    if output_path:
        _save_results(results, output_path)
        # Clean up partial checkpoint
        partial = output_path + ".partial"
        import os
        if os.path.exists(partial):
            os.remove(partial)
        if verbose:
            print(f"  Saved to {output_path}")

    return results


def _save_results(results, path):
    """Save results list to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serialise)


def _json_serialise(obj):
    """Handle numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state(params, parametrisation, L=None):
    """Build state vector from parameters and parametrisation type."""
    if parametrisation == "symmetric":
        return build_state_symmetric(params[0], params[1])
    elif parametrisation == "bhh":
        return initial_conditions_from_params(params[0], params[1], L)
    else:
        raise ValueError(f"Unknown parametrisation: {parametrisation}")
