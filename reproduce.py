"""Reproduction suite: end-to-end validation against known orbits.

Runs 17 published orbits (Suvakov symmetric + Jankovic BHH, spanning
L = 0 to 1.07 and word lengths 3 to 22) through the full pipeline:

    published params → Newton refine → RPF residual → Floquet → word
    → cross-reference

and checks each stage against expectations. Use as a pre-scan sanity
check — if this fails, scan results can't be trusted.

Usage:
    python reproduce.py             Full suite (~17 orbits, parallel)
    python reproduce.py --quick     5-orbit subset
    python reproduce.py --workers N

Exit code 0 = all pass, 1 = failures.
"""

import argparse
import multiprocessing as mp
import sys
import time

from three_body import (
    build_state_symmetric,
    initial_conditions_from_params,
    return_proximity,
    integrate_orbit,
    read_free_group_word,
    SUVAKOV_TABLE1,
    SUVAKOV_TABLE2,
    ALL_ORBITS,
)

# Refined params must stay near published values — a large jump means
# Newton wandered to a different orbit.
PARAM_DRIFT_TOL = 0.01
D_MIN_TOL = 1e-4
DET_TOL = 1e-3


def _suvakov(name, accept=None):
    for n, vx, vy, T in SUVAKOV_TABLE1:
        if n == name:
            return {"name": n, "parametrisation": "symmetric",
                    "params": (vx, vy), "L": None, "T": T,
                    "accept": accept or {n}, "k": None}
    raise KeyError(name)


def _satellite(name, accept):
    for n, vx, vy, T, k in SUVAKOV_TABLE2:
        if n == name:
            return {"name": f"satellite {n}", "parametrisation": "symmetric",
                    "params": (vx, vy), "L": None, "T": T,
                    "accept": accept, "k": None}
    raise KeyError(name)


def _jankovic(nr, accept=None):
    for n, L, a, c, T, k in ALL_ORBITS:
        if n == nr:
            return {"name": f"Jankovic #{nr}", "parametrisation": "bhh",
                    "params": (a, c), "L": L, "T": T,
                    "accept": accept or {f"Jankovic #{nr}"}, "k": k}
    raise KeyError(nr)


def build_cases():
    """The reference set. Periods kept under ~25 so the suite stays fast.

    Where two catalogue orbits at the same L share a free group word
    (cross-reference matches by word alone), both names are acceptable.
    """
    fig8 = {"name": "figure-eight", "parametrisation": "symmetric",
            "params": (0.3471128135672417, 0.532726851767674), "L": None,
            "T": 6.3250, "accept": {"figure-eight"}, "k": None}
    return [
        fig8,
        _suvakov("I.A.1 butterfly I"),
        # butterfly II shares butterfly I's word (different T, same topology)
        _suvakov("I.A.2 butterfly II",
                 accept={"I.A.1 butterfly I", "I.A.2 butterfly II"}),
        _suvakov("I.B.1 moth I"),
        _suvakov("I.B.5 goggles"),
        _suvakov("I.B.7 dragonfly"),
        _suvakov("II.C.2b yin-yang I",
                 accept={"II.C.2a yin-yang I", "II.C.2b yin-yang I"}),
        # S8 shares the figure-eight's word — topology can't separate them
        _satellite("S8", accept={"figure-eight", "satellite M8",
                                 "satellite S8"}),
        _jankovic(1),
        _jankovic(3),
        _jankovic(5),
        _jankovic(10),
        _jankovic(18, accept={"Jankovic #18", "Jankovic #22"}),
        _jankovic(28, accept={"Jankovic #28", "Jankovic #29"}),
        _jankovic(34),
        _jankovic(51),
        _jankovic(54),
    ]


QUICK_NAMES = {"figure-eight", "I.A.1 butterfly I", "Jankovic #1",
               "Jankovic #10", "Jankovic #34"}


def run_case(case):
    """Refine + classify one known orbit. Returns a result dict with a list
    of failure strings (empty = pass). Runs in a worker process."""
    from floquet import newton_refine_symmetric, newton_refine_bhh, analyse_orbit

    failures = []
    out = {"case": case, "failures": failures, "word": "?", "T_refined": None}
    p1, p2 = case["params"]

    try:
        if case["parametrisation"] == "symmetric":
            p1_r, p2_r, T_r, converged, info = newton_refine_symmetric(
                p1, p2, case["T"], max_iter=30)
            state0 = build_state_symmetric(p1_r, p2_r)
        else:
            p1_r, p2_r, T_r, converged, info = newton_refine_bhh(
                p1, p2, case["L"], case["T"], max_iter=30)
            state0 = initial_conditions_from_params(p1_r, p2_r, case["L"])
    except Exception as e:
        failures.append(f"Newton refinement raised: {e}")
        return out

    out["T_refined"] = T_r
    if not converged:
        failures.append("Newton did not converge")
        return out

    drift = max(abs(p1_r - p1), abs(p2_r - p2))
    if drift > PARAM_DRIFT_TOL:
        failures.append(f"refined params drifted {drift:.2g} "
                        f"(> {PARAM_DRIFT_TOL}) from published values")

    try:
        d_min, _, _ = return_proximity(state0, T_r * 1.2, n_samples=2000)
        out["d_min"] = d_min
        if d_min > D_MIN_TOL:
            failures.append(f"d_min = {d_min:.2e} (> {D_MIN_TOL})")
    except Exception as e:
        failures.append(f"return_proximity raised: {e}")
        return out

    try:
        result = analyse_orbit(state0, T_r, verbose=False)
        det = result["stability"]["determinant"]
        out["is_stable"] = result["stability"]["is_stable"]
        if abs(det - 1.0) > DET_TOL:
            failures.append(f"det(M) = {det:.6f} (|det-1| > {DET_TOL})")
        if not result["valid"]:
            failures.append("monodromy validation failed")
    except Exception as e:
        failures.append(f"Floquet analysis raised: {e}")
        return out

    try:
        sol = integrate_orbit(state0, T_r)
        word = read_free_group_word(sol, T_r)
        out["word"] = word
        if "?" in word:
            failures.append(f"unreadable free group word: {word}")
        elif case["k"] is not None and len(word) != case["k"]:
            failures.append(f"word length {len(word)} != published k={case['k']}")
    except Exception as e:
        failures.append(f"word reading raised: {e}")

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="run a 5-orbit subset")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cases = build_cases()
    if args.quick:
        cases = [c for c in cases if c["name"] in QUICK_NAMES]

    n_workers = args.workers or min(mp.cpu_count(), len(cases))
    print(f"=== Reproduction suite: {len(cases)} known orbits, "
          f"{n_workers} workers ===\n")

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = list(pool.imap_unordered(run_case, cases))

    # Cross-reference in the main process (it owns the word cache)
    from pipeline import cross_reference
    order = {c["name"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order[r["case"]["name"]])

    n_pass = 0
    for r in results:
        case = r["case"]
        if r["word"] != "?" and not r["failures"]:
            xref = cross_reference(r["word"], r["T_refined"],
                                   case["parametrisation"], case["L"])
            if xref is None:
                r["failures"].append("cross-reference found no match")
            elif xref["matched_name"] not in case["accept"]:
                r["failures"].append(
                    f"cross-reference matched {xref['matched_name']!r}, "
                    f"expected one of {sorted(case['accept'])}")

        status = "PASS" if not r["failures"] else "FAIL"
        n_pass += status == "PASS"
        L_str = f"{case['L']:.3f}" if case["L"] is not None else "0 (sym)"
        word = r["word"] if len(r["word"]) <= 14 else r["word"][:11] + "..."
        print(f"  [{status}] {case['name']:<24} L={L_str:<8} word={word:<15}"
              f" T={r['T_refined'] or 0:.4f}")
        for f in r["failures"]:
            print(f"         - {f}")

    elapsed = time.time() - t0
    print(f"\n{n_pass}/{len(cases)} passed in {elapsed:.0f}s")
    sys.exit(0 if n_pass == len(cases) else 1)


if __name__ == "__main__":
    main()
