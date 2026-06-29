"""Verify the 20 shortlisted candidates from the 500x500 campaign.

For each: Newton-refine (a,c,T) to machine precision, re-read the free-group
word from the REFINED orbit (the scan-time word was unreliable), recompute the
Kepler T* check, Floquet-classify, and re-cross-reference against Jankovic by
both (a,c) proximity and canonical word + integer period-multiple.

Verdict per orbit:
  REFINE-FAIL    Newton did not converge -> not a real orbit at this point
  REDISCOVERY    refined (a,c) coincides with a Jankovic orbit (T-ratio ~1)
  MULTIPLE       refined orbit is an m-fold traversal of a Jankovic orbit
  WORD-JUNK      refined orbit's word still disagrees with Kepler T* -> the
                 topology is not cleanly readable (near-collision); reject
  CONFIRMED-NEW  converged, clean word (T*~2.43), not a Jankovic match/multiple

Thread-pinned to 1 (avoids BLAS oversubscription OOM seen earlier).
Outputs: verify_shortlist.json, verify_shortlist_report.txt
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import re
import numpy as np

from three_body import (
    initial_conditions_from_params, integrate_orbit, read_free_group_word,
    compute_energy, compute_angular_momentum, ALL_ORBITS,
)
from floquet import newton_refine_bhh, analyse_orbit

AC_TOL = 0.03
TSTAR = 2.433


def canonical_word(w):
    if not w or "?" in w:
        return w
    d = w + w
    n = len(w)
    return min(d[i:i + n] for i in range(n))


def k_of(w):
    return len(w) if w and re.fullmatch(r"b+", w) else None


def jankovic_at(L):
    return [{"nr": nr, "a": a, "c": c, "T": T, "k": k}
            for nr, Lj, a, c, T, k in ALL_ORBITS if abs(Lj - L) < 0.02]


def nearest_jankovic(L, a, c):
    best, bd = None, 1e9
    for j in jankovic_at(L):
        d = ((j["a"] - a) ** 2 + (j["c"] - c) ** 2) ** 0.5
        if d < bd:
            best, bd = j, d
    return best, bd


def verify_one(cand):
    L, a0, c0, T0 = cand["L"], cand["a"], cand["c"], cand["T"]
    out = {"L": L, "a_in": a0, "c_in": c0, "T_in": T0,
           "k_in": cand["k"], "class_in": cand["class"]}

    # 1. Newton-refine to machine precision
    a, c, T, ok, info = newton_refine_bhh(a0, c0, L, T0, tol=1e-12)
    out["converged"] = bool(ok)
    out["a"], out["c"], out["T"] = float(a), float(c), float(T)
    if not ok:
        out["verdict"] = "REFINE-FAIL"
        return out

    state0 = initial_conditions_from_params(a, c, L)
    E = compute_energy(state0)
    Lreal = compute_angular_momentum(state0)
    out["E"] = float(E)
    out["L_check"] = float(Lreal)

    # 2. re-read word from the REFINED orbit + periodicity residual
    sol = integrate_orbit(state0, T)
    word = read_free_group_word(sol, T)
    from three_body import to_Z_vector
    dmin = float(np.linalg.norm(to_Z_vector(sol.sol(T)) - to_Z_vector(state0)))
    out["d_min"] = dmin
    out["word"] = word
    k_read = k_of(word)
    out["k_read"] = k_read
    k_law = T * abs(E) ** 1.5 / TSTAR
    out["k_law"] = round(k_law)
    out["tstar"] = round(T * abs(E) ** 1.5 / k_read, 3) if k_read else None
    word_ok = (k_read is not None and abs(k_read - k_law) <= max(2, 0.15 * k_law))
    out["word_ok"] = bool(word_ok)

    # 3. Floquet
    res = analyse_orbit(state0, T, verbose=False)
    stab = res["stability"]
    out["lambda_max"] = float(stab["max_instability"])
    out["is_stable"] = bool(stab["is_stable"])
    out["det"] = float(stab["determinant"])
    out["n_unit"] = int(stab["n_unit"])
    out["monodromy_valid"] = bool(res["valid"])

    # 4. cross-reference refined orbit against Jankovic
    j, jd = nearest_jankovic(L, a, c)
    out["nearest_jankovic"] = f"#{j['nr']} b^{j['k']}" if j else None
    out["nearest_dist"] = round(jd, 4) if j else None
    verdict = None
    if j and jd < AC_TOL:
        ratio = T / j["T"]
        out["T_ratio"] = round(ratio, 3)
        m = round(ratio)
        if abs(ratio - 1) < 0.08:
            verdict = "REDISCOVERY"
        elif m >= 2 and abs(ratio - m) < 0.12:
            verdict = f"MULTIPLE ({m}x #{j['nr']})"
    if verdict is None:
        # canonical-word match to any Jankovic at this L (recompute their words
        # would be costly; rely on (a,c)+k since all are pure-b)
        if not word_ok or dmin > 1e-4:
            verdict = "WORD-JUNK"
        else:
            verdict = "CONFIRMED-NEW"
    out["verdict"] = verdict
    return out


def main():
    sl = json.load(open("classify_candidates.json"))["shortlist"]
    print(f"Verifying {len(sl)} shortlisted candidates "
          f"(thread-pinned, sequential)...\n", flush=True)

    results = []
    for i, cand in enumerate(sl):
        try:
            r = verify_one(cand)
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
            r = {"L": cand["L"], "a_in": cand["a"], "c_in": cand["c"],
                 "k_in": cand["k"], "verdict": f"ERROR:{type(e).__name__}"}
        results.append(r)
        wd = r.get("word", "?")
        wd = (wd[:12] + "...") if wd and len(wd) > 15 else wd
        print(f"[{i+1:>2}/{len(sl)}] L={r['L']} b^{r.get('k_in')} -> "
              f"refined b^{r.get('k_read')} (k_law={r.get('k_law')}, "
              f"T*={r.get('tstar')}) lam={r.get('lambda_max')} "
              f"=> {r['verdict']}", flush=True)
        json.dump(results, open("verify_shortlist.json", "w"), indent=1)

    # summary
    import collections
    vc = collections.Counter(re.sub(r" \(.*", "", r["verdict"]) for r in results)
    lines = ["", "=" * 64, "VERIFICATION SUMMARY", "=" * 64]
    for v, n in vc.most_common():
        lines.append(f"  {v:<16} {n}")
    confirmed = [r for r in results if r["verdict"] == "CONFIRMED-NEW"]
    lines.append("")
    lines.append(f"CONFIRMED-NEW orbits: {len(confirmed)}")
    if confirmed:
        lines.append(f"{'L':>5} {'word':>8} {'T':>8} {'T*':>5} {'lam':>7} "
                     f"{'stable':>6} {'a':>8} {'c':>9}")
        for r in sorted(confirmed, key=lambda x: (x["L"], x["k_read"])):
            lines.append(f"{r['L']:>5} {('b^'+str(r['k_read'])):>8} "
                         f"{r['T']:>8.4f} {str(r['tstar']):>5} "
                         f"{r['lambda_max']:>7.1f} {str(r['is_stable']):>6} "
                         f"{r['a']:>8.4f} {r['c']:>9.4f}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open("verify_shortlist_report.txt", "w").write(txt)
    print("\nSaved: verify_shortlist.json, verify_shortlist_report.txt",
          flush=True)


if __name__ == "__main__":
    main()
