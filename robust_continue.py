"""Robust L-continuation of ALL not-yet-resolved confirmed-new orbits.

continue_borderline.py settled 6 borderline orbits as CONNECTS (known family at
a new L). Two gaps remain:
  1. 3 borderline traces collapsed to a single point (arclength corrector could
     not step) and were mislabeled INDEPENDENT -- they are really inconclusive.
  2. The 9 "solid" orbits (far from any Jankovic AT THEIR OWN L) were never
     continued -- but b^16@0.9 proved a family can travel far in (a,c) and
     connect to a Jankovic orbit at a DIFFERENT L. So "far at same L" does not
     prove novelty; they must be continued against the full catalogue too.

This re-continues every confirmed-new orbit that is NOT already a known
connector, with a robust tracer: pseudo-arclength first, and if that yields
< MIN_PTS points, fall back to NAIVE L-stepping (step L by +/-dL, Newton-refine
(a,c,T) from the previous point). Then test connection to the FULL Jankovic
catalogue (match a, c AND L).

Outputs: robust_continue.json, robust_continue_report.txt
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import multiprocessing as mp

from three_body import ALL_ORBITS, initial_conditions_from_params
from continuation import trace_family
from floquet import newton_refine_bhh

AC_TOL = 0.03
L_MATCH = 0.02
MIN_PTS = 5
N_WORKERS = 4


def naive_trace(a0, c0, T0, L0, dL=0.02, nsteps=40):
    """Fallback: step L by +/-dL, Newton-refine from the previous point.
    Cheaper than arclength (no variational integration) and robust where the
    arclength corrector stalls. Cannot round folds, but is enough to detect a
    sweep through a catalogued Jankovic orbit."""
    pts = []
    for sign in (+1, -1):
        a, c, T, L = a0, c0, T0, L0
        for _ in range(nsteps):
            L = L + sign * dL
            if not (0.45 <= L <= 1.15):
                break
            try:
                a, c, T, ok, _ = newton_refine_bhh(a, c, L, T, tol=1e-11)
            except (RuntimeError, FloatingPointError):
                break
            if not ok:
                break
            pts.append({"a": a, "c": c, "T": T, "L": L})
    return pts


def connection(family):
    best, bdist, hits = None, 1e9, []
    for nr, Lj, aj, cj, Tj, kj in ALL_ORBITS:
        for p in family:
            if abs(p["L"] - Lj) >= L_MATCH:
                continue
            d = ((p["a"] - aj) ** 2 + (p["c"] - cj) ** 2) ** 0.5
            if d < bdist:
                bdist, best = d, (nr, kj, Lj)
            if d < AC_TOL:
                hits.append({"nr": nr, "k": kj, "L": round(Lj, 3),
                             "dist": round(d, 4)})
    return best, bdist, hits


def resolve(orb):
    L0 = orb["L"]
    method = "arclength"
    try:
        fam = trace_family(orb["a"], orb["c"], orb["T"], L0,
                           L_min=0.45, L_max=1.15,
                           ds0=0.015, ds_max=0.06, max_steps=160,
                           verbose=False)
    except (ValueError, RuntimeError, FloatingPointError):
        fam = []
    if len(fam) < MIN_PTS:
        method = "naive-L"
        fam = naive_trace(orb["a"], orb["c"], orb["T"], L0)

    tag = {"L": L0, "word": f"b^{orb['k_read']}", "a": orb["a"],
           "c": orb["c"], "T": orb["T"], "method": method,
           "n_points": len(fam)}
    if len(fam) < MIN_PTS:
        return {**tag, "verdict": "TRACE-FAIL"}
    Ls = [p["L"] for p in fam]
    best, bdist, hits = connection(fam)
    tag["L_range"] = [round(min(Ls), 3), round(max(Ls), 3)]
    tag["min_dist_to_jankovic"] = round(bdist, 4)
    tag["nearest_on_family"] = f"#{best[0]} b^{best[1]} @L{best[2]:.3f}" if best else None
    if hits:
        h = min(hits, key=lambda x: x["dist"])
        tag["verdict"] = f"CONNECTS #{h['nr']} (b^{h['k']} @L{h['L']})"
    else:
        tag["verdict"] = "INDEPENDENT"
    return tag


def main():
    confirmed = [x for x in json.load(open("verify_shortlist.json"))
                 if x["verdict"] == "CONFIRMED-NEW"]
    # already-resolved connectors from the borderline pass
    border = json.load(open("continue_borderline.json"))
    known = {(round(r["L"], 3), r["word"]) for r in border
             if r["verdict"].startswith("CONNECTS")}
    # RESUME: keep results already in robust_continue.json (the laptop crashed
    # mid-run) so we only (re)continue orbits not yet done.
    try:
        results = json.load(open("robust_continue.json"))
        done = {(round(r["L"], 3), r["word"]) for r in results}
    except (FileNotFoundError, ValueError):
        results, done = [], set()
    todo = [x for x in confirmed
            if (round(x["L"], 3), f"b^{x['k_read']}") not in known
            and (round(x["L"], 3), f"b^{x['k_read']}") not in done]
    print(f"Confirmed-new: {len(confirmed)}; already-CONNECTS: {len(known)}; "
          f"already-done (resume): {len(done)}; to (re)continue: {len(todo)}\n",
          flush=True)

    with mp.Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(resolve, todo):
            results.append(r)
            print(f"  L={r['L']} {r['word']:>6} [{r['method']:>9}, "
                  f"{r['n_points']:>3} pts] -> {r['verdict']}  "
                  f"(min-d={r.get('min_dist_to_jankovic')})", flush=True)
            json.dump(results, open("robust_continue.json", "w"), indent=1)

    conn = [r for r in results if r["verdict"].startswith("CONNECTS")]
    indep = [r for r in results if r["verdict"] == "INDEPENDENT"]
    fail = [r for r in results if r["verdict"] == "TRACE-FAIL"]

    # final tally: known connectors (6) + this pass
    n_known_total = len(known) + len(conn)
    lines = ["", "=" * 66, "FINAL CENSUS (all confirmed-new orbits)", "=" * 66,
             f"  Confirmed genuine periodic orbits:        {len(confirmed)}",
             f"  -> CONNECT to a known Jankovic family:    {n_known_total}",
             f"  -> INDEPENDENT (genuinely new family):    {len(indep)}",
             f"  -> still TRACE-FAIL (inconclusive):       {len(fail)}", ""]
    lines.append("INDEPENDENT (genuinely-new) orbits:")
    for r in sorted(indep, key=lambda x: (x["L"], x["word"])):
        lines.append(f"  L={r['L']} {r['word']:>6}  a={r['a']:.4f} "
                     f"c={r['c']:.4f} T={r['T']:.4f}  "
                     f"(L-range {r['L_range']}, min-d {r['min_dist_to_jankovic']})")
    if fail:
        lines.append("")
        lines.append("Inconclusive (trace still failed):")
        for r in fail:
            lines.append(f"  L={r['L']} {r['word']}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open("robust_continue_report.txt", "w").write(txt)
    print("\nSaved: robust_continue.json, robust_continue_report.txt", flush=True)


if __name__ == "__main__":
    main()
