"""L-continuation of the borderline confirmed-new orbits to settle distinctness.

The verify_shortlist pass confirmed 19 genuine periodic orbits, but ~half sit
within 0.03-0.15 in (a,c) of a lower-k Jankovic orbit at the same L. Two orbits
at the same L with nearby (a,c) are either (i) two points on one family that
folds in L, or (ii) genuinely distinct families. The decisive test is to trace
each borderline orbit's family in L and check whether the family curve sweeps
THROUGH a catalogued Jankovic orbit (matching a, c AND L):

  CONNECTS #nr   the traced family passes within AC_TOL of Jankovic #nr at that
                 orbit's L  -> SAME family as a known orbit; NOT new
  INDEPENDENT    the family never coincides with any Jankovic orbit across its
                 whole L-range  -> genuinely new family

Borderline set = CONFIRMED-NEW orbits with nearest_dist < 0.15 (others are
already far from the catalogue and need no continuation).

Thread-pinned; parallel over orbits. Outputs: continue_borderline.json + report.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import multiprocessing as mp

from three_body import ALL_ORBITS
from continuation import trace_family

AC_TOL = 0.03         # (a,c) match for "same orbit"
L_MATCH = 0.015       # |L_trace - L_jankovic| to count as "at that L"
BORDER_TOL = 0.15     # nearest_dist below this = borderline
N_WORKERS = 5


def connection_to_jankovic(family):
    """For each Jankovic orbit, find the closest traced point at its L.
    Returns (best_nr, best_dist, all_hits) where a hit is (nr,k,L,dist)."""
    best_nr, best_dist = None, 1e9
    hits = []
    for nr, Lj, aj, cj, Tj, kj in ALL_ORBITS:
        # closest traced point in L to this Jankovic L
        cand = [p for p in family if abs(p["L"] - Lj) < L_MATCH]
        if not cand:
            continue
        for p in cand:
            d = ((p["a"] - aj) ** 2 + (p["c"] - cj) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_nr = d, (nr, kj, Lj)
            if d < AC_TOL:
                hits.append({"nr": nr, "k": kj, "L": Lj, "dist": round(d, 4)})
    return best_nr, best_dist, hits


def trace_one(orb):
    L0 = orb["L"]
    try:
        fam = trace_family(orb["a"], orb["c"], orb["T"], L0,
                           L_min=max(0.45, L0 - 0.35),
                           L_max=min(1.15, L0 + 0.35),
                           ds0=0.02, ds_max=0.06, max_steps=130,
                           verbose=False)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        return {**_tag(orb), "verdict": f"TRACE-FAIL:{type(e).__name__}",
                "n_points": 0}

    Ls = [p["L"] for p in fam]
    best, bdist, hits = connection_to_jankovic(fam)
    out = {**_tag(orb), "n_points": len(fam),
           "L_range": [round(min(Ls), 4), round(max(Ls), 4)],
           "nearest_jankovic_on_family": (
               f"#{best[0]} b^{best[1]} @L{best[2]}" if best else None),
           "nearest_dist_on_family": round(bdist, 4),
           "hits": hits}
    if hits:
        h = min(hits, key=lambda x: x["dist"])
        out["verdict"] = f"CONNECTS #{h['nr']} (b^{h['k']} @L{h['L']})"
    else:
        out["verdict"] = "INDEPENDENT"
    return out


def _tag(orb):
    return {"L": orb["L"], "word": f"b^{orb['k_read']}", "a": orb["a"],
            "c": orb["c"], "T": orb["T"],
            "scan_nearest": orb["nearest_jankovic"],
            "scan_dist": orb["nearest_dist"]}


def main():
    allc = [x for x in json.load(open("verify_shortlist.json"))
            if x["verdict"] == "CONFIRMED-NEW"]
    border = [x for x in allc if x["nearest_dist"] is not None
              and x["nearest_dist"] < BORDER_TOL]
    print(f"Borderline orbits to continue: {len(border)} "
          f"(of {len(allc)} confirmed)\n", flush=True)

    with mp.Pool(N_WORKERS) as pool:
        results = []
        for r in pool.imap_unordered(trace_one, border):
            results.append(r)
            print(f"  L={r['L']} {r['word']}: {r['verdict']}  "
                  f"[{r['n_points']} pts, L in {r.get('L_range')}, "
                  f"min-d-to-Jankovic={r.get('nearest_dist_on_family')}]",
                  flush=True)
            json.dump(results, open("continue_borderline.json", "w"), indent=1)

    conn = [r for r in results if r["verdict"].startswith("CONNECTS")]
    indep = [r for r in results if r["verdict"] == "INDEPENDENT"]
    fail = [r for r in results if "FAIL" in r["verdict"]]
    lines = ["", "=" * 64, "L-CONTINUATION SUMMARY (borderline orbits)",
             "=" * 64,
             f"  CONNECTS to a Jankovic family (NOT new): {len(conn)}",
             f"  INDEPENDENT (genuinely new family):      {len(indep)}",
             f"  TRACE-FAIL:                              {len(fail)}", ""]
    for r in sorted(results, key=lambda x: (x["L"], x["word"])):
        lines.append(f"  L={r['L']} {r['word']:>6}  ->  {r['verdict']}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open("continue_borderline_report.txt", "w").write(txt)
    print("\nSaved: continue_borderline.json, continue_borderline_report.txt",
          flush=True)


if __name__ == "__main__":
    main()
