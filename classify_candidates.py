"""Dedup + classify the 89 unmatched candidates from the 500x500 BHH campaign.

The pipeline's "is_new" only means "word didn't match a Jankovic orbit at the
SAME L (within 0.01)". That over-counts badly. This does a proper census:

  1. DEDUP within the campaign: cluster candidates at each L by refined (a,c).
     A cluster with >1 member = the same orbit found twice (or a multiple
     traversal: same (a,c), period ~ integer x base). Keep the primitive.

  2. CLASSIFY each distinct orbit against the FULL Jankovic catalogue:
     - REDISCOVERY: a Jankovic orbit exists at this L with matching (a,c).
       (Gold standard -- (a,c) match is more reliable than the b^k word,
       which gets noisy at large k.)
     - EXTENSION: word-length k is present in Jankovic at OTHER L values but
       not catalogued here -> known family, new L point.
     - BEYOND-CATALOGUE: k > 35 (Jankovic's max) -> genuinely new territory.
     - GAP-FILLER: k <= 35 but that (L, k) pair absent from Jankovic AND no
       (a,c) match -> a new orbit in Jankovic's k-range.

  3. FLAG suspects: k >= 60 (near-collision territory; word-reader + refiner
     unreliable), and positive-c band (different region).

Outputs: classify_candidates.json, classify_candidates_report.txt
"""

import json
import glob
import re
import collections

from three_body import ALL_ORBITS

CAND_DIR = "mini_results"
AC_TOL = 0.03      # (a,c) Euclidean distance for "same orbit" / Jankovic match
HIGH_K = 60        # k at/above this is flagged suspect


def k_of(word):
    return len(word) if word and re.fullmatch(r"b+", word) else None


def load_candidates():
    rows = []
    for f in sorted(glob.glob(f"{CAND_DIR}/scan_bhh_L*_500x500_candidates.json")):
        L = float(re.search(r"L([0-9.]+)_", f).group(1))
        for c in json.load(open(f)):
            if not c.get("converged"):
                continue
            a_r, c_r = c["params_refined"]
            k = k_of(c["free_group_word"])
            E = c.get("E")
            T = c["T"]
            # topological Kepler law: T* = T|E|^{3/2}/k ~ 2.43 for genuine orbits.
            tstar = (T * abs(E) ** 1.5 / k) if (E and k) else None
            # invert it to recover the word length the law predicts:
            k_law = (T * abs(E) ** 1.5 / 2.433) if E else None
            rows.append({
                "L": L, "a": a_r, "c": c_r, "T": T, "E": E,
                "k": k, "k_law": round(k_law) if k_law else None,
                "tstar": round(tstar, 2) if tstar else None,
                "word_len": len(c["free_group_word"]),
                "lam_max": c["max_instability"],
                "d_min": c["d_min"],
                "is_new_pipeline": c.get("is_new"),
                "id": c.get("id"),
            })
    return rows


def jankovic_by_L():
    byL = collections.defaultdict(list)
    kset_all = set()
    kset_byL = collections.defaultdict(set)
    for nr, L, a, c, T, k in ALL_ORBITS:
        Lr = round(L, 2)
        byL[Lr].append({"nr": nr, "a": a, "c": c, "T": T, "k": k})
        kset_all.add(k)
        kset_byL[Lr].add(k)
    return byL, kset_all, kset_byL


def dedup(rows):
    """Cluster by (L, a, c). Returns (distinct_rows, n_collapsed)."""
    distinct = []
    collapsed = 0
    for r in rows:
        hit = None
        for d in distinct:
            if d["L"] == r["L"] and \
               ((d["a"] - r["a"]) ** 2 + (d["c"] - r["c"]) ** 2) ** 0.5 < AC_TOL:
                hit = d
                break
        if hit is None:
            r["dup_count"] = 1
            distinct.append(r)
        else:
            collapsed += 1
            hit["dup_count"] += 1
            # keep the lower-T (primitive) representative
            if r["T"] < hit["T"]:
                hit.update({k: r[k] for k in ("a", "c", "T", "k",
                            "word_len", "lam_max", "d_min", "id")})
    return distinct, collapsed


def classify(r, byL, kset_all, kset_byL):
    Lr = round(r["L"], 2)
    # nearest Jankovic orbit at this L by (a,c)
    nearest, ndist = None, 1e9
    for j in byL.get(Lr, []):
        dd = ((j["a"] - r["a"]) ** 2 + (j["c"] - r["c"]) ** 2) ** 0.5
        if dd < ndist:
            nearest, ndist = j, dd
    r["nearest_jankovic"] = (f"#{nearest['nr']} b^{nearest['k']}"
                             if nearest else None)
    r["nearest_dist"] = round(ndist, 4) if nearest else None

    k = r["k"]
    r["suspect_highk"] = (k is not None and k >= HIGH_K)
    r["positive_c"] = r["c"] > 0
    r["T_ratio"] = None
    r["multiple_of"] = None

    # (a,c) match to a Jankovic orbit: classify by the PERIOD ratio, not the
    # word length (the b^k reader over-counts badly at large k).
    if nearest and ndist < AC_TOL:
        ratio = r["T"] / nearest["T"]
        r["T_ratio"] = round(ratio, 3)
        m = round(ratio)
        if abs(ratio - 1.0) < 0.08:
            r["class"] = "REDISCOVERY"
        elif m >= 2 and abs(ratio - m) < 0.12:
            r["class"] = "MULTIPLE"          # m-fold traversal of a Jankovic orbit
            r["multiple_of"] = f"{m}x {nearest['nr']}"
        else:
            r["class"] = "SAME-AC-ODD-T"     # same IC, non-integer T (rare/suspect)
        return r

    # No (a,c) match. It could still be a multiple whose refined (a,c) drifted:
    # check if T is an integer multiple of ANY Jankovic orbit at this L whose
    # PRIMITIVE k divides this k (within +/-2 to allow word-reader slop).
    if k is not None:
        for j in byL.get(Lr, []):
            ratio = r["T"] / j["T"]
            m = round(ratio)
            if m >= 2 and abs(ratio - m) < 0.10 and \
               abs(k - m * j["k"]) <= max(2, 0.1 * k):
                r["class"] = "LIKELY-MULTIPLE"
                r["multiple_of"] = f"~{m}x {j['nr']} (b^{j['k']})"
                r["T_ratio"] = round(ratio, 3)
                return r

    # Use the Kepler-law word length (trustworthy) not the read word (noisy).
    kl = r["k_law"]
    r["word_ok"] = (k is not None and kl is not None and abs(k - kl) <= max(2, 0.15 * kl))
    keff = kl if kl else k
    if keff is None:
        r["class"] = "NON-PURE-B?"
    elif keff > max(kset_all):
        r["class"] = "BEYOND-CATALOGUE"
    elif keff in kset_byL.get(Lr, set()) or any(
            abs(keff - kk) <= 1 for kk in kset_byL.get(Lr, set())):
        r["class"] = "NEW-SAME-K-FAMILY"
    elif keff in kset_all or any(abs(keff - kk) <= 1 for kk in kset_all):
        r["class"] = "EXTENSION"
    else:
        r["class"] = "GAP-FILLER"
    return r


def main():
    rows = load_candidates()
    byL, kset_all, kset_byL = jankovic_by_L()
    distinct, collapsed = dedup(rows)
    for r in distinct:
        classify(r, byL, kset_all, kset_byL)

    cls_count = collections.Counter(r["class"] for r in distinct)
    n_suspect = sum(1 for r in distinct if r["suspect_highk"])
    n_posc = sum(1 for r in distinct if r["positive_c"])

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 72)
    P("DEDUP + CLASSIFY: 500x500 BHH campaign candidates")
    P("=" * 72)
    P(f"Raw converged candidates:        {len(rows)}")
    P(f"Collapsed as (L,a,c) duplicates: {collapsed}")
    P(f"Distinct orbits:                 {len(distinct)}")
    P("")
    P("Classification of distinct orbits:")
    order = ["REDISCOVERY", "MULTIPLE", "LIKELY-MULTIPLE", "SAME-AC-ODD-T",
             "NEW-SAME-K-FAMILY", "EXTENSION", "GAP-FILLER", "BEYOND-CATALOGUE",
             "NON-PURE-B?"]
    for cls in order:
        if cls_count.get(cls):
            P(f"  {cls:<20} {cls_count[cls]:>3}")
    P("")
    P(f"  (of which suspect k>={HIGH_K}, near-collision): {n_suspect}")
    P(f"  (of which positive-c band):           {n_posc}")
    P("")
    known = ("REDISCOVERY", "MULTIPLE", "LIKELY-MULTIPLE")
    new_classes = ("NEW-SAME-K-FAMILY", "EXTENSION", "GAP-FILLER",
                   "BEYOND-CATALOGUE", "SAME-AC-ODD-T")
    genuinely_new = [r for r in distinct if r["class"] in new_classes]
    trustworthy_new = [r for r in genuinely_new if not r["suspect_highk"]
                       and not r["positive_c"]]
    n_known = sum(1 for r in distinct if r["class"] in known)
    P(f"Already-known (rediscovery or multiple of a Jankovic orbit): {n_known}")
    P(f"Candidate NEW (not a Jankovic orbit or multiple):           "
      f"{len(genuinely_new)}")
    P(f"  -> of those, trustworthy (k<{HIGH_K}, negative-c band):   "
      f"{len(trustworthy_new)}")
    P("")
    n_badword = sum(1 for r in distinct
                    if r["class"] not in known and not r.get("word_ok", True))
    P(f"  -> NOTE: {n_badword} of the 'new' have an UNRELIABLE word "
      f"(read k disagrees with Kepler-law k); their k is noise.")
    P("")
    P("--- distinct orbits by L (sorted). k_read vs k_law (Kepler) ---")
    P(f"{'L':>5} {'k_rd':>5} {'k_law':>5} {'T*':>5} {'T':>7} {'lam':>7} "
      f"{'class':>16} {'T/Tj':>5} {'note':>24}")
    for r in sorted(distinct, key=lambda x: (x["L"], x["k_law"] or x["k"] or 0)):
        flag = "*" if r["suspect_highk"] else ("+c" if r["positive_c"] else " ")
        wbad = "" if r.get("word_ok", True) or r["class"] in known else " !w"
        if r["multiple_of"]:
            note = r["multiple_of"]
        elif r["nearest_jankovic"]:
            note = f"{r['nearest_jankovic']}({r['nearest_dist']})"
        else:
            note = "-"
        tr = f"{r['T_ratio']}" if r["T_ratio"] is not None else ""
        P(f"{r['L']:>5} {r['k']:>5} {str(r['k_law']):>5} "
          f"{str(r['tstar']):>5} {r['T']:>7.3f} {r['lam_max']:>7.1f} "
          f"{r['class']:>16} {tr:>5} {note:>24} {flag}{wbad}")
    P("")
    P("Legend: k_rd=read word len; k_law=Kepler-law len (T|E|^1.5/2.433); "
      "T*=T|E|^1.5/k_rd (~2.43 if word ok); * suspect k>=60; +c pos-c; "
      "!w unreliable word")

    # SHORTLIST: the only defensibly-new orbits -- self-consistent word
    # (T* near 2.43), not a Jankovic match/multiple, moderate k, negative-c.
    shortlist = [r for r in distinct
                 if r["class"] in new_classes and r.get("word_ok")
                 and not r["suspect_highk"] and not r["positive_c"]
                 and r["lam_max"] < 100]
    P("")
    P("=" * 72)
    P(f"SHORTLIST: defensibly-new orbits worth individual Newton-verification "
      f"({len(shortlist)})")
    P("=" * 72)
    P(f"{'L':>5} {'b^k':>6} {'T*':>5} {'T':>7} {'lam':>7} {'class':>16} "
      f"{'a':>8} {'c':>9}")
    for r in sorted(shortlist, key=lambda x: (x["L"], x["k"])):
        P(f"{r['L']:>5} {('b^'+str(r['k'])):>6} {str(r['tstar']):>5} "
          f"{r['T']:>7.3f} {r['lam_max']:>7.1f} {r['class']:>16} "
          f"{r['a']:>8.4f} {r['c']:>9.4f}")

    json.dump({"distinct": distinct, "shortlist": shortlist,
               "counts": dict(cls_count),
               "n_raw": len(rows), "n_collapsed": collapsed},
              open("classify_candidates.json", "w"), indent=1)
    open("classify_candidates_report.txt", "w").write("\n".join(lines))
    P("")
    P("Saved: classify_candidates.json, classify_candidates_report.txt")


if __name__ == "__main__":
    main()
