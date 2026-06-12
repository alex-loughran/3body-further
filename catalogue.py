
"""Orbit catalogue database: schema, ingestion, and query API.

A single SQLite database (orbits.db) holding every orbit the project has
found or analysed, regardless of which scan campaign or catalogue it came
from. The database is fully regenerable from the JSON files it ingests
(*_candidates.json from pipeline.py, floquet_catalogue.json from main.py
catalogue), so it is gitignored — rebuild any time with:

    python catalogue.py ingest

Usage:
    python catalogue.py ingest [files...]   Ingest JSON files (default: all
                                            *_candidates.json + floquet_catalogue.json)
    python catalogue.py summary             Per-L summary table
    python catalogue.py query [filters]     Query orbits, e.g.:
        python catalogue.py query --L 0.8 --new
        python catalogue.py query --word-min 20 --unstable
        python catalogue.py query --stable --parametrisation symmetric
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = "orbits.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orbits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    orbit_key TEXT UNIQUE NOT NULL,
    name TEXT,
    source TEXT NOT NULL,
    source_file TEXT,
    parametrisation TEXT NOT NULL,
    L REAL NOT NULL,
    param1_raw REAL,
    param2_raw REAL,
    param1 REAL NOT NULL,
    param2 REAL NOT NULL,
    T REAL NOT NULL,
    E REAL,
    d_min REAL,
    converged INTEGER,
    word TEXT,
    word_canonical TEXT,
    word_length INTEGER,
    is_stable INTEGER,
    lambda_max REAL,
    n_unstable INTEGER,
    det_M REAL,
    n_unit_eigenvalues INTEGER,
    monodromy_valid INTEGER,
    matched_name TEXT,
    k_multiple INTEGER,
    is_new INTEGER,
    multiplier_magnitudes TEXT,
    ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_orbits_L ON orbits (L);
CREATE INDEX IF NOT EXISTS idx_orbits_word ON orbits (word_canonical);
CREATE INDEX IF NOT EXISTS idx_orbits_new ON orbits (is_new);
"""

# Magnitude tolerance for counting a Floquet multiplier as unstable.
# Matches the unit-circle tolerance used in floquet.py stability checks.
_UNSTABLE_TOL = 1e-3


def canonical_word(word):
    """Lexicographically smallest cyclic rotation of a free group word.

    Same convention as pipeline.canonical_word — duplicated here so the
    catalogue stays importable without pulling in scipy/numba.
    """
    if not word or "?" in word:
        return word
    doubled = word + word
    n = len(word)
    return min(doubled[i:i + n] for i in range(n))


def word_length(word):
    """Letter count of a free group word, or None if unreadable."""
    if not word or "?" in word:
        return None
    return len(word)


def _orbit_key(source, parametrisation, L, param1, param2, T):
    """Deduplication key: source + parametrisation + L + refined params + period.

    Rounding absorbs run-to-run Newton refinement jitter. Distinct orbits
    are separated by far more than 1e-6 in parameter space. The key is
    scoped by source ('scan' vs 'known') so a scan re-detection of a known
    orbit keeps its own row — re-detections are data (recovery rate), not
    duplicates. Use families() to group them back together by word.
    """
    return (f"{source}|{parametrisation}|L={L:.6f}|p1={param1:.6f}"
            f"|p2={param2:.6f}|T={T:.4f}")


def _n_unstable(magnitudes):
    """Count unstable directions from sorted multiplier magnitudes.

    Multipliers come in reciprocal pairs (symplectic), so the count of
    |lambda| > 1 equals the number of expanding directions.
    """
    if not magnitudes:
        return None
    return sum(1 for m in magnitudes if m > 1.0 + _UNSTABLE_TOL)


class OrbitDB:
    """Query and ingestion API for the orbit catalogue."""

    def __init__(self, path=DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self):
        self.conn.close()

    # -- Ingestion -----------------------------------------------------

    def _upsert(self, row):
        """Insert or replace a single orbit row. Returns True if new."""
        existing = self.conn.execute(
            "SELECT id FROM orbits WHERE orbit_key = ?",
            (row["orbit_key"],)).fetchone()
        cols = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        if existing:
            assignments = ", ".join(f"{c} = ?" for c in row)
            self.conn.execute(
                f"UPDATE orbits SET {assignments} WHERE id = ?",
                (*row.values(), existing["id"]))
            return False
        self.conn.execute(
            f"INSERT INTO orbits ({cols}) VALUES ({placeholders})",
            tuple(row.values()))
        return True

    def ingest_candidates(self, json_path):
        """Ingest a pipeline *_candidates.json file. Returns (added, updated)."""
        with open(json_path) as f:
            entries = json.load(f)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fname = Path(json_path).name
        added = updated = 0
        for e in entries:
            word = e.get("free_group_word", "")
            p1, p2 = e["params_refined"]
            mags = e.get("multiplier_magnitudes")
            match = e.get("match") or {}
            row = {
                "orbit_key": _orbit_key("scan", e["parametrisation"], e["L"], p1, p2, e["T"]),
                "name": f"{fname.replace('_candidates.json', '')}:{e['id']}",
                "source": "scan",
                "source_file": fname,
                "parametrisation": e["parametrisation"],
                "L": e["L"],
                "param1_raw": e["params_raw"][0],
                "param2_raw": e["params_raw"][1],
                "param1": p1,
                "param2": p2,
                "T": e["T"],
                "E": e.get("E"),
                "d_min": e.get("d_min"),
                "converged": int(bool(e.get("converged"))),
                "word": word,
                "word_canonical": canonical_word(word),
                "word_length": word_length(word),
                "is_stable": int(bool(e.get("is_stable"))),
                "lambda_max": e.get("max_instability"),
                "n_unstable": _n_unstable(mags),
                "det_M": e.get("determinant"),
                "n_unit_eigenvalues": e.get("n_unit_eigenvalues"),
                "monodromy_valid": int(bool(e.get("monodromy_valid"))),
                "matched_name": match.get("matched_name"),
                "k_multiple": match.get("k_multiple"),
                "is_new": int(bool(e.get("is_new"))),
                "multiplier_magnitudes": json.dumps(mags) if mags else None,
                "ingested_at": now,
            }
            if self._upsert(row):
                added += 1
            else:
                updated += 1
        self.conn.commit()
        return added, updated

    def ingest_floquet_catalogue(self, json_path):
        """Ingest floquet_catalogue.json (known orbits). Returns (added, updated)."""
        with open(json_path) as f:
            entries = json.load(f)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fname = Path(json_path).name
        added = updated = 0
        for e in entries:
            word = e.get("word", "")
            p1, p2 = e["params"]
            mags = e.get("multiplier_magnitudes")
            row = {
                "orbit_key": _orbit_key("known", e["parametrisation"], e["L"], p1, p2, e["T"]),
                "name": e["name"],
                "source": "known",
                "source_file": fname,
                "parametrisation": e["parametrisation"],
                "L": e["L"],
                "param1_raw": None,
                "param2_raw": None,
                "param1": p1,
                "param2": p2,
                "T": e["T"],
                "E": e.get("E"),
                "d_min": None,
                "converged": 1,
                "word": word,
                "word_canonical": canonical_word(word),
                "word_length": word_length(word),
                "is_stable": int(bool(e.get("is_stable"))),
                "lambda_max": e.get("max_instability"),
                "n_unstable": _n_unstable(mags),
                "det_M": e.get("determinant"),
                "n_unit_eigenvalues": e.get("n_unit"),
                "monodromy_valid": int(bool(e.get("valid"))),
                "matched_name": e["name"],
                "k_multiple": 1,
                "is_new": 0,
                "multiplier_magnitudes": json.dumps(mags) if mags else None,
                "ingested_at": now,
            }
            if self._upsert(row):
                added += 1
            else:
                updated += 1
        self.conn.commit()
        return added, updated

    # -- Queries -------------------------------------------------------

    def query(self, L=None, L_tol=0.005, parametrisation=None, word=None,
              stable=None, new=None, source=None, word_min=None,
              word_max=None, converged_only=True):
        """Filter orbits. Returns a list of dicts.

        Parameters mirror the CLI flags: word matches the canonical cyclic
        form, stable/new are tri-state (None = don't filter).
        """
        clauses, params = [], []
        if converged_only:
            clauses.append("converged = 1")
        if L is not None:
            clauses.append("ABS(L - ?) <= ?")
            params += [L, L_tol]
        if parametrisation:
            clauses.append("parametrisation = ?")
            params.append(parametrisation)
        if word:
            clauses.append("word_canonical = ?")
            params.append(canonical_word(word))
        if stable is not None:
            clauses.append("is_stable = ?")
            params.append(int(stable))
        if new is not None:
            clauses.append("is_new = ?")
            params.append(int(new))
        if source:
            clauses.append("source = ?")
            params.append(source)
        if word_min is not None:
            clauses.append("word_length >= ?")
            params.append(word_min)
        if word_max is not None:
            clauses.append("word_length <= ?")
            params.append(word_max)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM orbits{where} ORDER BY L, word_length, T"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def families(self, **kwargs):
        """Group query results into families by (L, canonical word).

        Returns {(L, word_canonical): [orbit dicts]} — multiple rows in one
        family are either duplicates across scans or genuine re-detections.
        """
        groups = {}
        for r in self.query(**kwargs):
            key = (round(r["L"], 4), r["word_canonical"])
            groups.setdefault(key, []).append(r)
        return groups

    def summary(self):
        """Per-L counts: total, new, stable, word-length range."""
        sql = """
        SELECT L,
               COUNT(*) AS total,
               SUM(is_new) AS new,
               SUM(is_stable) AS stable,
               MIN(word_length) AS k_min,
               MAX(word_length) AS k_max,
               COUNT(DISTINCT word_canonical) AS families
        FROM orbits WHERE converged = 1
        GROUP BY ROUND(L, 4) ORDER BY L
        """
        return [dict(r) for r in self.conn.execute(sql)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_ingest(args):
    db = OrbitDB(args.db)
    paths = [Path(p) for p in args.files]
    if not paths:
        paths = sorted(Path(".").glob("*_candidates.json"))
        if Path("floquet_catalogue.json").exists():
            paths.append(Path("floquet_catalogue.json"))
    if not paths:
        print("No JSON files found to ingest.")
        return
    for p in paths:
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        if p.name == "floquet_catalogue.json":
            added, updated = db.ingest_floquet_catalogue(p)
        else:
            added, updated = db.ingest_candidates(p)
        print(f"  {p.name}: {added} added, {updated} updated")
    total = db.conn.execute("SELECT COUNT(*) FROM orbits").fetchone()[0]
    print(f"Database now holds {total} orbits ({args.db})")
    db.close()


def _cmd_summary(args):
    db = OrbitDB(args.db)
    rows = db.summary()
    if not rows:
        print("Database is empty. Run: python catalogue.py ingest")
        return
    print(f"{'L':>8} {'total':>6} {'new':>5} {'stable':>7} "
          f"{'families':>9} {'word len':>10}")
    print("-" * 50)
    for r in rows:
        k_range = (f"{r['k_min']}-{r['k_max']}"
                   if r["k_min"] is not None else "?")
        print(f"{r['L']:>8.4f} {r['total']:>6} {r['new'] or 0:>5} "
              f"{r['stable'] or 0:>7} {r['families']:>9} {k_range:>10}")
    db.close()


def _cmd_query(args):
    db = OrbitDB(args.db)
    stable = True if args.stable else (False if args.unstable else None)
    new = True if args.new else (False if args.known else None)
    rows = db.query(L=args.L, parametrisation=args.parametrisation,
                    word=args.word, stable=stable, new=new,
                    source=args.source, word_min=args.word_min,
                    word_max=args.word_max)
    if not rows:
        print("No orbits match.")
        return
    print(f"{'name':<42} {'L':>6} {'T':>9} {'word':<18} "
          f"{'k':>4} {'stab':>4} {'new':>3} {'λ_max':>9}")
    print("-" * 105)
    for r in rows:
        w = r["word_canonical"] or "?"
        if len(w) > 16:
            w = w[:13] + "..."
        lam = f"{r['lambda_max']:.3g}" if r["lambda_max"] else "?"
        print(f"{r['name']:<42} {r['L']:>6.3f} {r['T']:>9.4f} {w:<18} "
              f"{r['word_length'] or '?':>4} "
              f"{'Y' if r['is_stable'] else 'N':>4} "
              f"{'Y' if r['is_new'] else 'N':>3} {lam:>9}")
    print(f"\n{len(rows)} orbits")
    db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest JSON result files")
    p_ingest.add_argument("files", nargs="*")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_summary = sub.add_parser("summary", help="Per-L summary table")
    p_summary.set_defaults(func=_cmd_summary)

    p_query = sub.add_parser("query", help="Query orbits")
    p_query.add_argument("--L", type=float)
    p_query.add_argument("--parametrisation", choices=["symmetric", "bhh"])
    p_query.add_argument("--word")
    p_query.add_argument("--word-min", type=int, dest="word_min")
    p_query.add_argument("--word-max", type=int, dest="word_max")
    p_query.add_argument("--stable", action="store_true")
    p_query.add_argument("--unstable", action="store_true")
    p_query.add_argument("--new", action="store_true")
    p_query.add_argument("--known", action="store_true")
    p_query.add_argument("--source", choices=["scan", "known"])
    p_query.set_defaults(func=_cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
