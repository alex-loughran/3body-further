"""Li & Liao 695 periodic orbit families (arXiv 1705.00527).

Data extracted from: https://numericaltank.sjtu.edu.cn/three-body/three-body-movies.htm

All orbits use the symmetric parametrisation:
  m1 = m2 = m3 = 1, G = 1
  positions: (-1, 0), (1, 0), (0, 0)
  velocities: (v1, v2), (v1, v2), (-2v1, -2v2)

Format: (name, v1, v2, T, T_star, Lf)
  name: class and number (e.g. "I.A-1")
  v1, v2: initial velocity components
  T: period
  T_star: scale-invariant period T|E|^{3/2}
  Lf: free group word length
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "ll_orbits.json"
_cache = None


def load_ll_orbits():
    """Load all 695 Li & Liao orbits. Cached after first call."""
    global _cache
    if _cache is not None:
        return _cache

    with open(_DATA_PATH) as f:
        raw = json.load(f)

    _cache = [(o["name"], o["v1"], o["v2"], o["T"], o["T_star"], o["Lf"])
              for o in raw]
    return _cache


def get_ll_orbit(name):
    """Get a single orbit by name (e.g. 'I.A-1'). Returns tuple or None."""
    for o in load_ll_orbits():
        if o[0] == name:
            return o
    return None
