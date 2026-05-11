---
name: Li & Liao 695-family data — now available
description: LL numerical data (v1, v2, T, T*, Lf) extracted from SJTU website HTML tables; stored in ll_orbits.json
type: project
---

Li & Liao's 695 orbit families (arXiv 1705.00527) were previously thought inaccessible. The numerical data (initial velocities, periods, word lengths) is actually embedded in HTML tables at https://numericaltank.sjtu.edu.cn/three-body/three-body-movies.htm — not in downloadable files, which is why earlier attempts failed.

**Status:** Extracted and parsed. `fetch_ll_data.py` downloads and parses the HTML. Data stored in `ll_orbits.json` (695 entries). `ll_data.py` provides `load_ll_orbits()` API. Wired into `pipeline.py` cross-reference for symmetric orbits.

**How to apply:** Cross-referencing now covers: figure-eight + 15 Suvakov + 19 satellites + 75 Jankovic BHH + 695 Li & Liao symmetric. Computing words for all 695 LL orbits is expensive (~1 hour) but cached to `known_words_cache.json`.
