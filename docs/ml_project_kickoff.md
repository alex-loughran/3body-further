# ML project kickoff

A self-contained starter for the ML layer on top of the three-body orbit
hunter. Everything here is callable against the **already-packaged** physics
engine — `pip install -e .` exposes the stable modules; the ML repo imports
them and never re-implements physics.

> **Honest framing first.** ML is an *accelerator*, not the highest-value work
> right now. The headline result so far (first stable orbit at L≠0) came from
> **continuation**, with no ML. So the clean, publishable ML deliverable is a
> **comparison**: "stable-orbits-found-per-CPU-hour, ML-guided vs uniform grid."
> Build toward that, not toward a flashy model. Use **classical ML, not deep
> learning** for the load-bearing tasks — inputs are 2–3 dimensional and
> stability labels are scarce/expensive, exactly where GPs and gradient boosting
> beat neural nets and DL overfits. (DL has a place; see the deferred track.)

---

## 1. What to build first (week 1)

A **stability surrogate** with an honest baseline, in this order:

1. Assemble a labelled table `(a, c, L) → λ_max, stable?` from existing
   continuation/classification outputs (Section 4).
2. Fit a Gaussian Process (and an XGBoost/LightGBM comparator) to predict
   `log10(λ_max)` and the binary `stable?`.
3. Wrap it in an **active-learning loop**: train → propose the next `(a,c,L)` to
   evaluate where the model is most uncertain/promising → call the physics
   engine for the true label → retrain.
4. Measure the loop against a uniform grid on the same budget. That comparison
   *is* the result.

Scope ceiling to set expectations up front: the known stable window is ~5×10⁻⁴
wide in L, far below what a coarse-label surrogate resolves. So the surrogate's
realistic job is **triage** ("this family/region is hopelessly unstable, skip
it" / rank candidates to trace precisely), not pinpointing windows.

---

## 2. Setup

```bash
# In the physics repo (this one), once:
pip install -e .            # exposes: three_body, floquet, scanner,
                            # parametrisations, continuation, pipeline,
                            # catalogue, ll_data, config

# In the new ML repo:
pip install scikit-learn        # GP + baselines; GP is in sklearn.gaussian_process
pip install xgboost lightgbm    # gradient-boosted comparators
# optional, only if sklearn GP can't scale to millions of scan points:
pip install gpytorch            # sparse/inducing-point GP (Track B)
```

The new ML repo should be a **separate git repo** that depends on
`threebody-physics`, not a folder in this one. Keep the physics engine a stable
importable library; keep ML experiments out of its history.

---

## 3. The physics API you'll call

All verified against the current package.

```python
import numpy as np
from parametrisations import BHHBuilder, SymmetricBuilder
from floquet import analyse_orbit, newton_refine_bhh, compute_monodromy, \
                     floquet_multipliers
from three_body import ALL_ORBITS, initial_conditions_from_params
from ll_data import load_ll_orbits

# --- label any (a, c, L): refine to a true orbit, then get its stability ---
a, c, L, T_guess = 0.2462, -2.0353, 0.8308, 4.88
a_r, c_r, T_r, ok, info = newton_refine_bhh(a, c, L, T_guess)   # Newton to 1e-12
if ok:
    state0 = initial_conditions_from_params(a_r, c_r, L)
    res = analyse_orbit(state0, T_r, verbose=False)
    # res = {"monodromy", "final_state", "multipliers", "stability", "valid", ...}
    lam_max = max(abs(m) for m in res["multipliers"])
    stable  = lam_max < 1.0 + 1e-3        # all |λ| on the unit circle

# --- seed catalogue: 75 Jankovic BHH orbits, tuple layout (n, L, a, c, T, k) ---
for n, L, a, c, T, k in ALL_ORBITS:       # k = free-group word length (b^k)
    ...

# --- 695 Li & Liao families (L=0 symmetric plane): name, v1, v2, T, T_star, Lf ---
ll = load_ll_orbits()
```

Cost note: one `analyse_orbit`/`compute_monodromy` call is **one variational
integration of the 156-component extended state** — the expensive operation the
surrogate exists to avoid. Newton refinement is a few more. Budget your
label-generation in these units.

---

## 4. Data inventory (what's already on disk)

| Source | What it gives the ML | Where |
| --- | --- | --- |
| 500×500 BHH RPF maps, 9 L values | `(a,c)` → `-log10(d_min)` (periodicity score). Input for the **discovery** selection function. | `mini_results/scan_bhh_L*_500x500.npz` (keys: `row_vals`=a, `col_vals`=c, `rpf_map`, `completed_rows`) |
| Syzygy regime census | `(a,c)` → ordered-domain vs chaotic-sea label. Cheap pre-classifier; explains where peaks are detectable. | `bk_bias_census.npz` (+ `bk_bias.py`) |
| Continuation families | `(a,c,L)` → `λ_max`, `n_unstable`, `E` along family curves — **the cleanest stability labels.** | `continuation_family_*.json`, `robust_continue.json`, `continue_borderline.json`, `pd_branch_v2.json` |
| Campaign classification | 100 distinct orbits with class + word + stability | `classify_candidates*.json`, `verify_shortlist.json` |
| Seed catalogues | 75 Jankovic (`ALL_ORBITS`) + 695 LL (`ll_orbits.json`) + Suvakov named | in-package |

**To generate fresh labels** (when the above runs thin): sweep `(a,c,L)`, call
`newton_refine_bhh` then `analyse_orbit`, store `(a,c,L,λ_max,stable,word,E)`.
Continuation (`continuation.trace_family`) is the *cheapest* label factory — it
walks a family producing one labelled point per step without re-searching.

Loading a scan map:
```python
d = np.load("mini_results/scan_bhh_L0.8_500x500.npz")
A, C, RPF = d["row_vals"], d["col_vals"], d["rpf_map"]   # RPF.shape == (500, 500)
# RPF[i,j] = -log10(d_min) at (a=A[i], c=C[j]); high = near-periodic. NaN = skipped.
```

---

## 5. Track A — stability surrogate (recommended first)

**Dataset schema** (one row per refined orbit):

```
features:  a, c, L           (+ derived: a*c, L - a*c, sign(c))
labels:    y_reg = log10(lam_max)        # regression target
           y_cls = (lam_max < 1.001)     # binary stable/unstable
meta:      word (b^k), E, T, source, converged
```

**Models:** `sklearn.gaussian_process.GaussianProcessRegressor` (Matérn kernel)
for `y_reg` with calibrated uncertainty for the acquisition step; XGBoost/
LightGBM as a comparator and for the `y_cls` coarse classifier. Start with the
GP — its uncertainty is what the active-learning loop needs.

**Active-learning loop** (the actual deliverable):

```
seed with catalogue labels
repeat under a fixed budget of N monodromy evaluations:
    fit GP on labelled set
    score a candidate pool (grid or Sobol over (a,c,L))
    pick argmax acquisition  (uncertainty, or expected stability)
    label it: newton_refine_bhh -> analyse_orbit   # <-- the one expensive call
    append to labelled set
compare: stable-orbits-found vs a uniform grid spending the same N evaluations
```

The refuted close-approach result showed 1–2 orders of magnitude of scatter in
λ_max for nearby orbits — real signal for a flexible model to learn, but also
the reason labels must stay classical and the surrogate stays *triage-grade*.

---

## 6. Track B — discovery selection function (lower priority now)

Learn which regions of the `(a,c)` / `(vx,vy)` plane yield detectable periodic
orbits, to propose seeds where no catalogue covers, then hand them to
continuation. Labels come free from the 500×500 RPF maps (Section 4) and the
syzygy census (ordered vs chaos). Model: GP Bayesian optimisation for the
fine selection; gradient-boosted trees for the coarse ordered-vs-chaos gate.
Over millions of scan points the GP needs a sparse/inducing-point variant
(gpytorch).

**Why it's lower value today:** there's a seed *backlog* (75 + 695 known
families), not seed scarcity, and continuation already supersedes scanning for
the high-value work. Peak-sharpness also caps the gain — you still must land on
needle-thin RPF peaks. Revisit once known families are exhausted and the goal is
genuinely *novel* topologies.

---

## 7. Deferred — deep-learning track (a deliberate learning goal)

DL is the right tool only when switching from 2–3 parameters to learning from
the **raw dynamics**: full trajectory time series, shape-sphere paths, or
symbolic syzygy words → 1D-CNN / small transformer (sequences) or
autoencoder/contrastive model (embeddings). Payoff: a similarity space where
families cluster, novel orbits are outliers, and "what's structurally near the
stable orbit?" becomes askable (representation-learning phase). Keep it on the
roadmap as a research-extension thread and an upskilling goal — don't let it
displace the classical core that does the near-term work.

---

## 8. Suggested new-repo layout

```
threebody-ml/
  pyproject.toml          # depends on threebody-physics
  data/
    build_labels.py       # (a,c,L) -> analyse_orbit -> labelled parquet/csv
    load_scans.py         # read mini_results/*.npz into feature/label arrays
  surrogate/
    gp.py  trees.py       # models
    active_loop.py        # the train -> propose -> label -> retrain loop
  baselines/
    uniform_grid.py       # the comparison arm
  experiments/
    exp01_surrogate_vs_grid.py
  results/                # metrics, plots (gitignored data, tracked reports)
```

---

## 9. Memory-safety (do not skip — this bit the project twice)

Label generation and active-learning loops launch the same parallel/integration
machinery that twice OOM-killed a 16 GB machine (see `docs/memory_incident.md`).

- **Import the guardrail.** Reuse `memguard` from the physics package in any
  parallel ML job: `import memguard; memguard.install()`, and size pools with
  `memguard.safe_worker_count()` — never `mp.cpu_count()`.
- **Never run two heavy multiprocessing jobs concurrently.** Each watchdog only
  sees its own process tree; two jobs can still exhaust RAM. Serialise them.
- Tune via env: `THREEBODY_MAX_WORKERS`, `THREEBODY_MEM_LIMIT_GB`.
- An active-learning loop is naturally serial (one expensive label at a time) —
  keep it that way; parallelise only the candidate-pool scoring, which is cheap.

---

## 10. Decision log & open gates

- **Method:** classical (GP / gradient boosting) for load-bearing tasks; DL only
  on raw-dynamics representation learning. (Decided 2026-06-16.)
- **Primary target:** stability surrogate as triage (Track A) over the discovery
  selection function (Track B) — Track B's value is gated on exhausting the seed
  backlog.
- **Success metric:** stable-orbits- (or orbits-) found per CPU-hour vs uniform
  grid. Everything is built to make this measurable.
- **Open gate:** Korn / CERN-academic feedback (pending) may redirect the aim —
  don't over-invest in ML infrastructure before it lands. The continuation
  science (more stable windows, bifurcation structure, the L=0 connection) is
  still the higher-value vein and needs no ML.

See project memory `project_ml_roadmap.md` for the full reasoning behind these.
