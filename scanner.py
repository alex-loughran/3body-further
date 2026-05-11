"""Parametrisation-agnostic parallel scanner for periodic orbit hunting.

The scanner knows nothing about physics or coordinate systems. It takes:
  - A picklable callable that maps parameter tuples to state vectors (or None)
  - Grid values for a 2D parameter space
  - Integration settings

It dispatches RPF evaluations across multiple cores and saves results
incrementally so crashes don't lose progress.
"""

import multiprocessing as mp
import sys
import time
import numpy as np
from pathlib import Path

from three_body import return_proximity

# Use 'spawn' on Linux for container/AWS compatibility.
# macOS defaults to 'spawn' already; 'fork' can be unsafe in containers.
if sys.platform == "linux" and mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")


# ---------------------------------------------------------------------------
# Per-point worker — top-level function, receives builder + params
# ---------------------------------------------------------------------------

def _evaluate_point(args):
    """Evaluate RPF at a single grid point.

    args: (i, j, row_val, col_val, state_builder, T_max, n_samples, t_min_frac)

    The state_builder is called inside the worker so only parameters
    cross the process boundary, not full state vectors.

    Returns (i, j, -log10(d_min)) or (i, j, nan) on failure/skip.
    """
    i, j, row_val, col_val, state_builder, T_max, n_samples, t_min_frac = args
    state0 = state_builder((row_val, col_val))
    if state0 is None:
        return (i, j, np.nan)
    try:
        d_min, _, _ = return_proximity(state0, T_max,
                                       t_min_frac=t_min_frac,
                                       n_samples=n_samples)
        return (i, j, -np.log10(max(d_min, 1e-15)))
    except (RuntimeError, FloatingPointError):
        # RuntimeError: integration failure (near-collision, step size too small)
        # FloatingPointError: overflow/underflow in force computation
        return (i, j, np.nan)


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_parallel(row_vals, col_vals, state_builder, T_max=8.0,
                  n_samples=800, t_min_frac=0.15, n_workers=None,
                  save_path=None, save_every=10, resume=True, verbose=True):
    """Scan a 2D parameter grid in parallel, computing RPF at each point.

    Parameters
    ----------
    row_vals : 1D array
        Values for the first parameter (rows of the output map).
    col_vals : 1D array
        Values for the second parameter (columns of the output map).
    state_builder : picklable callable
        Maps (row_val, col_val) -> 12-component state vector, or None to skip.
        Must be picklable (use a class with __call__, not a closure).
    T_max : float
        Maximum integration time.
    n_samples : int
        Number of time samples for RPF coarse search.
    t_min_frac : float
        Skip early times (fraction of T_max) to avoid trivial self-match.
    n_workers : int or None
        Number of parallel workers. None = cpu_count().
    save_path : str or None
        Path to .npz file for incremental saves.
    save_every : int
        Save after this many rows complete.
    resume : bool
        If True and save_path exists, load completed rows and skip them.
    verbose : bool
        Print progress.

    Returns
    -------
    rpf_map : (len(row_vals), len(col_vals)) array
        Contains -log10(d_min) at each grid point. NaN for skipped/failed points.
    """
    n_rows = len(row_vals)
    n_cols = len(col_vals)
    rpf_map = np.full((n_rows, n_cols), np.nan)
    completed_rows = set()

    # Resume from partial results if available
    if resume and save_path and Path(save_path).exists():
        try:
            prev = np.load(save_path)
            rpf_map = prev["rpf_map"].copy()
            completed_rows = set(int(x) for x in prev.get("completed_rows", []))
            if verbose:
                print(f"Resumed: {len(completed_rows)}/{n_rows} rows already done")
        except (KeyError, ValueError, OSError) as e:
            if verbose:
                print(f"Could not resume from saved file ({e}), starting fresh")

    if n_workers is None:
        n_workers = mp.cpu_count()

    # Build argument tuples — just indices + parameter values, no heavy data
    all_args = []
    for i, row_val in enumerate(row_vals):
        if i in completed_rows:
            continue
        for j, col_val in enumerate(col_vals):
            all_args.append((i, j, row_val, col_val, state_builder,
                             T_max, n_samples, t_min_frac))

    total_points = len(all_args)
    if total_points == 0:
        if verbose:
            print("  All rows already completed")
        return rpf_map

    # Track how many points remain per row to detect row completion
    pending_per_row = {}
    for i in range(n_rows):
        if i not in completed_rows:
            pending_per_row[i] = n_cols

    if verbose:
        print(f"  {total_points} points to evaluate "
              f"({n_rows - len(completed_rows)} rows, {n_workers} workers)")

    t_start = time.time()
    points_done = 0
    rows_since_save = 0

    with mp.Pool(n_workers) as pool:
        for i, j, rpf_val in pool.imap_unordered(_evaluate_point, all_args):
            rpf_map[i, j] = rpf_val
            points_done += 1
            pending_per_row[i] -= 1

            if pending_per_row[i] == 0:
                completed_rows.add(i)
                rows_since_save += 1

                if save_path and rows_since_save >= save_every:
                    np.savez(save_path,
                             row_vals=row_vals, col_vals=col_vals,
                             rpf_map=rpf_map,
                             completed_rows=np.array(sorted(completed_rows)))
                    rows_since_save = 0

            if verbose and points_done % max(1, total_points // 100) == 0:
                elapsed = time.time() - t_start
                rate = points_done / elapsed
                remaining = (total_points - points_done) / rate if rate > 0 else 0
                print(f"\r  {points_done}/{total_points} points "
                      f"({100 * points_done / total_points:.0f}%) "
                      f"| {len(completed_rows)}/{n_rows} rows "
                      f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s left]",
                      end="", flush=True)

    if verbose:
        total = time.time() - t_start
        print(f"\n  Done in {total:.1f}s ({n_workers} workers)")

    # Final save
    if save_path:
        np.savez(save_path,
                 row_vals=row_vals, col_vals=col_vals,
                 rpf_map=rpf_map,
                 completed_rows=np.array(sorted(completed_rows)))

    return rpf_map


# ---------------------------------------------------------------------------
# Adaptive refinement scan
# ---------------------------------------------------------------------------

def adaptive_scan(coarse_path, state_builder, n_refine=200, margin=2,
                  threshold=3.0, T_max=8.0, n_samples=800, t_min_frac=0.15,
                  save_dir=None, verbose=True):
    """Run fine-resolution scans around peaks found in a coarse scan.

    This is a ZOOM tool, not a replacement for a full uniform scan. It
    takes an existing coarse scan, identifies peaks above threshold,
    and re-scans each peak's neighbourhood at higher resolution. This
    improves the Newton-Raphson starting guess but will NOT find orbits
    that the coarse scan missed entirely.

    Intended workflow:
      1. Run full uniform scan at target resolution (e.g. 1000x1000)
      2. Run process-scan to extract and refine candidates
      3. Optionally run refine-scan for higher-resolution RPF maps
         around the best peaks

    Parameters
    ----------
    coarse_path : str -- path to coarse scan .npz
    state_builder : callable -- same builder used for the coarse scan
    n_refine : int -- grid resolution for each refined region
    margin : int -- number of coarse grid cells to extend around each peak
    threshold : float -- RPF threshold for peak detection
    T_max, n_samples, t_min_frac : integration settings
    save_dir : str or None -- directory to save refined scans
    verbose : bool

    Returns
    -------
    regions : list of dicts, each with keys:
        peak, row_range, col_range, rpf_map, save_path
    """
    from three_body import find_candidates
    from pathlib import Path

    data = np.load(coarse_path)
    if "row_vals" in data:
        row_vals = data["row_vals"]
        col_vals = data["col_vals"]
    elif "vx_vals" in data:
        row_vals = data["vx_vals"]
        col_vals = data["vy_vals"]
    else:
        row_vals = data["a_vals"]
        col_vals = data["c_vals"]
    rpf_map = data["rpf_map"]

    candidates = find_candidates(row_vals, col_vals, rpf_map,
                                 threshold=threshold, min_separation=0.01)
    if verbose:
        print(f"Found {len(candidates)} peaks above threshold {threshold}")

    if not candidates:
        return []

    dr = row_vals[1] - row_vals[0] if len(row_vals) > 1 else 0.01
    dc = col_vals[1] - col_vals[0] if len(col_vals) > 1 else 0.01

    regions = []
    for idx, (r_peak, c_peak, rpf_val) in enumerate(candidates):
        r_lo = r_peak - margin * dr
        r_hi = r_peak + margin * dr
        c_lo = c_peak - margin * dc
        c_hi = c_peak + margin * dc

        fine_rows = np.linspace(r_lo, r_hi, n_refine)
        fine_cols = np.linspace(c_lo, c_hi, n_refine)

        if save_dir:
            Path(save_dir).mkdir(exist_ok=True)
            sp = str(Path(save_dir) /
                     f"refined_{idx:03d}_r{r_peak:.4f}_c{c_peak:.4f}.npz")
        else:
            sp = None

        if verbose:
            print(f"\nRegion {idx + 1}: peak at ({r_peak:.4f}, {c_peak:.4f}), "
                  f"rpf={rpf_val:.2f}")
            print(f"  Zooming to [{r_lo:.4f}, {r_hi:.4f}] x "
                  f"[{c_lo:.4f}, {c_hi:.4f}] at {n_refine}x{n_refine}")

        fine_rpf = scan_parallel(
            fine_rows, fine_cols, state_builder,
            T_max=T_max, n_samples=n_samples, t_min_frac=t_min_frac,
            save_path=sp, verbose=verbose,
        )

        regions.append({
            "peak": (r_peak, c_peak),
            "row_range": (r_lo, r_hi),
            "col_range": (c_lo, c_hi),
            "rpf_map": fine_rpf,
            "save_path": sp,
        })

    return regions
