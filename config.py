"""Scan campaign configuration.

A campaign is one TOML file in configs/ specifying everything needed to
reproduce a set of scans: parametrisation, grid resolution and ranges,
integration settings, candidate threshold, and the L values to sweep.
Replaces the hardcoded parameters scattered through main.py — a scan run
from a config is reproducible from the file alone.

Uses stdlib tomllib (read-only TOML), no extra dependencies.

Example config:

    name = "bhh_500x500_jankovic"
    parametrisation = "bhh"
    n_grid = 500
    row_range = [0.05, 0.6]       # a  (vx for symmetric)
    col_range = [-3.5, 4.0]       # c  (vy for symmetric)
    T_max = 16.0
    n_samples = 800
    t_min_frac = 0.15
    threshold = 3.5
    L_values = [0.65, 0.7, 0.8]
"""

import tomllib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanConfig:
    name: str
    parametrisation: str        # 'symmetric' | 'bhh'
    n_grid: int
    row_range: tuple            # (min, max) of first param: a (bhh) or vx
    col_range: tuple            # (min, max) of second param: c (bhh) or vy
    T_max: float
    n_samples: int
    t_min_frac: float = 0.15
    threshold: float = 3.5
    L_values: tuple = (0.0,)

    @classmethod
    def from_toml(cls, path):
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        for key in ("row_range", "col_range", "L_values"):
            if key in raw:
                raw[key] = tuple(raw[key])
        cfg = cls(**raw)
        if cfg.parametrisation not in ("symmetric", "bhh"):
            raise ValueError(f"parametrisation must be 'symmetric' or 'bhh', "
                             f"got {cfg.parametrisation!r}")
        if cfg.parametrisation == "symmetric" and cfg.L_values != (0.0,):
            raise ValueError("symmetric parametrisation is L=0 only; "
                             "remove L_values from the config")
        return cfg

    def row_vals(self):
        return np.linspace(self.row_range[0], self.row_range[1], self.n_grid)

    def col_vals(self):
        return np.linspace(self.col_range[0], self.col_range[1], self.n_grid)

    def save_path(self, L=None):
        """Scan output filename — same convention as the main.py commands,
        so configs can resume scans started by hand."""
        if self.parametrisation == "symmetric":
            return f"scan_symmetric_{self.n_grid}x{self.n_grid}.npz"
        return f"scan_bhh_L{L}_{self.n_grid}x{self.n_grid}.npz"

    def builder(self, L=None):
        from parametrisations import SymmetricBuilder, BHHBuilder
        if self.parametrisation == "symmetric":
            return SymmetricBuilder()
        return BHHBuilder(L=L)

    def describe(self):
        lines = [
            f"Campaign: {self.name}",
            f"  parametrisation: {self.parametrisation}",
            f"  grid: {self.n_grid}x{self.n_grid}",
            f"  rows: [{self.row_range[0]}, {self.row_range[1]}]",
            f"  cols: [{self.col_range[0]}, {self.col_range[1]}]",
            f"  T_max={self.T_max}, n_samples={self.n_samples}, "
            f"t_min_frac={self.t_min_frac}",
            f"  threshold={self.threshold}",
        ]
        if self.parametrisation == "bhh":
            lines.append(f"  L values: {list(self.L_values)}")
        return "\n".join(lines)


def run_campaign(config_path, L_only=None, scan_only=False, verbose=True):
    """Run a full campaign from a config file: scan then process, per L.

    L_only restricts to a single L value from the config (must be listed).
    Returns {L: candidates_list} (empty dict if scan_only).
    """
    from scanner import scan_parallel

    cfg = ScanConfig.from_toml(config_path)
    if verbose:
        print(cfg.describe())

    L_values = cfg.L_values
    if L_only is not None:
        matches = [L for L in L_values if abs(L - L_only) < 1e-9]
        if not matches:
            raise ValueError(f"L={L_only} not in config L_values {list(L_values)}")
        L_values = matches

    all_results = {}
    for L in L_values:
        L_arg = None if cfg.parametrisation == "symmetric" else L
        save_path = cfg.save_path(L_arg)
        if verbose:
            print(f"\n=== Scan: {save_path} ===")
        scan_parallel(cfg.row_vals(), cfg.col_vals(), cfg.builder(L_arg),
                      T_max=cfg.T_max, n_samples=cfg.n_samples,
                      t_min_frac=cfg.t_min_frac,
                      save_path=save_path, verbose=verbose)
        if scan_only:
            continue

        from pipeline import process_scan
        out_path = save_path.replace(".npz", "_candidates.json")
        if verbose:
            print(f"\n=== Process: {out_path} ===")
        all_results[L] = process_scan(
            save_path, cfg.parametrisation, L=L_arg,
            threshold=cfg.threshold, T_max=cfg.T_max,
            output_path=out_path, verbose=verbose)
    return all_results
