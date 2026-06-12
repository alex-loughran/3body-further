"""Three-Body Periodic Orbit Hunter — main entry point.

Usage:
    python main.py validate                  Smoke-test the figure-eight orbit
    python main.py scan-symmetric [N]        Run a symmetric (vx, vy) scan at NxN
    python main.py scan-bhh L [N]            Run a BHH (a, c) scan at angular momentum L
    python main.py floquet vx vy T           Floquet analysis for a symmetric orbit
    python main.py refine-symmetric vx vy T  Newton-refine a symmetric orbit
    python main.py refine-bhh a c L T        Newton-refine a BHH orbit
    python main.py process-scan FILE sym     Process symmetric scan candidates
    python main.py process-scan FILE bhh L   Process BHH scan candidates
    python main.py campaign FILE [L]         Run a campaign from a configs/*.toml
                                             file (scan + process; optional single L)
    python main.py catalogue                 Floquet analysis of all known orbits
    python main.py compare-floquet vx vy T   Compare standard vs compound Floquet
"""

import sys
import numpy as np
from three_body import (
    build_state_symmetric,
    return_proximity,
    compute_energy,
    read_free_group_word,
)
from scanner import scan_parallel
from parametrisations import SymmetricBuilder, BHHBuilder


def validate():
    """Validate the figure-eight orbit as a smoke test."""
    vx, vy = 0.347111, 0.532727
    T_max = 8.0

    state0 = build_state_symmetric(vx, vy)
    E = compute_energy(state0)
    d_min, T, sol = return_proximity(state0, T_max, n_samples=3000)

    print("=== Figure-eight orbit validation ===")
    print(f"  vx = {vx}, vy = {vy}")
    print(f"  Energy E = {E:.10f}")
    print(f"  d_min = {d_min:.6e}  (< 1e-4 = periodic)")
    print(f"  Period T = {T:.10f}")

    word = read_free_group_word(sol, T)
    print(f"  Free group word: {word}")
    print()

    if d_min < 1e-4:
        print("  PASS")
    else:
        print("  FAIL")


def scan_symmetric(n_grid=200, T_max=8.0, n_samples=800):
    """Scan the symmetric (vx, vy) plane."""
    vx = np.linspace(0.01, 0.6, n_grid)
    vy = np.linspace(0.01, 0.6, n_grid)
    save_path = f"scan_symmetric_{n_grid}x{n_grid}.npz"

    print(f"=== Symmetric scan {n_grid}x{n_grid} ===")
    rpf = scan_parallel(vx, vy, SymmetricBuilder(),
                        T_max=T_max, n_samples=n_samples,
                        save_path=save_path, verbose=True)
    print(f"  Saved to {save_path}")


def scan_bhh(L, n_grid=200, T_max=16.0, n_samples=800):
    """Scan the BHH (a, c) plane at fixed angular momentum L."""
    a_vals = np.linspace(0.05, 0.6, n_grid)
    c_vals = np.linspace(-3.5, 4.0, n_grid)
    save_path = f"scan_bhh_L{L}_{n_grid}x{n_grid}.npz"

    print(f"=== BHH scan {n_grid}x{n_grid} at L={L} ===")
    rpf = scan_parallel(a_vals, c_vals, BHHBuilder(L=L),
                        T_max=T_max, n_samples=n_samples,
                        save_path=save_path, verbose=True)
    print(f"  Saved to {save_path}")


def floquet_cmd(vx, vy, T):
    """Run Floquet analysis on a symmetric orbit."""
    from floquet import analyse_orbit

    state0 = build_state_symmetric(vx, vy)
    E = compute_energy(state0)

    print(f"=== Floquet analysis ===")
    print(f"  vx = {vx}, vy = {vy}, T = {T}")
    print(f"  Energy E = {E:.10f}")
    print()

    result = analyse_orbit(state0, T, verbose=True)
    mults = result["multipliers"]
    stab = result["stability"]

    print()
    print("Floquet multipliers:")
    for i, m in enumerate(mults):
        print(f"  λ_{i+1} = {m.real:+.10f} {m.imag:+.10f}i   |λ| = {abs(m):.10f}")
    print()
    print(f"Stable: {stab['is_stable']}, unit eigenvalues: {stab['n_unit']}, "
          f"det(M) = {stab['determinant']:.8f}")


def refine_symmetric_cmd(vx, vy, T):
    """Newton-refine a symmetric orbit and print results."""
    from floquet import newton_refine_symmetric, analyse_orbit

    print(f"=== Refining symmetric orbit ===")
    print(f"  Initial: vx={vx}, vy={vy}, T={T}")
    print()

    vx_r, vy_r, T_r, converged, info = newton_refine_symmetric(
        vx, vy, T, verbose=True)

    print()
    print(f"  Refined: vx={vx_r:.15f}, vy={vy_r:.15f}, T={T_r:.12f}")
    print(f"  Converged: {converged}")

    if converged:
        state0 = build_state_symmetric(vx_r, vy_r)
        E = compute_energy(state0)
        print(f"  Energy: {E:.10f}")
        print()
        print("  Running Floquet analysis on refined orbit...")
        result = analyse_orbit(state0, T_r, verbose=True)
        stab = result["stability"]
        print(f"  Stable: {stab['is_stable']}, det(M) = {stab['determinant']:.8f}")


def refine_bhh_cmd(a, c, L, T):
    """Newton-refine a BHH orbit and print results."""
    from floquet import newton_refine_bhh, analyse_orbit
    from three_body import initial_conditions_from_params

    print(f"=== Refining BHH orbit ===")
    print(f"  Initial: a={a}, c={c}, L={L}, T={T}")
    print()

    a_r, c_r, T_r, converged, info = newton_refine_bhh(
        a, c, L, T, verbose=True)

    print()
    print(f"  Refined: a={a_r:.15f}, c={c_r:.15f}, T={T_r:.12f}")
    print(f"  Converged: {converged}")

    if converged:
        state0 = initial_conditions_from_params(a_r, c_r, L)
        E = compute_energy(state0)
        print(f"  Energy: {E:.10f}")


def catalogue_cmd():
    """Run Floquet analysis on all known orbits and save catalogue."""
    import json
    import time
    from three_body import (
        initial_conditions_from_params, integrate_orbit,
        read_free_group_word, SUVAKOV_TABLE1, SUVAKOV_TABLE2, ALL_ORBITS,
    )
    from floquet import newton_refine_symmetric, newton_refine_bhh, analyse_orbit

    results = []
    failures = []

    def process(name, params, T_guess, parametrisation, L=0.0):
        try:
            if parametrisation == "symmetric":
                vx_r, vy_r, T_r, conv, info = newton_refine_symmetric(
                    params[0], params[1], T_guess, max_iter=30)
                if not conv:
                    return None, "Newton did not converge"
                state0 = build_state_symmetric(vx_r, vy_r)
                params_r = [vx_r, vy_r]
            else:
                a_r, c_r, T_r, conv, info = newton_refine_bhh(
                    params[0], params[1], L, T_guess, max_iter=30)
                if not conv:
                    return None, "Newton did not converge"
                state0 = initial_conditions_from_params(a_r, c_r, L)
                params_r = [a_r, c_r]

            result = analyse_orbit(state0, T_r, verbose=False)
            sol = integrate_orbit(state0, T_r)
            word = read_free_group_word(sol, T_r)
            E = compute_energy(state0)
            stab = result["stability"]

            return {
                "name": name, "parametrisation": parametrisation,
                "L": L, "params": params_r, "T": T_r, "E": E,
                "word": word, "is_stable": stab["is_stable"],
                "max_instability": stab["max_instability"],
                "determinant": stab["determinant"],
                "n_unit": stab["n_unit"], "valid": result["valid"],
                "multiplier_magnitudes": sorted(
                    float(abs(m)) for m in result["multipliers"]),
            }, None
        except Exception as e:
            return None, str(e)

    t0 = time.time()

    # Figure-eight
    print("Processing figure-eight...")
    entry, err = process("figure-eight",
        (0.3471128135672417, 0.532726851767674), 6.3250, "symmetric")
    if entry: results.append(entry)
    else: failures.append(("figure-eight", err))

    # Suvakov Table 1
    for name, vx, vy, T in SUVAKOV_TABLE1:
        print(f"Processing {name}...")
        entry, err = process(name, (vx, vy), T, "symmetric")
        if entry: results.append(entry)
        else: failures.append((name, err))

    # Suvakov Table 2
    for name, vx, vy, T, k in SUVAKOV_TABLE2:
        print(f"Processing satellite {name}...")
        entry, err = process(f"satellite {name}", (vx, vy), T, "symmetric")
        if entry: results.append(entry)
        else: failures.append((f"satellite {name}", err))

    # BHH orbits
    for nr, L, a, c, T, k in ALL_ORBITS:
        print(f"Processing Jankovic #{nr} (L={L})...")
        entry, err = process(f"Jankovic #{nr}", (a, c), T, "bhh", L)
        if entry: results.append(entry)
        else: failures.append((f"Jankovic #{nr}", err))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Succeeded: {len(results)}, Failed: {len(failures)}")
    for name, err in failures:
        print(f"  FAIL: {name}: {err}")

    # Save
    def ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        raise TypeError(f"Cannot serialise {type(obj)}")

    with open("floquet_catalogue.json", "w") as f:
        json.dump(results, f, indent=2, default=ser)
    print(f"Saved to floquet_catalogue.json")

    # Summary table
    print(f"\n{'Name':<30} {'L':>5} {'T':>10} {'E':>10} {'Word':<15} "
          f"{'Stab':>4} {'λ_max':>8} {'det':>8}")
    print("-" * 100)
    for r in results:
        w = r["word"][:13] + ".." if len(r["word"]) > 15 else r["word"]
        s = "Y" if r["is_stable"] else "N"
        print(f"{r['name']:<30} {r['L']:>5.2f} {r['T']:>10.4f} "
              f"{r['E']:>10.4f} {w:<15} {s:>4} "
              f"{r['max_instability']:>8.4f} {r['determinant']:>8.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "validate":
        validate()
    elif cmd == "scan-symmetric":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        scan_symmetric(n_grid=n)
    elif cmd == "scan-bhh":
        if len(sys.argv) < 3:
            print("Usage: python main.py scan-bhh <L> [n_grid]")
            sys.exit(1)
        L = float(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        scan_bhh(L, n_grid=n)
    elif cmd == "floquet":
        if len(sys.argv) < 5:
            print("Usage: python main.py floquet <vx> <vy> <T>")
            sys.exit(1)
        floquet_cmd(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif cmd == "refine-symmetric":
        if len(sys.argv) < 5:
            print("Usage: python main.py refine-symmetric <vx> <vy> <T>")
            sys.exit(1)
        refine_symmetric_cmd(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif cmd == "refine-bhh":
        if len(sys.argv) < 6:
            print("Usage: python main.py refine-bhh <a> <c> <L> <T>")
            sys.exit(1)
        refine_bhh_cmd(float(sys.argv[2]), float(sys.argv[3]),
                       float(sys.argv[4]), float(sys.argv[5]))
    elif cmd == "process-scan":
        from pipeline import process_scan
        if len(sys.argv) < 4:
            print("Usage: python main.py process-scan <scan.npz> symmetric [threshold]")
            print("       python main.py process-scan <scan.npz> bhh <L> [threshold]")
            sys.exit(1)
        scan_file = sys.argv[2]
        ptype = sys.argv[3]
        if ptype in ("symmetric", "sym"):
            thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
            out = scan_file.replace(".npz", "_candidates.json")
            process_scan(scan_file, "symmetric", threshold=thresh, output_path=out)
        elif ptype == "bhh":
            if len(sys.argv) < 5:
                print("Usage: python main.py process-scan <scan.npz> bhh <L> [threshold]")
                sys.exit(1)
            L = float(sys.argv[4])
            thresh = float(sys.argv[5]) if len(sys.argv) > 5 else 3.5
            out = scan_file.replace(".npz", "_candidates.json")
            process_scan(scan_file, "bhh", L=L, threshold=thresh, output_path=out)
        else:
            print(f"Unknown parametrisation: {ptype}")
            sys.exit(1)
    elif cmd == "campaign":
        from config import run_campaign
        if len(sys.argv) < 3:
            print("Usage: python main.py campaign <config.toml> [L]")
            sys.exit(1)
        L_only = float(sys.argv[3]) if len(sys.argv) > 3 else None
        run_campaign(sys.argv[2], L_only=L_only)
    elif cmd == "catalogue":
        catalogue_cmd()
    elif cmd == "compare-floquet":
        if len(sys.argv) < 5:
            print("Usage: python main.py compare-floquet <vx> <vy> <T>")
            sys.exit(1)
        from compound import compare_floquet_methods
        vx, vy, T = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        state0 = build_state_symmetric(vx, vy)
        E = compute_energy(state0)
        print(f"Orbit: vx={vx}, vy={vy}, T={T}, E={E:.10f}\n")
        compare_floquet_methods(state0, T, verbose=True)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
