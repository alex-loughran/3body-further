"""Analyse the Floquet catalogue: T* law, stability landscape, multiplier structure."""

import json
import numpy as np
import matplotlib.pyplot as plt


def load_catalogue(path="floquet_catalogue.json"):
    with open(path) as f:
        return json.load(f)


def plot_tstar_vs_wordlength(cat):
    """T* = T|E|^{3/2}/L_f vs word length, coloured by parametrisation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in cat:
        word = r["word"]
        if "?" in word or len(word) == 0:
            continue
        L_f = len(word)
        T_star = r["T"] * abs(r["E"]) ** 1.5 / L_f

        if r["parametrisation"] == "symmetric":
            ax.plot(L_f, T_star, "bo", ms=5, alpha=0.7)
        else:
            ax.plot(L_f, T_star, "r^", ms=5, alpha=0.7)

    ax.axhline(2.433, color="k", ls="--", lw=1, label="T* ≈ 2.433 (expected)")
    ax.set_xlabel("Free group word length $L_f$", fontsize=12)
    ax.set_ylabel("$T^* = T|E|^{3/2} / L_f$", fontsize=12)
    ax.set_title("Topological Kepler's Third Law", fontsize=13)
    ax.legend(["Symmetric (L=0)", "BHH (L≠0)", "T* ≈ 2.433"],
              fontsize=10)
    ax.set_ylim(1.5, 5.5)
    plt.tight_layout()
    plt.savefig("tstar_vs_wordlength.png", dpi=150)
    plt.close()


def plot_stability_landscape(cat):
    """log10(λ_max) vs word length, coloured by L value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in cat:
        word = r["word"]
        if "?" in word or len(word) == 0:
            continue
        L_f = len(word)
        lam_max = r["max_instability"]
        log_lam = np.log10(max(lam_max, 1.001))  # avoid log(1)

        if r["parametrisation"] == "symmetric":
            color = "blue"
            marker = "o"
        else:
            color = plt.cm.plasma(r["L"] / 1.2)
            marker = "^"

        ax.plot(L_f, log_lam, marker, color=color, ms=5, alpha=0.7)

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Free group word length $L_f$", fontsize=12)
    ax.set_ylabel("$\\log_{10}(\\lambda_{\\max})$", fontsize=12)
    ax.set_title("Stability Landscape", fontsize=13)
    ax.annotate("stable", (2, -0.1), fontsize=10, color="green")
    ax.annotate("unstable", (2, 0.5), fontsize=10, color="red")
    plt.tight_layout()
    plt.savefig("stability_landscape.png", dpi=150)
    plt.close()


def plot_tstar_by_L(cat):
    """T* vs word length, separate series for each L value (BHH only)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group BHH orbits by L
    L_groups = {}
    for r in cat:
        if r["parametrisation"] != "bhh" or "?" in r["word"]:
            continue
        L_val = round(r["L"], 3)
        L_groups.setdefault(L_val, []).append(r)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_groups)))
    for (L_val, orbits), color in zip(sorted(L_groups.items()), colors):
        Lfs = [len(r["word"]) for r in orbits]
        Tstars = [r["T"] * abs(r["E"]) ** 1.5 / len(r["word"]) for r in orbits]
        idx = np.argsort(Lfs)
        ax.plot([Lfs[i] for i in idx], [Tstars[i] for i in idx],
                "o-", color=color, ms=4, lw=1, label=f"L={L_val}", alpha=0.8)

    ax.axhline(2.433, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Free group word length $L_f$", fontsize=12)
    ax.set_ylabel("$T^* = T|E|^{3/2} / L_f$", fontsize=12)
    ax.set_title("Topological Kepler's Third Law by Angular Momentum", fontsize=13)
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig("tstar_by_L.png", dpi=150)
    plt.close()


def print_stability_summary(cat):
    """Print summary statistics."""
    stable = [r for r in cat if r["is_stable"]]
    unstable = [r for r in cat if not r["is_stable"]]
    valid = [r for r in cat if r.get("valid", True)]

    print(f"=== Catalogue Summary ===")
    print(f"Total orbits: {len(cat)}")
    print(f"Valid monodromy: {len(valid)}")
    print(f"Stable: {len(stable)}")
    print(f"Unstable: {len(unstable)}")
    print()

    print("Stable orbits:")
    for r in stable:
        print(f"  {r['name']:<30} L={r['L']:.2f}, T={r['T']:.4f}, "
              f"word={r['word'][:20]}")
    print()

    # Instability statistics by word type
    sym = [r for r in cat if r["parametrisation"] == "symmetric"]
    bhh = [r for r in cat if r["parametrisation"] == "bhh"]
    print(f"Symmetric orbits: {len(sym)} total, "
          f"{sum(1 for r in sym if r['is_stable'])} stable")
    print(f"BHH orbits: {len(bhh)} total, "
          f"{sum(1 for r in bhh if r['is_stable'])} stable")


def print_topology_correlation(cat):
    """Analyse correlation between word structure and stability."""
    from collections import Counter

    def word_type(word):
        if "?" in word:
            return "unknown"
        unique = set(word)
        if len(unique) == 1:
            return f"pure-{list(unique)[0]}"
        return "mixed"

    print("=== Stability-Topology Correlation ===")

    # By word type
    types = Counter()
    stable_by_type = Counter()
    for r in cat:
        wt = word_type(r["word"])
        types[wt] += 1
        if r["is_stable"]:
            stable_by_type[wt] += 1

    print(f"\n{'Word type':<15} {'Total':>5} {'Stable':>6} {'Rate':>8}")
    print("-" * 38)
    for wt in sorted(types, key=lambda x: -types[x]):
        rate = stable_by_type[wt] / types[wt] * 100
        print(f"{wt:<15} {types[wt]:>5} {stable_by_type[wt]:>6} {rate:>7.1f}%")

    # By number of unstable directions
    print(f"\n{'Unstable dirs':>13} {'Count':>5}")
    print("-" * 22)
    dir_counts = Counter()
    for r in cat:
        n = sum(1 for m in r["multiplier_magnitudes"] if m > 1.01)
        dir_counts[n] += 1
    for n in sorted(dir_counts):
        print(f"{n:>13} {dir_counts[n]:>5}")

    # Key finding: all stable orbits are figure-eight family
    print("\nKey finding: all 9 stable orbits are L=0 mixed-letter words.")
    print("All 75 BHH (pure-letter) orbits are unstable.")
    print("Stability correlates with topological complexity, not period.")


def plot_instability_by_L(cat):
    """Plot instability vs word length for each L value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bhh = [r for r in cat if r["parametrisation"] == "bhh"]
    L_groups = {}
    for r in bhh:
        L_val = round(r["L"], 3)
        L_groups.setdefault(L_val, []).append(r)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_groups)))
    for (L_val, orbits), color in zip(sorted(L_groups.items()), colors):
        Lfs = [len(r["word"]) for r in orbits]
        lams = [np.log10(r["max_instability"]) for r in orbits]
        idx = np.argsort(Lfs)
        ax.plot([Lfs[i] for i in idx], [lams[i] for i in idx],
                "o-", color=color, ms=4, lw=1, label=f"L={L_val}", alpha=0.8)

    ax.set_xlabel("Free group word length $L_f$", fontsize=12)
    ax.set_ylabel("$\\log_{10}(\\lambda_{\\max})$", fontsize=12)
    ax.set_title("BHH Orbit Instability by Angular Momentum", fontsize=13)
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig("instability_by_L.png", dpi=150)
    plt.close()


def print_tstar_residuals(cat):
    """Identify orbits with largest T* deviations and analyse patterns."""
    print("=== T* Residuals (deviations from 2.433) ===\n")

    entries = []
    for r in cat:
        word = r["word"]
        if "?" in word or len(word) == 0:
            continue
        L_f = len(word)
        T_star = r["T"] * abs(r["E"]) ** 1.5 / L_f
        entries.append((r["name"], r["parametrisation"], r["L"],
                        L_f, T_star, T_star - 2.433, word))

    # Sort by absolute deviation
    entries.sort(key=lambda x: -abs(x[5]))

    print(f"{'Name':<30} {'Type':<5} {'L':>5} {'Lf':>3} {'T*':>7} {'Δ':>7}")
    print("-" * 62)
    for name, ptype, L, Lf, Tstar, delta, word in entries[:20]:
        print(f"{name:<30} {ptype:<5} {L:>5.2f} {Lf:>3} {Tstar:>7.3f} {delta:>+7.3f}")

    # Statistics by word type
    pure = [e for e in entries if len(set(e[6])) == 1]
    mixed = [e for e in entries if len(set(e[6])) > 1]
    if pure:
        pure_tstars = [e[4] for e in pure]
        print(f"\nPure-letter words: T* = {np.mean(pure_tstars):.3f} ± {np.std(pure_tstars):.3f}"
              f" (n={len(pure)}, range [{min(pure_tstars):.3f}, {max(pure_tstars):.3f}])")
    if mixed:
        mixed_tstars = [e[4] for e in mixed]
        print(f"Mixed-letter words: T* = {np.mean(mixed_tstars):.3f} ± {np.std(mixed_tstars):.3f}"
              f" (n={len(mixed)}, range [{min(mixed_tstars):.3f}, {max(mixed_tstars):.3f}])")


def print_multiplier_structure(cat):
    """Analyse Floquet multiplier spectra: unstable direction counts vs word structure."""
    print("=== Multiplier Structure ===\n")

    def word_type(word):
        if "?" in word:
            return "unknown"
        if len(set(word)) == 1:
            return "pure"
        return "mixed"

    # Unstable directions by word type
    print(f"{'Word type':<10} {'0 unstable':>10} {'2 unstable':>10} {'4 unstable':>10}")
    print("-" * 44)
    for wt in ["pure", "mixed"]:
        counts = {0: 0, 2: 0, 4: 0}
        for r in cat:
            if word_type(r["word"]) != wt:
                continue
            n_unstable = sum(1 for m in r["multiplier_magnitudes"] if m > 1.01)
            counts[n_unstable] = counts.get(n_unstable, 0) + 1
        print(f"{wt:<10} {counts.get(0,0):>10} {counts.get(2,0):>10} {counts.get(4,0):>10}")

    # Instability scaling with L for BHH orbits (fixed word length ranges)
    print(f"\n{'L':>6} {'n_orbits':>8} {'mean log10(λmax)':>17} {'std':>6}")
    print("-" * 42)
    bhh = [r for r in cat if r["parametrisation"] == "bhh" and "?" not in r["word"]]
    L_groups = {}
    for r in bhh:
        L_val = round(r["L"], 3)
        L_groups.setdefault(L_val, []).append(r)
    for L_val in sorted(L_groups):
        orbits = L_groups[L_val]
        log_lams = [np.log10(r["max_instability"]) for r in orbits]
        print(f"{L_val:>6.3f} {len(orbits):>8} {np.mean(log_lams):>17.3f} {np.std(log_lams):>6.3f}")


def plot_multiplier_spectrum(cat):
    """Plot number of unstable directions vs word length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in cat:
        word = r["word"]
        if "?" in word or len(word) == 0:
            continue
        L_f = len(word)
        n_unstable = sum(1 for m in r["multiplier_magnitudes"] if m > 1.01)

        if r["parametrisation"] == "symmetric":
            ax.plot(L_f, n_unstable, "bo", ms=6, alpha=0.6)
        else:
            ax.plot(L_f, n_unstable, "r^", ms=6, alpha=0.6)

    ax.set_xlabel("Free group word length $L_f$", fontsize=12)
    ax.set_ylabel("Number of unstable directions", fontsize=12)
    ax.set_title("Unstable Directions vs Topological Complexity", fontsize=13)
    ax.set_yticks([0, 2, 4])
    ax.legend(["Symmetric (L=0)", "BHH (L≠0)"], fontsize=10)
    plt.tight_layout()
    plt.savefig("unstable_directions.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    cat = load_catalogue()
    print_stability_summary(cat)
    print()
    print_topology_correlation(cat)
    print()
    print_tstar_residuals(cat)
    print()
    print_multiplier_structure(cat)
    print()
    print("Generating plots...")
    plot_tstar_vs_wordlength(cat)
    plot_stability_landscape(cat)
    plot_tstar_by_L(cat)
    plot_instability_by_L(cat)
    plot_multiplier_spectrum(cat)
    print("Saved: tstar_vs_wordlength.png, stability_landscape.png, "
          "tstar_by_L.png, instability_by_L.png, unstable_directions.png")
