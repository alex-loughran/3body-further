"""Pre-compute free group words for all 695 Li & Liao orbits.

Saves to known_words_cache.json. Takes ~1 hour on first run.
Also cross-references LL orbits against our Suvakov catalogue
to identify which LL families correspond to which Suvakov names.
"""

import json
import time
from ll_data import load_ll_orbits
from three_body import build_state_symmetric, integrate_orbit, read_free_group_word
from pipeline import (
    canonical_word, detect_period_multiple,
    _load_word_cache, _save_word_cache, _known_words_cache,
    _get_known_symmetric_words,
)


def precompute():
    ll_orbits = load_ll_orbits()
    print(f"Li & Liao orbits: {len(ll_orbits)}")

    _load_word_cache()
    already = sum(1 for i in range(len(ll_orbits)) if ("ll", i) in _known_words_cache)
    print(f"Already cached: {already}")
    print(f"To compute: {len(ll_orbits) - already}")

    t0 = time.time()
    computed = 0

    for idx, (name, v1, v2, T, T_star, Lf) in enumerate(ll_orbits):
        key = ("ll", idx)
        if key in _known_words_cache:
            continue

        state0 = build_state_symmetric(v1, v2)
        try:
            sol = integrate_orbit(state0, T)
            word = read_free_group_word(sol, T)
        except Exception as e:
            word = f"?ERROR:{e}"

        _known_words_cache[key] = word
        computed += 1

        if computed % 10 == 0:
            elapsed = time.time() - t0
            rate = computed / elapsed
            remaining = (len(ll_orbits) - already - computed) / rate if rate > 0 else 0
            print(f"  {already + computed}/{len(ll_orbits)} "
                  f"({computed} new, {elapsed:.0f}s elapsed, ~{remaining:.0f}s left)")

        if computed % 50 == 0:
            _save_word_cache()

    _save_word_cache()
    elapsed = time.time() - t0
    print(f"\nDone: {computed} new words in {elapsed:.0f}s")
    print(f"Total cached: {len(_known_words_cache)} words")

    return ll_orbits


def cross_reference_against_suvakov():
    """Check which LL orbits match Suvakov named orbits."""
    print("\n" + "=" * 60)
    print("CROSS-REFERENCE: Li & Liao vs Suvakov")
    print("=" * 60)

    _load_word_cache()
    ll_orbits = load_ll_orbits()

    # Get Suvakov words
    suvakov = _get_known_symmetric_words(verbose=False)

    # Build LL word list
    ll_words = []
    for idx, (name, v1, v2, T, T_star, Lf) in enumerate(ll_orbits):
        key = ("ll", idx)
        word = _known_words_cache.get(key, "?")
        ll_words.append((name, word, T, Lf))

    # Match each Suvakov orbit against LL
    print(f"\nSuvakov orbits ({len(suvakov)}) vs LL families ({len(ll_words)}):\n")
    print(f"{'Suvakov name':<30} {'LL match':<12} {'Word match':>10} {'Period mult':>11}")
    print("-" * 70)

    matched_ll = set()
    for suv_name, suv_word, suv_T in suvakov:
        if "?" in suv_word:
            print(f"{suv_name:<30} {'?':<12} {'bad word':>10}")
            continue

        suv_canon = canonical_word(suv_word)
        best_match = None

        for ll_name, ll_word, ll_T, ll_Lf in ll_words:
            if "?" in ll_word:
                continue
            ll_canon = canonical_word(ll_word)

            # Exact match
            if suv_canon == ll_canon:
                best_match = (ll_name, "exact", 1)
                matched_ll.add(ll_name)
                break

            # Period multiple
            k = detect_period_multiple(suv_word, ll_word)
            if k is not None and k > 1:
                best_match = (ll_name, "period", k)
                matched_ll.add(ll_name)
                break

            k_rev = detect_period_multiple(ll_word, suv_word)
            if k_rev is not None and k_rev > 1:
                best_match = (ll_name, "sub-period", k_rev)
                matched_ll.add(ll_name)
                break

        if best_match:
            ll_name, match_type, k = best_match
            print(f"{suv_name:<30} {ll_name:<12} {match_type:>10} {k:>11}")
        else:
            print(f"{suv_name:<30} {'NO MATCH':<12}")

    # Summary
    n_matched = len(matched_ll)
    print(f"\n--- Summary ---")
    print(f"Suvakov orbits matched to LL: {sum(1 for s, w, t in suvakov if any(canonical_word(w) == canonical_word(ll_w) for _, ll_w, _, _ in ll_words if '?' not in ll_w and '?' not in w))}")
    print(f"LL families matched: {n_matched}")
    print(f"LL families with no Suvakov match: {len(ll_words) - n_matched}")

    # Check for LL words with errors
    n_errors = sum(1 for _, w, _, _ in ll_words if "?" in w)
    if n_errors:
        print(f"LL orbits with word errors: {n_errors}")
        for name, word, T, Lf in ll_words:
            if "?" in word:
                print(f"  {name}: {word[:50]}")


if __name__ == "__main__":
    precompute()
    cross_reference_against_suvakov()
