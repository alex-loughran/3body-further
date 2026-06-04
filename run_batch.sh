#!/bin/bash
# BHH 500x500 scan batch — resume from L=0.7 processing onwards

set -e  # stop on first error

# L=0.7 — scan done, needs processing
python main.py process-scan scan_bhh_L0.7_500x500.npz bhh 0.7

# Remaining L values — scan + process
for L in 0.8 0.85 0.9 0.935 1.0 1.03 1.07; do
    python main.py scan-bhh $L 500
    python main.py process-scan scan_bhh_L${L}_500x500.npz bhh $L
done

echo "=== All scans complete ==="

# Push results to GitHub
echo "=== Pushing results to GitHub ==="
git add -f scan_bhh_L*_500x500.npz
git add -f scan_bhh_L*_500x500_candidates.json
git add -f scan_bhh_L*_500x500_candidates_plots/
git add -f scan_bhh_L*_200x200.npz
git add -f scan_bhh_L*_200x200_candidates.json
git add -f scan_bhh_L*_200x200_candidates_plots/
git commit -m "Add BHH scan results: 500x500 at 9 Jankovic L values + 200x200 at L=0.5, 1.0, 1.5"
git push
echo "=== Done ==="
