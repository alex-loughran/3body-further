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
