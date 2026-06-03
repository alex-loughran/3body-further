"""Quick check: RPF distribution of candidates from a scan."""
import sys
from pipeline import load_scan, extract_candidates

scan_path = sys.argv[1] if len(sys.argv) > 1 else "scan_bhh_L1.0_200x200.npz"
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5

scan = load_scan(scan_path)
cands = extract_candidates(scan, threshold=threshold)

print(f"Scan: {scan_path}, threshold={threshold}, candidates={len(cands)}\n")
for i, c in enumerate(sorted(cands, key=lambda x: -x["rpf"])):
    print(f"  {i+1:3d}  rpf={c['rpf']:.2f}  params=({c['params'][0]:.6f}, {c['params'][1]:.6f})")
