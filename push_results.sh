#!/bin/bash
# Wait for scans to finish, then push all results to GitHub.
# Run this in a separate terminal tab while run_batch.sh is running.

echo "Waiting for scans to finish..."
while pgrep -f "main.py" > /dev/null; do
    sleep 60
done

echo "Scans finished. Pushing results..."
git add -f scan_bhh_L*_500x500.npz
git add -f scan_bhh_L*_500x500_candidates.json
git add -f scan_bhh_L*_500x500_candidates_plots/
git add -f scan_bhh_L*_200x200.npz
git add -f scan_bhh_L*_200x200_candidates.json
git add -f scan_bhh_L*_200x200_candidates_plots/
git commit -m "Add BHH scan results: 500x500 at 9 Jankovic L values + 200x200 at L=0.5, 1.0, 1.5"
git push
echo "Done."
