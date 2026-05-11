"""Fetch and parse Li & Liao 695 orbit data from the SJTU website.

Source: https://numericaltank.sjtu.edu.cn/three-body/three-body-movies.htm

The HTML uses <th> tags and encodes class names via GIF links like data/I.A-1.gif.
"""

import json
import re
import urllib.request

URL = "https://numericaltank.sjtu.edu.cn/three-body/three-body-movies.htm"


def fetch_and_parse():
    print(f"Fetching {URL}...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    print(f"  Got {len(html)} bytes")

    # Strategy: split by <tr> tags, then parse <th> cells in each row.
    # The orbit name is encoded in the GIF link: data/I.A-1.gif
    rows = html.split("<tr>")
    print(f"  Found {len(rows)} rows")

    orbits = []
    for row in rows:
        # Extract orbit name from GIF link
        gif_match = re.search(r'data/(I{1,2}\.[ABC]-\d+)\.gif', row)
        if not gif_match:
            continue
        name = gif_match.group(1)

        # Extract all numbers from <th> cells
        cells = re.findall(r'<th>\s*([\d.]+)\s*</th>', row)
        if len(cells) < 5:
            continue

        # cells should be: v1, v2, T, T*, Lf
        try:
            v1 = float(cells[0])
            v2 = float(cells[1])
            T = float(cells[2])
            T_star = float(cells[3])
            Lf = int(float(cells[4]))
        except (ValueError, IndexError):
            continue

        # Sanity checks
        if v1 > 1.0 or v2 > 1.0 or T < 1.0 or Lf < 1:
            continue

        orbits.append({
            "name": name,
            "v1": v1,
            "v2": v2,
            "T": T,
            "T_star": T_star,
            "Lf": Lf,
        })

    print(f"  Parsed {len(orbits)} orbits")

    # Save
    with open("ll_orbits.json", "w") as f:
        json.dump(orbits, f, indent=2)
    print(f"  Saved to ll_orbits.json")

    # Summary by class
    from collections import Counter
    classes = Counter(o["name"].split("-")[0] for o in orbits)
    for cls, count in sorted(classes.items()):
        print(f"    {cls}: {count} orbits")

    # Verify known orbits
    for o in orbits[:3]:
        print(f"  {o['name']}: v1={o['v1']}, v2={o['v2']}, T={o['T']}, Lf={o['Lf']}")

    return orbits


if __name__ == "__main__":
    fetch_and_parse()
