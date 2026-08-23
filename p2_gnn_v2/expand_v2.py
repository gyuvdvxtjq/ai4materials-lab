"""P2 数据扩充 v2：三元电池体系结构拉取（目标再增1500+）"""
import os
import requests, json, time
from pymatgen.core import Structure

BASE = os.getenv("MP_BASE_URL", "http://localhost:8000").rstrip("/")
SYSTEMS = [
    "Li-Ni-Co-O", "Li-Mn-Co-O", "Li-Ni-Mn-O", "Li-Fe-Mn-O", "Li-Cu-O-F",
    "Li-Ti-O-F", "Li-V-Fe-O", "Na-Fe-Mn-O", "Na-Cr-Mn-O", "Na-Fe-P-O",
    "Li-Fe-Si-O", "Li-Mn-Si-O", "Li-Co-Si-O", "Li-Fe-B-O", "Li-Mn-B-O",
    "Li-Ni-S-O", "Li-Co-S-O", "Li-Cu-S-O", "Na-Co-O-F", "Na-Ni-O-F",
    "K-Fe-S-O", "K-Mn-S-O", "Li-Fe-F", "Li-Mn-F", "Li-Co-F", "Li-Ni-F",
    "Na-Cr-O-F", "Na-Ti-S-O", "Li-Ti-S-O", "Li-Nb-O", "Li-Mo-O",
    "Na-Mo-O", "K-W-O", "Li-W-O", "Na-W-O", "Li-Ta-O", "Na-Ta-O",
    "Ca-Fe-O", "Ca-Mn-O", "Ca-Co-O", "Mg-Fe-O", "Mg-Mn-O", "Mg-Co-O",
    "Zn-Fe-O", "Zn-Co-O", "Zn-Mn-O", "Cu-Fe-O", "Cu-Mn-O", "Cu-Co-O",
    "Ni-Fe-O", "Ni-Mn-O", "Co-Fe-O", "Co-Mn-O", "Fe-Cr-O", "Fe-V-O",
    "Mn-Cr-O", "Mn-V-O", "Ti-V-O", "Ti-Cr-O", "V-Cr-O", "V-Mo-O",
    "Ti-Nb-O", "Ti-Ta-O", "Nb-Ta-O", "Mo-W-O", "Cr-Mo-O", "Cr-W-O",
]

rows = json.load(open("dataset.json"))
seen = set(r["formula"] for r in rows)
base_n = len(rows)
for sys_ in SYSTEMS:
    try:
        r = requests.get(f"{BASE}/materials/search",
                         params={"chemsys": sys_, "limit": 100}, timeout=90)
        res = r.json().get("results", [])
    except Exception:
        continue
    kept = 0
    for m in res:
        bg, st = m.get("band_gap"), m.get("structure")
        if bg is None or st is None:
            continue
        try:
            struct = Structure.from_dict(st)
        except Exception:
            continue
        if not (2 <= len(struct) <= 30):
            continue
        key = m["formula_pretty"]
        if key in seen:
            continue
        seen.add(key)
        rows.append({"formula": key, "band_gap": float(bg), "structure": st})
        kept += 1
    if kept:
        print(f"{sys_:14s} kept={kept:3d} total={len(rows)}", flush=True)
    time.sleep(0.1)

json.dump(rows, open("dataset.json", "w"))
print(f"\nFINAL: {len(rows)} (+{len(rows)-base_n})")
