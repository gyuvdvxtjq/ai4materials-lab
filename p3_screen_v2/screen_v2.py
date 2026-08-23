"""P3 v2：用 magpie 特征的 P1 模型重新筛选（升级版闭环）
生成 -> magpie模型预测 -> MP验证 -> 榜单
"""
import itertools
import json
import joblib
import os
import requests
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

# 不在代码中提交内部服务地址；运行时通过环境变量注入。
BASE = os.getenv("MP_BASE_URL", "http://localhost:8000").rstrip("/")
featurizer = ElementProperty.from_preset("magpie")

CATIONS = ["Li", "Na"]
TMS = ["V", "Cr", "Mn", "Fe", "Co", "Ni", "Ti", "Cu"]
ANIONS = ["O2", "S2", "PO4"]
STOICH = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 2)]

def gen_candidates():
    cands = set()
    for a, tm, an, (x, y) in itertools.product(CATIONS, TMS, ANIONS, STOICH):
        try:
            formula = f"{a}{x if x>1 else ''}{tm}{y if y>1 else ''}{an}"
            c = Composition(formula)
            if len(c.elements) >= 3 and c.num_atoms <= 12:
                cands.add(c.reduced_formula)
        except Exception:
            continue
    return sorted(cands)

def predict(model_pack, formulas):
    comps = [Composition(f) for f in formulas]
    X = np.array(featurizer.featurize_many(comps), dtype=float)
    keep = model_pack["keep_cols"]
    X = X[:, keep]
    return model_pack["model"].predict(X)

def check_mp(formula):
    try:
        r = requests.get(f"{BASE}/materials/search",
                         params={"formula": formula, "limit": 3}, timeout=30)
        res = r.json().get("results", [])
        if not res:
            return {"in_mp": False}
        best = sorted(res, key=lambda x: x.get("energy_above_hull") or 9e9)[0]
        return {"in_mp": True, "mp_band_gap": best.get("band_gap"),
                "e_above_hull": best.get("energy_above_hull"),
                "is_stable": best.get("is_stable"), "n_polymorphs": len(res)}
    except Exception:
        return {"in_mp": None}

def main(lo=1.0, hi=3.0):
    print("=" * 60)
    print(f"P3 v2 · magpie模型筛选（目标带隙 {lo}-{hi} eV）")
    print("=" * 60)
    cands = gen_candidates()
    print(f"[1/3] 枚举 {len(cands)} 候选")
    pack = joblib.load("p1_opt/bandgap_magpie.joblib")
    preds = predict(pack, cands)
    print(f"[2/3] magpie模型预测完成")
    df = pd.DataFrame({"formula": cands, "pred_bandgap": np.round(preds, 3)})
    df = df[(df.pred_bandgap >= lo) & (df.pred_bandgap <= hi)]
    print(f"      落入窗口: {len(df)} 个")
    print(f"[3/3] MP 验证（前30）...")
    records = []
    for f in df.head(30)["formula"]:
        info = check_mp(f)
        records.append({"formula": f,
                        "pred_bandgap": df[df.formula == f].pred_bandgap.values[0],
                        **info})
    out = pd.DataFrame(records)
    out["known_struct"] = out["in_mp"].map({True: "已有结构", False: "全新候选", None: "查询失败"})
    out["mp_band_gap"] = out["mp_band_gap"].astype(float).round(3)
    show = out[["formula", "pred_bandgap", "known_struct", "mp_band_gap",
                "e_above_hull", "is_stable"]].sort_values("pred_bandgap")
    print(show.to_string(index=False))
    out.to_csv("p3_screen_v2/screening_results_v2.csv", index=False)
    # 误差统计（有MP数据的行）
    known = out[out.in_mp == True]
    if len(known):
        err = (known["pred_bandgap"] - known["mp_band_gap"]).abs().mean()
        print(f"\n与MP实测平均偏差: {err:.3f} eV ({len(known)} 个可对照样本)")
    print("saved p3_screen_v2/screening_results_v2.csv")

if __name__ == "__main__":
    main()
