"""P1 优化版：万条数据 + 多模型对比 + 扩充特征
模型：RF / GradientBoosting / HistGradientBoosting
特征：在35维基础上扩充（原子半径/熔沸点/摩尔体积等 + 各元素占比）
"""
import ast
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.metrics import mean_absolute_error, r2_score

PROPS = ["X", "Z", "atomic_mass", "electron_affinity", "ionization_energy",
         "average_ionic_radius", "group", "row", "atomic_radius",
         "average_anion_radius", "average_cation_radius", "mendeleev_no",
         "iupac_ordering"]

def comp_features(formula: str) -> list:
    try:
        c = Composition(formula)
    except Exception:
        return [0.0] * (2 + len(PROPS) * 4 + 12)
    feats = [c.num_atoms, len(c.elements)]
    for prop in PROPS:
        vals = []
        for el, frac in c.items():
            try:
                vals.append(float(getattr(el, prop)) * frac)
            except Exception:
                pass
        v = np.array(vals, dtype=float)
        if len(v) == 0:
            feats += [0, 0, 0, 0]
        else:
            feats += [v.sum(), v.mean(), v.std() if len(v) > 1 else 0.0, v.max()]
    # 补充：极值类特征
    xs = [el.X for el in c.elements if el.X is not None]
    zs = [el.Z for el in c.elements]
    rows_ = [el.row for el in c.elements if el.row]
    groups = [el.group for el in c.elements if el.group]
    feats += [
        max(xs) - min(xs) if xs else 0.0,           # 电负性极差（离子性）
        max(zs) - min(zs) if zs else 0.0,           # 原子序数差
        max(rows_) - min(rows_) if rows_ else 0.0,  # 周期跨度
        max(groups) - min(groups) if groups else 0.0,  # 族跨度
        1.0 if "O" in [str(e) for e in c.elements] else 0.0,  # 是否氧化物
        len([e for e in c.elements if e.is_chalcogen]),          # 硫族元素数
        len([e for e in c.elements if e.is_transition_metal]),  # 过渡金属数
        len([e for e in c.elements if e.is_alkali]),             # 碱金属数
        len([e for e in c.elements if e.is_alkaline]),           # 碱土金属数
        len([e for e in c.elements if e.is_halogen]),            # 卤素数
        len([e for e in c.elements if e.is_metal]),              # 金属元素数
        len(c.elements) - len([e for e in c.elements if e.is_metal or e.is_metalloid]),  # 非金属数
    ]
    return feats

def main():
    df = pd.read_csv("p1_opt/materials_final.csv").dropna(subset=["band_gap", "formula"])
    df["key"] = df["formula"].apply(lambda f: Composition(f).reduced_formula)
    df = df.drop_duplicates("key", keep="first")
    print(f"去重后数据: {len(df)}")

    X = np.array([comp_features(f) for f in df["formula"]], dtype=float)
    y = df["band_gap"].values.astype(float)
    print(f"特征维度: {X.shape[1]}")

    bins = np.quantile(y, [0.25, 0.5, 0.75])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=np.digitize(y, bins))
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42,
                                              n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        r2 = r2_score(yte, pred)
        results[name] = {"mae": mae, "r2": r2}
        print(f"{name:22s} MAE={mae:.3f} eV  R2={r2:.3f}")
    const = np.mean(np.abs(yte - yte.mean()))
    print(f"{'常数基线':22s} MAE={const:.3f} eV")
    return results

if __name__ == "__main__":
    main()
