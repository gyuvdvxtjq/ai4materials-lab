"""P1 最终版：matminer magpie 描述符（材料信息学标准145维）+ 多模型对比
数据：8000条拉取 -> 去重后 6772 条 Materials Project 真实DFT数据
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
P1_DIR = Path(__file__).resolve().parent
featurizer = ElementProperty.from_preset("magpie")

def main():
    df = pd.read_csv(P1_DIR / "materials_final.csv")
    df["key"] = df["formula"].apply(lambda f: Composition(f).reduced_formula)
    df = df.drop_duplicates("key", keep="first")
    print(f"去重后: {len(df)}")

    comps = [Composition(f) for f in df["formula"]]
    print("featurizing with magpie ...")
    X = featurizer.featurize_many(comps)
    X = np.array(X, dtype=float)
    # 去掉NaN列
    keep = ~np.isnan(X).any(axis=0)
    X = X[:, keep]
    y = df["band_gap"].values.astype(float)
    print(f"magpie 特征维度（去NaN后）: {X.shape[1]}")

    bins = np.quantile(y, [0.25, 0.5, 0.75])
    strat = np.digitize(y, bins)
    Xtrva, Xte, ytrva, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    Xtr, Xva, ytr, yva = train_test_split(
        Xtrva, ytrva, test_size=0.125, random_state=42,
        stratify=np.digitize(ytrva, bins),
    )
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        results[name] = {
            "mae": float(mean_absolute_error(yva, pred)),
            "r2": float(r2_score(yva, pred)),
        }
        print(f"{name:22s} validation MAE={results[name]['mae']:.3f} eV  "
              f"R2={results[name]['r2']:.3f}")
    best_name = min(results, key=lambda name: results[name]["mae"])
    best = models[best_name]
    best.fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
    test_pred = best.predict(Xte)
    test_metrics = {"mae": float(mean_absolute_error(yte, test_pred)),
                    "r2": float(r2_score(yte, test_pred))}
    print(f"{best_name:22s} test MAE={test_metrics['mae']:.3f} eV  R2={test_metrics['r2']:.3f}")
    const = np.mean(np.abs(yte - ytrva.mean()))
    print(f"{'常数基线':22s} MAE={const:.3f} eV")

    # 按验证集 MAE 选择并保存最佳模型，避免文档与实际权重不一致。
    import joblib
    artifact = {
        "model": best,
        "model_name": best_name,
        "keep_cols": keep,
        "featurizer_preset": "magpie",
        "feature_dim": int(X.shape[1]),
        "random_state": 42,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, P1_DIR / "bandgap_magpie.joblib")
    pd.DataFrame(results).T.to_json(P1_DIR / "metrics_magpie.json", indent=2)
    print(f"saved {P1_DIR / 'bandgap_magpie.joblib'} ({best_name})")

if __name__ == "__main__":
    main()
