"""P3 v3 前置：训练稳定性分类器（is_stable），供筛选闭环使用。

为什么需要它
------------
P3 v2 的筛选只看预测带隙，但「一个候选化学式能不能稳定存在」才是新材料
筛选的第一道判据（MatBench-Discovery 范式：预测形成能/稳定性 → 再看性质）。
materials_final.csv 自带 MP 的 `is_stable`（基于 energy_above_hull < 50 meV/atom），
正好用来训练一个组分层面的稳定性预筛器。

注意（诚实边界）
----------------
- is_stable 是 DFT+凸包判据的标签，本身有 DFT 误差和化学空间覆盖偏差；
- 组分特征预测不了结构依赖的稳定性（同一化学式不同构型 Ehull 可差很大）；
- 因此该模型用于**候选粗排**（过滤明显不稳的），不用于最终判定。

运行：python p3_screen_v2/train_stability.py
产物：p3_screen_v2/stability.joblib（模型，不入库）、
      p3_screen_v2/stability_metrics.json（5 折 CV 指标）
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

P3_DIR = Path(__file__).resolve().parent
P1_CSV = P3_DIR.parent / "p1_opt" / "materials_final.csv"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(P1_CSV)
    df["_key"] = df["formula"].apply(lambda f: Composition(f).reduced_formula)
    df = df.drop_duplicates("_key", keep="first").reset_index(drop=True)
    comps = [Composition(f) for f in df["formula"]]
    feat = ElementProperty.from_preset("magpie")
    X = np.array([feat.featurize(c) for c in comps], dtype=float)
    keep = ~np.isnan(X).any(axis=0)
    y = df["is_stable"].astype(int).values
    return df, X[:, keep], y, keep


def main() -> None:
    df, X, y, keep = load_data()
    n_stable = int(y.sum())
    print(f"n={len(df)}  stable={n_stable} ({n_stable/len(df)*100:.1f}%)  特征={X.shape[1]}")

    # ---- 5 折 CV ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv_results = {}
    for name, proto in models.items():
        aucs, accs, f1s = [], [], []
        for tr, te in skf.split(X, y):
            m = type(proto)(**proto.get_params())
            m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])[:, 1]
            pred = (proba >= 0.5).astype(int)
            aucs.append(roc_auc_score(y[te], proba))
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred))
        cv_results[name] = {
            "roc_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
        }
        r = cv_results[name]
        print(f"  {name:22s} AUC={r['roc_auc']['mean']:.3f}±{r['roc_auc']['std']:.3f} "
              f"acc={r['accuracy']['mean']:.3f} F1={r['f1']['mean']:.3f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc"]["mean"])
    best = models[best_name]
    best.fit(X, y)  # 全量训练最终模型

    artifact = {
        "model": best,
        "model_name": best_name,
        "keep_cols": keep,
        "featurizer_preset": "magpie",
        "cv": cv_results,
    }
    joblib.dump(artifact, P3_DIR / "stability.joblib")
    (P3_DIR / "stability_metrics.json").write_text(
        json.dumps({"n": len(df), "n_stable": n_stable, "cv": cv_results,
                    "best_model": best_name}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n最佳: {best_name} → saved {P3_DIR / 'stability.joblib'}")


if __name__ == "__main__":
    main()
