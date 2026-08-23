"""MatBench 标准评估（离线可复现版）：官方数据 + 官方 5 折划分。

与 run_matbench.py 相同的协议（MatbenchTask 官方 folds），但把结果落盘为
benchmark/matbench_results.json，并内置官方榜单对照，便于直接引用。

覆盖任务（组分输入，与 P1 同类）：
- matbench_expt_gap      组分→实验带隙（回归, MAE）
- matbench_expt_is_metal 组分→金属性（分类, ROC-AUC）

官方榜单对照（matbench.materialsproject.org, v0.1）：
- expt_gap:      RF-SCM/Magpie 0.446 | AMMExpress 0.416 | MODNet 0.333 | Darwin 0.287
- expt_is_metal: RF-SCM/Magpie 0.936 ROC-AUC | MODNet 0.974（本表以实测为准）

运行（需联网首次下载，约 40KB×2，之后走本地缓存）：
    python benchmark/run_matbench_official.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RANDOM_STATE = 42
OUT = Path(__file__).resolve().parent / "matbench_results.json"

LEADERBOARD = {
    "matbench_expt_gap": {
        "RF-SCM/Magpie (官方)": 0.4461,
        "AMMExpress v2020 (官方)": 0.4161,
        "MODNet v0.1.12 (官方)": 0.3327,
        "Darwin (官方)": 0.2865,
    },
    "matbench_expt_is_metal": {
        "RF-SCM/Magpie (官方)": 0.9356,
        "AMMExpress v2020 (官方)": 0.9471,
        "MODNet v0.1.12 (官方)": 0.9739,
    },
}


def featurize(comps) -> np.ndarray:
    from pymatgen.core import Composition
    comps = [c if isinstance(c, Composition) else Composition(str(c)) for c in comps]
    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(1)  # 沙箱/CI 环境下多进程池可能挂起，串行更稳
    X = np.array([featurizer.featurize(c) for c in comps], dtype=float)
    keep = ~np.isnan(X).any(axis=0)
    return X[:, keep]


def main() -> None:
    mb = MatbenchBenchmark(autoload=False)
    results = {}
    for task_name in ["matbench_expt_gap", "matbench_expt_is_metal"]:
        task = getattr(mb, task_name)
        task.load()
        target = task.metadata.target
        is_clf = task.metadata.task_type == "classification"
        metric_name = "roc_auc" if is_clf else "mae"
        print(f"\n===== {task_name} (n={len(task.df)}, target={target}, {'分类' if is_clf else '回归'}) =====")

        scores = []
        for fold in range(5):
            tr_X_raw, tr_y = task.get_train_and_val_data(fold_number=fold)
            te_X_raw, te_y = task.get_test_data(fold_number=fold, include_target=True)
            X_tr, y_tr = featurize(tr_X_raw.tolist()), np.asarray(tr_y)
            X_te, y_te = featurize(te_X_raw.tolist()), np.asarray(te_y)
            if is_clf:
                m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
                m.fit(X_tr, y_tr.astype(int))
                from sklearn.metrics import roc_auc_score
                s = roc_auc_score(y_te.astype(int), m.predict_proba(X_te)[:, 1])
            else:
                m = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
                m.fit(X_tr, y_tr.astype(float))
                from sklearn.metrics import mean_absolute_error
                s = mean_absolute_error(y_te.astype(float), m.predict(X_te))
            scores.append(s)
            print(f"  fold{fold}: {metric_name}={s:.4f}")

        mean, std = float(np.mean(scores)), float(np.std(scores))
        results[task_name] = {
            "model": "magpie + RandomForest(200 trees)",
            "metric": metric_name,
            "mean": mean,
            "std": std,
            "folds": [float(s) for s in scores],
            "n_samples": int(len(task.df)),
            "leaderboard_reference": LEADERBOARD[task_name],
        }
        print(f"  → 5折 mean±std: {mean:.4f} ± {std:.4f}  ({metric_name})")
        for k, v in LEADERBOARD[task_name].items():
            print(f"     对照 {k}: {v:.4f}")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
