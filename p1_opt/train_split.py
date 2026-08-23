"""P1 v3：金属/半导体分类 + 带隙回归拆分 + 不确定性估计。

为什么需要这一版
----------------
P1 v2 直接在全部 6772 条数据上回归 band_gap，得到 R²=0.714。但这个数字是
"虚高"的：约 60% 的样本带隙恰好为 0（金属），回归器只要学会"认出金属"就能
拿到一个好看但会误导的 R²。本脚本把任务拆成两个更诚实、也更贴近 MatBench
官方定义的问题：

1. 分类：用 magpie 组分特征预测 metal / non-metal（对应 MatBench 的
   matbench_mp_is_metal / matbench_expt_is_metal）。
2. 回归：只在非金属子集上预测 band_gap（对应 matbench_mp_gap 的"半导体/绝缘体"
   部分），并单独报告非金属子集的 MAE / RMSE / R²。

同时给出回归的不确定性（随机森林各树预测的标准差），供 P3 筛选时排序使用。

运行：python p1_opt/train_split.py
产物：p1_opt/bandgap_split.joblib（模型）、p1_opt/p1_split_metrics.json（指标）
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

P1_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def load_and_featurize() -> tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, ElementProperty]:
    """读取数据、按还原式去重、构造 magpie 描述符。"""
    df = pd.read_csv(P1_DIR / "materials_final.csv")
    df["_key"] = df["formula"].apply(lambda f: Composition(f).reduced_formula)
    df = df.drop_duplicates("_key", keep="first").reset_index(drop=True)

    featurizer = ElementProperty.from_preset("magpie")
    comps = [Composition(f) for f in df["formula"]]
    X = np.array(featurizer.featurize_many(comps), dtype=float)
    keep = ~np.isnan(X).any(axis=0)  # 去掉含 NaN 的列
    X = X[:, keep]

    y_metal = df["is_metal"].astype(int).values
    y_gap = df["band_gap"].astype(float).values
    return df, X, y_metal, y_gap, keep, featurizer


def evaluate_classifier(clf, Xtr, ytr, Xte, yte) -> dict:
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "f1": float(f1_score(yte, pred)),
    }


def evaluate_regressor(reg, Xtr, ytr, Xte, yte) -> dict:
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    return {
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(np.sqrt(np.mean((yte - pred) ** 2))),
        "r2": float(r2_score(yte, pred)),
        "const_mae": float(mean_absolute_error(yte, np.full_like(yte, ytr.mean()))),
    }


def main() -> None:
    df, X, y_metal, y_gap, keep, featurizer = load_and_featurize()
    n = len(df)
    n_metal = int(y_metal.sum())
    print(f"数据: {n} 条（金属 {n_metal}，非金属 {n - n_metal}）")
    print(f"magpie 特征维度（去 NaN 后）: {X.shape[1]}")

    # ---- 按金属/非金属分层划分 ----
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE,
                                      stratify=y_metal)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.125, random_state=RANDOM_STATE,
                                      stratify=y_metal[idx_tr])
    print(f"train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

    # ---- 任务 1：金属/半导体分类 ----
    print("\n===== 任务 1：metal / non-metal 分类（test 指标）=====")
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cls_results = {}
    for name, clf in classifiers.items():
        cls_results[name] = evaluate_classifier(clf, X[idx_tr], y_metal[idx_tr],
                                                X[idx_te], y_metal[idx_te])
        r = cls_results[name]
        print(f"  {name:20s} acc={r['accuracy']:.3f}  ROC-AUC={r['roc_auc']:.3f}  F1={r['f1']:.3f}")

    # ---- 任务 2：非金属子集带隙回归 ----
    print("\n===== 任务 2：非金属子集 band_gap 回归（test 指标）=====")
    nonmetal_tr = idx_tr[y_metal[idx_tr] == 0]
    nonmetal_te = idx_te[y_metal[idx_te] == 0]
    print(f"非金属子集: train={len(nonmetal_tr)} test={len(nonmetal_te)}")
    regressors = {
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    reg_results = {}
    for name, reg in regressors.items():
        reg_results[name] = evaluate_regressor(reg, X[nonmetal_tr], y_gap[nonmetal_tr],
                                               X[nonmetal_te], y_gap[nonmetal_te])
        r = reg_results[name]
        print(f"  {name:20s} MAE={r['mae']:.3f} eV  RMSE={r['rmse']:.3f}  R2={r['r2']:.3f}  (常数基线 MAE={r['const_mae']:.3f})")

    # ---- 参考：全量回归（说明为何 R² 虚高）----
    print("\n===== 参考：全量 band_gap 回归（说明 R² 为何虚高）=====")
    full_reg = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    full_reg.fit(X[idx_tr], y_gap[idx_tr])
    full_pred = full_reg.predict(X[idx_te])
    full_mae = mean_absolute_error(y_gap[idx_te], full_pred)
    full_r2 = r2_score(y_gap[idx_te], full_pred)
    zero_frac = float((y_gap[idx_te] == 0).mean())
    print(f"  RF 全量回归: MAE={full_mae:.3f} eV  R2={full_r2:.3f}  (test 中 {zero_frac*100:.1f}% 为金属)")
    print(f"  说明：R²={full_r2:.3f} 主要来自识别 {zero_frac*100:.0f}% 的零带隙金属，而非预测带隙数值。")

    # ---- 选择最佳模型并打包 ----
    best_clf_name = max(cls_results, key=lambda k: cls_results[k]["roc_auc"])
    best_reg_name = min(reg_results, key=lambda k: reg_results[k]["mae"])
    best_clf = classifiers[best_clf_name]
    best_reg = regressors[best_reg_name]
    # 用 train+val 重训最终模型
    best_clf.fit(X[np.concatenate([idx_tr, idx_va])], y_metal[np.concatenate([idx_tr, idx_va])])
    nonmetal_trva = np.concatenate([idx_tr, idx_va])[y_metal[np.concatenate([idx_tr, idx_va])] == 0]
    best_reg.fit(X[nonmetal_trva], y_gap[nonmetal_trva])

    # 不确定性：用随机森林各树预测的标准差（RF 天然提供树间方差，
    # 与最佳回归器是否为 RF 无关，因为这里单独训练一个 RF 专门算不确定度）
    rf_unc = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    rf_unc.fit(X[nonmetal_trva], y_gap[nonmetal_trva])
    n_trees_preds = np.array([t.predict(X[nonmetal_te]) for t in rf_unc.estimators_])
    reg_unc_std = float(np.mean(np.std(n_trees_preds, axis=0)))

    metrics = {
        "n_total": n,
        "n_metal": n_metal,
        "n_nonmetal": n - n_metal,
        "classification": {"best_model": best_clf_name, **cls_results[best_clf_name]},
        "regression_nonmetal": {"best_model": best_reg_name, **reg_results[best_reg_name]},
        "regression_full_reference": {"mae": full_mae, "r2": full_r2},
        "mean_tree_std_uncertainty": reg_unc_std,
        "random_state": RANDOM_STATE,
        "featurizer": "magpie",
    }
    artifact = {
        "classifier": best_clf,
        "regressor": best_reg,
        "rf_unc": rf_unc,  # 用于逐样本不确定度（树间标准差）
        "classifier_name": best_clf_name,
        "regressor_name": best_reg_name,
        "keep_cols": keep,
        "featurizer_preset": "magpie",
        "feature_dim": int(X.shape[1]),
        "metrics": metrics,
    }
    joblib.dump(artifact, P1_DIR / "bandgap_split.joblib")
    (P1_DIR / "p1_split_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {P1_DIR / 'bandgap_split.joblib'}")
    print(f"saved {P1_DIR / 'p1_split_metrics.json'}")
    print(f"平均树标准差（回归不确定性）: {reg_unc_std:.3f} eV")


if __name__ == "__main__":
    main()
