"""P1 v3 严格评估：5 折交叉验证 + 超参搜索 + mean±std。

与 train_split.py（训练最终模型）不同，本脚本回答两个评审人必问的问题：
1. 「指标稳不稳」→ 用 5 折分层交叉验证，报告 mean ± std，而不是单次划分的单点数字。
2. 「调参了吗」→ 对 HistGradientBoosting 做小规模网格搜索，报告最优超参与其 CV 分数，
   并与默认超参对比，展示调参带来的真实增益（而非靠默认值蒙混）。

运行：python p1_opt/evaluate_cv.py
产物：p1_opt/p1_cv_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

P1_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_FOLDS = 5
RF_TREES = 200  # 交叉验证用 200 棵树（最终模型仍用 400）；已固定，保证可复现


def load_and_featurize():
    df = pd.read_csv(P1_DIR / "materials_final.csv")
    df["_key"] = df["formula"].apply(lambda f: Composition(f).reduced_formula)
    df = df.drop_duplicates("_key", keep="first").reset_index(drop=True)
    featurizer = ElementProperty.from_preset("magpie")
    comps = [Composition(f) for f in df["formula"]]
    X = np.array(featurizer.featurize_many(comps), dtype=float)
    keep = ~np.isnan(X).any(axis=0)
    return df, X[:, keep], df["is_metal"].astype(int).values, df["band_gap"].astype(float).values


def cv_scores(estimator, X, y, groups, is_clf):
    """按给定分层标签做 K 折交叉验证，返回每折分数。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in skf.split(X, groups):
        est = estimator
        est.fit(X[tr], y[tr])
        if is_clf:
            scores.append(roc_auc_score(y[te], est.predict_proba(X[te])[:, 1]))
        else:
            scores.append(mean_absolute_error(y[te], est.predict(X[te])))
    return np.array(scores)


def nested_tuned_scores(estimator, param_grid, X, y, groups, is_clf):
    """嵌套 CV：外层评估，内层只在训练折上调参，避免调参泄漏。"""
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores, params = [], []
    for tr, te in outer.split(X, groups):
        inner = GridSearchCV(estimator, param_grid,
                             scoring="roc_auc" if is_clf else "neg_mean_absolute_error",
                             cv=3, n_jobs=-1)
        inner.fit(X[tr], y[tr])
        best = inner.best_estimator_
        if is_clf:
            score = roc_auc_score(y[te], best.predict_proba(X[te])[:, 1])
        else:
            score = mean_absolute_error(y[te], best.predict(X[te]))
        scores.append(score)
        params.append(inner.best_params_)
    # 仅用于部署时获取全量数据上的最终参数；不参与外层分数计算。
    final = GridSearchCV(estimator, param_grid,
                         scoring="roc_auc" if is_clf else "neg_mean_absolute_error",
                         cv=3, n_jobs=-1)
    final.fit(X, y)
    return np.asarray(scores), final.best_params_, params


def main() -> None:
    df, X, y_metal, y_gap = load_and_featurize()
    n = len(df)
    print(f"数据: {n} 条（金属 {y_metal.sum()}，非金属 {n - y_metal.sum()}）")
    print(f"协议: {N_FOLDS} 折分层交叉验证（shuffle, random_state={RANDOM_STATE}），RF 固定 {RF_TREES} 棵树\n")

    results = {"protocol": f"{N_FOLDS}-fold stratified CV, shuffle, random_state={RANDOM_STATE}",
               "random_state": RANDOM_STATE, "n_folds": N_FOLDS}

    # ---------- 任务 1：分类（金属 vs 非金属） ----------
    print("===== 任务 1：metal/non-metal 分类（ROC-AUC, mean±std）=====")
    cls_models = {
        "RandomForest(默认)": RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting(默认)": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cls_cv = {}
    for name, m in cls_models.items():
        s = cv_scores(m, X, y_metal, y_metal, is_clf=True)
        cls_cv[name] = {"mean": float(s.mean()), "std": float(s.std())}
        print(f"  {name:28s} {s.mean():.4f} ± {s.std():.4f}")

    # 调参：HGB 分类
    clf_grid = {"learning_rate": [0.05, 0.1, 0.2], "max_iter": [100, 200], "max_leaf_nodes": [31, 63]}
    s, best_params, fold_params = nested_tuned_scores(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE), clf_grid,
        X, y_metal, y_metal, is_clf=True)
    cls_cv["HistGradientBoosting(调参后)"] = {"mean": float(s.mean()), "std": float(s.std())}
    results["classification"] = {"cv": cls_cv, "best_hgb_params": best_params,
                                 "outer_cv_roc_auc": float(s.mean()),
                                 "outer_fold_params": fold_params}
    print(f"  {'HistGradientBoosting(调参后)':28s} {s.mean():.4f} ± {s.std():.4f}")
    print(f"  最优超参（全量部署）: {best_params}  （外层5折无偏评估）\n")

    # ---------- 任务 2：非金属子集带隙回归 ----------
    print("===== 任务 2：非金属子集 band_gap 回归（MAE, mean±std）=====")
    nm = np.where(y_metal == 0)[0]
    Xnm, ynm = X[nm], y_gap[nm]
    # 回归也分层：按带隙分位数分组，保证每折金属/带隙分布一致
    gap_bins = np.digitize(ynm, np.quantile(ynm, [0.33, 0.67]))
    reg_models = {
        "RandomForest(默认)": RandomForestRegressor(n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting(默认)": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    reg_cv = {}
    for name, m in reg_models.items():
        s = cv_scores(m, Xnm, ynm, gap_bins, is_clf=False)
        reg_cv[name] = {"mean": float(s.mean()), "std": float(s.std())}
        print(f"  {name:28s} {s.mean():.4f} ± {s.std():.4f} eV")

    # 调参：HGB 回归
    reg_grid = {"learning_rate": [0.05, 0.1, 0.2], "max_iter": [100, 200], "max_leaf_nodes": [31, 63]}
    s, best_params, fold_params = nested_tuned_scores(
        HistGradientBoostingRegressor(random_state=RANDOM_STATE), reg_grid,
        Xnm, ynm, gap_bins, is_clf=False)
    reg_cv["HistGradientBoosting(调参后)"] = {"mean": float(s.mean()), "std": float(s.std())}
    results["regression_nonmetal"] = {"cv": reg_cv, "best_hgb_params": best_params,
                                      "outer_cv_mae": float(s.mean()),
                                      "outer_fold_params": fold_params}
    print(f"  {'HistGradientBoosting(调参后)':28s} {s.mean():.4f} ± {s.std():.4f} eV")
    print(f"  最优超参（全量部署）: {best_params}  （外层5折无偏评估）")

    (P1_DIR / "p1_cv_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {P1_DIR / 'p1_cv_metrics.json'}")


if __name__ == "__main__":
    main()
