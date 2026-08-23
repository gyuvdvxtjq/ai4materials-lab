"""MatBench 标准评估脚本（可选，需 `matbench` 包 + 联网）。

用途
----
在 MatBench 官方数据集和划分（5 折嵌套 CV）上评估本项目的**组分模型基线**
（magpie 描述符 + 随机森林），并与官方榜单对标，让指标"可比较"。

对应任务与官方榜单（MAE / ROC-AUC，单位 eV）：
- matbench_expt_gap     组分→实验带隙   RF-SCM/Magpie 0.446  MODNet 0.333
- matbench_expt_is_metal 组分→金属性    （分类，ROC-AUC）
- matbench_mp_gap       结构→DFT 带隙   RF-SCM/Magpie 0.345  CGCNN 0.297  ALIGNN 0.186
- matbench_mp_is_metal  结构→金属性    （分类，ROC-AUC）

榜单：https://matbench.materialsproject.org

注意
----
- 首次运行会从 Materials Project 下载数据集（mp_gap 约数百 MB），需联网。
- 结构类任务（mp_gap / mp_is_metal）本脚本只做"组分基线"（从结构提取组成），
  用于与结构模型（CGCNN/ALIGNN）对照，展示"结构信息"的增量价值。

运行
----
    pip install matbench
    python benchmark/run_matbench.py                 # 跑全部四个任务
    python benchmark/run_matbench.py --task matbench_expt_gap
"""
from __future__ import annotations

import argparse

import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

TASKS = [
    "matbench_expt_gap",
    "matbench_expt_is_metal",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
]
RANDOM_STATE = 42


def featurize_compositions(comps: list[Composition]) -> np.ndarray:
    """magpie 组分描述符，去掉含 NaN 的列。"""
    featurizer = ElementProperty.from_preset("magpie")
    X = np.array(featurizer.featurize_many(comps), dtype=float)
    keep = ~np.isnan(X).any(axis=0)
    return X[:, keep]


def run_task(task) -> dict:
    """在 MatBench 的 5 折嵌套 CV 上跑 magpie+RF 基线。"""
    task.load()
    df = task.df
    target = task.metadata.target
    input_type = task.metadata.input_type
    is_clf = task.is_classification

    if input_type == "composition":
        comps = df["composition"].tolist()
    else:  # structure 类任务：只取组成，作为"组分基线"
        comps = [s.composition for s in df["structure"]]

    X = featurize_compositions(comps)
    y = df[target].values

    scores = []
    for tr_idx, te_idx in task.folds:
        if is_clf:
            m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
            m.fit(X[tr_idx], y[tr_idx].astype(int))
            proba = m.predict_proba(X[te_idx])[:, 1]
            scores.append(roc_auc_score(y[te_idx].astype(int), proba))
        else:
            m = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
            m.fit(X[tr_idx], y[tr_idx].astype(float))
            scores.append(mean_absolute_error(y[te_idx].astype(float), m.predict(X[te_idx])))

    metric = "ROC-AUC" if is_clf else "MAE"
    return {
        "task": task.dataset_name,
        "metric": metric,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_samples": int(len(df)),
        "input": input_type,
        "classification": is_clf,
    }


def main() -> None:
    from matbench.bench import MatbenchBenchmark

    parser = argparse.ArgumentParser(description="MatBench 组分基线评估")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="只跑指定任务；默认跑全部")
    args = parser.parse_args()

    mb = MatbenchBenchmark(autoload=False)
    tasks = [args.task] if args.task else TASKS
    for name in tasks:
        task = getattr(mb, name)
        result = run_task(task)
        print(f"{result['task']:28s} {result['metric']}={result['mean']:.4f} ± {result['std']:.4f} "
              f"(n={result['n_samples']}, {result['input']})")


if __name__ == "__main__":
    main()
