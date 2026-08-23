"""MODNet 基线：P1 数据 + 与 train_split.py 完全相同的划分。

目的
----
回答「P1 的组分路线换更强的模型（MODNet，专为小数据设计）能好多少」。
MODNet（Material Optimal Descriptor Network）= matminer 特征 + 特征选择 +
浅层 NN，MatBench 榜单上 13 个任务中 7 个第一（2021），mp_gap 0.220。

协议
----
与 p1_opt/train_split.py 完全一致：同样的去重、同样的分层 80/10/10 划分
（random_state=42，按 is_metal 分层）。报告：
- 全量 band_gap 回归（对照 RF 全量：MAE 0.337 / R² 0.689，单次划分）
- 非金属子集 MAE（对照 HGB：0.522±0.012，5 折 CV）

环境注意
--------
- 需要 tensorflow（CPU 即可）+ modnet（`pip install --no-deps modnet`）。
- MODData.featurize() 默认多进程，受限环境会死锁 → 强制 n_jobs=1。

运行：python benchmark/run_modnet.py
产物：benchmark/modnet_results.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
P1_CSV = ROOT / "p1_opt" / "materials_final.csv"
OUT = Path(__file__).resolve().parent / "modnet_results.json"
RANDOM_STATE = 42


def main() -> None:
    os_env = __import__("os").environ
    os_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # ---- 数据与划分（与 train_split.py 严格一致）----
    df = pd.read_csv(P1_CSV)
    df["_key"] = df["formula"].apply(lambda f: Composition(f).reduced_formula)
    df = df.drop_duplicates("_key", keep="first").reset_index(drop=True)
    idx = np.arange(len(df))
    y_metal = df["is_metal"].astype(int).values
    itr, ite = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_metal)
    itr, iva = train_test_split(itr, test_size=0.125, random_state=RANDOM_STATE,
                                stratify=y_metal[itr])
    print(f"n={len(df)} train={len(itr)} val={len(iva)} test={len(ite)}")

    # ---- MODData 特征化（全量一次，按索引切分）----
    from modnet.preprocessing import MODData
    t0 = time.time()
    comp_objs = [Composition(f) for f in df["formula"]]
    md_all = MODData(
        materials=[comp_objs[i] for i in range(len(df))],
        targets=[[v] for v in df["band_gap"].astype(float).tolist()],
        target_names=["gap"],
    )
    # MODNet 自带 featurize 太慢（默认 ~35 分钟）且 fast 模式需从 figshare 下载
    # 预计算库（受限环境不可达）→ 手动用 magpie 132 维填 df_featurized（~16 秒，
    # 且与 P1 的 RF/HGB 基线特征完全一致，对比更公平）
    from matminer.featurizers.composition import ElementProperty
    _feat = ElementProperty.from_preset("magpie")
    X_all = np.array([_feat.featurize(c) for c in comp_objs], dtype=float)
    feat_cols = [f"magpie_{j}" for j in range(X_all.shape[1])]
    df_feat = pd.DataFrame(X_all, columns=feat_cols)
    df_feat = df_feat.loc[:, ~df_feat.isna().any(axis=0)]
    md_all.df_featurized = df_feat
    print(f"magpie 特征填充完成: {df_feat.shape}")

    md_train = md_all.split_frames(itr.tolist() + iva.tolist()) if hasattr(md_all, "split_frames") else None
    # 通用做法：用 MODData 子集构造
    from modnet.preprocessing import MODData as _MD
    trva = np.concatenate([itr, iva])
    md_train = _MD(
        materials=[comp_objs[i] for i in trva],
        targets=[[v] for v in df["band_gap"].astype(float).iloc[trva].tolist()],
        target_names=["gap"],
    )
    # 复用已算好的特征（避免二次 featurize）
    md_train.df_featurized = md_all.df_featurized.iloc[trva]
    md_train.df_targets = md_all.df_targets.iloc[trva]
    md_test = _MD(
        materials=[comp_objs[i] for i in ite],
        targets=[[v] for v in df["band_gap"].astype(float).iloc[ite].tolist()],
        target_names=["gap"],
    )
    md_test.df_featurized = md_all.df_featurized.iloc[ite]
    md_test.df_targets = md_all.df_targets.iloc[ite]

    # ---- 特征选择 + 训练 ----
    from modnet.models import MODNetModel
    t0 = time.time()
    # MODNet 自带特征选择（基于互信息）；实际可用特征 117 个（132 减全 NaN 列）
    md_train.feature_selection(n=117)
    model = MODNetModel(
        [[["gap"]]],
        weights={"gap": 1},
        num_neurons=[[256], [128], [64], [32]],
        n_feat=117,
    )
    model.fit(md_train, val_fraction=0.15, lr=0.001, epochs=150,
              batch_size=64, verbose=0)
    print(f"训练完成 {time.time() - t0:.0f}s")

    # ---- 评估 ----
    pred = model.predict(md_test)["gap"].values
    y_te = df["band_gap"].astype(float).iloc[ite].values
    mae_full = float(mean_absolute_error(y_te, pred))
    r2_full = float(r2_score(y_te, pred))
    nm_mask = y_metal[ite] == 0
    mae_nm = float(mean_absolute_error(y_te[nm_mask], pred[nm_mask]))

    results = {
        "model": "MODNet (num_neurons=[[256],[128],[64],[32]], n_feat=200, 150 epochs)",
        "protocol": "与 train_split.py 同划分（分层 80/10/10, random_state=42）",
        "n": int(len(df)),
        "mae_full": mae_full,
        "r2_full": r2_full,
        "mae_nonmetal": mae_nm,
        "reference": {
            "RF 全量（train_split.py, 同划分）": {"mae": 0.337, "r2": 0.689},
            "HGB 非金属子集（evaluate_cv.py, 5折）": {"mae_mean": 0.522, "std": 0.012},
            "官方 MODNet matbench_mp_gap (5折)": 0.2199,
        },
        "seconds_total": round(time.time() - t0),
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n全量: MAE={mae_full:.4f} R2={r2_full:.4f}  (RF 同划分: 0.337/0.689)")
    print(f"非金属子集: MAE={mae_nm:.4f}  (HGB 5折CV: 0.522±0.012)")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
