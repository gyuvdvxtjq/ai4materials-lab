"""P3 v3：带稳定性判据的筛选闭环（本地、无网络依赖）。

v2 的局限（已 documented）：只看预测带隙，「未检索到 MP」≈查字典。
v3 的升级——对齐 MatBench-Discovery 范式的最小可行版本：

    枚举化学式
      → P1 分类器：P(金属) 过滤      [bandgap_split.joblib]
      → P1 回归器：预测带隙 ± 不确定度，落入目标窗口
      → 稳定性模型：P(stable) 过滤/排序  [stability.joblib, AUC 0.90]
      → 输出「带隙 + 稳定性 + 不确定度」综合榜单（CSV）

「新候选」的语义升级：v3 的候选经过「模型认为可能是稳定半导体」的过滤，
不再是「数据库里没有」这么弱的条件。但仍然只是**组分层面的粗筛**——
不含结构预测、形成能计算或实验可行性（诚实边界，见 README）。

依赖：p1_opt/bandgap_split.joblib（train_split.py 生成）
      p3_screen_v2/stability.joblib（train_stability.py 生成）
MP 验证（可选）：服务可用时对照 MP 实测（与 v2 相同），不可用则跳过。

运行：python p3_screen_v2/screen_v3.py [--lo 1.0] [--hi 3.0] [--top 30]
产物：p3_screen_v2/screening_results_v3.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

P3_DIR = Path(__file__).resolve().parent
ROOT = P3_DIR.parent

CATIONS = ["Li", "Na"]
TMS = ["V", "Cr", "Mn", "Fe", "Co", "Ni", "Ti", "Cu"]
ANIONS = ["O2", "S2", "PO4"]
STOICH = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 2)]


def gen_candidates() -> list[str]:
    cands = set()
    for a, tm, an, (x, y) in itertools.product(CATIONS, TMS, ANIONS, STOICH):
        try:
            formula = f"{a}{x if x > 1 else ''}{tm}{y if y > 1 else ''}{an}"
            c = Composition(formula)
            if len(c.elements) >= 3 and c.num_atoms <= 12:
                cands.add(c.reduced_formula)
        except Exception:
            continue
    return sorted(cands)


def featurize(formulas: list[str]) -> np.ndarray:
    feat = ElementProperty.from_preset("magpie")
    comps = [Composition(f) for f in formulas]
    X = np.array([feat.featurize(c) for c in comps], dtype=float)
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 v3 筛选闭环（带稳定性）")
    parser.add_argument("--lo", type=float, default=1.0, help="目标带隙下界 (eV)")
    parser.add_argument("--hi", type=float, default=3.0, help="目标带隙上界 (eV)")
    parser.add_argument("--top", type=int, default=30, help="输出榜单长度")
    parser.add_argument("--p-metal-max", type=float, default=0.5, help="金属概率上限")
    parser.add_argument("--p-stable-min", type=float, default=0.5, help="稳定性概率下限")
    args = parser.parse_args()

    print("=" * 64)
    print(f"P3 v3 · 带隙 {args.lo}-{args.hi} eV + 稳定性 + 不确定度（本地筛选）")
    print("=" * 64)

    # ---- 加载三个模型 ----
    p1 = joblib.load(ROOT / "p1_opt" / "bandgap_split.joblib")
    st = joblib.load(P3_DIR / "stability.joblib")
    keep_p1 = np.asarray(p1["keep_cols"], dtype=bool)
    keep_st = np.asarray(st["keep_cols"], dtype=bool)

    # ---- 1. 枚举 ----
    cands = gen_candidates()
    print(f"[1/4] 枚举候选: {len(cands)}")
    X = featurize(cands)

    # ---- 2. 金属性过滤 + 带隙预测 ----
    p_metal = p1["classifier"].predict_proba(X[:, keep_p1])[:, 1]
    gap = p1["regressor"].predict(X[:, keep_p1])
    unc = p1.get("metrics", {}).get("mean_tree_std_uncertainty", np.nan)
    df = pd.DataFrame({"formula": cands, "p_metal": p_metal, "pred_bandgap": gap})
    n0 = len(df)
    df = df[df.p_metal < args.p_metal_max]
    print(f"[2/4] 金属性过滤 (P(金属)<{args.p_metal_max}): {n0} → {len(df)}")

    # ---- 3. 带隙窗口 ----
    n1 = len(df)
    df = df[(df.pred_bandgap >= args.lo) & (df.pred_bandgap <= args.hi)]
    print(f"[3/4] 带隙窗口 [{args.lo},{args.hi}] eV: {n1} → {len(df)}")

    # ---- 4. 稳定性过滤 + 综合排序 ----
    n2 = len(df)
    if len(df):
        Xs = featurize(df["formula"].tolist())
        df["p_stable"] = st["model"].predict_proba(Xs[:, keep_st])[:, 1]
        df = df[df.p_stable >= args.p_stable_min]
    print(f"[4/4] 稳定性过滤 (P(稳定)≥{args.p_stable_min}): {n2} → {len(df)}")

    if not len(df):
        print("无候选通过全部过滤；可放宽 --p-stable-min 或带隙窗口。")
        return

    # 综合分：P(stable) × 带隙窗口内的高斯权重（离窗口中心越近越好）
    center = (args.lo + args.hi) / 2
    width = (args.hi - args.lo) / 2
    df["gap_score"] = np.exp(-((df.pred_bandgap - center) / width) ** 2)
    df["score"] = df.p_stable * df.gap_score
    df["uncertainty"] = unc
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # ---- 可选 MP 验证（与 v2 相同协议，服务可用才查）----
    base = os.getenv("MP_BASE_URL", "").rstrip("/")
    out_cols = ["formula", "pred_bandgap", "p_metal", "p_stable", "score", "uncertainty"]
    if base:
        import requests
        def check(f):
            try:
                r = requests.get(f"{base}/materials/search", params={"formula": f, "limit": 3}, timeout=30)
                res = r.json().get("results", [])
                if not res:
                    return {"in_mp": False}
                best = sorted(res, key=lambda x: x.get("energy_above_hull") or 9e9)[0]
                return {"in_mp": True, "mp_band_gap": best.get("band_gap"),
                        "e_above_hull": best.get("energy_above_hull"),
                        "is_stable": best.get("is_stable")}
            except Exception:
                return {"in_mp": None}
        verif = [check(f) for f in df.head(args.top)["formula"]]
        df = pd.concat([df, pd.DataFrame(verif)], axis=1)
        out_cols += ["in_mp", "mp_band_gap", "e_above_hull"]

    show = df.head(args.top)[out_cols]
    print("\n===== 候选榜单（前 {}）=====".format(min(args.top, len(df))))
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out_path = P3_DIR / "screening_results_v3.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")
    print(f"漏斗: {len(cands)} 枚举 → 金属性 → 带隙 → 稳定性 → {len(df)} 候选")
    print("注: 候选=「模型认为可能是稳定半导体」的化学式，非实验可行的材料；"
          "不含结构预测与形成能计算。")


if __name__ == "__main__":
    main()
