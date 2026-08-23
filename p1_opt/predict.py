"""P1 v3 推理入口：金属/半导体分类 + 带隙回归 + 不确定度。

用法（在项目根目录执行）：
    python p1_opt/predict.py LiFePO4
    python p1_opt/predict.py Fe2O3
    python p1_opt/predict.py Cu

首次使用前请先运行 ``python p1_opt/train_split.py`` 生成 ``bandgap_split.joblib``。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

P1_DIR = Path(__file__).resolve().parent


def load_model(path: Path | None = None):
    path = Path(path) if path else (P1_DIR / "bandgap_split.joblib")
    if not path.exists():
        raise FileNotFoundError(
            f"模型不存在: {path}\n请先运行 python p1_opt/train_split.py"
        )
    return joblib.load(path)


def predict(formula: str, artifact=None) -> dict:
    """返回 {reduced_formula, p_metal, band_gap, uncertainty, label}。"""
    comp = Composition(formula)
    artifact = artifact or load_model()
    featurizer = ElementProperty.from_preset(artifact["featurizer_preset"])
    x = np.asarray(featurizer.featurize(comp), dtype=float).reshape(1, -1)
    keep = np.asarray(artifact["keep_cols"], dtype=bool)
    x = x[:, keep]

    p_metal = float(artifact["classifier"].predict_proba(x)[0, 1])
    gap = float(artifact["regressor"].predict(x)[0])
    # 平均不确定度：随机森林树间标准差（训练时估计，见 train_split.py）
    unc = float(artifact.get("metrics", {}).get("mean_tree_std_uncertainty", np.nan))

    if p_metal >= 0.5:
        label = "金属（预测）"
    elif gap < 2.0:
        label = "半导体"
    else:
        label = "宽带隙半导体/绝缘体"
    return {
        "reduced_formula": comp.reduced_formula,
        "p_metal": p_metal,
        "band_gap": gap,
        "uncertainty": unc,
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="由化学式预测金属性 + DFT 带隙")
    parser.add_argument("formula", help="例如 LiFePO4、TiO2、Cu")
    parser.add_argument("--model", type=Path, default=None, help="模型文件路径")
    args = parser.parse_args()
    try:
        r = predict(args.formula, load_model(args.model))
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))
    print(f"化学式: {r['reduced_formula']}")
    print(f"P(金属): {r['p_metal']:.3f}")
    print(f"预测带隙: {r['band_gap']:.3f} eV (±{r['uncertainty']:.2f} eV)")
    print(f"类别: {r['label']}")


if __name__ == "__main__":
    main()
