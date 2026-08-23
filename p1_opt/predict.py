"""P1 推理入口。

用法（在项目根目录执行）：
    python p1_opt/predict.py LiFePO4

首次使用前请确保 ``bandgap_magpie.joblib`` 已生成，或从项目发布页下载。
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
    path = path or (P1_DIR / "bandgap_magpie.joblib")
    if not path.exists():
        raise FileNotFoundError(
            f"模型不存在: {path}\n请先运行 python p1_opt/train_magpie.py"
        )
    return joblib.load(path)


def predict(formula: str, artifact=None) -> tuple[float, str]:
    comp = Composition(formula)
    artifact = artifact or load_model()
    featurizer = ElementProperty.from_preset(artifact.get("featurizer_preset", "magpie"))
    x = np.asarray(featurizer.featurize(comp), dtype=float).reshape(1, -1)
    keep = np.asarray(artifact["keep_cols"], dtype=bool)
    x = x[:, keep]
    value = float(artifact["model"].predict(x)[0])
    if value < 0.05:
        label = "金属/近零带隙"
    elif value < 2.0:
        label = "半导体"
    else:
        label = "宽带隙半导体/绝缘体"
    return value, label


def main() -> None:
    parser = argparse.ArgumentParser(description="由化学式预测 DFT 带隙")
    parser.add_argument("formula", help="例如 LiFePO4、TiO2")
    parser.add_argument("--model", type=Path, default=None, help="模型文件路径")
    args = parser.parse_args()
    try:
        value, label = predict(args.formula, load_model(args.model))
    except Exception as exc:
        parser.error(str(exc))
    print(f"化学式: {Composition(args.formula).reduced_formula}")
    print(f"预测带隙: {value:.3f} eV")
    print(f"类别: {label}")


if __name__ == "__main__":
    main()
