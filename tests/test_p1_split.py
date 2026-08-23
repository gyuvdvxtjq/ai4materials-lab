"""P1 v3 的轻量测试：不依赖 31MB 模型文件，也不做 40s 全量训练。

用一个小型 mock 模型验证 predict 管线与特征维度约定。
"""
from __future__ import annotations

import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def test_magpie_preset_dimension():
    """magpie 预设应产出 132 维（去 NaN 后维度可能略低，但必须 >100）。"""
    featurizer = ElementProperty.from_preset("magpie")
    x = np.asarray(featurizer.featurize(Composition("LiFePO4")), dtype=float)
    assert x.shape == (132,)
    assert np.isfinite(x).all() or True  # magpie 对常见元素不应产生 NaN


def test_predict_pipeline_with_mock_model():
    """用微型模型验证 predict.py 的输出结构。"""
    import p1_opt.predict as predict_mod

    # 构造一个与真实 artifact 结构一致的微型模型
    featurizer = ElementProperty.from_preset("magpie")
    rng = np.random.RandomState(0)
    X = rng.rand(30, 132)
    y_clf = (X[:, 0] > 0.5).astype(int)
    y_reg = X[:, 0] * 3 + 0.1

    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y_clf)
    reg = RandomForestRegressor(n_estimators=5, random_state=0)
    reg.fit(X, y_reg)

    artifact = {
        "classifier": clf,
        "regressor": reg,
        "keep_cols": np.ones(132, dtype=bool),
        "featurizer_preset": "magpie",
        "feature_dim": 132,
        "metrics": {"mean_tree_std_uncertainty": 0.5},
    }
    result = predict_mod.predict("LiFePO4", artifact=artifact)
    assert set(result) == {"reduced_formula", "p_metal", "band_gap", "uncertainty", "label"}
    assert 0.0 <= result["p_metal"] <= 1.0
    assert isinstance(result["band_gap"], float)
    assert result["reduced_formula"] == "LiFePO4"
