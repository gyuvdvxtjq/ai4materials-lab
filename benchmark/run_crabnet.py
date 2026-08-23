"""CrabNet 基线：在 matbench_expt_gap 官方 5 折上评估（含 crabnet 1.0.1 兼容补丁）。

目的
----
给 P1 的「纯组分」路线补一个更强的基线：CrabNet（组分编码 + 注意力，
官方榜单 expt_gap 5 折 MAE 0.346，优于 RF-SCM/Magpie 的 0.446）。

环境兼容补丁（crabnet 1.0.1 + torch>=2.x 需要以下全部，缺一不可）
------------------------------------------------------------------
1. 包名 bug：crabnet.model 内部 import 大写 `CrabNet.*` → 注入模块别名。
2. 数据文件缺失：pip 包不带 `data/element_properties/mat2vec.csv` → 自动从
   CrabNet 仓库下载到包预期路径。
3. 优化器不兼容：Lookahead/SWA 缺 torch>=2 的 step hook 属性 → 替换为透传
   包装（底层 Lamb 照常工作，禁用 SWA 权重平均，不影响训练正确性）。
4. DataLoader 死锁：容器/受限环境多进程 worker 偶发挂起 → 强制 num_workers=0。
5. BLAS 死锁：容器内 CPU 多线程偶发死锁 → torch.set_num_threads(1)。

注意
----
- 这里用缩减版配置（d_model=128 vs 原作 512、40 epochs、无 SWA），
  CPU 上单折约 2-4 分钟；结果应视为 CrabNet 的「下限」，不是官方复现。
- 返回元组顺序：(真实值, 预测值, 化学式, 不确定度)。

运行：python benchmark/run_crabnet.py [--folds 0 1 2 3 4] [--epochs 40]
产物：benchmark/crabnet_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)  # 补丁 5：避免容器内 BLAS 多线程死锁

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.show = lambda *a, **k: None

import pandas as pd  # noqa: E402


def _apply_crabnet_patches():
    """应用 crabnet 1.0.1 的全部兼容补丁，返回打好补丁的 crabnet.model 模块。"""
    import crabnet

    # 补丁 1：包名大小写 bug
    pkg_dir = os.path.dirname(crabnet.__file__)
    if "CrabNet" not in sys.modules:
        alias = types.ModuleType("CrabNet")
        alias.__path__ = [pkg_dir]
        alias.__package__ = "CrabNet"
        sys.modules["CrabNet"] = alias

    # 补丁 2：缺失的 mat2vec.csv
    mat2vec_path = os.path.join(os.path.dirname(pkg_dir), "data", "element_properties", "mat2vec.csv")
    if not os.path.exists(mat2vec_path):
        os.makedirs(os.path.dirname(mat2vec_path), exist_ok=True)
        import requests
        urls = [
            "https://raw.githubusercontent.com/kjappelbaum/CrabNet/master/crabnet/data/element_properties/mat2vec.csv",
            "https://raw.githubusercontent.com/anthony-wang/CrabNet/master/crabnet/data/element_properties/mat2vec.csv",
        ]
        for u in urls:
            r = requests.get(u, timeout=90)
            if r.status_code == 200 and len(r.content) > 5000:
                open(mat2vec_path, "wb").write(r.content)
                break
        else:
            raise RuntimeError(f"无法下载 mat2vec.csv，请手动放置到 {mat2vec_path}")

    import crabnet.model as cm

    # 补丁 3：优化器兼容
    class _PassthroughOptim(torch.optim.Optimizer):
        def __init__(self, *args, **kwargs):
            optimizer = (kwargs.get("base_optimizer") or kwargs.get("optimizer")
                         or (args[0] if args else None))
            if optimizer is None:
                raise TypeError("no optimizer supplied")
            self.optimizer = optimizer
            self.param_groups = optimizer.param_groups
            self.state = {}
            self.defaults = {}
            self.discard_count = 0
            self.minimum_found = False

        def step(self, closure=None):
            return self.optimizer.step(closure)

        def zero_grad(self, set_to_none=True):
            return self.optimizer.zero_grad(set_to_none)

        def update_swa(self, mae_v):  # SWA 权重平均禁用（兼容性裁剪）
            pass

        def swap_swa_sgd(self):
            pass

        def state_dict(self):
            return self.optimizer.state_dict()

        def load_state_dict(self, sd):
            return self.optimizer.load_state_dict(sd)

    cm.SWA = _PassthroughOptim
    cm.Lookahead = _PassthroughOptim

    # 补丁 4：单进程 DataLoader
    _OrigLoader = cm.EDM_CsvLoader

    class _SingleProcLoader(_OrigLoader):
        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            super().__init__(*args, **kwargs)

    cm.EDM_CsvLoader = _SingleProcLoader
    return cm


def run_fold(cm, fold: int, epochs: int = 40, d_model: int = 128) -> dict:
    """在 matbench_expt_gap 指定折上训练 + 评估 CrabNet。"""
    from crabnet.kingcrab import CrabNet
    from matbench.bench import MatbenchBenchmark
    from sklearn.metrics import mean_absolute_error

    mb = MatbenchBenchmark(autoload=False)
    task = mb.matbench_expt_gap
    task.load()
    tr_X, tr_y = task.get_train_and_val_data(fold_number=fold)
    te_X, te_y = task.get_test_data(fold_number=fold, include_target=True)
    train_df = pd.DataFrame({"formula": tr_X.astype(str), "target": tr_y.astype(float)})
    test_df = pd.DataFrame({"formula": te_X.astype(str), "target": te_y.astype(float)})

    torch.manual_seed(42)
    net = CrabNet(out_dims=3, d_model=d_model, N=3, heads=4, compute_device="cpu")
    m = cm.Model(net, model_name=f"crabnet_fold{fold}", verbose=False)
    m.load_data(train_df, batch_size=2 ** 7, train=True)
    m.load_data(test_df, batch_size=2 ** 7)
    t0 = time.time()
    m.fit(epochs=epochs, checkin=None, losscurve=False, learningcurve=False)
    act, pred, _, unc = m.predict(test_df)
    mae = float(mean_absolute_error(act, pred))
    return {"fold": fold, "mae": mae, "mean_uncertainty": float(np.mean(unc)),
            "seconds": round(time.time() - t0), "epochs": epochs, "d_model": d_model}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "crabnet_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {"folds": {}}
    results.setdefault("folds", {})

    cm = _apply_crabnet_patches()
    for fold in args.folds:
        print(f"\n===== fold {fold} =====", flush=True)
        r = run_fold(cm, fold, epochs=args.epochs)
        results["folds"][str(fold)] = r
        print(f"fold{fold}: MAE={r['mae']:.4f} ({r['seconds']}s)", flush=True)
        done = [v["mae"] for v in results["folds"].values()]
        results["n_done"] = len(done)
        if done:
            results["mean"] = float(np.mean(done))
            results["std"] = float(np.std(done))
        results["reference"] = {
            "本项目 magpie+RF (官方5折)": 0.4459,
            "官方 CrabNet (完整版, 5折)": 0.3463,
            "官方 RF-SCM/Magpie (5折)": 0.4461,
        }
        results["note"] = ("缩减版配置(d_model=128, 40ep, 无SWA)，是 CrabNet 的性能下限，"
                           "非官方复现；官方完整版见 leaderboard。")
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"已完成 {results['n_done']} 折, 当前 mean={results.get('mean', float('nan')):.4f}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
