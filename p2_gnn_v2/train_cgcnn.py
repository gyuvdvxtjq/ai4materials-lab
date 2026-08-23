"""P2 v3：修正版 CGCNN（对齐原作关键组件）+ 与旧 GNN / RF 同划分对照。

旧版（train_distaware.py）的三个偏差与本次修正：
1. 原子特征：旧版只用 7 个标量属性 → CGCNN 用**可学习 one-hot 元素嵌入**
   （元素身份本身是信息量最大的原子特征，让网络自己学元素表示）。
2. 归一化：旧版无归一化 → 每层卷积后接 **BatchNorm1d**（CGCNN 原作做法，
   对回归稳定性至关重要）。
3. 激活：旧版 ReLU → **softplus**（CGCNN 原作做法，平滑且避免死节点）。
4. 池化：旧版 mean pooling → **sum pooling**（CGCNN 原作用求和池化聚合晶胞）。

协议：与旧版完全一致的数据集（2012 结构）与划分（分层随机，random_state=42），
checkpoint 断点续训（适配不稳定环境）：python p2_gnn_v2/train_cgcnn.py [n_epochs]
产物：p2_gnn_v2/results_cgcnn.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core import Composition, Structure
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

torch.manual_seed(42)
np.random.seed(42)
P2_DIR = Path(__file__).resolve().parent
CKPT = P2_DIR / "ckpt_cgcnn.pt"
MAX_EPOCHS = 250
PATIENCE = 40  # 早停耐心

# 元素表：Z 1..100 的嵌入索引
def z_index(specie) -> int:
    z = specie.Z if hasattr(specie, "Z") else specie.element.Z
    return min(int(z) - 1, 99)


# ---------- 高斯距离展开（与旧版一致，CGCNN 原作做法） ----------
class GaussianDistance:
    def __init__(self, dmin=0.5, dmax=5.0, step=0.25):
        self.centers = np.arange(dmin, dmax + step, step)
        self.width = step

    def expand(self, distances):
        return np.exp(-((distances[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)


GAUSS = GaussianDistance()


def structure_to_data(struct: Structure, y: float, cutoff: float = 6.0, n_neighbors: int = 12) -> Data:
    """CGCNN 原作做法：每个原子只保留 12 个最近邻（不是 cutoff 内所有邻居）。

    实测本数据集 5Å 全邻居平均 601 边/图、12NN 约 169 边——更稀疏、更快，
    且与 CGCNN 原作协议一致。
    """
    x = torch.tensor([[z_index(s.specie)] for s in struct], dtype=torch.long)
    src, dst, dists = [], [], []
    for i, nbrs in enumerate(struct.get_all_neighbors(r=cutoff)):
        nbrs = sorted(nbrs, key=lambda n: n.nn_distance)[:n_neighbors]
        for n in nbrs:
            j = n.index
            if i != j:
                src.append(j)
                dst.append(i)
                dists.append(n.nn_distance)
    if not src:
        src = list(range(len(struct)))
        dst = list(range(len(struct)))
        dists = [2.0] * len(struct)
    edge_attr = torch.tensor(GAUSS.expand(np.array(dists)), dtype=torch.float)
    return Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=edge_attr, y=torch.tensor([y], dtype=torch.float))


# ---------- 修正版 CGCNN ----------
class CGCNNConv(torch.nn.Module):
    """CGCNN 原作卷积：h_i' = BN(h_i + Σ_j h_j ⊙ σ(W_e e_ij))，softplus 激活。"""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.src_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.dst_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden), torch.nn.Sigmoid())
        self.bn = torch.nn.BatchNorm1d(hidden)

    def forward(self, h, edge_index, edge_attr):
        s, d = edge_index
        msg = self.src_proj(h[s]) * self.edge_proj(edge_attr)
        agg = torch.zeros(h.size(0), msg.size(1), device=h.device)
        agg.index_add_(0, d, msg)
        return self.bn(F.softplus(self.dst_proj(h) + agg))


class CGCNN(torch.nn.Module):
    def __init__(self, n_elements: int = 100, edge_dim: int = 19, hidden: int = 64, n_conv: int = 3):
        super().__init__()
        self.embed = torch.nn.Embedding(n_elements, hidden)
        self.convs = torch.nn.ModuleList([CGCNNConv(hidden, edge_dim) for _ in range(n_conv)])
        self.drop = torch.nn.Dropout(0.2)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 64), torch.nn.Softplus(),
            torch.nn.Linear(64, 32), torch.nn.Softplus(),
            torch.nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.embed(x).squeeze(-2)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                h = self.drop(h)
        h = global_add_pool(h, batch)  # CGCNN 原作：sum pooling
        return self.head(h).squeeze(-1)


# ---------- 与旧版一致的组分特征（RF 对照用，来自 train_distaware） ----------
def comp_features(formula):
    c = Composition(formula)
    feats = [c.num_atoms, len(c.elements)]
    for prop in ["X", "Z", "atomic_mass", "electron_affinity",
                 "ionization_energy", "average_ionic_radius", "group", "row"]:
        vals = [float(getattr(el, prop)) * frac for el, frac in c.items()
                if hasattr(el, prop)]
        if not vals:
            feats += [0, 0, 0, 0]
            continue
        v = np.array(vals)
        feats += [v.sum(), v.mean(), v.std() if len(v) > 1 else 0.0, v.max()]
    xs = [el.X for el in c.elements if el.X is not None]
    feats.append(max(xs) - min(xs) if xs else 0.0)
    return feats


def build():
    cache = P2_DIR / "graphs_cache.pt"
    if cache.exists():
        payload = torch.load(cache, weights_only=False)
        graphs, formulas, y_list = payload["graphs"], payload["formulas"], payload["y_list"]
    else:
        rows = json.loads((P2_DIR / "dataset.json").read_text(encoding="utf-8"))
        graphs, formulas, y_list = [], [], []
        for r in rows:
            try:
                graphs.append(structure_to_data(Structure.from_dict(r["structure"]), r["band_gap"]))
                formulas.append(r["formula"])
                y_list.append(r["band_gap"])
            except Exception:
                continue
        torch.save({"graphs": graphs, "formulas": formulas, "y_list": y_list}, cache)
    y_all = np.array(y_list)
    idx = np.arange(len(graphs))
    bins = np.quantile(y_all, [0.25, 0.5, 0.75])
    tr, te = train_test_split(idx, test_size=0.2, random_state=42,
                              stratify=np.digitize(y_all, bins))
    tr, va = train_test_split(tr, test_size=0.125, random_state=42,
                              stratify=np.digitize(y_all[tr], bins))
    return graphs, formulas, y_all, tr, va, te


def evaluate(model, loader):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for b in loader:
            ps.append(model(b.x, b.edge_index, b.edge_attr, b.batch))
            ys.append(b.y.view(-1))
    return torch.cat(ps), torch.cat(ys)


def main() -> None:
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_EPOCHS
    t_start = time.time()
    graphs, formulas, y_all, tr, va, te = build()
    print(f"graphs={len(graphs)}  train={len(tr)} val={len(va)} test={len(te)}  (build {time.time()-t_start:.0f}s)")

    # ===== RF 对照（同划分，与旧版一致） =====
    Xc = np.array([comp_features(f) for f in formulas], dtype=float)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xc[tr], y_all[tr])
    rf_pred = rf.predict(Xc[te])
    rf_mae = mean_absolute_error(y_all[te], rf_pred)
    rf_r2 = r2_score(y_all[te], rf_pred)
    print(f"[RF 对照] MAE={rf_mae:.3f}  R2={rf_r2:.3f}")

    tr_l = DataLoader([graphs[i] for i in tr], batch_size=32, shuffle=True)
    va_l = DataLoader([graphs[i] for i in va], batch_size=64)
    te_l = DataLoader([graphs[i] for i in te], batch_size=64)

    model = CGCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    start_ep, best_mae, best_state, best_ep = 0, 1e9, None, 0
    if CKPT.exists():
        ck = torch.load(CKPT, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep, best_mae = ck["epoch"], ck["best_mae"]
        best_state, best_ep = ck.get("best_state"), ck.get("best_ep", 0)
        print(f"从 ep{start_ep} 续跑 (best val MAE={best_mae:.3f} @ep{best_ep})")
    for _ in range(start_ep):
        sch.step()

    for ep in range(start_ep, min(start_ep + n_ep, MAX_EPOCHS)):
        model.train()
        for b in tr_l:
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.edge_attr, b.batch)
            F.mse_loss(out, b.y.view(-1)).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 5 == 0:
            p, t = evaluate(model, va_l)
            vmae = F.l1_loss(p, t).item()
            if vmae < best_mae:
                best_mae, best_ep, best_state = vmae, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep+1:3d} val_MAE={vmae:.3f}  (best {best_mae:.3f} @ep{best_ep+1})")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_mae": best_mae,
                    "best_state": best_state, "best_ep": best_ep}, CKPT)
        if best_ep and (ep - best_ep) >= PATIENCE:
            print(f"早停：{PATIENCE} 折无改善")
            break
        if time.time() - t_start > 500:  # 单次调用时间预算（适配不稳定环境）
            print("达到单次时间预算，保存 checkpoint 后退出（再次运行继续）")
            return

    # ===== 最终测试评估 =====
    model.load_state_dict(best_state)
    p, t = evaluate(model, te_l)
    gnn_mae = F.l1_loss(p, t).item()
    gnn_r2 = (1 - (((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())).item()
    print(f"\n===== P2 v3 修正版 CGCNN 结果（n={len(graphs)}）=====")
    print(f"RF  (组分特征)     : MAE={rf_mae:.3f}  R2={rf_r2:.3f}")
    print(f"GNN (修正版CGCNN) : MAE={gnn_mae:.3f}  R2={gnn_r2:.3f}  (best ep{best_ep+1})")
    print(f"旧版 GNN 参考     : MAE=0.594（distance-aware，ReLU/无BN/标量特征/mean-pool）")
    out = {"rf_mae": rf_mae, "rf_r2": rf_r2, "gnn_mae": gnn_mae, "gnn_r2": gnn_r2,
           "n": len(graphs), "arch": "cgcnn-fixed",
           "upgrades": ["one-hot embedding", "BatchNorm", "softplus", "sum pooling"],
           "best_epoch": best_ep + 1,
           "reference_old_gnn": 0.594}
    (P2_DIR / "results_cgcnn.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {P2_DIR / 'results_cgcnn.json'}")


if __name__ == "__main__":
    main()
