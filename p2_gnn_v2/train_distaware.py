"""P2 架构升级版：距离感知消息传递网络（CGCNN 风格）+ 扩充数据
关键升级（对比之前 GCN 版）：
 1. 边特征 = 高斯距离展开(GaussianDistance, CGCNN原作做法)，不再丢弃距离信息
 2. 消息传递显式融合边特征：h_j * e_ij 求和聚合
 3. 结合同数据 RF 对照，验证"结构信息+正确架构"的增益
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from pymatgen.core import Structure, Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

torch.manual_seed(42)
np.random.seed(42)
P2_DIR = Path(__file__).resolve().parent

# ---------- 高斯距离展开（CGCNN 风格） ----------
class GaussianDistance:
    def __init__(self, dmin=0.5, dmax=5.0, step=0.25):
        self.centers = np.arange(dmin, dmax + step, step)
        self.width = step
    def expand(self, distances):
        return np.exp(-((distances[:, None] - self.centers[None, :]) ** 2)
                      / self.width ** 2)

GAUSS = GaussianDistance()

def atom_features(el):
    feats = []
    for p in ["X", "Z", "atomic_mass", "atomic_radius", "electron_affinity", "row", "group"]:
        try:
            feats.append(float(getattr(el, p)))
        except Exception:
            feats.append(0.0)
    return feats

def structure_to_data(struct, y, cutoff=5.0):
    x = torch.tensor([atom_features(s.specie) for s in struct], dtype=torch.float)
    src, dst, dists = [], [], []
    for i, nbrs in enumerate(struct.get_all_neighbors(r=cutoff)):
        for n in nbrs:
            j = n.index
            if i != j:
                d = n.nn_distance
                src.append(j); dst.append(i); dists.append(d)  # 有向：j -> i
    if not src:
        src = list(range(len(struct))); dst = list(range(len(struct)))
        dists = [2.0] * len(struct)
    edge_attr = torch.tensor(GAUSS.expand(np.array(dists)), dtype=torch.float)
    return Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))

# ---------- 距离感知消息传递网络 ----------
class DistAwareConv(torch.nn.Module):
    """CGCNN 风格：h_i' = h_i + sum_j( h_j ⊙ σ(W_e · e_ij) )"""
    def __init__(self, in_h, in_e, out_h):
        super().__init__()
        self.src_proj = torch.nn.Linear(in_h, out_h, bias=False)
        self.dst_proj = torch.nn.Linear(in_h, out_h, bias=True)
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(in_e, out_h), torch.nn.Sigmoid())
    def forward(self, h, edge_index, edge_attr):
        s, d = edge_index
        msg = self.src_proj(h[s]) * self.edge_proj(edge_attr)
        agg = torch.zeros(h.size(0), msg.size(1), device=h.device)
        agg.index_add_(0, d, msg)
        return self.dst_proj(h) + agg

class CrystalNet(torch.nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden=64):
        super().__init__()
        self.embed = torch.nn.Linear(atom_dim, hidden)
        self.c1 = DistAwareConv(hidden, edge_dim, hidden)
        self.c2 = DistAwareConv(hidden, edge_dim, hidden)
        self.c3 = DistAwareConv(hidden, edge_dim, hidden)
        self.drop = torch.nn.Dropout(0.2)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.embed(x))
        h = F.relu(self.c1(h, edge_index, edge_attr)); h = self.drop(h)
        h = F.relu(self.c2(h, edge_index, edge_attr)); h = self.drop(h)
        h = F.relu(self.c3(h, edge_index, edge_attr))
        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)

# ---------- P1 式组分特征（RF 对照） ----------
def comp_features(formula):
    c = Composition(formula)
    feats = [c.num_atoms, len(c.elements)]
    for prop in ["X", "Z", "atomic_mass", "electron_affinity",
                 "ionization_energy", "average_ionic_radius", "group", "row"]:
        vals = [float(getattr(el, prop)) * frac for el, frac in c.items()
                if hasattr(el, prop)]
        if not vals:
            feats += [0, 0, 0, 0]; continue
        v = np.array(vals)
        feats += [v.sum(), v.mean(), v.std() if len(v) > 1 else 0.0, v.max()]
    xs = [el.X for el in c.elements if el.X is not None]
    feats.append(max(xs) - min(xs) if xs else 0.0)
    return feats

def evaluate(model, loader):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for b in loader:
            ps.append(model(b.x, b.edge_index, b.edge_attr, b.batch))
            ys.append(b.y.view(-1))
    return torch.cat(ps), torch.cat(ys)

def main(dataset_path=None):
    dataset_path = Path(dataset_path) if dataset_path else P2_DIR / "dataset.json"
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"dataset: {len(rows)} structures")

    graphs, formulas, y_list = [], [], []
    for r in rows:
        try:
            graphs.append(structure_to_data(Structure.from_dict(r["structure"]),
                                            r["band_gap"]))
            formulas.append(r["formula"]); y_list.append(r["band_gap"])
        except Exception:
            continue
    y_all = np.array(y_list)
    print(f"graphs: {len(graphs)}, edge_dim={graphs[0].edge_attr.shape[1]}")

    # 标准化节点特征
    Xall = torch.cat([g.x for g in graphs]); mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    for g in graphs:
        g.x = (g.x - mu) / sd

    idx = np.arange(len(graphs))
    bins = np.quantile(y_all, [0.25, 0.5, 0.75])
    tr, te = train_test_split(idx, test_size=0.2, random_state=42,
                              stratify=np.digitize(y_all, bins))
    tr2, va = train_test_split(tr, test_size=0.125, random_state=42,
                               stratify=np.digitize(y_all[tr], bins))
    tr_l = DataLoader([graphs[i] for i in tr2], batch_size=32, shuffle=True)
    va_l = DataLoader([graphs[i] for i in va], batch_size=64)
    te_l = DataLoader([graphs[i] for i in te], batch_size=64)
    print(f"train={len(tr2)} val={len(va)} test={len(te)}")

    # ===== RF 对照 =====
    Xc = np.array([comp_features(f) for f in formulas], dtype=float)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xc[tr], y_all[tr])
    rf_pred = rf.predict(Xc[te])
    rf_mae = mean_absolute_error(y_all[te], rf_pred)
    rf_r2 = r2_score(y_all[te], rf_pred)
    print(f"\n[RF 组分特征]  MAE={rf_mae:.3f}  R2={rf_r2:.3f}")

    # ===== 距离感知 GNN =====
    m = CrystalNet(graphs[0].x.shape[1], graphs[0].edge_attr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

    best_mae, best_state, best_ep = 1e9, None, 0
    for ep in range(401):
        m.train()
        for b in tr_l:
            opt.zero_grad()
            out = m(b.x, b.edge_index, b.edge_attr, b.batch)
            F.mse_loss(out, b.y.view(-1)).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            p, t = evaluate(m, va_l)
            vmae = F.l1_loss(p, t).item()
            if vmae < best_mae:
                best_mae, best_ep = vmae, ep
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
        if ep % 100 == 0:
            p, t = evaluate(m, va_l)
            print(f"  ep{ep:3d} val_MAE={F.l1_loss(p, t):.3f}")

    m.load_state_dict(best_state)
    p, t = evaluate(m, te_l)
    gnn_mae = F.l1_loss(p, t).item()
    gnn_r2 = (1 - (((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())).item()
    print(f"\n===== 结果（早停ep{best_ep}, n={len(graphs)}）=====")
    print(f"RF  (组分特征)      : MAE={rf_mae:.3f}  R2={rf_r2:.3f}")
    print(f"GNN (距离感知消息传递): MAE={gnn_mae:.3f}  R2={gnn_r2:.3f}")
    imp = "GNN赢" if gnn_mae < rf_mae else "RF赢"
    print(f"结论: {imp}（差距 {abs(rf_mae-gnn_mae):.3f} eV）")
    torch.save({"state_dict": m.state_dict(), "atom_mean": mu,
                "atom_std": sd, "edge_dim": int(graphs[0].edge_attr.shape[1]),
                "atom_dim": int(graphs[0].x.shape[1]), "hidden": 64,
                "arch": "distaware"}, P2_DIR / "crystal_distaware.pt")
    json.dump({"rf_mae": rf_mae, "rf_r2": rf_r2, "gnn_mae": gnn_mae,
               "gnn_r2": gnn_r2, "n": len(graphs), "arch": "distaware",
               "split": "train/validation/test, random_state=42"},
              (P2_DIR / "results_distaware.json").open("w", encoding="utf-8"))

if __name__ == "__main__":
    main()
