"""P2 距离感知GNN：checkpoint分段训练版（适配不稳定的执行环境）
每次调用跑 N 个epoch（默认40），从checkpoint续跑，多次调用接力
用法: python train_ckpt.py [n_epochs]
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pymatgen.core import Structure, Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from train_distaware import (GaussianDistance, atom_features,
                              DistAwareConv, CrystalNet, comp_features)

torch.manual_seed(42)
np.random.seed(42)
GAUSS = GaussianDistance()
P2_DIR = Path(__file__).resolve().parent
CKPT = P2_DIR / "ckpt.pt"

def structure_to_data(struct, y, cutoff=5.0):
    x = torch.tensor([atom_features(s.specie) for s in struct], dtype=torch.float)
    src, dst, dists = [], [], []
    for i, nbrs in enumerate(struct.get_all_neighbors(r=cutoff)):
        for n in nbrs:
            j = n.index
            if i != j:
                src.append(j); dst.append(i); dists.append(n.nn_distance)
    if not src:
        src = list(range(len(struct))); dst = list(range(len(struct)))
        dists = [2.0] * len(struct)
    edge_attr = torch.tensor(GAUSS.expand(np.array(dists)), dtype=torch.float)
    return Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=edge_attr, y=torch.tensor([y], dtype=torch.float))

def build():
    rows = json.loads((P2_DIR / "dataset.json").read_text(encoding="utf-8"))
    graphs, formulas, y_list = [], [], []
    for r in rows:
        try:
            graphs.append(structure_to_data(Structure.from_dict(r["structure"]), r["band_gap"]))
            formulas.append(r["formula"]); y_list.append(r["band_gap"])
        except Exception:
            continue
    y_all = np.array(y_list)
    Xall = torch.cat([g.x for g in graphs]); mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    for g in graphs:
        g.x = (g.x - mu) / sd
    idx = np.arange(len(graphs))
    bins = np.quantile(y_all, [0.25, 0.5, 0.75])
    tr, te = train_test_split(idx, test_size=0.2, random_state=42,
                               stratify=np.digitize(y_all, bins))
    tr, va = train_test_split(tr, test_size=0.125, random_state=42,
                              stratify=np.digitize(y_all[tr], bins))
    return graphs, formulas, y_all, tr, va, te, mu, sd

def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    graphs, formulas, y_all, tr, va, te, atom_mean, atom_std = build()
    tr_l = DataLoader([graphs[i] for i in tr], batch_size=32, shuffle=True)
    va_l = DataLoader([graphs[i] for i in va], batch_size=64)
    te_l = DataLoader([graphs[i] for i in te], batch_size=64)

    # RF对照（若已算过直接读缓存）
    import os
    rf_path = P2_DIR / "rf_baseline.json"
    if not rf_path.exists():
        Xc = np.array([comp_features(f) for f in formulas], dtype=float)
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(Xc[tr], y_all[tr])
        rf_pred = rf.predict(Xc[te])
        json.dump({"mae": mean_absolute_error(y_all[te], rf_pred),
                   "r2": r2_score(y_all[te], rf_pred)},
                  rf_path.open("w", encoding="utf-8"))
    rf_res = json.loads(rf_path.read_text(encoding="utf-8"))
    print(f"[RF] MAE={rf_res['mae']:.3f} R2={rf_res['r2']:.3f}")

    m = CrystalNet(graphs[0].x.shape[1], graphs[0].edge_attr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1.5e-3, weight_decay=1e-5)
    start_ep, best_mae, done = 0, 1e9, False
    best_model = None
    if os.path.exists(CKPT):
        ck = torch.load(CKPT)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_mae = ck["epoch"], ck["best_mae"]
        best_model = ck.get("best_model")
        print(f"从 ep{start_ep} 续跑 (best_MAE={best_mae:.3f})")
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    for _ in range(start_ep):
        sch.step()

    for ep in range(start_ep, min(start_ep + n_ep, 150)):
        m.train()
        for b in tr_l:
            opt.zero_grad()
            out = m(b.x, b.edge_index, b.edge_attr, b.batch)
            F.mse_loss(out, b.y.view(-1)).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 20 == 0 or ep == 149:
            m.eval(); ps, ys = [], []
            with torch.no_grad():
                for b in va_l:
                    ps.append(m(b.x, b.edge_index, b.edge_attr, b.batch))
                    ys.append(b.y.view(-1))
            p, t = torch.cat(ps), torch.cat(ys)
            tmae = F.l1_loss(p, t).item()
            if tmae < best_mae:
                best_mae = tmae
                best_model = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            print(f"ep{ep+1:3d} val_MAE={tmae:.3f}")
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_mae": best_mae,
                    "best_model": best_model}, CKPT)
    if (ep + 1) >= 150:
        done = True

    if done:
        if best_model is not None:
            m.load_state_dict(best_model)
        m.eval(); ps, ys = [], []
        with torch.no_grad():
            for b in te_l:
                ps.append(m(b.x, b.edge_index, b.edge_attr, b.batch))
                ys.append(b.y.view(-1))
        p, t = torch.cat(ps), torch.cat(ys)
        gnn_mae = F.l1_loss(p, t).item()
        gnn_r2 = (1 - (((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())).item()
        print(f"\n===== 最终（n={len(graphs)}）=====")
        print(f"RF  : MAE={rf_res['mae']:.3f}  R2={rf_res['r2']:.3f}")
        print(f"GNN : MAE={gnn_mae:.3f}  R2={gnn_r2:.3f}")
        print("GNN 赢!" if gnn_mae < rf_res["mae"] else "RF 仍优")
        torch.save({"state_dict": m.state_dict(), "atom_mean": atom_mean,
                    "atom_std": atom_std, "edge_dim": int(graphs[0].edge_attr.shape[1]),
                    "atom_dim": int(graphs[0].x.shape[1]), "hidden": 64,
                    "arch": "distaware"}, P2_DIR / "crystal_distaware_final.pt")
        json.dump({"rf_mae": rf_res["mae"], "rf_r2": rf_res["r2"],
                   "gnn_mae": gnn_mae, "gnn_r2": gnn_r2, "n": len(graphs)},
                   (P2_DIR / "results_distaware_final.json").open("w", encoding="utf-8"))
        print("训练完成，已保存最终结果")

if __name__ == "__main__":
    main()
