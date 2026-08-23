"""P2 CIF 推理入口（需要使用修复后的训练脚本重新训练权重）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pymatgen.core import Structure
from torch_geometric.loader import DataLoader

from train_distaware import CrystalNet, structure_to_data

P2_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="由 CIF 结构预测带隙")
    parser.add_argument("cif", type=Path)
    parser.add_argument("--model", type=Path, default=P2_DIR / "crystal_distaware.pt")
    args = parser.parse_args()
    artifact = torch.load(args.model, map_location="cpu")
    if "state_dict" not in artifact or artifact.get("atom_mean") is None:
        parser.error("模型缺少节点特征归一化统计量，请用新版 train_distaware.py 重新训练")
    struct = Structure.from_file(args.cif)
    graph = structure_to_data(struct, 0.0)
    graph.x = (graph.x - artifact["atom_mean"]) / artifact["atom_std"]
    loader = DataLoader([graph], batch_size=1)
    model = CrystalNet(artifact["atom_dim"], artifact["edge_dim"], artifact["hidden"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        value = float(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)[0])
    print(f"结构: {args.cif}")
    print(f"预测带隙: {value:.3f} eV")


if __name__ == "__main__":
    main()
