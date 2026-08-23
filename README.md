# AI4Materials Lab

面向 AI for Science / AI4Materials 实习的三个递进 Demo：

1. **P1：组分特征带隙预测**：Materials Project DFT 数据 + Magpie 描述符 + RF/HGB。
2. **P2：晶体结构 GNN 对照**：周期性原子图 + 距离感知消息传递，并与同划分 RF 对照。
3. **P3：候选筛选闭环**：化学式枚举 → 模型预筛 → Materials Project 查询。

## 已验证结果

| 项目 | 数据 | 指标 | 说明 |
|---|---:|---:|---|
| P1 | 6772 个去重化学式 | MAE 0.324 eV，R² 0.714 | Magpie 组分描述符；结果见 `p1_final_metrics.json` |
| P2 | 2012 个结构 | GNN MAE 0.594，RF MAE 0.591 | 同一数据集、同一划分；GNN 与基线基本持平 |
| P3 | 240 个枚举候选 | 30 个候选完成数据库查询 | “未检索到”不等于现实中的新材料 |

数据指标是项目快照，不是 MatBench 官方榜单结果；如需严格 MatBench 对标，请单独使用官方数据集和 split。

## 数据与 API 说明

当前仓库已经包含训练所需的数据快照，因此 **P1/P2 首次运行不需要 Materials Project API key，也不需要联网**：

| 用途 | 仓库文件 | 是否需要 API |
|---|---|---|
| P1 训练 | `p1_opt/materials_final.csv`（6772 条） | 否 |
| P2 训练 | `p2_gnn_v2/dataset.json`（2012 个结构） | 否 |
| P1/P2 推理 | 已保存模型文件 | 否 |
| P3 数据库验证 | `/materials/search` 服务 | 需要一个兼容的 HTTP 服务 |
| P2 数据扩充 | `p2_gnn_v2/expand_v2.py` | 需要一个兼容的 HTTP 服务 |

P3 使用的是可配置的 `MP_BASE_URL`，仓库不再写入内部服务地址。若改用 Materials Project 官方服务重新拉取数据，则需要自行申请 MP API key，并把抓取结果保存为本地快照；这不是当前 Demo 的必要步骤。

## 环境

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

所有命令均从仓库根目录执行：

```bash
# P1：训练并自动按验证集 MAE 保存最佳模型
python p1_opt/train_magpie.py

# P1：化学式推理 Demo
python p1_opt/predict.py LiFePO4

# P2：完整训练（数据量较大，CPU 可能耗时较久）
python p2_gnn_v2/train_distaware.py

# P2：资源有限时分段训练，每次接着 ckpt.pt 继续
python p2_gnn_v2/train_ckpt.py 40

# P2：CIF 推理（需先用新版训练脚本生成带归一化统计量的权重）
python p2_gnn_v2/predict_cif.py path/to/structure.cif

# P3：候选筛选
python p3_screen_v2/screen_v2.py
```

P3 和数据扩充脚本需要一个兼容 `/materials/search` 的服务。为避免提交内部地址，运行时设置：

```bash
# PowerShell
$env:MP_BASE_URL = "http://localhost:8000"
# Bash
export MP_BASE_URL=http://localhost:8000
```

服务不可用时，P3 仍可完成本地枚举和模型预测，但数据库验证会标记为查询失败。

## 目录

```text
p1_opt/
  materials_final.csv
  train_magpie.py
  predict.py
  bandgap_magpie.joblib
p2_gnn_v2/
  dataset.json
  train_distaware.py
  train_ckpt.py
  predict_cif.py
p3_screen_v2/
  screen_v2.py
  screening_results_v2.csv
INTERVIEW_CARDS.md
results_final.png
```

## 方法与限制

- P1 在化学式归一化后去重，再进行分层随机划分，避免同一成分的多晶型泄漏。
- P2 使用 `Structure.get_all_neighbors` 处理周期性邻居，并用高斯距离展开构造边特征。
- P2 的测试集只在最终评估时使用；checkpoint 依据 validation MAE 保存。
- 当前数据中约 60% 样本带隙为 0，R² 需要结合金属/非金属分类指标和非零带隙子集误差解读。
- P3 是候选化学式预筛，不包含晶体结构生成、形成能计算或实验可行性证明。

## GitHub 发布注意

- 仓库内没有 API key、密码、个人路径或内部服务 IP；服务地址通过 `MP_BASE_URL` 注入。
- `ckpt.pt`、Python 缓存和本地日志已加入 `.gitignore`。
- `*.joblib` 模型文件由 Git LFS 管理；克隆仓库时请确保已安装 Git LFS。
- 项目采用 MIT License，详见 `LICENSE`。
- 上传前建议执行：

```bash
git grep -n -i -E "token|api[_-]?key|password|secret"
git grep -n -E "http://|https://|[A-Za-z]:/"
python -m compileall -q .
```

## 面试材料

`INTERVIEW_CARDS.md` 记录 P1/P2/P3 的迭代过程、指标和局限，面试时应以代码和可复现实验为准，不夸大“全新材料”或 GNN 的性能优势。
