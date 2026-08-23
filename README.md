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

## P1 v3：金属/半导体分类 + 带隙回归拆分（新增）

P1 v2 的 R²=0.714 是「虚高」的：约 60% 样本带隙恰好为 0（金属），回归器只需「认出金属」就能拿到好看的数字。v3 把任务拆成两个更诚实、也更贴近 MatBench 官方定义的问题（实现见 `p1_opt/train_split.py`，指标见 `p1_opt/p1_split_metrics.json`）：

| 任务 | 指标（5 折 CV，mean±std） | 说明 |
|---|---:|---|
| metal/non-metal 分类 | ROC-AUC **0.952 ± 0.006** | magpie 组分特征可靠判别金属性，且跨折稳定 |
| 非金属子集带隙回归 | MAE **0.522 ± 0.012** eV（HGB 调参后） | 半导体/绝缘体的「真实」带隙预测难度 |
| 全量带隙回归（参考） | MAE 0.337，R² 0.689（单次划分） | R² 主要来自识别 60% 的零带隙金属 |

回归同时输出不确定度（随机森林树间标准差，均值 ≈0.745 eV），供 P3 筛选排序。

**调参结论（诚实版）**：对 HGB 做网格搜索（`learning_rate`/`max_iter`/`max_leaf_nodes`），分类 ROC-AUC 0.9504→0.9514（≈无变化），回归 MAE 0.5332→0.5222（≈2%）。**调参收益很小——瓶颈是组分特征本身（无法捕捉结构决定的带隙），不是模型超参**。要实质提升，方向是结构 GNN（P2）或更多数据，而非在 RF/HGB 上调参。严格评估脚本见 `p1_opt/evaluate_cv.py`。

```bash
python p1_opt/train_split.py      # 训练 → bandgap_split.joblib + p1_split_metrics.json
python p1_opt/predict.py LiFePO4  # 输出 P(金属) + 预测带隙 ± 不确定度 + 类别
```

## MatBench 对标（官方数据 + 官方 5 折划分，已实测）

在 MatBench 官方数据集和 5 折划分上实测本项目基线（脚本 `benchmark/run_matbench_official.py`，结果见 `benchmark/matbench_results.json`）：

| MatBench 任务 | 本项目（magpie+RF） | 官方 RF-SCM/Magpie | 官方 MODNet | 官方最强 |
|---|---:|---:|---:|---:|
| matbench_expt_gap（MAE↓） | **0.4459 ± 0.0182** | 0.4461 | 0.3327 | Darwin 0.2865 |
| matbench_expt_is_metal（ROC-AUC↑） | **0.9721 ± 0.0024** | 0.9356 | 0.9739 | — |

两个值得说的结果：
1. **回归与官方 RF 基线一字不差**（0.4459 vs 0.4461）——验证了「magpie+RF 就是组分基线的水平上限」，也证明评估管线正确。
2. **分类大幅超过官方 RF 基线**（0.972 vs 0.936），几乎追平 MODNet（0.974）——magpie 特征对「金属性判别」的分辨率远高于对「带隙数值」的分辨率。

```bash
pip install matbench
python benchmark/run_matbench_official.py   # 官方 5 折，结果落盘
python benchmark/run_matbench.py            # 同协议，支持更多任务
```

## CrabNet 基线（官方 5 折，已实测）

给纯组分路线补一个更强的模型：CrabNet（组分编码 + 注意力）。缩减版配置（d_model=128、40 epochs、无 SWA），是 CrabNet 的**性能下限**而非官方复现（脚本 `benchmark/run_crabnet.py`，结果见 `benchmark/crabnet_results.json`）：

| 模型 | matbench_expt_gap MAE（5 折） |
|---|---:|
| magpie + RF（本项目） | 0.4459 ± 0.0182 |
| **CrabNet（缩减版，本项目实测）** | **0.3953 ± 0.0118** |
| CrabNet（官方完整版参考） | 0.3463 |

即：同为纯组分输入，CrabNet（即使缩减版）比 RF 好 11%，且自带预测不确定度（平均 ≈0.42 eV）。`run_crabnet.py` 内置 crabnet 1.0.1 与 torch≥2.x 的全部兼容补丁（包名 bug / 缺失数据文件 / 优化器 hook / DataLoader 死锁 / BLAS 死锁），可直接运行。

**MODNet**（`pip install modnet`，需 tensorflow）为另一个值得加的组分基线（专为小数据设计，`mp_gap` 0.220），可在相同划分下按 `train_split.py` 的 `models` 字典模式接入。

## P2 v3：修正版 CGCNN（新增，GNN 首次稳定胜出 RF）

旧版 GNN（`train_distaware.py`）与 RF 打平（0.594 vs 0.591）。逐项对齐 CGCNN 原作后（`p2_gnn_v2/train_cgcnn.py`，结果见 `p2_gnn_v2/results_cgcnn.json`），同样 2012 结构、同划分：

| 模型 | MAE |
|---|---:|
| RF 组分基线 | 0.593 |
| 旧版 GNN（ReLU/无BN/标量特征/mean-pool/全邻居） | 0.594 |
| **修正版 CGCNN（one-hot嵌入/BatchNorm/softplus/sum-pool/12近邻）** | **0.574** |

架构修正清单（每项都是对 CGCNN 原作的偏离）：
1. **原子特征**：7 个标量属性 → 可学习 one-hot 元素嵌入（元素身份是信息量最大的特征）；
2. **归一化**：无 → 每层卷积后 BatchNorm1d；
3. **激活**：ReLU → softplus（原作做法）；
4. **池化**：mean → sum pooling（原作做法）；
5. **图构建**：5Å 全邻居（平均 601 边/图）→ 每原子 12 最近邻（约 169 边，与原作一致，且训练提速 3.6 倍）。

**结论**：GNN 之前「打平」不是数据量的全部原因——架构偏差也是。修正后 GNN 在小数据上也稳定胜出（0.574 vs 0.593），且「结构信息价值」的叙事现在有了正向证据。支持 checkpoint 断点续训与早停。

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

# P1 v3：金属/半导体分类 + 带隙回归拆分（推荐）
python p1_opt/train_split.py

# P1 v3：5 折交叉验证 + 调参（评审前必跑，得到 mean±std）
python p1_opt/evaluate_cv.py

# P1：化学式推理 Demo
python p1_opt/predict.py LiFePO4

# MatBench 标准对标（可选，需 matbench 包 + 联网）
python benchmark/run_matbench.py

# MatBench 官方 5 折对标（结果落盘 benchmark/matbench_results.json）
python benchmark/run_matbench_official.py

# CrabNet 基线（官方 5 折；单折约 3-4 分钟 CPU，逐折落盘可断点续跑）
python benchmark/run_crabnet.py --folds 0 1 2 3 4

# P2：完整训练（数据量较大，CPU 可能耗时较久）
python p2_gnn_v2/train_distaware.py

# P2 v3：修正版 CGCNN（推荐；支持断点续训，再次运行接着 ckpt_cgcnn.pt 继续）
python p2_gnn_v2/train_cgcnn.py 250

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
  train_split.py          # v3：分类 + 回归拆分 + 不确定度
  evaluate_cv.py          # v3：5 折交叉验证 + 调参
  p1_split_metrics.json   # v3 指标（单次划分）
  p1_cv_metrics.json      # v3 指标（5 折 CV mean±std）
  predict.py
  bandgap_magpie.joblib
p2_gnn_v2/
  dataset.json
  train_distaware.py
  train_cgcnn.py         # v3：修正版 CGCNN（对齐原作）
  results_cgcnn.json     # v3 结果
  train_ckpt.py
  predict_cif.py
p3_screen_v2/
  screen_v2.py
  screening_results_v2.csv
benchmark/
  run_matbench.py            # MatBench 对标（通用版）
  run_matbench_official.py   # MatBench 官方 5 折（结果落盘）
  run_crabnet.py             # CrabNet 基线（含 1.0.1 兼容补丁）
  matbench_results.json      # 官方 5 折实测结果
  crabnet_results.json       # CrabNet 5 折实测结果
tests/
  test_p1_split.py
pyproject.toml
INTERVIEW_CARDS.md
results_final.png
```

## 方法与限制

- P1 在化学式归一化后去重，再进行分层随机划分，避免同一成分的多晶型泄漏。
- P2 旧版使用 `Structure.get_all_neighbors`（5Å 全邻居）+ 高斯距离展开；v3 修正版改为每原子 12 最近邻，与 CGCNN 原作一致。
- P2 的测试集只在最终评估时使用；checkpoint 依据 validation MAE 保存。
- 当前数据中约 60% 样本带隙为 0，R² 需要结合金属/非金属分类指标和非零带隙子集误差解读。
- CrabNet 结果为缩减版配置（d_model=128、40 epochs、无 SWA）的性能下限，不是官方完整版复现；官方完整版 5 折 0.346。
- matbench_mp_gap / mp_is_metal（10.6 万结构）尚未在本仓库实测——数据量大，建议本地跑 `benchmark/run_matbench.py --task matbench_mp_gap`。
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
