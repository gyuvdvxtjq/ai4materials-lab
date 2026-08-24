# AI4Materials Lab

可复现的 AI for Materials 实验项目，围绕三个递进问题组织：

1. **P1：组分基线**：用 Magpie 描述符预测金属性和带隙。
2. **P2：结构模型**：用周期晶体图和 CGCNN 对照组分模型，检验结构信息的增益。
3. **P3：候选预筛**：枚举化学式，串联金属性、带隙和稳定性模型。

项目重点不是宣称“发现新材料”，而是展示一条有明确评估协议、可复现实验和诚实限制说明的材料机器学习工作流。

## 结果概览

仓库中的 JSON/CSV 是已运行的实验快照；修改训练协议后应重新运行脚本再更新结果。当前快照包括：

| 实验 | 数据/协议 | 快照结果 |
| --- | --- | --- |
| P1 MatBench `expt_gap` | 官方 5 折，Magpie + RF | MAE `0.4459 ± 0.0182` eV |
| P1 MatBench `expt_is_metal` | 官方 5 折，Magpie + RF | ROC-AUC `0.9721 ± 0.0024` |
| P1 `mp_gap` 组分基线 | 官方 5 折，Magpie + HGB | MAE `0.3313 ± 0.0041` eV |
| P2 修正版 CGCNN | 2012 个结构，固定划分 | MAE `0.574` eV |
| P3 稳定性预筛 | Magpie + 5 折分类 CV | ROC-AUC `0.899 ± 0.004` |

这些数字不等价于官方榜单成绩：CrabNet 是缩减配置的本地基线，MODNet 是小数据/统一 Magpie 特征下的对照，P3 只做组分层面的粗筛。

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

运行轻量测试：

```bash
python -m pytest -q
python -m compileall -q p1_opt p2_gnn_v2 p3_screen_v2 benchmark
```

## 推荐流程

### P1：组分模型

```bash
python p1_opt/train_split.py
python p1_opt/evaluate_cv.py
python p1_opt/predict.py LiFePO4
```

`train_split.py` 使用验证集选择模型，测试集只用于最终报告；`evaluate_cv.py` 使用外层 5 折、内层 3 折的嵌套交叉验证，避免调参泄漏。首次训练会生成被 `.gitignore` 排除的 `bandgap_split.joblib`。

### P2：晶体结构模型

可选依赖：`python -m pip install -e ".[gnn]"`

```bash
python p2_gnn_v2/train_cgcnn.py 250
python p2_gnn_v2/predict_cif.py path/to/structure.cif
```

训练支持 checkpoint 续跑。图缓存和 checkpoint 不纳入版本控制。

### P3：候选预筛

```bash
python p3_screen_v2/train_stability.py
python p3_screen_v2/screen_v3.py --lo 1.0 --hi 3.0 --top 30
```

P3 默认完全离线运行。若要做可选数据库核验，通过环境变量提供兼容服务地址：

```powershell
$env:MP_BASE_URL = "http://localhost:8000"
```

`P(stable)` 是组分模型概率，不是形成能计算，也不能替代结构预测、实验验证或 Materials Project 查询。

## MatBench / 基线实验

可选依赖：`python -m pip install -e ".[benchmark]"`

```bash
python benchmark/run_matbench_official.py
python benchmark/run_matbench_mp_gap.py --fold 0
python benchmark/run_crabnet.py --folds 0 1 2 3 4
python benchmark/run_modnet.py
```

其中 `run_matbench_mp_gap.py` 按折运行并落盘，首次执行会下载约 137 MB 数据并生成本地特征缓存。CrabNet/MODNet 需要额外深度学习依赖，详见脚本头部说明。

## 目录结构

```text
.
├── p1_opt/                  # 组分特征、P1 训练、CV 和推理
├── p2_gnn_v2/               # 晶体图、CGCNN 训练和 CIF 推理
├── p3_screen_v2/            # 稳定性模型和候选预筛
├── benchmark/               # MatBench、CrabNet、MODNet 基线
├── tests/                   # 轻量回归测试
├── pyproject.toml           # 安装元数据和可选依赖
├── requirements.txt         # 基础运行环境
└── LICENSE                  # MIT License
```

训练数据快照和已发布结果保留在仓库中，运行时缓存、模型 checkpoint、API 配置和个人文件均由 `.gitignore` 排除。

## 数据与限制

- P1 数据是仓库内的 6772 条去重化学式快照；P2 数据是 2012 个结构快照。
- 约 60% 的 P1 样本带隙为 0，必须同时看金属性分类和非金属子集误差，不能只看全量 R²。
- 组分描述符无法表达同一化学式不同晶体结构的差异；P2 的结构增益应在相同划分上比较。
- 所有外部榜单数字仅作协议对照，不代表本项目重新复现了官方完整模型。
- P3 输出的是候选化学式预筛结果，不是“新材料”证明。

## 开发与贡献

请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。提交指标时请附数据、划分、随机种子、特征和完整命令；不要提交 API key、个人路径、缓存、checkpoint 或面试准备材料。

## 许可证

本项目以 [MIT License](LICENSE) 发布。
