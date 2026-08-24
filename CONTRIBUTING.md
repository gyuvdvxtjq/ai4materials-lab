# Contributing

感谢关注 AI4Materials Lab。这个仓库优先接受可复现、边界清晰的实验改进。

## 提交前检查

```bash
python -m compileall -q p1_opt p2_gnn_v2 p3_screen_v2 benchmark
python -m pytest -q
```

涉及指标时，请同时说明数据快照、划分协议、随机种子、特征输入和完整命令。不要把调参结果与外部测试集混用，也不要把组分模型结果描述成结构模型结果。

## Pull Request

- 说明改动动机和影响范围；
- 对新增指标提供可落盘的 JSON 或 CSV 结果；
- 不提交 API key、个人路径、缓存、checkpoint 或面试准备材料；
- 说明无法在 CI 中运行的重量级步骤及其本地验证方式。

项目采用 MIT License。提交代码即表示同意在该许可证下发布。
