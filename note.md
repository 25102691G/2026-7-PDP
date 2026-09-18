# 原始流程
是的。数据集已经准备完成的情况下，只运行这一条即可：

```bash
.venv/bin/python experiments/run_reproduction.py \
  --device auto \
  --skip-prepare
```

它会依次完成：

1. 训练 6 个数据集；
2. 训练 ε=1–6；
3. 训练 `PrivIM`、`PrivIM+SCS` 和 `PrivIM+`；
4. 每种配置运行 5 个随机种子；
5. 训练每个数据集的非隐私 `PrivIM+`；
6. 自动评估所有模型并生成 `outputs/evaluation/summary.csv`。

需要注意：

- 这是完整实验，共 `6 × 3 × 6 × 5 + 6 × 5 = 570` 次训练，耗时会很长。
- 已存在的同名 checkpoint 会被重新训练并覆盖，脚本没有断点续跑或自动跳过机制。
- 历史 checkpoint 若仍位于可识别的目录中，最终评估可能会把它们一起统计。建议开跑前清理历史产物，或者先用下面的命令查看它将执行什么：

```bash
.venv/bin/python experiments/run_reproduction.py \
  --device auto \
  --skip-prepare \
  --dry-run
```

# 9.17
已按 P1–P5 完成代码调整。现有 `outputs` 和 prepared 数据没有覆盖，新的实验会进入独立 run 目录。

## 已完成的修改

- 隐私会计：
  - 默认 `paper_binomial`，对应论文 Theorem 3。
  - 保留 `fixed_batch_hypergeometric` 作为显式扩展。
  - accountant、敏感度和优化器噪声统一计算。
  - 记录 achieved ε、α、clip、σ、梯度敏感度和最终噪声尺度。
  - 见 [privacy.py](/home/zhaowentao/user_workspace/PrivIM/experiments/privacy.py)。

- 训练协议：
  - 移除论文未说明的 momentum。
  - 私有与非私有训练都按 batch 平均梯度更新。
  - 记录逐轮 loss、梯度范数、裁剪比例和训练时间。
  - 见 [train.py](/home/zhaowentao/user_workspace/PrivIM/experiments/train.py)。

- 采样诊断：
  - 保存 q、τ、L、r、θ、M、μ、BES divisor、Stage 1/2 子图数、最大观测频率和预处理时间。
  - 见 [load_graph.py](/home/zhaowentao/user_workspace/PrivIM/experiments/load_graph.py)。

- 孤立节点协议：
  - 新增 `active` 和 `preserve` 两种明确策略。
  - 论文复现默认 `active`，即划分后仅保留仍参与边的节点。
  - 默认在准备阶段统一删除自环，避免训练采样删除而评估保留。
  - 见 [prepare_datasets.py](/home/zhaowentao/user_workspace/PrivIM/dataset/prepare_datasets.py)。

- 实验隔离：
  - 新实验写入 `outputs/runs/<run_id>`。
  - 生成 `manifest.json`，记录论文 hash、git revision/dirty 状态、数据元信息、完整命令和执行状态。
  - 支持 `--resume`。
  - 评估只扫描本次 run，不再混入历史 checkpoint。
  - 见 [run_reproduction.py](/home/zhaowentao/user_workspace/PrivIM/experiments/run_reproduction.py)。

- 结果审计：
  - 恢复推理时间和内存指标。
  - 新增 [validate_results.py](/home/zhaowentao/user_workspace/PrivIM/experiments/validate_results.py)，可比较论文 Table II 并标记高方差配置。

## 验证结果

- 静态编译检查通过。
- 10 个单元测试全部通过。
- 非隐私端到端 smoke 通过。
- `fixed_batch_hypergeometric` 私有 smoke 通过。
- 默认 `paper_binomial` 私有 smoke 通过：
  - target ε=4；
  - achieved ε=3.9983；
  - 敏感度、噪声和训练诊断均正确写入元数据。
- 隔离模型目录的评估和 summary 生成通过。

## 接下来怎么跑

旧 prepared 数据属于 `legacy-preserve`，因此原来的：

```bash
--skip-prepare
```

现在会主动拒绝运行，避免混用协议。第一阶段需要重新准备 Email 和 Facebook。

先验证非隐私基线：

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset email facebook \
  --only-non-private \
  --round 5 \
  --device auto \
  --run-id validation-non-private
```

检查结果：

```bash
.venv/bin/python experiments/validate_results.py \
  outputs/runs/validation-non-private/evaluation/summary.csv
```

非隐私结果接近论文且稳定后，再跑 ε=4 三种方法：

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset email facebook \
  --settings PrivIM PrivIM+SCS PrivIM+ \
  --epsilon 4 \
  --skip-non-private \
  --skip-prepare \
  --round 5 \
  --device auto \
  --run-id validation-eps4
```

验收 ε=4：

```bash
.venv/bin/python experiments/validate_results.py \
  outputs/runs/validation-eps4/evaluation/summary.csv
```

暂时不要启动完整 570 次实验。建议先确认：

- 非隐私覆盖率接近论文约 98%；
- 单配置标准差不超过约 5 个百分点；
- `PrivIM* / PrivIM+SCS` 整体优于基础 `PrivIM`；
- 没有 top-50 全部落到零出度节点的失败运行。

使用方式和新协议已同步更新到 [README.md](/home/zhaowentao/user_workspace/PrivIM/README.md)。
