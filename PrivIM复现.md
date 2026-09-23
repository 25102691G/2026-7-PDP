# PrivIM Table II：复现命令与当前结果

记录日期：2026-09-23

本文档保存当前代码复现 Table II 所使用的完整命令，以及已经完成的实验结果。
以下命令应在仓库根目录运行：

```bash
cd /home/zhaowentao/user_workspace/PrivIM
```

## 1. 完整复现命令

### 1.1 私有方法：epsilon = 1 和 4

该命令运行六个数据集、两个隐私预算、三种方法和五个随机种子，共
`6 x 2 x 3 x 5 = 180` 个训练轮次组合。

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset email bitcoin lastfm hepph facebook gowalla \
  --epsilon 1 4 \
  --settings PrivIM PrivIM+SCS PrivIM+ \
  --loss-penalty 0.5 \
  --iterations 100 \
  --round 5 \
  --device cuda \
  --skip-non-private \
  --run-id final-table2-private
```

说明：代码参数 `PrivIM+` 对应论文 Table II 中的完整方法 `PrivIM*`
（SCS + BES），不是表中的 `PrivIM+SCS`。

### 1.2 Non-Private

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset email bitcoin lastfm hepph facebook gowalla \
  --only-non-private \
  --loss-penalty 0.5 \
  --iterations 500 \
  --round 5 \
  --skip-prepare \
  --device cuda \
  --run-id final-table2-nonprivate
```

两条命令按顺序执行。第一条会准备数据；因此第二条使用 `--skip-prepare`。
若第一条运行前数据已经准备完成，也可以给第一条添加 `--skip-prepare`。

如果运行中断，保持原命令和 `run-id` 不变，并在末尾增加 `--resume`：

```bash
# 示例：恢复私有方法实验
.venv/bin/python experiments/run_reproduction.py \
  --dataset email bitcoin lastfm hepph facebook gowalla \
  --epsilon 1 4 \
  --settings PrivIM PrivIM+SCS PrivIM+ \
  --loss-penalty 0.5 \
  --iterations 100 \
  --round 5 \
  --device cuda \
  --skip-non-private \
  --run-id final-table2-private \
  --resume
```

不要删除同名运行目录后再使用 `--resume`。如果需要从头重新运行，建议更换
`--run-id`，以保留已有结果。

## 2. 本次实验配置

| 项目 | 私有方法 | Non-Private |
|---|---:|---:|
| 数据集 | Email、Bitcoin、LastFM、HepPh、Facebook、Gowalla | 相同 |
| 随机运行次数 | 5 | 5 |
| loss penalty（lambda） | 0.5 | 0.5 |
| 训练迭代数（T） | 100 | 500 |
| batch size | 16 | 16 |
| momentum | 0 | 0 |
| epsilon | 1、4 | infinity |
| 评估指标 | Coverage Ratio（%），均值 +/- 样本标准差 | 相同 |

论文 Equation (5) 定义了正的权衡系数 `lambda`，但没有公开 Table II
实验采用的具体数值；论文也没有报告训练迭代数 `T`。因此，这里的
`lambda=0.5`、私有方法 `T=100` 是当前代码经过受控实验选出的统一配置，
不是论文作者明确公布的超参数。released-code 兼容配置仍为 `lambda=1`。

## 3. 当前代码的实测结果

以下结果来自已经完成的运行：

- Non-Private：`outputs/runs/uniform-lambda05-nonprivate`，HepPh 补充结果来自
  `outputs/runs/diagnostic-hepph-lambda05`。
- PrivIM*、epsilon=1：`outputs/runs/uniform-lambda05-eps1-t100`。
- PrivIM*、epsilon=4：`outputs/runs/uniform-lambda05-eps4-t100`。
- PrivIM 和 PrivIM+SCS：
  `outputs/runs/paper-ablation-lambda05-t100-eps1-4`。

### 3.1 Non-Private

| 数据集 | 当前代码 | 论文 Table II |
|---|---:|---:|
| Email | 96.85 +/- 1.86 | 98.09 +/- 1.69 |
| Bitcoin | 98.82 +/- 0.69 | 98.99 +/- 0.46 |
| LastFM | 96.15 +/- 0.94 | 97.93 +/- 0.97 |
| HepPh | 92.12 +/- 7.98 | 98.42 +/- 0.31 |
| Facebook | 93.94 +/- 3.49 | 97.50 +/- 0.34 |
| Gowalla | 99.12 +/- 0.61 | 99.78 +/- 0.03 |
| **六数据集平均** | **96.17** | **98.45** |

相对论文 Table II 的六数据集平均绝对误差（MAE）为 **2.28**。

### 3.2 epsilon = 1

| 数据集 | PrivIM | PrivIM+SCS | PrivIM* |
|---|---:|---:|---:|
| Email | 72.50 +/- 33.08 | 83.52 +/- 9.30 | 82.59 +/- 9.49 |
| Bitcoin | 74.00 +/- 37.60 | 92.11 +/- 5.13 | 89.16 +/- 6.68 |
| LastFM | 74.14 +/- 35.63 | 83.42 +/- 18.99 | 90.23 +/- 6.39 |
| HepPh | 63.32 +/- 32.00 | 70.09 +/- 26.04 | 83.71 +/- 13.57 |
| Facebook | 68.85 +/- 37.53 | 87.02 +/- 8.66 | 87.40 +/- 10.85 |
| Gowalla | 76.77 +/- 42.52 | 95.69 +/- 1.88 | 97.78 +/- 1.36 |
| **六数据集平均** | **71.60** | **85.31** | **88.48** |
| **论文六数据集平均** | **54.06** | **82.41** | **85.16** |
| **相对 Table II 的 MAE** | **17.99** | **6.53** | **4.34** |

### 3.3 epsilon = 4

| 数据集 | PrivIM | PrivIM+SCS | PrivIM* |
|---|---:|---:|---:|
| Email | 71.57 +/- 28.82 | 86.06 +/- 11.43 | 86.48 +/- 11.90 |
| Bitcoin | 74.10 +/- 36.38 | 93.93 +/- 3.45 | 96.64 +/- 2.86 |
| LastFM | 78.30 +/- 34.56 | 94.55 +/- 1.76 | 96.03 +/- 1.48 |
| HepPh | 68.41 +/- 32.49 | 79.24 +/- 26.64 | 81.83 +/- 1.77 |
| Facebook | 74.33 +/- 37.27 | 90.00 +/- 10.22 | 93.44 +/- 1.70 |
| Gowalla | 76.94 +/- 42.37 | 98.01 +/- 1.13 | 98.55 +/- 0.79 |
| **六数据集平均** | **73.94** | **90.30** | **92.16** |
| **论文六数据集平均** | **77.89** | **92.86** | **94.85** |
| **相对 Table II 的 MAE** | **4.16** | **5.64** | **4.89** |

## 4. 结果结论与稳定性

- epsilon=1 时，SCS 相对 PrivIM 平均提升 **13.71** 个百分点，BES 再提升
  **3.17** 个百分点。
- epsilon=4 时，SCS 相对 PrivIM 平均提升 **16.36** 个百分点，BES 再提升
  **1.86** 个百分点。论文的对应提升约为 14.97 和 1.99 个百分点，消融趋势
  基本一致。
- 基础 PrivIM 的五次运行标准差仍然很大，不能把单个 seed 的结果当作稳定结果。
- 完整 PrivIM* 明显更稳定，但 epsilon=1 的 HepPh 等结果仍有一定波动；正式报告
  应始终保留五次运行的均值和标准差。
- 在同一机器、同一软件环境与相同代码下，结果应具有较好的统计可复现性；CUDA、
  DGL 或 GPU 型号改变时，不保证逐位一致。

## 5. 输出位置

上述完整命令分别写入：

```text
outputs/runs/final-table2-private/
outputs/runs/final-table2-nonprivate/
```

运行结束后，以各目录中的 `manifest.json` 和汇总文件为准。为避免混合旧实验，
重新测试时优先使用新的 `run-id`，不要直接覆盖已有目录。

## 6. 论文与实现对应关系

- 主要对照目标：`PrivIM_Report.pdf` 的 Table II。
- 损失函数权衡系数：论文 Equation (5)。
- 方法名对应关系：论文 `PrivIM*` = 代码 `PrivIM+` = SCS + BES。
- 本文所列 `lambda=0.5` 和私有方法 `T=100` 是当前复现实验的有意偏离，目的在于
  降低当前严格隐私实现下的方差并提升平均结果，不能表述成论文公开参数。
