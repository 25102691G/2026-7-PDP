# Node-PDP 实现与六数据集评估报告

日期：2026-10-27。原始复现基线 HEAD：
`fff03e7a49fdb3bcff3bce53d3536b926cfe6c13`。

## 结论及当前决策

已经完成独立 Node-PDP 机制、数学证明、实现检查、预注册实验与六数据集评估。
**尚未证实个性化私有训练带来额外 Coverage 收益。不能把工程完成写成主效用
研究目标成功，也不能把公开结构先验带来的高 Coverage 归于个性化学习。**

当前决策：

- 原 PrivIM* 的出现次数会计仍是 No-Go；没有修改原复现代码或认证历史模型。
- 新机制的模型参数保证，在所列邻接、输入域、独立槽位计算及秘密随机位模型下成立。
- “PDP 优于同结构公开初始化或统一 DP”的效用主张为 No-Go，当前实验不支持。
- 已达到两次开发机制修订的预注册上限，停止新增机制/参数/seed 搜索。
  后续研究方向与新的实验额度需要重新明确，不能自动增加一轮来挑好结果。

## 可复查交付

| 部分 | 位置 |
|---|---|
| 原实现审计与反例 | [p0_privim_audit.md](p0_privim_audit.md)、[p0_go_nogo.md](p0_go_nogo.md) |
| 定义、保护边界、保守训练级证明 | [pdp_privacy_design.md](pdp_privacy_design.md) |
| 一次节点抽样 PDP 的双向证明 | [p0_pdp_feasibility.md](p0_pdp_feasibility.md) |
| 整数 B0 与精确噪声 | [b0_integer_privacy.md](b0_integer_privacy.md)、`pdp/accounting.py`、`pdp/lattice.py`、`pdp/discrete_noise.py` |
| 稳定槽位及逐节点会计 | [pdp_stable_slots_design.md](pdp_stable_slots_design.md)、`pdp/stable_slots.py` |
| 最后一轮公开编码器与私有读出 | [pdp_positive_head_design.md](pdp_positive_head_design.md)、`pdp/positive_head.py` |
| 任意外生节点预算策略 | `pdp/policies.py`、[pdp_usage.md](pdp_usage.md) |
| 预注册及停止规则 | [pdp_experiment_protocol.md](pdp_experiment_protocol.md)、`configs/pdp/` |
| 全部逐单元指标与核验 | [all_runs.csv](../outputs/pdp_reports/final-v1/all_runs.csv)、[verification.json](../outputs/pdp_reports/final-v1/verification.json) |
| 全部私有模型预算数值复核 | [privacy_verification.json](../outputs/pdp_reports/final-v1/privacy_verification.json) |
| 配对差值与种子集重合度 | [paired_differences.json](../outputs/pdp_reports/final-v1/paired_differences.json) |
| 完整结果表及图 | [results.md](../outputs/pdp_reports/final-v1/results.md)、[PDF 图](../outputs/pdp_reports/final-v1/coverage_comparison.pdf) |

## 隐私保证究竟是什么

无界节点增删邻接：删除 v 同时删除其属性及全部关联边，存活节点稳定 ID 和
外生预算策略不变。保护从逻辑 prepared training graph 开始；认证输出是固定
架构的模型参数与公共证书。原始数据切分脚本没有由此获得认证。

整数 B0 对每步至多 B 个整数球贡献使用和敏感度 2BR，固定 B 均值的物理界
至多 2C。T 步条件组合包括隐藏容器后验差异，两个邻接方向均已证明。
在此基础上，一次独立非均匀节点抽样的界为：

~~~text
p_v ≤ min(1, expm1(epsilon_v)/expm1(epsilon_0), delta_v/delta_0)；
epsilon_bound_v = log1p(p_v*expm1(epsilon_0))；
delta_bound_v = p_v*delta_0。
~~~

概率向下包络为有理数；外层只抽一次，不按 T 重复放大。该路线在首轮被完整评估。

后两轮使用独立的直接槽位 PDP：所有槽位预先存在，节点 v 只路由到 M_v 个
固定槽位；其他槽位的诱导图、局部随机流和查询不变。每槽位查询精确限制在
整数球，故双向和敏感度为 2M_v R，且：

~~~text
rho_v = T*(2M_v R)^2/(2*sigma_lattice^2)；
epsilon_cert_v ≤ rho_v + 2sqrt(rho_v*log(1/delta_v))。
~~~

共同噪声按所有公共预算组的最大需求标定，包括实际图中没有出现的组。
基准预算组为 {1,2,4}、delta_v=10^-6，参与上限为 {1,2,4}。
实际认证 epsilon 上界约为 {0.952387,1.936516,4.000000}。
该 M_v 是已证明的改变槽位数，不是原 SCS 的内部出现频率 M。

噪声采用整数/Fraction 精确离散高斯拒绝采样，先整数相加再转浮点。
校准使用 Fraction 与 Decimal 向外舍入，不用经验 PMF 检查代替证明。
生产路径没有公开噪声 seed。数学保证依赖秘密、独立、无偏随机位及受控的
确定性 CPU 查询环境；不宣称认证未审查的 GPU、外部可见计时或系统故障渠道。

目标图必须是固定公开辅助输入。重新读取私有图产生分数、Top-k 或 coverage，
不由训练参数保证自动覆盖。偏好策略本身是公共外生输入，未保护偏好文件。
单模型证书也不是多个模型、非私有控制、输入 hash 或整套研究目录的联合证书。

## 研究过程与全部轮次

| 阶段 | 评估单元 | 结果及动作 |
|---|---:|---|
| 整数 B0 + 一次节点抽样 v1 | 60 | base2 在 Email 83.56%、Bitcoin 51.72%；base4 的 Bitcoin 26.03%；不扩大该配置 |
| 稳定槽位全参数 v2 | 60，其中 20 个控制复用 | PDP 63.98% / 25.19%；敏感度证明有效，但效用未改善 |
| 公开编码器 + 私有读出 v3 | 50 | PDP 93.06% / 98.36%，通过预设稳定性关口；公开初始化已经近乎相同 |
| 冻结配置四数据集确认 | 120 | 不再调参；全部方法、全部 5 个模型 seed 保留 |

共 290 个评估单元、150 个私有模型调用。20 个复用控制都在新目录标注来源，
没有重复训练已完成控制或剔除较差 seed。首轮因阶段范围调整被明确中断的
单元尚无模型，后续恢复完成；所有已完成单元保留。没有使用观察超时推断作业失败。

非私有全参数 T=500 是前两轮的上参照；最后一轮匹配控制是相同公开编码器、
槽位、裁剪、T=100、无噪声的读出训练。两者不能混作同一个 Non-Private。
最终 `full` 与 `positive_head` 两种训练模式都保留，但后者冻结编码器，只私有
训练 33 个读出参数；不能称为原 PrivIM 的全参数复现。

## 最终六数据集结果

相同 prepared split，IC(p=1)、一步、k=50、相同原 CELF 分母。
所有方法均有完整的 5 个模型 seed；下面只列主要对照的均值，标准差及其他
控制见完整结果表，不省略失败开发轮。

| 数据集 | 公开初始化 | 统一 DP：低参与 | 统一 DP：全参与 | Node-PDP |
|---|---:|---:|---:|---:|
| Email | 93.0556 | 93.0556 | 93.0556 | 93.0556 |
| Bitcoin | 98.3967 | 98.3967 | 98.3967 | 98.3610 |
| LastFM | 94.8779 | 94.8779 | 94.8779 | 94.8779 |
| HepPh | 81.3900 | 81.3900 | 81.3900 | 81.3900 |
| Facebook | 89.0592 | 89.1986 | 89.0592 | 89.0592 |
| Gowalla | 96.8172 | 96.8172 | 96.8069 | 96.8172 |
| 六数据集等权均值 | **92.266084** | **92.289313** | **92.264376** | **92.260146** |

Node-PDP 相对公开初始化为 -0.005938 个百分点，相对较强的统一 DP 对照为
-0.029167 个百分点。不能将这些结果写成个性化方法更优或显著改善。
公开初始化不读取受保护训练图；这里的“数据独立”仅针对训练输入，推理仍读取
固定公开目标图。该任务设置下，公开结构映射已经解释了几乎全部最终 Coverage。

读出梯度确实非零，冻结参数实际保持不变，并非误把冻结参数传给 autograd 后
全部失败回退。但参数被训练不意味着 Top-k 或 Coverage 一定受益。
checkpoint 诊断显示特征变化高度集中。确认后进一步采用向外区间支配、
固定核心和 14,029 个候选集合的精确计数，得到固定读出家族的六数据集
宏平均 Coverage 上界 93.136820%，相对当前 PDP 最多剩余 0.876674 个百分点；
两个开发数据集最多剩余 0.486826 个百分点。完整假设、逐数据集结果与有理数
重算见[家族容量分析](pdp_readout_capacity_results.md)。
它限制当前固定编码器与读出盒，**没有证明所有 GNN/PDP 方法不能改进，也没有
证明当前家族不能取得任何正增益**。停止依据是实验、预注册规则及上述家族限制。

## 核验及兼容事项

完整 121 项检查通过，18.159 秒。独立核验了 290 个单元的完整性、产物摘要、
原 prepared 摘要、固定扩散协议，并逐单元从公开 seed 集重新计算 spread，
对六个目标图重新计算相同 CELF 分母。150 个私有模型的 rho、转换上界、
逐组 epsilon/delta、外层一次性抽样条件均以高精度重算。
这些是实现和证据核验，数学隐私保证仍来自前述证明。
确认后的只读区间/集合分析另有四项专项检查通过；没有因此重跑训练或覆盖旧结果。

最终 API 又显式规范了无向边的端点顺序，避免未来依赖库改变简化排序时破坏
稳定查询。已执行环境为 python-igraph 0.11.5 / C igraph 0.10.12；其新构图与
随后强制 simplify 已完成相同规范化，故此次强化不改变已执行实验的查询。
各轮旧源码快照保持不动；不得因当前源码摘要不同而覆盖或重跑旧目录。
依赖行为依据为 igraph 0.10.12 的官方
[igraph_create](https://github.com/igraph/igraph/blob/0.10.12/src/constructors/basic_constructors.c)、
[igraph_add_edges](https://github.com/igraph/igraph/blob/0.10.12/src/graph/type_indexededgelist.c) 与
[igraph_simplify](https://github.com/igraph/igraph/blob/0.10.12/src/operators/simplify.c)：
新边构造使多重边缓存失效；强制简化按规范端点顺序构造结果。
该源码核对只用于固定环境中历史查询与显式规范化的对应，不推广到任意未来依赖版本。

论文依据：Definition 2、Algorithm 2、Equation (5)、Lemma 2、Theorem 3、
Algorithm 3、Appendix D/E 和 V-A。整数格点、2BC 范围会计、稳定公开槽位、
外生逐节点预算、冻结公开编码器及读出盒均为明确的新设计，不归因于原论文。
原 PrivIM 入口、默认行为、数据准备和既有结果未改。

## 尚未完成的主研究预期

当前结果可作为严格机制构造、实现审计和负结果材料；不足以作为“个性化隐私
改善 IM Coverage”的实证主结论。整体效用目标仍未证实，不能标记为成功。
若继续，需要新的研究假设及预注册，而非继续同配置重复抽噪声。
改变扩散概率、hop 数或其他论文评估默认值属于另一个评估问题，必须单独明确，
不能用来替换当前不利的六数据集比较。
