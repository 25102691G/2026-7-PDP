# P0：当前 PrivIM 回退基线的代码与论文审计

初审日期：2026-10-26；收尾复核：2026-10-27（Asia/Shanghai）。
HEAD：fff03e7a49fdb3bcff3bce53d3536b926cfe6c13。
结论：当前 PrivIM* 不能认证为所声明预算下的无界节点级 DP，
不能直接用作 Node-PDP 的基础黑盒。

本文件审查当前工作区文件，不把历史版本的修复记录、已有结果的 epsilon 标签、
strict_privacy_claim 字段或单元测试通过视为理论证据。
理论与噪声推导见 [pdp_privacy_design.md](pdp_privacy_design.md)。

## 1. 起始文件状态与范围

首次检查时间：2026-10-26 23:23（Asia/Shanghai）。
起始 git status --short 为：

~~~text
 D tests/test_config.py
 D tests/test_models.py
 D tests/test_oracle_audit.py
 D tests/test_prepare_datasets.py
 D tests/test_privacy.py
 D tests/test_reproduction.py
 D tests/test_sampling.py
 D tests/test_seeding.py
 D tests/test_training_protocol.py
?? docs/PDP实现计划.md
~~~

没有起始已跟踪文件内容修改或暂存变更。原始 P0 审计阶段未恢复这些删除，
未覆盖既有计划，只形成用户指定的四份报告。
审计期间九个测试文件被外部恢复，收尾已逐一核对与 HEAD 内容一致；
这不是本轮执行的操作。其余源码、原 PDF、既有文档及已跟踪 outputs/results
的内容摘要与本轮保存的起始内容快照一致，既有计划内容也未改变。

### 2026-10-27 收尾状态与范围偏差

本次复核发现会话中还存在超出 P0 范围的独立实现及 CPU 试验。
已停止该会话的训练进程（退出码 130），没有删除、重跑或覆盖其文件。
因此不能再把整个会话描述成“只新增四份报告、未实现、未训练、未安装环境”。
这些扩展和试验不作为本次回退基线审计的实现依据，也不用于推翻 No-Go。
本次收尾仅编辑四份指定审计文档，不继续扩展代码或训练。

2026-10-27（Asia/Shanghai）复核状态：

~~~text
已跟踪文件：相对 HEAD 无内容修改、无删除、无暂存变更。
未跟踪：configs/
        docs/PDP实现计划.md
        docs/b0_integer_privacy.md
        docs/p0_go_nogo.md
        docs/p0_pdp_feasibility.md
        docs/p0_privim_audit.md
        docs/pdp_experiment_protocol.md
        docs/pdp_privacy_design.md
        pdp/
        requirements-pdp.txt
        tests/test_b0_integer.py
        tests/test_b0_training.py
        tests/test_pdp_runner.py
        tests/test_pdp_sampling.py
另有被 git 忽略的 .venv/ 与 outputs/pdp_runs/integer-pdp-pilot-v1/。
~~~

已逐个核对全部已跟踪文件与 HEAD 的内容摘要，没有变化或缺失；
原始 PDF 摘要仍符合 AGENTS.md。停止时上述额外运行的 manifest
记录 46 个 completed、1 个 running、13 个 pending；其中 running 是中断前
遗留标记，不表示进程仍运行，不表示该任务完成。没有修改该 manifest。
这批结果不作为隐私证明或本轮要求的实验交付。

已阅读 AGENTS.md、PrivIM复现.md、docs/PDP实现计划.md。
PDF 的实际 SHA-256 与 AGENTS.md 一致：
cf7fc97430366bec111d230d1a945a79c844980dd791373c1d04374514c647ad。
计划文件只作为待审计建议，不能替代原论文或当前实现。

审查主路径：

~~~text
prepare_datasets.py
 -> load_graph.load_train
 -> train_models.train_models
 -> load_graph.random_walking_with_restart
 -> sampling.sample_privim / sample_privim_star
 -> train.train / train_single
 -> privacy.PrivateSGD / multiterm_rdp / rdp_to_dp
 -> state_dict + JSON + manifest + log
 -> evaluate_models + igraph_loader + CELF
~~~

论文映射：PrivIM=Algorithm 1+2；PrivIM+SCS=Algorithm 3 第一阶段；
PrivIM*=Algorithm 3 两阶段；代码 PrivIM+ 是完整方法的兼容别名。
以下“默认”指当前主启动器/CLI，不把 train() 函数内部备用默认值混同为启动器默认值。

风险分级：“阻断”表示不能据此发出当前 Node-DP/PDP 声明；
“高”表示在特定发布对象或选项下可直接破坏前提；
“低”表示该局部实现一致，但不代表整个机制已证明。

## 2. 逐项审计

### I1：裁剪位置与 C / 2C

- 论文依据：Algorithm 2 第 4–8 行；Lemma 2 / Equation (7)；
  Appendix D / Equations (20)–(22)。
- 代码位置：[train.py:983](../experiments/train.py#L983)、
  [privacy.py:112](../experiments/privacy.py#L112)。
- 实际实现：对每个子图单独反传；跨所有可训练参数计算一个全局 L2 范数；
  系数 min(C/(norm+10^-6),1)，再把该子图梯度加入累加器。
  它是每子图全参数裁剪，不是每节点裁剪，也不是先聚合再裁剪。
- 必要假设：梯度有限、C>0；如果使用 C 作为两侧差界，还必须证明改变项
  是相对于零的真正增删，而不是两个非零裁剪梯度间的替换。
- 潜在反例：h=Cu、h'=-Cu 给出 2C。Appendix D 的 [0,C] 标量区间推理
  不适用于向量 L2 球；第 3 节进一步给出当前损失和 GRAT 参数下的相反梯度。
- 风险程度：阻断。
- 是否已证明：有限输入下的单梯度范数≤C 可以证明；梯度差≤C 未证明且一般不成立。

### I2：M 是否约束改变的梯度数

- 论文依据：Algorithm 3、IV-A/D；将 N_g 替换为 M 的 Theorem 3 应用。
- 代码位置：[sampling.py:174](../experiments/sampling.py#L174)、
  [sampling.py:211](../experiments/sampling.py#L211)。
- 实际实现：接受一个子图后，对其唯一节点逐一增加频率；后续候选排除 f≥M；
  两个阶段共用 frequencies，末尾检查最大值不超过 M。
- 必要假设：必须另证明 G 与 G-v 的子图槽位耦合、改变槽位上界；
  仅有每侧频率上界不够。
- 潜在反例：第 3 节的有向路径，M=1，删除端点后 k 个子图全部改变；
  加上 BES 后两侧容器数均为 k，排除了只由容器大小造成的解释。
- 风险程度：阻断。
- 是否已证明：f≤M 的单次运行不变量成立；受影响梯度数≤M 不成立为一般命题。
  多子图影响不仅来自 v 所在子图，还来自其他节点频率和后续游走的变化。

### I3：全局频率、游走与 BES 的数据依赖

- 论文依据：Algorithm 3、Equation (9)、IV-B 的“不增加预算”论述。
- 代码位置：[sampling.py:106](../experiments/sampling.py#L106)、
  [sampling.py:189](../experiments/sampling.py#L189)、
  [sampling.py:247](../experiments/sampling.py#L247)。
- 实际实现：邻居权重取决于此前接受的所有子图；BES 只在 remaining 内游走，
  使用同一原图对象但 allowed 限制使其邻接等价于 remaining 诱导图；
  大小为 max(1,n//s)，全程共享频率。
- 必要假设：要用 M 的隐私界，须分析节点删除引起的整个采样分布改变；
  包括成功/失败游走、随机数消耗、剩余节点集合和补充子图。
- 潜在反例：端点删除使配对整体错位；BES 还改变子图尺寸组成。
  共享频率保证不突破 M，却不保证对邻接图的稳定性。
- 风险程度：阻断。
- 是否已证明：BES 的频率限制与代码结构基本对应论文；“无额外成本”在当前
  出现次数型证明下未成立。若改用独立的路线 B，可作为内部选择而无需额外单列成本，
  但这不是原始 M 敏感度证明。

### I4：基础 PrivIM 的投影与 r-hop 界

- 论文依据：Algorithm 1、Lemma 1、Equation (6)、Appendix C。
- 代码位置：[sampling.py:35](../experiments/sampling.py#L35)、
  [sampling.py:142](../experiments/sampling.py#L142)、
  [train_models.py:101](../experiments/train_models.py#L101)。
- 实际实现：先限制入度，再在每个起点 out r-hop 范围内采样；
  N_g=sum(theta**i,i=0..r)，默认 theta=10,r=3。
- 必要假设：固定图中出现次数界与跨相邻输入重新投影的稳定性是两件事。
  两侧投影后共同节点之间的边也可能不同，需要重新建立耦合。
- 潜在反例：某节点入度 theta+1，在 G 上随机删除一个入边；
  若删除 v 使其入度变 theta，G-v 会保留其他全部入边。
  两侧变化不只是投影图里删除 v。无向图逐顶点投影还有顺序依赖。
- 风险程度：高/阻断基础 PrivIM 的直接认证。
- 是否已证明：单图度数和出现次数的意图对应论文；完整输入图敏感度未证明。
  不能绕回基础 PrivIM 来自动解决 PrivIM* 的缺口。

### I5：真实 batch law 与 RDP accountant

- 论文依据：Algorithm 2 第 3 行；Theorem 3 / Equation (8)；
  Appendix E Definition 8、Equations (23)–(28)。
- 代码位置：[train.py:643](../experiments/train.py#L643)、
  [privacy.py:189](../experiments/privacy.py#L189)、
  [run_reproduction.py:379](../experiments/run_reproduction.py#L379)、
  [train_models.py:541](../experiments/train_models.py#L541)。
- 实际实现：fixed-iterations 下每步重新构造 iterator，从 RandomSampler
  的无放回随机排列取 B 个样本；在理想随机语义下，是固定 m 中均匀 B 子集。
  两个 CLI 默认均为 paper_binomial；直接调用 train()/multiterm_rdp 的
  备用默认却是 fixed_batch_hypergeometric。
  另外，实际 PRNG 从公开 batch_seed 初始化：上述均匀子集描述是对理想随机
  sampler 的 law 描述，不是说固定已知种子下仍有秘密的独立批次随机性。
  当前代码不能仅凭每次重建 iterator 就获得数学上所需的随机概率空间。
- 必要假设：固定、可对齐的共同 m 个槽位；K 个改变槽位；相同混合权重；
  有效的逐槽位差界；对持久预处理和历史适用；噪声方差在邻接输入间一致。
- 潜在反例：第 3 节 m=B=16,M=1，hypergeometric 预测 I=1，
  但实际全部 16 个子图的梯度都可改变。二项式选项也没有正确的“受影响集合”。
- 风险程度：阻断。
- 是否已证明：实际批次的意图可由当前调用确定；所填“受影响计数”的分布
  没有成立。二项式与无放回不相等本身并不证明低估，可能存在保守替代，
  但本轮没有用这种可能性掩盖更根本的耦合缺口。

补充纸面一致性：论文把二项式计数求和上限印为 N_g，实际 Binomial(B,N_g/m)
的支持集为 0..B。N_g<B 时截断会漏概率质量。当前 paper_binomial 代码使用
完整 0..B，是一项有利于规范化的纸面修正，不是逐字照抄 Equation (8)。
Appendix E 又把独立 Bernoulli 纳入和 B 次独立抽取混合描述，不能当作同一 law。
代码 hypergeometric 分支正确调用固定 K,m,B 的超几何 PMF；
但 PMF 数学正确不证明 K 真的是当前机制的受影响槽位数。

### I6：梯度和、平均与噪声尺度

- 论文依据：Algorithm 2 第 7–9 行。
- 代码位置：[train.py:566](../experiments/train.py#L566)、
  [train.py:614](../experiments/train.py#L614)、
  [privacy.py:127](../experiments/privacy.py#L127)。
- 实际实现：私有路径每子图裁剪后求和，只添加一次各参数坐标高斯噪声，
  然后除固定配置 B。令 f=sensitivity_noise_factor，
  S_code=C*sensitivity_occurrences，则实际和噪声 std=sigma*f*S_code，
  均值噪声 std=sigma*f*S_code/B。strict 默认 f=1。
  paper 模式 occurrence=min(N_g,m)，hyper 模式为 min(N_g,m,B)。
- 必要假设：S_code 必须是真正的梯度和敏感度；和/均值不得混用；
  同一对相邻图噪声须有统一分析，梯度及噪声坐标均为有限值。
- 潜在反例：当前采用 CM 但真实改变梯度可达约 2BC；
  固定 B 并不会把错误敏感度变正确。
- 风险程度：局部缩放一致为低；所用敏感度整体为阻断。
- 是否已证明：求和—加噪—除 B 的实现与 Algorithm 2 的结构一致；
  参数 sigma 的隐私含义未证明。

### I7：迭代次数、优化器状态与后处理

- 论文依据：Algorithm 2 的 T 次更新；Definition 5 组合。
- 代码位置：[train.py:412](../experiments/train.py#L412)、
  [train.py:495](../experiments/train.py#L495)、
  [train.py:662](../experiments/train.py#L662)、
  [train.py:738](../experiments/train.py#L738)、
  [train.py:795](../experiments/train.py#L795)。
- 实际实现：默认 fixed-iterations 的实际更新数等于配置 T，
  标定默认使用实际步数；early stopping 调用已注释。默认 momentum=0；
  StepLR 依更新数推进，EMA（若开启）只基于参数更新。
  同一个采样容器会跨 T 步复用。
- 必要假设：每一步条件隐私核正确，隐藏持久状态已纳入证明；
  不能因为外层容器只采一次，就把逐步隐私成本忽略。
- 潜在反例：直接 CLI 允许显式设置 privacy_accounting_steps，
  也允许 strict 配合 released-epochs 或非默认 factor；strict 标签没有
  检查所有理论条件。代码另有实际步数诊断，但标签本身不是认证。
- 风险程度：默认步数结构为低；人为模式覆盖和无条件认证标签为高。
- 是否已证明：默认 T 的实现匹配；完整条件组合未由现有局部会计证明。
  momentum/EMA 可在正确基础上作为已私有化轨迹后处理，不修复错误的基础核。

### I8：n、q、M、delta、m 及噪声的私有依赖

- 论文依据：Algorithm 1/3；Theorem 3 的 m,N_g；IV-C、Appendix H；
  V-A 的 delta 与训练节点数关系。
- 代码位置：[load_graph.py:442](../experiments/load_graph.py#L442)、
  [train_models.py:173](../experiments/train_models.py#L173)、
  [config.py:44](../experiments/config.py#L44)、
  [train.py:325](../experiments/train.py#L325)、
  [train.py:529](../experiments/train.py#L529)、
  [privacy.py:211](../experiments/privacy.py#L211)。
- 实际实现：q=min(1,256/max(1,|V_train|))；主启动器总传 auto-parameters，
  由 metadata.original_nodes 选择 n,M；delta=min(数据集常数,0.9/N_train)；
  实际容器 m 进入 sigma 的数值反解与 occurrence 截断。
- 必要假设：这些量是真正公共的预先配置，或整个数据相关策略有统一证明；
  仅对一次观测到的 m 或 n,M 标定，不足以证明整个算法。
- 潜在反例：删除节点改变其他所有起点的 q；参数接近网格阈值时 n/M 跳变；
  即便均值相同，不同方差的高斯也不符合等协方差 RDP 公式，
  高阶 RDP 甚至可能发散。两个邻接输入各自显示 epsilon≤目标并不解决此问题。
- 风险程度：阻断。
- 是否已证明：参数计算可复现；跨输入的统一机制保证未证明。
  在某些具体样本上 delta 的 min 分支未改变，不代表整个域上都固定。

### I9：数据准备与属性协议

- 论文依据：Definition 2；V-A 的 train/test 划分；论文没有给出
  当前 active/重编号预处理对节点邻接的证明。
- 代码位置：[prepare_datasets.py:158](../dataset/prepare_datasets.py#L158)、
  [prepare_datasets.py:208](../dataset/prepare_datasets.py#L208)、
  [prepare_datasets.py:252](../dataset/prepare_datasets.py#L252)、
  [load_graph.py:82](../experiments/load_graph.py#L82)、
  [graph_util.py:40](../experiments/graph_util.py#L40)。
- 实际实现：从边端点推断节点；按当前实际节点集 shuffle/切半；
  默认主启动器使用 active；写稳定原 ID 到本次局部 ID 的映射；
  loader 依赖 metadata 节点数。默认 ones32；degree/degree32 诊断选项
  在完整消息图上计算并按全图最大度归一化，再诱导子图。
- 必要假设：要继承 prepared-DP 到 raw-DP，须邻接保持；
  保护 all-induced 域时须显式保留孤点/空图；输入属性必须按节点删除一起删除。
- 潜在反例：星形图删除中心后 active 清除全部叶子；
  seed=0、节点 0..5 的 train_order=[4,2,1]，
  删除节点 0 后 train_order=[3,2]，多个存活节点跨 split 改变。
  删除最大度节点还可能改变所有 degree32 特征。
- 风险程度：阻断原始图保证；高于 prepared 边界的其他声明。
- 是否已证明：当前预处理不满足普遍的一节点邻接保持。可在新机制中明确从
  prepared graph 起保护；路线 B 若把整个预处理留在内部，可另走全范围证明。
  默认 ones 并没有 degree32 的全图特征依赖；额外 degree 张量虽被计算，
  当前 GRAT 主路径只读取 feat，不能据此误报默认模型使用了度数特征。

### I10：模型初始化与真正的噪声随机性

- 论文依据：Algorithm 2 第 1、8 行；Appendix A Lemma 5 的随机高斯核。
- 代码位置：[train_models.py:91](../experiments/train_models.py#L91)、
  [train_models.py:262](../experiments/train_models.py#L262)、
  [train_models.py:425](../experiments/train_models.py#L425)、
  [seeding.py:21](../experiments/seeding.py#L21)、
  [privacy.py:152](../experiments/privacy.py#L152)。
- 实际实现：采样后重置初始化 RNG；GRAT3 参数维度不随节点数变化，
  没有发现 BatchNorm 等原始数据运行缓冲区。噪声生成器却由默认 seed=0
  及 round_index 确定地 manual_seed；已实现的是可复现 PRNG 实验流。
- 必要假设：初始化可以是公共且数据独立的；隐私噪声需来自秘密、新鲜、
  与私有选择独立的随机源。有限精度采样器的保证必须明确。
- 潜在反例：给定全部公共种子和确定性 CPU 运算，机制是输入的确定性函数；
  若一对相邻图得到不同参数，取单点输出事件将迫使 delta≥1。
  不公开 JSON 也不能把默认公开常量 0 自动变成随机秘密。
- 风险程度：阻断字面上的已部署 DP 声明。
- 是否已证明：初始化与数据独立这一局部性质可以确认；
  当前公开固定种子模式没有论文假设的高斯概率空间。
  不能把偶然的 CUDA 不确定性当作经过校准的隐私噪声。

### I11：失败、空图、非有限值和条件化成功

- 论文依据：DP 对整个输入域的输出分布定义；Algorithm 2 的固定 T；
  原论文未提供本仓库异常路径的隐私分析。
- 代码位置：[train_models.py:200](../experiments/train_models.py#L200)、
  [train.py:415](../experiments/train.py#L415)、
  [train.py:508](../experiments/train.py#L508)、
  [train.py:293](../experiments/train.py#L293)、
  [privacy.py:112](../experiments/privacy.py#L112)、
  [run_reproduction.py:799](../experiments/run_reproduction.py#L799)。
- 实际实现：m<B 直接 RuntimeError；训练再次检查 m；诊断访问第一个子图；
  sigma 搜索上界固定 100，超出则异常；裁剪没有非有限梯度总定义。
  运行失败写入 manifest，模型文件可能不存在。
- 必要假设：所有诱导图上有共同输出域和机制，失败状态若可观察就必须计入隐私；
  NaN/Inf 必须先映射为合法有界贡献，而不是进入 optimizer。
- 潜在反例：空诱导图必无 B 个子图，却在 PDP 外层有正概率出现；
  “只保留成功的抽样”改变采样分布；无限重抽直到成功不再是所证明的一次 Bernoulli。
  将数据相关异常留作可见失败事件，不能通过只认证成功 checkpoint 来规避。
- 风险程度：阻断 all-induced 基础训练器。
- 是否已证明：没有全输入总定义和可见失败事件的保证。

### I12：模型保存、路径、元数据和日志

- 论文依据：Algorithm 2 只返回 W_T；额外输出必须另证明。
- 代码位置：[train.py:760](../experiments/train.py#L760)、
  [train.py:808](../experiments/train.py#L808)、
  [train.py:845](../experiments/train.py#L845)、
  [load_graph.py:520](../experiments/load_graph.py#L520)、
  [run_reproduction.py:676](../experiments/run_reproduction.py#L676)、
  [prepare_datasets.py:530](../dataset/prepare_datasets.py#L530)。
- 实际实现：pt 只保存 state_dict；JSON 同时保存真实节点数、m、sigma、
  采样子图 hash、初始/末尾探测图输出诊断、真实 loss/梯度范数、种子和计时。
  文件路径含数据相关 n、N_g；manifest 包含 prepared 元数据、命令和状态。
- 必要假设：只对 W_T 宣称 DP；其余产物处于可信内部，公开路径也须独立于私有量。
- 潜在反例：发布 train_nodes 的精确值可在 N 与 N-1 间以概率 1 区分；
  固定 seed 下确定性的子图 hash 不是隐私保护。初始探测图输出在任何私有训练前
  就读取子图，也不属于 W_T 的后处理。
- 风险程度：阻断整套运行产物的 DP 发布。
- 是否已证明：state_dict 导出范围可确认；当前整套目录/日志作为联合输出
  明确不由现有会计覆盖。“strict_privacy_claim=true”只检查 with_dp 和 profile，
  不代表上述假设已验证。

### I13：评估、分数和 Top-k

- 论文依据：III-C 的输出打分与 Top-k，V-A 的 IC/CELF 评估；
  DP 后处理只能读取已私有化输出及固定公开辅助输入。
- 代码位置：[evaluate_models.py:51](../experiments/evaluate_models.py#L51)、
  [evaluate_models.py:183](../experiments/evaluate_models.py#L183)、
  [evaluate_models.py:317](../experiments/evaluate_models.py#L317)、
  [igraph_loader.py:64](../experiments/igraph_loader.py#L64)。
- 实际实现：读取 prepared test graph 和特征，输出 logits 排名；
  统计 spread、CELF、图 n/m、计时和排名诊断。主表 k=50、d=1；
  d=1 时按确定的一跳覆盖计算，等价 IC 边概率 1。
- 必要假设：若要作为训练图 DP 的后处理，目标图及辅助特征必须对相邻训练输入固定；
  需要额外声明目标图是否公共，不能自动声称保护 test 节点。
- 潜在反例：在同一私有图直接发布 v 的 Top-k 入选事件；或发布私有目标图的精确 n/m。
  原始图重新划分会使 test 图也改变，不能把它当作固定辅助信息。
- 风险程度：高；阻断未声明边界的端到端“私有种子选择”声明。
- 是否已证明：评估协议可用于公开数据效用比较；
  当前训练 DP 假设即便修复，也不自动赋予私有目标图 DP。

### I14：多轮复现、恢复与选择结果

- 论文依据：Definition 5 顺序组合；Algorithm 2 的 T 是单次训练步数。
- 代码位置：[train_models.py:138](../experiments/train_models.py#L138)、
  [run_reproduction.py:507](../experiments/run_reproduction.py#L507)、
  [run_reproduction.py:689](../experiments/run_reproduction.py#L689)。
- 实际实现：多 dataset/epsilon/method/round 分别训练；恢复通过已有文件和
  元数据跳过已完成任务，并不是一个已证明的全局隐私过滤器。
- 必要假设：同一真实私有数据上重复发布多个模型/调参结果，须全局组合；
  每次真正新增随机训练不能重新使用一次预算的标签。
- 潜在反例：五次各标 epsilon=1 的模型共同公开，被误写为联合 epsilon=1。
  依据真实 test 效用选择模型也不能未经分析当成公共超参数选择。
- 风险程度：高。
- 是否已证明：实验隔离、恢复功能是复现工程能力；没有多次发布级别的隐私证书。

## 3. 一个与当前代码对应的形式化反例

该反例否定一般的“M 频率界⇒M 个改变梯度”，不是用经验精度估计隐私。
参数取当前采样代码支持的一组值，不声称是 Table II 的默认数据集或默认超参数：
n=2、M=1、s=2、num_subgraph=256、tau=0.3、L=200、mu=1。

取顶点 0..2k-1。加入所有 i→i+1 的前向边，并为偶数 i 加反向边 i+1→i。
因此偶数位置的相邻对是双向边，跨对连接是单向边。
取 k=16 时 N=32；在 G 与 G-0 上，q 都为 1。

从小到大遍历起点时：

~~~text
G 的 SCS：   [0,1], [2,3], …, [30,31]；
G 的 BES：   无。
G-0 的 SCS： [1,2], [3,4], …, [29,30]；
G-0 的 BES： [31]。
~~~

每个成功起点只有一个频率未满的 out 邻居；tau 的重启不会改变第一次行走。
每个两节点子图在收集第二个节点后结束。失去伙伴的末端节点只能在 BES
以大小 1 被接受。因此这是由代码控制流确定的轨迹，不依赖选中特定随机数。
两侧 m 均为 16，所有节点频率均为 1；删除节点 0，却改变全部 16 个槽位。
k 可扩展，只要讨论固定公共 q=1 的采样函数即可；16 的例子本身已足够否定 M=1。

还可展示梯度变化并不只是“节点 ID 不同”：

- 采用当前 legacy_released GRAT3 和 ones 输入，令各层权重、前两层 bias 为零，
  最后一层 bias b=log(4)，故所有节点输出 p=sigmoid(b)=0.8。
- 最后层权重输入为零，且其权重为零阻断上游梯度，所以仅最后一层 bias 有非零梯度。
- 使用当前 KSetMaxCoverAdjLoss，lambda=0.5，d=1；
  baselines/heuristics.py:23 在 d≤1 时直接复制子图，graph_util.py:125 添加单位对角线。
- 双向两节点子图损失为 2(1-p)^2+2lambda*p，bias 梯度 +0.032。
- 单向两节点子图损失为 (1-p)+(1-p)^2+2lambda*p，bias 梯度 -0.064。
- 单节点子图损失为 (1-p)+lambda*p，bias 梯度 -0.08。

选择受支持的 C=0.01，按代码 norm+10^-6 裁剪公式进行实数/标量计算得到：

~~~text
h_bi     = +0.00999968750976532；
h_single = -0.00999984375244137；
h_node   = -0.00999987500156248。
B=m=16 时两侧梯度和之差 = 0.31999253144442813。
CM=0.01；即便换成 2CM=0.02 也不足。
~~~

这也是保守 2BC=0.32 范围界接近可达的例子。
它是对任意参数状态全局界的代数反例，不是声称已从默认初始化训练到这个状态，
也不是当前六数据集 checkpoint 的实测隐私攻击。当前 C=3 等具体配置不能仅由
这个 C=0.01 数字推断出其真实泄露量；其原有证明仍缺失，不能因此获得认证。

核查方式：本轮用标准库 AST 读取当前 sampling.py 中的五个纯采样函数，
在提供 vcount/neighbors 的最小图接口上执行上述 32/31 节点轨迹；
得到 stage1/stage2 分别 16/0 与 15/1，max_f 都为 1。
没有导入训练器，没有改源文件，没有使用 GPU。
代数推导是反例依据，这次执行仅检查报告没有误读控制流。

## 4. 本轮检查的能力与限制

初次审计使用的系统 Python 未安装 torch、igraph、scipy；九个原测试文件
首次检查时已删除，审计期间由外部恢复为 HEAD 内容。这是初审时的环境记录，
不是本次复核后的完整环境描述。当前还存在独立 .venv 和额外测试文件，
本次收尾没有执行这些测试、导入新增 PDP 训练器或继续训练。
已完成源码逐段阅读、论文对应附录阅读、标准库有限轨迹核查和公式数值计算。
2026-10-27 又从当前 sampling.py 的 AST 读取五个纯采样函数与结果类，
复核得到 G 的 stage1/stage2=16/0、G-0 的 15/1，两侧最大频率均为 1；
60 位 Decimal 代数计算的梯度和差为 0.319992531444428119…。
这些执行只验证反例没有误读控制流，不是经验隐私证明。

明确区分：

1. 单子图裁剪、共享频率和固定迭代计数有局部可证明性质。
2. 当前出现次数型全图敏感度与会计缺少关键假设，并有反例否定其一般推理。
3. 整套元数据发布有直接泄露，公开固定噪声种子也不提供所需高斯随机性。
4. 对最终参数的实际最紧隐私预算本轮未计算；“当前不能认证”不等于给出
   六数据集上某个新的实测 epsilon，也不否定已有复现的效用记录。

## 5. 最小必要修正的审计意见

若“最小”意味着只改一个常数，同时保留所有现有隐私声明：**不存在已论证的方案**。
2C 修正不处理全局重采样；换成超几何不处理错误的受影响集合；
只隐藏 JSON 不处理错误校准、空图失败及公开固定噪声种子。

保留复现入口和默认行为不动时，最小可审查的独立修正路线是新增 B0：
把现有 SCS/BES 留在可信内部，固定 B,C,T、架构、delta 和绝对噪声，
以 2BC/2C 范围界重新会计，所有诱导图上补零执行固定 T 步，
秘密新鲜噪声、有限梯度映射、参数专用导出和统一失败合同。
它是新的基础机制，不能继续沿用原 M 会计或认证历史 checkpoint。

路线 A 若坚持原样获得更小噪声，必须先提供对当前采样器成立的跨输入
稳定耦合与统一噪声分析；第 3 节表明不能简单以 K=M 完成。
