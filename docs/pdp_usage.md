# 独立 Node-PDP 入口与发布边界

原 PrivIM 复现命令保持不变。新增机制位于 `pdp/`，不调用旧出现次数会计，
也不给历史 checkpoint 补发 DP 标签。理论与差异见
[整数 B0](b0_integer_privacy.md)、[稳定槽位](pdp_stable_slots_design.md)、
[受约束读出层](pdp_positive_head_design.md)。

## 输入和外生预算

有效图是 igraph 图，包含显式孤点，非空图的每个顶点有唯一整数 `stable_id`。
删除节点时删除其属性及全部关联边，存活节点的稳定 ID 不变。
从现有 prepared split 读取可使用 `pdp.data.load_prepared`，但该读取和原始切分
不是额外获得认证的数据发布步骤。模型保护从逻辑 prepared training graph 开始。

`HashBudgetPolicy` 是预注册公开基准的合成预算分组。
真实外部指定的逐节点预算可使用 `FixedBudgetPolicy`：

~~~python
from pdp.personalization import NodeBudget
from pdp.policies import FixedBudgetPolicy

policy = FixedBudgetPolicy(
    default_budget=NodeBudget("1", "0.000001"),
    overrides={
        101: NodeBudget("2", "0.000001"),
        202: NodeBudget("4", "0.000001"),
    },
)
~~~

策略在所有可能整数 ID 上定义：未列出的 ID，包括当前不存在的节点，使用默认预算。
构造时复制并冻结映射，后续修改调用者字典不会改变策略；预算组顺序确定。
同一策略必须在相邻图两侧保持不变，不能按当前度数、成员或当前最大预算生成。
偏好策略是公共、外生输入；本机制不保护偏好文件本身的保密性。
证书报告公共策略指纹与预算组，不附加实际成员名单。

## 一次模型训练与导出

下面演示稳定槽位入口。`private_graph` 及整个运行过程均留在可信环境。
这是单次调用，不能把一个模型的预算用于任意次数训练。

~~~python
from decimal import Decimal
from pathlib import Path
import torch
from pdp.accounting import RangePrivacy
from pdp.training import TrainingConfig, save_release
from pdp.stable_slots import StableSlots, train_slots

torch.set_num_threads(1)  # 独立进程同时预设 OMP_NUM_THREADS=1，使用 CPU
limits = tuple(
    0 if Decimal(g.epsilon) == 0 or Decimal(g.delta) == 0
    else min(4, max(1, int(Decimal(g.epsilon))))
    for g in policy.groups
)
slots = StableSlots(appearance_limits=limits, trainable_mode="positive_head")
config = TrainingConfig(
    privacy=RangePrivacy(clip_norm="3", batch_size=16, steps=100),
    learning_rate=0.0001,
)
release = train_slots(private_graph, config, policy, slots)
save_release(release, Path("new_model_release"))
~~~

在该入口，实际 epsilon_v/delta_v 来自 `policy` 与逐组槽位会计；
`RangePrivacy` 仅为共享裁剪、格点、B/T 等配置容器，不把其默认标量预算
当成最终模型的证书。查看 `certificate.json` 的逐组上界与参数配置。
若任何可能预算组要求更强保护，校准必须包含该组，即使当前图没有该组节点。
上例参与数规则只是一种固定公共配置，不宣称对任意偏好分布都有最佳效用。

`positive_head` 仍是三层 GRAT，公开冻结编码器，只私有训练读出层；
`full` 模式私有训练全部参数。二者差异及证据必须分别报告。
`noise_enabled=False` 只能作为公开基准的显式非私有控制，不能用于私有部署。
已证明的一次节点抽样路线仍可通过 `pdp.personalization.train_pdp` 调用；
其基础 B0 配置、噪声和效用与直接槽位方案不同，不混用证书。

导出目录只创建新路径，含 `parameters.pt`、`certificate.json`。
实际训练图、槽位成员、掩码、原始损失、失败记录、计时、临时文件及诊断
不能随参数按该证书公开。系统随机源按秘密独立无偏随机位模型信任，
不能设置或导出可重建噪声的固定 seed。

## 公开基准与私有部署的区别

`python -m pdp.run` 只运行显式声明为公开基准的预注册研究协议。
其 manifest、输入摘要、原始诊断和多个模型在 `trusted/` 中，不属于单模型
epsilon_v/delta_v 证书的联合发布。真实私有数据上的调参、多模型公开和选择
需要预先分配/组合预算或采用另行证明的选择机制。

推理只能在固定公开目标图上作为已私有化参数的后处理。
若重新读取受保护图或另一私有图，分数、Top-k、coverage 及成员身份没有
由训练参数证书自动获得保护。不得公开实际私有节点 ID 名单并称为同一 Node-PDP。

已经验证的 CPU 环境与安装扩展见 `requirements-pdp.txt`。
可用 `DGLBACKEND=pytorch OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v`
检查实现不变量；测试不代替数学证明。
