# Phoenix P2 train-only 因果 sign-interior predictor v1（预注册协议）

协议冻结日期：2026-09-12

状态：**train-side protocol frozen；尚未读取 P2 dev，test 禁止使用**。

本协议承接冻结的 P1 No-Go。P2 只检验一个变化：把 P1 的 11 维汇总统计和逻辑回归换成
左右手分离、手形敏感、带固定短历史的严格因果时序表征。P0/P1 的文件、结果目录和声明保持
不变。

## 1. 问题与停止规则

P2 问的是：只使用当前及历史 HRNet WholeBody 关键点，更强的视觉表征能否在 train
calibration 上达到 P0 noisy-center 所要求的定位 gate？

- gate：center recall within `±1` 至少 `75%`，且匹配事件的 positive-delay fraction 不超过
  `25%`；
- gate 失败：冻结为 No-Go，不选择 dev 阈值、不读取 P2 dev、不运行 scheduler；
- gate 通过：冻结全部 train-side 输出后，才允许一次性读取 dev-only，先报告 detection，再连接
  `50% dense budget + span-15` scheduler；
- 无论结果如何，本阶段禁止读取或运行 test-only。

不得用 frame AUROC/AUPRC 代替 event gate，也不得因为 calibration 结果不理想而改变模型、训练
轮数、阈值、事件或匹配规则。

## 2. 数据、target 与隔离

- 只读取 SHA-256 为
  `18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763` 的
  train-only 关键点 pickle；拒绝任何非 `train/` key；
- 复用 P1 的 source-video 哈希划分：fit 5,655 个视频、calibration 1,440 个视频，并排除
  `train/25August_2009_Tuesday_heute-3301`；
- target 不变：alignment-derived nonblank segment midpoint `±1` 帧；blank 不产生正例；它是
  proxy，不是人工逐帧 ground truth；
- fit 用于归一化和权重拟合；calibration 只在固定 8 epochs 训练完成后用于一次阈值扫描与 gate；
- 禁止 gloss identity、reference token、ISLR logits、未来关键点、dev/test alignment 作为输入。

## 3. 固定帧表征

置信度阈值固定为 `0.2`。非有限关键点先记录为无效，再把坐标中和为 0。坐标按已冻结的
Phoenix 原始尺寸 `210 x 260` 归一化。

每帧表征固定包含：

1. 11 个上身/面部 pose 点的全局归一化坐标、有效 mask 和截断置信度；
2. 左右手分别处理的 21 点 wrist-relative 局部坐标和有效 mask；
3. 左右手的 wrist 加 5 个 fingertips 的全局归一化坐标；
4. 左右手有效率、平均置信度、手中心相对肩中心、手间距离；
5. 严格 past-only 的左右手全局速度、局部手形变化、pose 速度、手间距离变化和置信度变化。

所有缺失统计确定性地置零，并保留对应有效率/mask。fit-only mean/std 用于标准化；标准差为
0 的维度置为 1。

## 4. 固定 causal temporal head

- 模型：输入线性投影到 32 channels，随后 4 个 residual causal Conv1d block；
- kernel size `3`，dilation 依次 `1,2,4,8`，左侧补零，receptive field 为 31 帧；
- 每个 block 为 causal convolution、GELU、dropout `0.10`、残差相加；最后逐帧线性二分类；
- 模型在时刻 `t` 的输出只能依赖 `<=t`；单元测试必须用 future perturbation 验证；
- seed `20260912`；按 UUID 单卡绑定一张现场检查为空闲的健康 RTX 3090，禁止使用 PCI
  `01:00.0` 和 `25:00.0`；PyTorch deterministic algorithms；
- fit-only class-balanced BCE-with-logits；AdamW，learning rate `0.002`，weight decay `0.0001`；
- batch `64` 个按长度分桶的完整视频，右侧 padding 只用于组 batch 且不计入 loss；视频间不共享
  历史；gradient clip `1.0`；
- 固定训练 `8` epochs；不做 early stopping，不按 calibration 选择 epoch 或超参数。

## 5. 固定 calibration、事件与 gate

- threshold grid：`0.01` 到 `0.99`，步长 `0.01`；
- event：概率阈值向上穿越后发射，固定 4 帧 refractory；每个视频独立重置状态；
- 匹配：沿用 P1 的 center/event `±1` 帧规则和指标实现；
- 选择：在同时满足 gate 的阈值中选择最高阈值，precision 仅作同阈值确定性 tie-break；
- 没有合格阈值时 `selected_threshold=null`，不得打开 P2 dev。

次要报告包括 fit/calibration frame AUROC、AUPRC、prevalence、Brier、ECE-15，以及固定网格的
全部 event rows。它们不改变 gate。

## 6. 产物与 provenance

大型训练特征写入 `/tmp/phoenix_p2_train_center_sequences_v1.npz`。compact 输出使用新目录：

`Online/CSLR/results/phoenix-2014t_ISLR/p2_train_center_tcn_v1/`

至少保存 resolved config、fit/calibration split、模型权重、train calibration metrics、pre-dev
freeze manifest 和 protocol manifest。实现与测试完成后，manifest 必须记录输入、代码和输出
SHA-256。P2 不得覆盖 `p1_train_center_audit_v1` 或任何 A0/P0 目录。

## 7. 允许声明

在 calibration 完成前只能说“P2 协议已冻结并开始 train-only 实验”。若 gate 失败，只能否定
本协议中的关键点 TCN；若 gate 通过，也只能说它取得 scheduler 评估资格，不能在一次性 dev
replay 前声称 WER 改善。
