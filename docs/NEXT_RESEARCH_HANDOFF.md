# 下一阶段研究交接：decoder-aware 因果 ranking predictor

更新时间：2026-09-14

仓库：`/mnt/workspace/projects/haojun/SLRT`

当前冻结提交：

- `3540ac7`：decoder marginal-utility oracle 与同域二窗口诊断；
- `697cd63`：Phoenix P2 关键点 causal center TCN；
- 本交接所在提交：完整反事实 utility 数据集生成器与紧凑索引。

本文件只描述当前状态与下一步。出现冲突时，以对应实验的 `protocol_manifest.json`、
`dataset_manifest.json` 和机器可读摘要为准。

## 0. 交给新对话时怎么用

新对话直接读取：

```text
/mnt/workspace/projects/haojun/SLRT/docs/NEXT_RESEARCH_HANDOFF.md
```

建议任务描述：

> 核对 decoder-utility oracle、pair diagnostic 和反事实 utility 数据集。先设计 train-side
> dense replay 与互斥的 fit/calibration/evaluation 划分，再冻结严格因果 ranking predictor
> 协议；不得使用同一 dev 数据训练 predictor 后再把该 dev 的 WER 改善当成独立证据。

开始前检查工作区、冻结哈希和是否存在运行中的实验。不得覆盖 A0、P0、P1、P2 或 P3 的已有
结果目录，当前阶段禁止使用 test。

## 1. 当前研究状态

| 阶段 | 冻结结论 |
|---|---|
| A0 自适应步长 | 相对 dense 节省窗口与 wall time，但未显著优于等预算 uniform |
| P0 调度诊断 | label-derived center oracle 有上界；boundary 等简单 proxy 无可靠收益 |
| P1 最小因果 center predictor | 11 维廉价特征未通过 train calibration gate，No-Go |
| P2 关键点 causal TCN | AUROC/AUPRC 提升，但最大 center recall 38.89%，仍未通过 gate，No-Go |
| P3 decoder utility oracle | 50% 总预算、50% skeleton 主配置显著优于等预算 uniform，Strong-Go |
| P3 同域二窗口诊断 | 存在稀疏互补性；单步 utility 保持主标签，二步 rollout 仅作困难状态辅助监督 |
| P3 反事实 utility 数据集 | dev oracle 数据已完整生成；只能用于开发和标签审计，不能直接充当独立训练—评测证据 |

P2 的权威结论见 [`P2_CAUSAL_CENTER_TCN_V1.md`](P2_CAUSAL_CENTER_TCN_V1.md)。统一历史数字见
[`RESULTS.md`](RESULTS.md)。

## 2. Decoder-utility oracle 的决定性结果

实验固定修复后 Phoenix dev、逐样本 50% dense 窗口预算、triangular span-15 decoder，并以
uniform coverage skeleton 加 reference-aware greedy bonus。主配置把总预算的一半给 skeleton、
一半给 bonus：

- 等预算 uniform：WER `23.058447%`，864 errors；
- 50% skeleton + 50% bonus：WER `17.694155%`，663 errors；
- 差值 `-5.364291` 个百分点，少 201 errors；
- 配对 bootstrap 95% CI `[-6.110225, -4.603300]`，满足预注册 Strong-Go。

这是使用 reference 和 dense logits 的离线上界，只证明 decoder-aware 非均匀分配值得预测，
不证明存在可部署 scheduler，也不产生真实 wall time 或在线延迟结论。

同域 pair diagnostic 在 246 个初始单步 utility 全零样本中分层抽取 100 个，穷举 287,855 个
窗口对：8 个样本的最佳二窗口组合比两步 greedy 少 1 个错误；9 个样本存在“两个单独为零、
联合为正”的组合，共 24 对，最大联合 utility 为 1。互补性真实但稀疏，因此冻结为：单步
marginal utility 是主要监督，二步 rollout 只用于全零困难状态的辅助监督与诊断。

## 3. 完整反事实 utility 数据集

目录：

```text
Online/CSLR/results/phoenix-2014t_ISLR/p3_counterfactual_utility_dataset_v1_49faacc3/
```

它沿 50% skeleton 主配置的 reference-aware greedy 轨迹，在每个决策状态对所有未选候选窗口
进行反事实解码，标签定义为：

```text
utility = 当前 edit errors - 加入候选窗口后的 edit errors
```

冻结规模：519 个 dev 样本、13,881 个状态、1,095,130 条候选记录；其中正 utility 671 条、
零 utility 1,062,937 条、负 utility 31,522 条，6,947 个状态的全部单步 utility 为零。类别极度
不平衡，后续应使用候选排序、hard negatives、recall@K/NDCG/utility regret 等中间指标，最终仍以
逐样本等预算 WER 为准。

17 个压缩 shard 共 20,390,810 bytes，只保留本地；Git 提交生成器、schema、config、summary 和
manifest。manifest 保存每个 shard 的大小与 SHA-256，可验证本地数据而无需把约 19.45 MiB
训练记录写入仓库。

重要边界：

- 数据集来自 dev，包含 reference-aware oracle 标签；
- oracle 的集合构造步不是实际流式时间步；
- `predictor_inputs` 仅含无 oracle 泄漏的基础字段，但尚未接入 pose、phase/hazard 或 decoder
  prefix；
- exact per-sample budget 和 EOS 信息只存在于 `oracle_state`，禁止作为部署 predictor 输入；
- candidate logits、reference、当前错误和 future 信息只能生成标签或做审计，禁止作为输入。

## 4. 下一阶段：先解决数据隔离，再训练 predictor

不能在当前 519-sample dev utility 数据上拟合模型，再用同一 dev 的 WER 宣称 predictor 成功。
下一步优先级固定如下：

1. 在 train split 建立与 dev 同定义的 dense ISLR replay；若算力暂不允许，则对现有 dev 做按
   source/video group 的互斥开发拆分，并把结论明确限定为开发性结果；
2. 冻结 fit、calibration 和 evaluation 身份，任何归一化、模型选择、阈值和 early stopping
   只能读取 fit/calibration；
3. 给每个候选窗口按可审计时间索引接入截至当前的廉价 pose/手形/运动、coverage、距上次执行、
   token-bucket，以及可选 CTC prefix/decoder 不确定性；
4. 分别训练和评估 `0/4/8` 帧 lookahead，严格区分 causal 与 bounded-lookahead；
5. uniform skeleton 始终保底，predictor 只分配 bonus；先报告排序指标，再做逐样本等预算 replay；
6. evaluation 通过后才运行真实流式系统，计入 controller、数据搬运、队列积压、wall time 和
   P95 稳定提交延迟。

若 0 帧失败而 4/8 帧成功，只能声称 bounded-lookahead 有效并报告帧延迟。若排序指标好但 WER
不提升，应检查 ranking-to-schedule、coverage、互补性与 decoder 耦合，而不是直接扩大模型。

## 5. 不可变实验身份与边界

- 数据：修复后 Phoenix-2014T；现有 dev 为 519 个样本、55,775 帧、3,747 个 reference gloss；
- 视频 ZIP SHA-256：`49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`；
- checkpoint SHA-256：`b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`；
- 当前窗口/解码：16-frame centered window、triangular span-weighted-15；
- 当前 oracle：逐样本 50% 总预算，主配置 50% skeleton share；
- 改变数据、checkpoint、窗口、decoder、预算或 utility 时必须建立新版本；
- test 当前禁止使用；仓库中的历史 retrospective test 不能用于新方法选择或正式验证。

## 6. 允许与禁止的声明

当前可以说：decoder marginal-utility 在冻结 dev replay 中具有很大的 reference-aware 上界；
单步标签虽有稀疏互补反例，仍适合作为第一版主要监督；完整候选标签已经生成并可验证。

当前不能说：因果 predictor 已学会 utility、scheduler 已优于 uniform、严格在线系统已加速，或
当前 dev 同时提供了无偏训练和独立评测证据。
