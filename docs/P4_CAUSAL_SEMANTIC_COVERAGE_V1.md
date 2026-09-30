# P4：因果语义轨迹覆盖（CSC-L8-v1）预注册

冻结时间：2026-09-21。本文在任何 P4 target audit、probe OOF 或策略 outcome 产生前冻结。

## 1. 假设与实验身份

当前 robust counterfactual utility 标签依赖 reference、真实 EOS 和完整 future continuation，且在
366,802 个 side rows 中只有 769 beneficial。P4 不再预测这个标签，而检验以下不同假设：

> 一个昂贵 ISLR 窗口提供的**当前语义观测**，可由决策时可见的廉价视觉线索预测；在固定
> coverage skeleton 之外，按一段轨迹的语义重建收益分配 bonus，比逐候选 utility 分类更稳定。

本实验固定名为 `CSC-L8-v1`（causal semantic coverage, bounded lookahead 8）。现有16帧窗口和
candidate deadline 含 bounded `+8`，因此不得称为 zero-lookahead。未来的 `CSC-L4-*` 和
`CSC-L0-*` 必须重建相应 trailing/shifted window target、单独预注册、使用新输出目录，不能从
本实验结果外推。

## 2. 数据身份与禁止项

- 只使用完整 train fit：6,378 samples、578 source groups、183,401 blocks；
- folds 精确复用 `robust-oof-v1` 的5折 source-disjoint 映射；
- 视频、keypoints、checkpoint、旧 skeleton OOF 的 SHA-256 分别为
  `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`、
  `18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763`、
  `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`、
  `9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a`；
- Phase A 不读取 robust reward、reference、WER、future continuation、真实 EOS、dev 或 test；
- 在 Phase A gate 通过前，不训练正式 predictor，不读取 P4 train policy outcome；
- 在 249-error gate 通过前，不运行 unknown-EOS closed loop；
- 不覆盖任何 P3 目录，不对旧 utility OOF 做 threshold/model/feature tuning。

## 3. 新 target：局部语义曲率

昂贵 teacher 仅用于训练标签。对每个完整4帧 block，令 offset 0 与下一个 block 的 offset 0 为
两个 anchor；offset `c in {1,2,3}` 的 dense ISLR logit 经温度1 softmax 得到 `q_c`。线性语义轨迹为
`m_c=(1-c/4)q_0+(c/4)q_4`，主 target 为

`S_c = JS(q_c || m_c)`。

这一定义只依赖当前及 bounded-L8 可见范围内的昂贵窗口观测，不读 reference、EOS 或窗口后的
continuation。它是稠密连续 target，不等价于“该动作最终减少几个 WER error”。最后一个没有
`q_4` 的 block 不参与 target 审计，部署时使用冻结的 uniform fallback。

稳定性敏感度在看结果前固定为：softmax temperature `0.8/1.0/1.25`，以及 Hellinger residual
替代 JS。它们只用于审计，不允许事后选择 target；主 target 始终是 `T=1` 的 JS。

## 4. Phase A：不读 outcome 的稳定性与可辨识性审计

### A1 target gate

- target 覆盖率必须至少为全部 blocks 的95%；
- 三个 winning offsets 各自占比必须至少10%，防止 target 退化为固定 offset；
- `T=1` winner 与 `T=0.8`、`T=1.25` 的一致率都必须至少80%；
- `T=1 JS` 与 `T=1 Hellinger` winner 一致率必须至少75%；
- 上述一致率的578-source cluster bootstrap 95% CI 下界分别不得低于75%、75%、70%；
- 至少90%的 blocks 的最大 `S_c` 必须大于 `1e-8`。

任一项失败即停止 CSC target，不进入 probe OOF。

### A2 identifiability probe gate

输入冻结为决策 deadline 前的 K（31步 causal pose/hand/motion）加已经冻结的 fold-specific
finger/face self-supervised embedding；不得加入昂贵 candidate logits、reference、future 或 EOS。
每个 outer fold 只在另外四折拟合标准化、降维和 probe。模型固定为多输出 ridge，`alpha=10`，
不调参。主指标为三候选连续 target 的 source-disjoint OOF `R2` 及每 block winner accuracy。

进入正式 OOF 必须同时满足：相对 outer-train-mean baseline 的 pooled `R2 >= 0.10`、winner
accuracy `>=0.45`，且五折中至少四折 `R2 > 0`。否则停止，不读取 train policy outcome。

## 5. 冻结的序列级决策

coverage skeleton 与现有 robust 链相同：每个完整4帧 block 必选 offset 0，再从
`{1,2,3}` 恰选一个 bonus，逐样本预算不变。正式模型预测三个 `S_c`，选择最大者；若预测的
top-two margin 不为正、输入无效、或末尾 anchor 不可用，则选择 uniform bonus offset 2。
不做全局 top-K、不用样本总长度、不尾部补 token。

正式 predictor 架构、seed、epoch 和损失只有在 A1/A2 均通过后才能另建不可覆盖的
`resolved_config_preregistered.json`；A2 probe 本身不能升级为正式模型。

## 6. 正式 OOF、gate 与解锁

正式 source-disjoint OOF 的主检验是按 predicted CSC schedule 重放完整 train dense logits，和
逐样本等预算 uniform offset-2 比较总 error。只有改善至少
`ceil(0.5% * 49,613)=249 errors`，且578-source cluster bootstrap 的 WER 差95% CI上界小于0，
才解锁 unknown-EOS token-bucket closed loop。

闭环必须保持 controller 成本、逐帧频率、缓存口径一致，并分别报告明确命名的 L0/L4/L8；本
预注册只可能解锁 `CSC-L8-v1`，不能把它重命名为 L0/L4。闭环通过后才允许读取冻结 held-out
outcome；test、多重比较和论文级结论继续保持锁定。

## 7. 计算设备

Phase A 预期 CPU 可完成。所有 `nvidia-smi`、CUDA 检查和 GPU 实验必须在沙盒外运行；禁止
PCI `25:00.0`、`41:00.0` 和宿主机 GPU0。沙盒内 CUDA 不可见不能作为驱动故障证据。

## 8. 冻结结果（预注册后追加）

Phase A 全程 CPU，未读取 robust reward、reference、WER、future continuation、真实 EOS、dev 或
test。

- A1 通过：181,815/183,401 blocks 有 target（99.14%）；主 winner 的 offset 1/2/3 分布为
  30.28%/41.84%/27.88%；温度0.8与1.25的一致率分别为86.81%与86.86%，Hellinger 一致率
  97.12%，相应 source-cluster CI 下界均通过 gate。
- A2 未通过：source-disjoint pooled `R2=0.047806`（门槛0.10），winner accuracy
  `0.416352`（门槛0.45；多数 offset-2 基线为0.418376），尽管5/5 folds 的 R2 均为正。

结论：局部语义曲率标签本身稠密且对预注册扰动稳定，但冻结的廉价 K+finger+face 观测不足以
达到可辨识性门槛。按停止规则，不训练正式 CSC predictor，不读取 train policy outcome，不运行
闭环或 held-out。证据目录：

- `p4_csc_l8_target_audit_v1_49faacc3/metrics.json`；
- `p4_csc_l8_identifiability_probe_v1_49faacc3/metrics.json`。

## 9. 独立的 CSC→WER oracle relevance 检验

原预注册顺序把正式 policy outcome 放在可辨识性 gate 之后。为回答“CSC即使可以预测，是否真的
有助于WER”，另建了不可覆盖的新实验身份，并在生成任何该实验 outcome 前冻结配置：
`p4_csc_l8_oracle_relevance_v1_preregistered.json`。它只在完整train fit上读取reference计分，
没有读取calibration、dev、test或held-out。

协议固定为每个完整4帧block选择offset 0，再选择一个bonus。对照固定选offset 2；oracle选
`T=1 JS` CSC winner，target无效的1,586个block按预注册回退offset 2。两种schedule逐样本预算
完全相同，均选择371,594/743,147个窗口（window rate 0.5000276）。结果：

- uniform offset-2：2,325 errors，WER 4.6863%；
- perfect CSC winner：2,224 errors，WER 4.4827%；
- 差值：-101 errors、-0.2036 pp，578-source cluster bootstrap 95% CI
  `[-0.3100,-0.0965]`。

CI上界小于0，说明CSC与WER不是完全无关；但完美winner只减少101 errors，未达到预注册的249
error headroom。relevance gate 因此失败。这个上界比任何现实predictor更乐观，而现有K+U的
winner accuracy又没有超过固定offset-2基线，所以停止整个CSC target：不再训练predictor、
不运行unknown-EOS闭环，也不读取held-out。证据为
`p4_csc_l8_oracle_relevance_v1_49faacc3/metrics.json`。
