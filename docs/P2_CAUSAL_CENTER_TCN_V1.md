# Phoenix P2 因果 center TCN v1（冻结）

更新时间：2026-09-12

预注册协议见 [`P2_TRAIN_ONLY_PROTOCOL_V1.md`](P2_TRAIN_ONLY_PROTOCOL_V1.md)，机器索引为
[`results/phoenix_p2_causal_center_tcn_v1_frozen.json`](results/phoenix_p2_causal_center_tcn_v1_frozen.json)。
compact 证据位于 `Online/CSLR/results/phoenix-2014t_ISLR/p2_train_center_tcn_v1/`。

## 1. 结论

P2 为 **No-Go**。211 维左右手/手形表征与 31 帧严格因果 TCN 明显改善了 threshold-free
frame detection，但 train calibration 上仍没有阈值通过预注册的 event gate。因此 P2 没有
打开 dev、没有 scheduler replay、没有新 WER；test-only 也未打开或运行。

这表明关键点中确有比 P1 汇总运动更强的 sign-interior 信号，但现有信号和固定事件策略离
P0 要求仍很远。不得把 AUROC/AUPRC 的提升表述成调度效果。

## 2. 冻结身份

- train-only 关键点 SHA-256：
  `18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763`；
- 复用 P1 source-video split：fit 5,655 视频，calibration 1,440 视频；split SHA-256
  `33db242f4c2353ebcd71ca663d7b3faa59bff3b06856b45456cf5d27dbdcd65a`；
- 827,338 帧、165,672 个 radius-1 positive frame；
- target 仍是 alignment-derived nonblank midpoint `±1`，不是人工 ground truth；
- 211 维当前/历史可用特征：pose、左右手 wrist-relative shape、手部 anchor、有效性、置信度、
  躯干相对位置和当前运动；
- 32-channel、4-block causal TCN，dilation `1/2/4/8`，receptive field 31；
- 固定 8 epochs、无 early stopping、无 calibration model selection；
- 单卡 UUID `GPU-2c6a50cb-f770-c785-dc1d-7f8cc7b7b9aa`（PCI `41:00.0`）；未使用已知故障
  PCI `01:00.0` 或 `25:00.0`。

## 3. Train calibration

| 指标 | P1 | P2 |
|---|---:|---:|
| frame AUROC | 0.545158 | **0.737159** |
| frame AUPRC | 0.220340 | **0.392310** |
| prevalence | 0.201588 | 0.201588 |
| 任意阈值最大 center recall | 17.8307% | **38.8904%** |
| 该点 positive delay | 33.4167% | **30.0917%** |
| delay ≤25% 时最大 recall | 2.3727% | **22.4512%** |

P2 最大 recall 位于 threshold `0.59`：4,360/11,211 centers hit，event precision
33.8325%，positive delay 30.0917%。满足 delay 约束的最大 recall 位于 threshold `0.39`：
2,517/11,211 centers hit，event precision 19.5965%，positive delay 24.9901%。两者都远低于
75% recall gate，所以 `selected_threshold=null`。

固定 8 epochs 的 fit weighted BCE 从 `1.044327` 单调下降到 `0.945273`。最终确定性重跑与前次
训练逐 epoch 一致；模型训练和数值均正常。

## 4. 数据边界与运行审计

- P2 只读取 train-only 关键点、train bags、split manifest 和预注册协议；
- pre-dev freeze manifest 明确记录 `dev_opened_or_run=false`、`test_opened_or_run=false`；
- gate 失败后严格跳过 dev detection 和 scheduler；
- 大型 207MB feature archive 保留在 `/tmp`，compact 模型仅约 90KB；
- 训练开始前确认 8 张 3090 连续三次采样均为空闲。宿主机可见 GPU，但默认 Codex 隔离层未
  挂载 GPU device；使用经授权的 UUID 单卡绑定后 PyTorch 正常识别。

## 5. 下一步判断

P2 相比 P1 的提升足以排除“关键点完全没有信息”，但最大 recall 仍只有 38.9%。不建议继续
围绕同一个二分类 TCN 小幅调宽、调深或调 threshold。更有信息量的下一步应建立新协议，优先
做以下二选一：

1. **P3 RGB/预训练视觉特征的因果 probe**：只用 train 拟合，比较关键点、RGB/S3D feature
   和二者融合的 detection 上界；任何 centered clip/lookahead 必须显式标成 bounded-lookahead；
2. **W1 window/decoder 敏感性**：停止 center head，系统比较 causal/bounded-lookahead window
   与 span decoder 的精度—延迟曲线，判断 P0 上界是否依赖当前 centered-16/span-15。

在新协议冻结前不得打开 dev；当前阶段仍禁止 test。
