# 下一阶段研究交接：P1 之后

更新时间：2026-09-12

仓库：`/mnt/workspace/projects/haojun/SLRT`

研究冻结基线：`a57825d`（P1）

文档整理基线：`298ff76`

本文件只描述**当前状态和下一步**。P0/P1 启动时的历史计划不再混入当前任务；历史入口见
[`archive/NEXT_RESEARCH_HANDOFF_PRE_P1_2026-09-12.md`](archive/NEXT_RESEARCH_HANDOFF_PRE_P1_2026-09-12.md)。
若本文与实验日志冲突，以对应阶段冻结文档、机器索引和 protocol manifest 为准。

## 0. 交给新对话时怎么用

新对话不需要继承旧聊天记录。直接提供本文件绝对路径：

```text
/mnt/workspace/projects/haojun/SLRT/docs/NEXT_RESEARCH_HANDOFF.md
```

并发送：

> 阅读该 handoff，核对仓库与本地资产状态。先提出 P2 train-only protocol，再实施更强的
> 因果 sign-interior predictor；不得用 dev/test 做训练或选择，不得覆盖 A0/P0/P1。

新对话开始后应先确认：

- 工作目录使用 `/mnt/workspace/projects/haojun/SLRT`；
- `/home/haojun/projects/SLRT` 只是同一目录的符号链接别名；
- `git status --short --branch` 没有未知改动；
- A0/P0/P1 的冻结文件和本地数据 hash 仍匹配；
- 当前没有需要接管的运行进程，再决定是否启动新实验。

## 1. 当前研究状态

| 阶段 | 冻结结论 | 权威文档 |
|---|---|---|
| A0 自适应步长 | 约减少三分之一窗口，但未显著优于等预算 uniform | [`ADAPTIVE_BASELINE_V1.md`](ADAPTIVE_BASELINE_V1.md) |
| P0 调度诊断 | boundary/prediction-change No-Go；label-derived sign-center 是 offline Strong-Go | [`P0_SCHEDULE_DIAGNOSTICS_V1.md`](P0_SCHEDULE_DIAGNOSTICS_V1.md) |
| P1 最小因果 center predictor | 11 维廉价关键点统计无法通过检测 gate，No-Go | [`P1_CAUSAL_CENTER_PREDICTOR_V1.md`](P1_CAUSAL_CENTER_PREDICTOR_V1.md) |

统一数字与声明边界见 [`RESULTS.md`](RESULTS.md)，机器可读索引见
[`results/README.md`](results/README.md)。

## 2. P1 留下的决定性事实

P0 的 50% dense budget、span-15 offline midpoint proxy 将 WER 从 `23.058447%` 降到
`21.537230%`，少 57 errors。这只是 label/alignment-derived oracle 上界，不是可部署模型。

P1 只用 train 拟合和校准严格因果逻辑回归：

- calibration AUROC `0.5452`，AUPRC `0.2203`（prevalence `0.2016`）；
- 任意阈值最大 center recall 只有 `17.83%`，此时 positive delay 为 `33.42%`；
- 满足 positive delay ≤25% 时，最大 recall 只有 `2.37%`；
- 预注册 gate 为 recall ≥75% 且 positive delay ≤25%；
- gate 失败，因此没有选择阈值、没有 scheduler replay、没有新 WER；
- 模型正常收敛，失败原因是表征能力，不是 CPU 或程序错误；
- pre-dev freeze 后只做了一次 dev threshold-free detection，没有用 dev 改模型；
- P1 从未打开或运行 test-only 文件。

不要继续在 dev 上微调 P1 的逻辑回归阈值，也不要把 P0 oracle WER 写成 P1 成绩。

## 3. 不可变实验身份

- 数据：修复后 Phoenix-2014T；dev 519 个样本、55,775 帧、3,747 个清洗后 reference gloss；
- 视频 ZIP SHA-256：`49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`；
- checkpoint SHA-256：`b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`；
- 主窗口/解码：16-frame centered window、triangular span-weighted-15；
- A0/B2 及 P0 坐标、预算和清洗规则保持冻结；
- 改变数据、checkpoint、窗口、decoder 或 target 时必须建立新版本，不能覆盖已有目录。

关键点已经受控拆分：

| split | 视频 | 帧 | SHA-256 前缀 | 后续用途 |
|---|---:|---:|---|---|
| train-only | 7,096 | 827,354 | `18a045bf` | 拟合与内部 calibration |
| dev-only | 519 | 55,775 | `85d1f0e2` | train 方案冻结后评估 |
| test-only | 642 | 64,627 | `5cb767b9` | 当前阶段禁止使用 |

大型 pickle 保持本地、不得提交。拆分完整性与完整哈希已冻结在 P1 文档和机器索引中。

## 4. 下一步候选

### 推荐主线：P2 更强的因果 center/interior 表征

目标不是继续调整阈值，而是验证更有信息量的视觉表示能否接近 P0 门槛。优先顺序：

1. 左右手分离的速度、加速度、手间距离和相对躯干坐标；
2. 局部手形/姿态变化和置信度变化，而不是只用整体运动中位数；
3. 固定短历史的 causal temporal head；若使用 lookahead，必须明确建立 bounded-lookahead 版本；
4. 先在 train fit/calibration 报告 center/event detection，再决定是否打开 dev；
5. 只有 train calibration gate 通过，才连接 50% budget、span-15 scheduler。

新版本必须在读取 dev 前冻结：特征、归一化、模型、训练轮数、threshold grid、event emission、
refractory period、匹配规则和 gate。不得使用 gloss identity 或 reference token 作为预测输入。

### 备选主线：window/decoder 或模态计算门控

如果更强 P2 仍无法定位 sign interior，应停止继续堆 center head，转而检查：

- centered 16-frame window 和 span-15 是否抹平了调度差异；
- causal/bounded-lookahead window 的精度—延迟曲线；
- RGB/keypoint 两流是否应按可靠性选择性计算，而不是只改变时间采样；
- 固定等预算下 decoder 对窗口位置的敏感度。

这些方向必须新建协议，不能用 P0/P1 的 oracle 或 threshold-free detection 代替真实调度结果。

## 5. 数据与 test 边界

- train 用于模型拟合；train 内按 source video 划分 calibration；
- dev 只能在训练侧方案冻结后评估，不得循环查看并改规则；
- P1 的 test-only 文件未使用；
- 仓库存在更早的修复前 historical retrospective test 结果，说明 test 在历史上已被查看；它们
  不能用于当前选择，也不能改称修复后正式 test；
- alignment-derived midpoint 是 proxy，不是人工逐帧标注。

## 6. GPU 与存储

已知故障 GPU：PCI `01:00.0` 和 `25:00.0`，禁止使用。需要 GPU 时先现场检查，再用健康卡
UUID 单卡绑定，不能依赖逻辑编号。廉价 CPU 审计不需要为了形式切换 GPU；真正的时序 head
训练可使用健康 GPU。

NFS 长期接近 99% 使用率。大型特征、logits 和临时模型写入 `/tmp`；最终必须把 compact
manifest、config、aggregate、必要模型参数和日志持久化，不能把唯一结论留在 `/tmp`。

## 7. 新任务启动清单

1. 检查 `git status --short --branch` 和 `git diff --check`，不要 reset 用户改动；
2. 阅读本文件、P1 冻结文档、P0 冻结文档和 `RESULTS.md`；
3. 复核数据/模型 hash，不覆盖现有 `model_dir`；
4. 先写 P2 train-only protocol 和输出目录；
5. 先完成 train fit/calibration，gate 通过后才允许一次性 dev；
6. 当前阶段禁止 test。

建议首条任务：

> 基于冻结 P1，设计 P2 train-only 因果 sign-interior predictor。增强左右手与短历史表征，先
> 冻结特征、训练、校准、事件与 gate 协议；不得读取 dev/test 做选择，不得覆盖 A0/P0/P1。

## 8. 可声明与不可声明

可以说：P0 显示 label-derived sign-center 的 offline 上界；P1 当前廉价因果特征无法达到定位
门槛。不能说：已有可部署 center scheduler、P1 改善了 WER、GPU 会自动解决表征失败，或
当前存在未被历史查看的正式 test 结果。
