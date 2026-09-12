# Phoenix 自适应步长基线 v1（冻结）

本文是 Phoenix-2014T 自适应步长后续研究的单一权威入口。冻结对象是**修复后 Phoenix dev 基线**；旧数据实验和历史 test 只作审计材料，不属于当前资产版本的正式结果。

## 冻结结论

当前实现已经唯一确定为 `A0_adaptive_span15`：冻结 Two-Stream S3D 模型，在在线推理阶段根据历史关键点运动量因果选择 stride 1--3，再用真实时间 triangular span-15 投票解码。它不改变模型权重，也不重新训练模型。

现有证据支持的结论是：A0 相对原工程 B0 减少 32.56% 模型窗口，在三重复测量中减少 29.48% 完整命令时间；其 dev WER 增加 0.1868 pp。与等预算、同解码 B2 相比，A0 的 WER 低 0.1334 pp，但 95% CI 跨 0，不能宣称显著优于均匀采样。

## 不可变实验身份

| 项目 | 冻结值 |
|---|---|
| 数据集 / split | Phoenix-2014T dev |
| 样本 / 清洗后参考词数 | 519 / 3,747 |
| 视频 ZIP SHA-256 | `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457` |
| checkpoint | `Online/CSLR/results/phoenix-2014t_ISLR/ckpts/best.ckpt`，epoch 92 |
| checkpoint SHA-256 | `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b` |
| 模型 | Online/CSLR Two-Stream S3D block5；RGB + HRNet WholeBody keypoint heatmap；`triplehead_cat_bilateral` |
| prediction source | `ensemble` |
| clip / inference split size | 16 frames / 16 clips |
| blank / probability threshold | 0.5 / -1 |
| seed | 321 |
| 协议文件 | `Online/CSLR/configs/experiments/phoenix_adaptive_baselines_v1.yaml` |
| 协议 SHA-256 | `3d583b1430def1ae0b517c79bf66abc8dae9528118b325c58620fd83e22f9f65` |
| 修复后 manifest SHA-256 | `2e6889ec14777d938a3c7ac9265ab97f87118523e9258f5f93beba4b3e854b8f` |
| 代码 commit | `65c88e5d9686fe338f151890dc07878809abd94b` |

manifest 记录运行时工作树含日志类未提交改动，因此代码身份以 manifest 中逐文件 SHA-256 为最终依据；当前核心配置、采样、预测和评估文件的 hash 与该 manifest 完全一致。

## 对照定义

| ID | 采样 | 报告解码器 | 作用 |
|---|---|---|---|
| B0 | fixed stride 1 | 原工程 window-greedy-7 | 端到端原工程基线 |
| B1 | fixed stride 1，复用 B0 logits | span-weighted-15 | 隔离解码器变化 |
| B2 | deterministic uniform rate `1.4827861225574903` | span-weighted-15 | A0 的等窗口预算对照 |
| B3 | fixed stride 2 | span-weighted-15 | 固定步长效率点 |
| B4 | fixed stride 3 | span-weighted-15 | 固定步长效率点 |
| A0 | causal adaptive stride 1--3 | span-weighted-15 | 冻结自适应方法 |

B2 与 A0 相差 91 clips，即相对 A0 为 0.2419%，通过预注册的 2% 等预算上限。

## A0 冻结参数与规则

| 参数 | 值 |
|---|---:|
| min / max stride | 1 / 3 |
| keypoint confidence threshold | 0.2 |
| minimum valid keypoints | 4 |
| causal EMA decay | 0.4 |
| low / high motion quantile | 0.2 / 0.7 |
| calibration history / warmup | 48 / 16 frames |
| vote span / kernel / minimum weight | 15 / triangular / 0.05 |

每个决策仅使用当前及历史关键点。有效关键点的尺度归一化位移先聚合为运动分数并作 EMA；warmup 或关键点质量不足时保守使用 stride 1；之后按最近 48 帧历史的 0.2/0.7 分位点将运动量映射为 stride 3/2/1。配置 YAML 默认关闭 adaptive 和 span voting 是兼容旧命令的安全默认值；矩阵运行器根据变体显式覆盖，因此不构成参数冲突。

## 修复后权威 dev 正确性结果

| ID | WER | DEL / INS / SUB（错误数） | errors / ref | clips |
|---|---:|---:|---:|---:|
| B0 | 22.231118% | 385 / 130 / 318 | 833 / 3,747 | 55,775 |
| B1 | 22.604750% | 422 / 124 / 301 | 847 / 3,747 | 55,775 |
| B2 | 22.551374% | 425 / 121 / 299 | 845 / 3,747 | 37,706 |
| B3 | 22.818255% | 425 / 128 / 302 | 855 / 3,747 | 28,014 |
| B4 | 22.925007% | 437 / 121 / 301 | 859 / 3,747 | 18,766 |
| A0 | 22.417934% | 434 / 104 / 302 | 840 / 3,747 | 37,615 |

配对 bootstrap（1,000 次，seed 321）：A0−B0 为 `+0.186816 pp`，95% CI `[-0.525470,+0.852723]`；A0−B1 为 `-0.186816 pp`，CI `[-0.700818,+0.315969]`；A0−B2 为 `-0.133440 pp`，CI `[-0.668713,+0.418291]`。三个区间均跨 0。

## 修复后权威 runtime

B0、B2、A0 在同一健康 RTX 3090（PCI `81:00.0`）按交替顺序各运行三次。B1 复用 B0 前向；B3/B4 只有正确性运行的单次工程时间，不能与三重复统计混称。

| ID | wall R1/R2/R3 (s) | mean / median / std (s) | CV | forward median (s) |
|---|---:|---:|---:|---:|
| B0 | 1642.57 / 1651.48 / 1634.23 | 1642.76 / 1642.57 / 8.62 | 0.525% | 1036.95 |
| B2 | 1133.10 / 1132.77 / 1135.16 | 1133.68 / 1133.10 / 1.29 | 0.114% | 702.72 |
| A0 | 1158.29 / 1144.42 / 1161.04 | 1154.58 / 1158.29 / 8.91 | 0.771% | 701.09 |

以中位数计算：A0 相对 B0 的 wall time 减少 29.483%、加速 1.418×，model-forward 减少 32.390%、加速 1.479×；A0 相对 B2 的 wall time 增加 2.224%，但 forward 减少 0.232%，说明模型计算预算确实近似相等。

## 权威产物与版本边界

- 修复后正确性：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/aggregate/dev_summary.json`
- 修复后正确性 manifest：同目录 `protocol_manifest.json`
- 修复后三重复 runtime：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_runtime_49faacc3/aggregate/dev_summary.json`
- 可提交的冻结索引：`docs/results/phoenix_adaptive_baseline_v1_frozen.json`
- 修复前结果索引：`Online/CSLR/results/_archive/phoenix_pre_repair_81629b2f/`

2026-09-03 及更早结果绑定缺 61 帧的旧视频 hash `81629b2f...07e30`，仅为历史证据。历史 Phoenix test 虽然 test split 本身没有缺帧，但运行绑定旧资产和旧 manifest，且 test 已被查看，只能标记为 **historical retrospective control**。截至本冻结文档建立时，尚未使用修复后 hash 运行新的 test；不得把旧 test 改名为修复后正式结果。

## 后续研究约束

后续可靠性/边界感知方法必须以 A0 为继承起点，并至少报告 B0（原工程基线）、B1（同解码全预算）和 B2（同解码等预算）对照。继续固定上述 dev 数据、checkpoint、预处理、16 帧窗口、评估脚本与指标定义；算法选择、预算和消融只使用 dev。任何改变冻结项的实验必须使用新版本名和新 manifest，不能覆盖 v1。

当前**尚未冻结**的事项是：下一代可靠性/边界信号及其参数、不同预算层级、真正流式延迟协议、跨数据集验证，以及修复后 retrospective test。它们不是 A0 v1 已完成结果的一部分。
