# 统一实验结果口径

> **数据版本提示（2026-09-09）：** 本页 2026-09-03 及更早的 Phoenix 结果使用修复前视频 ZIP（SHA-256 `81629b2f...07e30`）。2026-09-08 恢复 61 帧后的标准 ZIP 为 `49faacc3...d457`。修复后完整 dev 矩阵得到相同 WER/clips，且同卡三重复 runtime 已完成；旧结果的非破坏性索引见 `Online/CSLR/results/_archive/phoenix_pre_repair_81629b2f/README.md`。test 本身没有缺帧，但尚未以修复后资产重新运行。详细证据见 `code_agent_logs/2026-09-08/phoenix_repaired_dev_matrix.md` 和 `phoenix_repaired_runtime_repeats.md`。

Phoenix 自适应步长 v1 的精确冻结定义以 [ADAPTIVE_BASELINE_V1.md](ADAPTIVE_BASELINE_V1.md) 为准。本页保留历史结果叙事；出现不同数字时，按以下优先级解释：

1. `ADAPTIVE_BASELINE_V1.md` 和 `docs/results/phoenix_adaptive_baseline_v1_frozen.json`：修复后 dev 冻结口径；
2. 修复后正确性矩阵和三重复 runtime manifest/aggregate；
3. 本页和 `docs/results/phoenix_adaptive_stride_summary.json`：包含修复前历史 test 的兼容摘要；
4. `code_agent_logs/2026-07-16/adaptive_stride_frozen_final_evaluation.md`：修复前 historical retrospective test 审计；
5. `adaptive_stride_dev_tuning.md`：dev 选择过程；
6. `adaptive_stride_full_comparison.md`：调参前 span-13 历史工程对照，不作为最终结果；
7. `implementation_plan.md` 中 21.86/24.18/39.11%：旧服务器材料，仅作实施前参考。

## 正式口径

正式算法是 **Phoenix-2014T、固定 checkpoint、因果自适应 stride 1--3、真实时间 triangular span-15 解码**。参数只在 dev 选择；现有 test 是冻结后产生的历史 retrospective control，但尚未绑定修复后资产。

### Dev：配置选择依据

| 指标 | 固定基线 | 冻结自适应 | 变化 |
|---|---:|---:|---:|
| 解码 | window-greedy-7 | span-weighted-15 | — |
| WER | 22.2311% | 22.4179% | +0.1868 pp |
| DEL / INS / SUB | 10.2749 / 3.4694 / 8.4868% | 11.5826 / 2.7756 / 8.0598% | — |
| clips | 55,775 | 37,615 | **-32.56%** |
| command wall time（三重复中位数） | 1,642.57 s | 1,158.29 s | **-29.48%** |
| wall-time speedup | 1.000x | 1.418x | +41.81% |

Dev 的 span-15 指标由 D2 同一次前向保存的 logits/窗口中心离线扫描得到，权威机器文件是 `dev_span_sweep.json`；原 `dev_evaluation_results.pkl` 只包含运行时启用的 span-13，不能用它否定离线预注册扫描结果。

Dev 预注册候选的完整选择表：

| ID | stride | decoder | WER | clips | clips 变化 | wall time | wall time 变化 |
|---|---|---|---:|---:|---:|---:|---:|
| D0 | 固定 1 | window-greedy-7 | 22.2311% | 55,775 | — | 1,450.60 s | — |
| D1 | 1--2 | span-weighted-13 | 22.3912% | 43,349 | -22.28% | 1,231.22 s | -15.12% |
| D2 | 1--3 | span-weighted-15 | 22.4179% | 37,615 | **-32.56%** | 1,084.18 s | **-25.26%** |

D1、D2 的 WER 增量都未超过预注册的 +0.75 pp 约束，因此按“满足精度约束后优先减少 clips”的规则选择 D2。

### Dev：基础对照矩阵 v1（2026-09-03）

在 commit `2393728` 上用统一采样/评估接口完成了B0--B4/A0全量dev正确性矩阵。所有变体使用同一checkpoint、519样本和3,747参考gloss；B1复用B0前向。

| ID | 采样 | 解码 | WER | Clips | 单次wall time |
|---|---|---|---:|---:|---:|
| B0 | fixed stride=1 | window-greedy-7 | 22.2311% | 55,775 | 1680.61s |
| B1 | fixed stride=1 | span-weighted-15 | 22.6048% | 55,775 | 复用B0 |
| B2 | uniform rate=1.482786... | span-weighted-15 | 22.5514% | 37,706 | 1183.30s |
| B3 | fixed stride=2 | span-weighted-15 | 22.8183% | 28,014 | 904.25s |
| B4 | fixed stride=3 | span-weighted-15 | 22.9250% | 18,766 | 641.32s |
| A0 | adaptive stride=1--3 | span-weighted-15 | 22.4179% | 37,615 | 1191.51s |

B2与A0的clips只差0.242%，满足预注册的2%等预算约束。A0相对B2的WER低0.1334 pp，paired bootstrap 95% CI为[-0.6687,+0.4183] pp；区间跨0，不能声称统计显著的精度改善。相对同解码B1，A0减少32.56%窗口且WER低0.1868 pp，95% CI同样跨0。当前最稳妥结论是：A0以约三分之一的窗口减少保持相近WER，尚无证据证明运动自适应显著优于等预算均匀采样。

这些wall time来自一次正确性运行，不替代3次同卡交替runtime benchmark。完整审计见`code_agent_logs/2026-09-03/phoenix_adaptive_baseline_matrix_dev.md`。

### Dev：P0 等预算调度诊断 v1（2026-09-12 冻结）

P0 只读取修复后 dev 的既有 dense B0 logits，不重新前向、不读取 test。所有候选固定
16 帧 centered window；主比较使用固定 B0 左补帧坐标与逐样本严格等预算。权威定义见
[P0_SCHEDULE_DIAGNOSTICS_V1.md](P0_SCHEDULE_DIAGNOSTICS_V1.md)。

| 诊断 | 主要结果 | 决策 |
|---|---|---|
| Dense replay | B1/B2/A0 的 WER 与 519/519 hypotheses 精确复现 | replay 有效 |
| Random 30 seeds | 逐样本等预算 WER `23.2123±0.2756%`，30/30 差于 B2/A0 | random No-Go |
| Prediction change | 最优 past-only top-1 change `22.6314%`，等于同预算 uniform | No-Go |
| Boundary proxy | 最优只比同预算 uniform 少 2 errors（`-0.0534 pp`） | No-Go |
| Sign-center proxy | 50% dense + span-15：`23.0584% → 21.5372%`，少 57 errors | offline Strong-Go |
| Noisy center | ±1 帧、75% recall 仍保留显著收益；组合噪声失效 | 可训练性脆弱 |

Sign-center 使用 dev label/alignment 派生中心，是不可部署的 offline oracle，不是正式模型
成绩，也不能与 B0/A0 的真实调度结果混称。它只证明当前系统在 sign interior 附近存在可
利用的窗口放置上界。下一阶段必须用 train-only target 训练 predictor，并在 dev 上报告
center detection 与 scheduler 结果；在方案完全冻结前不得运行 test。

### P1 train-only causal center predictor（2026-09-12 冻结）

P1 使用 alignment-derived train midpoint `±1` 帧作为 proxy target，以 11 个严格 past-only
关键点运动/有效性/人体框特征拟合固定逻辑回归。权威定义见
[P1_CAUSAL_CENTER_PREDICTOR_V1.md](P1_CAUSAL_CENTER_PREDICTOR_V1.md)。

| 阶段 | AUROC | AUPRC / prevalence | center recall | 正延迟比例 |
|---|---:|---:|---:|---:|
| Train calibration 最佳 recall 阈值 | 0.5452 | 0.2203 / 0.2016 | 17.83% | 33.42% |
| Train calibration 满足 delay ≤25% | — | — | 2.37% | 22.93% |
| Dev 一次性 threshold-free detection | 0.5522 | 0.2242 / 0.2015 | — | — |

预注册 gate 要求 recall ≥75% 且正延迟 ≤25%，没有阈值合格，结论为 No-Go。scheduler
replay 按协议跳过，P1 没有新的 WER；这不是 CPU 或优化器失败，而是当前廉价特征的定位
能力不足。test-only 未打开或运行。

### Test：修复前历史 retrospective control

以下结果绑定修复前视频资产，且 Phoenix test 已被历史实验查看。test split 自身没有缺帧，但这些结果不能重新标记成修复后资产的正式结果。截至 2026-09-10，尚未以修复后 hash 运行新的 test。

| 指标 | 固定基线 | 冻结自适应 | 变化 |
|---|---:|---:|---:|
| 解码 | window-greedy-7 | span-weighted-15 | — |
| WER | **22.0005%** | **23.0571%** | **+1.0566 pp** |
| DEL | 9.2510% | 11.1294% | +1.8784 pp |
| INS | 3.4985% | 3.4280% | -0.0704 pp |
| SUB | 9.2510% | 8.4996% | -0.7514 pp |
| 参考 gloss / errors | 4,259 / 937 | 4,259 / 982 | +45 errors |
| clips | 64,627 | 43,932 | **-32.02%** |
| 平均 clips/样本 | 100.67 | 68.43 | -32.24 |
| command wall time | 1,547.29 s | 1,141.08 s | **-26.25%** |
| wall-time speedup | 1.000x | 1.356x | +35.60% |

引用该历史 test 时必须写成：

> 自适应步长在 Phoenix-2014T test 上减少 32.02% 的模型窗口和 26.25% 的单次完整命令 wall time，但 WER 从 22.0005% 上升到 23.0571%（+1.0566 个百分点）。它是精度—效率折中，不是 WER 改进。

## 指标定义

- **WER**：Phoenix cleanup 后，在整个 split 上汇总的 `(DEL + INS + SUB) / reference gloss count`；
- **窗口/clips 减少率**：`1 - adaptive_clip_count / fixed_clip_count`，不是原始视频帧减少率；每个送入 S3D 的 clip 仍为完整 16 帧；
- **runtime**：完整命令的单次 wall-clock `real` 时间，包括初始化、数据加载、模型前向、解码和结果保存；不是纯 GPU kernel 时间，也不是平均/P95 在线延迟；
- **speedup**：`fixed_wall_time / adaptive_wall_time`；
- 百分点变化使用 `pp`，不与相对百分比混写。

WER 和 clips 已从当前本地 PKL/逐样本 `adaptive_stride_metadata` 重新核对。Runtime 来自当次 `/usr/bin/time` 记录，模型日志时间戳可近似佐证，但现有 PKL 不存储 wall time；因此 runtime 是单次工程测量，不能表述成多次重复的统计均值。

## 冻结配置

| 项目 | 正式值 |
|---|---|
| 模型 | SLRT Online Two-Stream S3D，RGB + HRNet WholeBody keypoints |
| checkpoint | `Online/CSLR/results/phoenix-2014t_ISLR/ckpts/best.ckpt`，epoch 92 |
| prediction source | `ensemble` |
| clip window | 16 frames |
| fixed baseline | stride=1，window-greedy-7 |
| adaptive stride | min=1，max=3 |
| keypoint confidence / minimum valid | 0.2 / 4 |
| causal EMA | 0.4 |
| motion quantiles | 0.2 / 0.7 |
| calibration window / warmup | 48 / 16 frames |
| decoder | triangular span-weighted-15，min weight=0.05 |
| inference split size | 16 |
| blank threshold / probability threshold | 0.5 / -1 |

配置文件故意保持 `adaptive_stride.enabled: false` 和 `span_weighted_voting.enabled: false`，确保旧命令默认仍是固定步长。正式复现必须使用 `scripts/reproduce/phoenix_adaptive_stride.sh adaptive <split>` 显式启用；因此“配置版本”应理解为 **YAML 公共参数 + 复现脚本 CLI 覆盖**。

## 版本关系

| ID | 定位 | max stride | decoder | Test WER | clips | wall time | 是否正式 |
|---|---|---:|---|---:|---:|---:|---|
| P0 | 固定工程基线 | 1 | window-greedy-7 | 22.0005% | 64,627 | 1,547.29 s | 对照 |
| P1 | 调参前历史对照 | 3 | span-weighted-13 | 22.7283% | 43,932 | 1,101.12 s | 否 |
| P2 | dev 冻结配置的历史 test | 3 | span-weighted-15 | 23.0571% | 43,932 | 1,141.08 s | 历史 retrospective |

P1 与 P2 的采样配置相同，所以窗口数和步长分布相同。WER 差异来自 span 13/15；runtime 差异来自两次独立完整运行的波动和解码设置，不能把 P1 的 -28.84% runtime 与 P2 的 23.0571% WER 拼成同一组结果。

## 权威数据位置

机器结果位于本地 `Online/CSLR/results/`，被 Git 忽略：

- Test 固定基线：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_full642/test/`
- Test 冻结 P2：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_dev_tuned_span15_final_test/test/`
- Test 历史 P1：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_full642/test/`
- Dev 固定 D0：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_dev_tune_d0_fixed_s16/dev/`
- Dev max-stride=2 D1：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_dev_tune_d1_max2_s16/dev/`
- Dev 冻结 D2：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_dev_tune_d2_max3_s16/dev/`
- Dev 基础对照矩阵v1：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1/`，统一摘要位于其`aggregate/dev_summary.{json,csv,md}`。
- 修复后 Dev 基础对照：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/`。
- 修复后同卡三重复 runtime：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_runtime_49faacc3/`。
- 修复前结果 provenance 索引：`Online/CSLR/results/_archive/phoenix_pre_repair_81629b2f/`（仅相对符号链接，不复制结果）。

每个 test 目录中的 `*_evaluation_results.pkl` 保存 WER 分项，`*_results.pkl` 保存逐样本预测和步长 metadata。Dev span 扫描结果在 `dev_span_sweep.json`。可提交的统一机器摘要位于 [results/phoenix_adaptive_stride_summary.json](results/phoenix_adaptive_stride_summary.json)。

## 不属于本 WER 口径的结果

CSL-Daily Top-800 R1 是 isolated dev 可靠性诊断，报告 accuracy/AUROC，不报告连续识别 WER，也没有验证自适应步长。其数据位于 `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/reliability_r1_dev/`，摘要见 `code_agent_logs/2026-07-26/reliability_r1_results.md`。

## 下一步实验

基础正确性对照、同卡交替三重复 runtime 和 P0 调度诊断均已完成。P0 不支持继续微调 motion、prediction-change 或 boundary-only 调度，但 label-derived sign-center oracle 显示了明确上界。下一步只使用 train 构造最小 causal sign-interior predictor，再在 repaired dev 评估 detection 与等预算 scheduler；仍不读取 test。由于 A0 对等预算 B2 的 dev WER 差异不显著，不能把本矩阵解释成现有自适应采样已经显著优于均匀采样。
