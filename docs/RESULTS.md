# 统一实验结果口径

本页是仓库当前结果的唯一对外摘要。出现不同数字时，按以下优先级解释：

1. 本页和 `docs/results/phoenix_adaptive_stride_summary.json`：冻结后的正式口径；
2. `code_agent_logs/2026-07-16/adaptive_stride_frozen_final_evaluation.md`：正式 test 审计；
3. `adaptive_stride_dev_tuning.md`：dev 选择过程；
4. `adaptive_stride_full_comparison.md`：调参前 span-13 历史工程对照，不作为最终结果；
5. `implementation_plan.md` 中 21.86/24.18/39.11%：旧服务器材料，仅作实施前参考。

## 正式口径

正式算法是 **Phoenix-2014T、固定 checkpoint、因果自适应 stride 1--3、真实时间 triangular span-15 解码**。参数只在 dev 选择，test 在冻结后运行一次。

### Dev：配置选择依据

| 指标 | 固定基线 | 冻结自适应 | 变化 |
|---|---:|---:|---:|
| 解码 | window-greedy-7 | span-weighted-15 | — |
| WER | 22.2311% | 22.4179% | +0.1868 pp |
| DEL / INS / SUB | 10.2749 / 3.4694 / 8.4868% | 11.5826 / 2.7756 / 8.0598% | — |
| clips | 55,775 | 37,615 | **-32.56%** |
| command wall time | 1,450.60 s | 1,084.18 s | **-25.26%** |
| wall-time speedup | 1.000x | 1.338x | +33.80% |

Dev 的 span-15 指标由 D2 同一次前向保存的 logits/窗口中心离线扫描得到，权威机器文件是 `dev_span_sweep.json`；原 `dev_evaluation_results.pkl` 只包含运行时启用的 span-13，不能用它否定离线预注册扫描结果。

Dev 预注册候选的完整选择表：

| ID | stride | decoder | WER | clips | clips 变化 | wall time | wall time 变化 |
|---|---|---|---:|---:|---:|---:|---:|
| D0 | 固定 1 | window-greedy-7 | 22.2311% | 55,775 | — | 1,450.60 s | — |
| D1 | 1--2 | span-weighted-13 | 22.3912% | 43,349 | -22.28% | 1,231.22 s | -15.12% |
| D2 | 1--3 | span-weighted-15 | 22.4179% | 37,615 | **-32.56%** | 1,084.18 s | **-25.26%** |

D1、D2 的 WER 增量都未超过预注册的 +0.75 pp 约束，因此按“满足精度约束后优先减少 clips”的规则选择 D2。

### Test：唯一冻结最终评估

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

正式结论必须写成：

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
| P2 | dev 冻结最终配置 | 3 | span-weighted-15 | 23.0571% | 43,932 | 1,141.08 s | **是** |

P1 与 P2 的采样配置相同，所以窗口数和步长分布相同。WER 差异来自 span 13/15；runtime 差异来自两次独立完整运行的波动和解码设置，不能把 P1 的 -28.84% runtime 与 P2 的 23.0571% WER 拼成同一组结果。

## 权威数据位置

机器结果位于本地 `Online/CSLR/results/`，被 Git 忽略：

- Test 固定基线：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_full642/test/`
- Test 冻结 P2：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_dev_tuned_span15_final_test/test/`
- Test 历史 P1：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_full642/test/`
- Dev 固定 D0：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_dev_tune_d0_fixed_s16/dev/`
- Dev max-stride=2 D1：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_dev_tune_d1_max2_s16/dev/`
- Dev 冻结 D2：`Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_dev_tune_d2_max3_s16/dev/`

每个 test 目录中的 `*_evaluation_results.pkl` 保存 WER 分项，`*_results.pkl` 保存逐样本预测和步长 metadata。Dev span 扫描结果在 `dev_span_sweep.json`。可提交的统一机器摘要位于 [results/phoenix_adaptive_stride_summary.json](results/phoenix_adaptive_stride_summary.json)。

## 不属于本 WER 口径的结果

CSL-Daily Top-800 R1 是 isolated dev 可靠性诊断，报告 accuracy/AUROC，不报告连续识别 WER，也没有验证自适应步长。其数据位于 `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/reliability_r1_dev/`，摘要见 `code_agent_logs/2026-07-26/reliability_r1_results.md`。

## 下一步基础对照

当前结果尚未消除采样器与解码器差异，也缺少固定stride和预算匹配均匀采样对照。下一步预注册实施协议见 [baseline_evaluation_plan.md](experiments/adaptive_stride/baseline_evaluation_plan.md)。在该矩阵完成前，不声称自适应策略优于相同计算预算的普通降采样。
