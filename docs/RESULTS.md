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

### P2 train-only causal center TCN（2026-09-12 冻结）

P2 复用 P1 的 train fit/calibration split、midpoint `±1` proxy target、事件规则和 gate，把输入
增强为 211 维左右手/手形/躯干相对表征，并使用 receptive field 31 帧的严格因果 TCN。权威
定义见 [P2_CAUSAL_CENTER_TCN_V1.md](P2_CAUSAL_CENTER_TCN_V1.md)。

| 阶段 | AUROC | AUPRC / prevalence | 最大 center recall | delay ≤25% 最大 recall |
|---|---:|---:|---:|---:|
| P1 train calibration | 0.5452 | 0.2203 / 0.2016 | 17.83% | 2.37% |
| P2 train calibration | **0.7372** | **0.3923 / 0.2016** | **38.89%** | **22.45%** |

P2 表征明显强于 P1，但仍没有阈值达到 recall ≥75% 且 positive delay ≤25% 的预注册 gate，
因此结论仍为 No-Go。P2 没有打开 dev、没有 scheduler replay、没有新 WER；test-only 未打开或
运行。当时冻结的后续建议是转向 RGB 与 window/decoder utility；该建议已经由下方 P3 完整链
执行并取代，不能再把它当作当前下一步。

### P3 complete-train robust utility predictability（2026-09-21 冻结）

P3 先在完整 train fit（6,378 samples、578 sources、183,401 blocks）复现 robust continuation
oracle：同一 coverage skeleton 下 errors 从 2,325 降至 1,688，少 637，说明窗口选择上界真实。
但部署 predictor 面对 366,802 个 side-candidate rows，其中只有 769 beneficial、2,015 harmful，
其余 364,018 neutral。全部模型使用冻结的5折 source-disjoint OOF 和 top-769 signed utility gate；
没有读取新的 calibration/dev/test 或运行闭环。

| Predictor | 主要新增信息 | PR-AUC | Recall@769 | Top-769 utility | 决策 |
|---|---|---:|---:|---:|---|
| K | causal skeleton/history | .002549 | .01040 | -2 | No-Go |
| R | 双手 RGB-DCT | .002042 | .00260 | -15 | No-Go |
| HF | fold-specific finger RGB tiny-CNN | .002491 | .00650 | -3 | No-Go |
| RF | fold-specific face RGB tiny-CNN | .002570 | .00260 | -5 | No-Go |
| DCV | decoder-prefix state + causal K | .002649 | .00780 | -9 | No-Go |
| L3 | 最丰富的可部署信息阶梯 | .005415 | .02341 | +5 | 仍远低于249-error gate |

这些结果说明瓶颈不是缺少 oracle headroom，而是该 future/reference/EOS-aware counterfactual
utility 标签难以由决策时可见信息稳定预测。细粒度 RGB 相比最初的手部 RGB-DCT 有局部改善，
但没有超过 skeleton K，也没有形成可用累计 utility。当前冻结结论是停止这一 per-window target，
不在相同 OOF 上继续调 crop、history、network 或 threshold。完整事实和产物见
[实验账本](ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md)。

### P5 H4 joint-schedule oracle（2026-09-22冻结）

P5把连续四个完整block作为一个horizon，精确枚举每个block的三个bonus位置，共`3^4=81`种
schedule，并在fixed-center/fixed-late两种future continuation下最大化最差终端收益。完整train fit
上，uniform为2,325 errors/4.6863%，旧逐block robust oracle为1,688/3.4023%，H4 joint oracle为
1,447/2.9166%。H4相对旧oracle再减少241 errors（-0.4858 pp，578-source cluster bootstrap
95% CI [-0.5511,-0.4223]），相对uniform共减少878 errors。三种策略逐样本预算完全一致，全部
预注册门控通过。

该结果证明多窗口schedule-level target的上界显著强于逐窗口utility，但oracle读取reference、未来
continuation和真实endpoint，不可部署。它只解锁新的source-disjoint H4 schedule predictor及
ordered heatmap-U对照，不直接解锁闭环或held-out。详见
[P5报告](P5_H4_ROBUST_JOINT_SCHEDULE_ORACLE_V1.md)。

### P5 H4 schedule-ranker OOF（2026-09-23冻结）

在43,437个完整H4 horizon和578-source五折上，固定81-output ridge直接预测全部schedule的signed
robust value。K、K+summary-U、K+ordered-U和K+time-shuffled-U的累计reward分别为
`-40/-24/-51/-19`；ordered-U相对K为-11、相对打乱对照为-32，六项门槛全部失败。H4 oracle
正headroom只出现在824个horizon，而KO在42,928个horizon离开center，说明当前模型缺少可靠的
rare-event abstention；保序heatmap也没有表现出超过时间打乱对照的证据。因此不解锁闭环或
held-out。详见[P5 ranker报告](P5_H4_SCHEDULE_RANKER_OOF_V1.md)。

失败后另行冻结了post-OOF rare-event detector + conditional ranker：每折只按训练正例率选择约
1.9%的horizon。KO detector AP为.02925（prevalence .01897），固定预算召回4.98%，最终reward
仍为-11，CI [-28,6]；相对K和time-shuffled只增加4和3 reward。9/9 gate失败，说明限制动作率
仍不能定位正机会。详见[P5 abstention报告](P5_H4_ABSTENTION_RANKER_OOF_V1.md)。

后续诊断排除了“source-disjoint OOF设计导致失败”这一主要解释。KO detector AP在训练内、
row-random、sample-disjoint、source-disjoint分别为.0483/.0302/.0280/.0292；row与source只差
.0009，label-shuffled为.0193，接近.0190 prevalence。如果预先知道真实正机会，KO conditional
ranker在source OOF可捕获44.8% oracle reward。两种future continuation的机会Jaccard为.821，
共同正机会上的best schedule一致率94.8%。因此主要瓶颈是从全部horizon定位稀疏机会，而不是
source fold或两future标签不稳定。详见[P5诊断报告](P5_H4_OOF_AND_TARGET_AUDITS_V1.md)。

固定CPU MLP容量probe进一步显示：KO训练内AP可升到.2784，但row/sample/source OOF仅
.0325/.0275/.0290，source相对ridge还低.00028；shuffled-label为.0200。网络可以记忆训练模式，
却没有泛化增益。因此当前失败也不能主要归因于线性ridge容量不足；应停止在相同target/输入上
继续换分类器。

### P6 local posterior-JS information gain（2026-09-23冻结）

P6改用局部、无reference target：每个block计算offset0与offset2 ISLR posterior的symmetric JS，
每个样本只保留一半center bonus。逐样本等预算下，uniform sparse为2,761 errors/5.5651%，JS oracle
为2,562/5.1640%，减少199 errors、-0.4011 pp，source bootstrap CI [-0.5340,-0.2642]；125-error
和CI门槛均通过。它仍弱于保留全部bonus的full center 2,325 errors，但证明局部信息增益能在相同
稀疏预算下显著改善分配。该结果只解锁source-disjoint causal JS predictor，不直接解锁闭环或
held-out。详见[P6报告](P6_LOCAL_JS_INFORMATION_GAIN_ORACLE_V1.md)。

随后完成source-disjoint predictor OOF。K/KS/KO/KX的R2为.256/.267/.275/.257，对应等预算errors
2,387/2,399/2,418/2,386，均显著优于uniform sparse 2,761。KO相对uniform少343 errors，CI
[-0.8285,-0.5556] pp；但KO比K和KX分别多31和32 errors，两个CI均明确为正。ordered-U提高R2
却损害WER，未显示时序增量。预注册总gate仅2/6通过，不解锁闭环或held-out；K-only只能作为
需要独立确认的post-OOF候选。详见[P6 predictor报告](P6_LOCAL_JS_PREDICTOR_OOF_V1.md)。

为直接检验heatmap运动，随后在不读取reference/WER的诊断身份下，从正质量heatmap均值坐标显式
计算body/face/左右手的逐帧位移、最近运动、净位移、路径长度和加速度。运动M单独OOF R2为
.01419；在K/history上增加M为+.00474，而时间打乱运动仅+.00074；运动与质量联合为+.01224。
右手最近一步位移与raw JS的相关最高（rho .1064），但多数分位曲线在中等运动后平台或回落，
不支持“运动越大越重要”的全局单调假设。该结果只证明弱heatmap时序信号，不解锁policy replay、
闭环或held-out。详见[P6 heatmap运动诊断](P6_HEATMAP_INTERPRETABLE_MOTION_JS_DIAGNOSTIC_V1.md)。

随后以冻结OOF分数做同一train-fit的等预算WER诊断。K/KM/KQ/KMQ/KMM2/KX分别得到
2,387/2,391/2,402/2,399/2,377/2,390 errors。KMQ比K多12 errors、比时间打乱对照多9，两个
CI均跨0；KMM2虽少10 errors，但CI `[-.0509,+.0102]` pp跨0且是次要比较。主检查仅2/6通过，
停止heatmap scheduler分支；不启动该分支的正式GPU延时、calibration、闭环或held-out。
详见[P6 heatmap WER报告](P6_HEATMAP_INTERPRETABLE_MOTION_WER_REPLAY_V1.md)。

K-only随后在此前未用于P6的718个calibration样本、65个独立source上正式确认。local-JS R2为
.25851；逐样本等预算uniform/K/full-center为331/283/287 errors。K相对uniform少48 errors，
CI `[-1.2802,-.4564]` pp，3/3门槛通过。随后解锁的unknown-EOS自身历史模拟使用固定fit-OOF
中位阈值和0.5-token/block、容量2、no-topup控制：K/online-uniform/matched-uniform/full-center
为311/325/330/287 errors。K方向上少14/19 errors，但CI上界`+.0188/+.0174`，五项门槛只通过
三项。停止K路线，不读取held-out或调控制器。详见
[P6 K独立确认与闭环报告](P6_K_HISTORY_CALIBRATION_AND_UNKNOWN_EOS_V1.md)。

factorial audit随后确认闭环退化来自controller而非state：teacher-forced top-K、teacher-forced
fixed-token、self-history fixed-token分别为283/312/311 errors；controller shift +29，state shift
-1，固定schedule分数相关`.9739`。据此新建不使用全局阈值的因果视频内中位数控制器。在完整fit
source-OOF unknown-EOS self-history模拟中，quantile/online-uniform/matched-uniform/full-center为
2,543/2,758/2,773/2,325 errors和273,882/278,307/273,882/371,594 windows。quantile相对两个
uniform少215/230 errors，CI全负，5/5 gate通过。它相对full-center少26.3%窗口但多218 errors，
仍是精度—效率折中。详见
[P6 K控制器与在线分位报告](P6_K_HISTORY_CONTROLLER_AUDIT_AND_ONLINE_QUANTILE_V1.md)。

在获准规划final held-out后、读取新outcome之前完成split readiness audit：train 7,096样本已全部
进入fit/calibration且两者均被P6使用；dev已参与多轮历史选择；test已有历史WER结果。修复后test
尚未重跑并不能恢复其“未见”身份。因此没有执行dev/test，当前quantile结果仍是train-fit
source-disjoint OOF证据而非外部确认。详见
[P6 final evaluation readiness audit](P6_FINAL_EVALUATION_READINESS_AUDIT.md)。

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

基础正确性对照、runtime、P0--P5 oracle、标签提取与完整train OOF均已完成。H4 joint oracle通过，
但H4 direct ranker与随后冻结的rare-event/abstention新身份都得到负signed robust reward，门槛
分别6/6和9/9失败。因此停止当前H4 utility预测链，不运行calibration、unknown-EOS闭环、held-out
或test。下一方向必须更换为决策时直接可观测的训练目标或端到端资源约束目标，不能继续在同一
future/reference/EOS-derived标签上调阈值。A0对等预算B2的dev WER差异不显著，也不能解释成
现有自适应采样显著优于均匀采样。
