# 下一阶段研究交接：Reliability- and Boundary-Aware Online CSLR

更新时间：2026-09-10

> **状态更新（2026-09-12）：** 本文规划的 P0 已完成并冻结。权威结论见
> `docs/P0_SCHEDULE_DIAGNOSTICS_V1.md` 与
> `docs/results/phoenix_p0_schedule_diagnostics_v1_frozen.json`。本文第 8--10 节保留为
> P0 启动时的历史计划，不再表示待执行事项。下一步是只使用 train 构造并校准最小
> causal sign-center predictor，继续以 repaired dev 评估；仍不得读取 test。

> **P1 状态更新（2026-09-12）：** 最小 train-only causal center predictor 已完成并冻结为
> No-Go。权威结论见 `docs/P1_CAUSAL_CENTER_PREDICTOR_V1.md` 与
> `docs/results/phoenix_p1_causal_center_predictor_v1_frozen.json`。固定 11 维关键点统计逻辑
> 回归在 calibration 上的最大 center recall 仅 17.83%，无法满足 recall ≥75% 且正延迟
> ≤25% 的 gate；因此没有运行 scheduler，也没有新的 WER。后续不要在 dev 上继续调该阈值，
> 应建立更强时序表征的新版本，或转向 window/decoder 与模态计算门控。

本文件用于开启新的工作对话。它冻结已经完成的 Phoenix-2014T 自适应步长 v1，说明历史结果边界，并把下一阶段限定为只使用 dev 的可靠性/边界诊断。若本文与早期日志冲突，以本文、`ADAPTIVE_BASELINE_V1.md` 和机器可读 frozen JSON 为准。

## 1. 新对话的研究问题

下一阶段不是继续微调 motion-only 步长，也不是立即运行 test，而是回答：

> 在与 uniform 相同的模型前向预算下，能否利用因果的边界、预测变化和跨模态可靠性信号，把窗口放到更有识别价值的位置？

当前方向为 **Reliability- and Boundary-Aware Online CSLR**。先完成无需新训练的 boundary/oracle 诊断，再决定是否实现轻量 boundary/signness head 或可靠性校准器。

## 2. 仓库与工作树状态

- 仓库：`/mnt/workspace/projects/haojun/SLRT`
- 分支：`main`
- 当前 HEAD：`65c88e5d9686fe338f151890dc07878809abd94b`
- `main` 相对 `origin/main`：ahead 5
- 当前有一组尚未提交、彼此关联的冻结/恢复/归档/hand-off 修改。新对话开始后先运行 `git status --short --branch`，不要 reset、覆盖或拆散这些改动。
- 本 hand-off 建立时没有正在运行的 GPU 实验。

当前变更包含：A0 v1 冻结文档与 frozen JSON；数据完整性和 61 帧恢复；修复后 dev 矩阵与三重复 runtime；修复前结果归档；A0/B2 逐样本分析；README、RESULTS、REPRODUCIBILITY 和 `.gitignore` 统一更新。开始新算法前建议审阅 diff，并在用户授权后把整组变更作为“冻结已有基线”提交。

## 3. 数据线不可混用

| 数据线 | 任务与指标 | 当前用途 |
|---|---|---|
| Phoenix-2014T | Online CSLR，WER | A0 及下一阶段边界调度主线 |
| CSL-Daily Top-800 | ISLR，accuracy/AUROC | R1 跨模态可靠性诊断 |

当前不存在“Top-800 自适应步长 WER”。不要用 Top-800 分类准确率解释 Phoenix WER。

## 4. 已冻结的 A0 v1

权威定义：`docs/ADAPTIVE_BASELINE_V1.md`
机器索引：`docs/results/phoenix_adaptive_baseline_v1_frozen.json`

### 4.1 不可变身份

| 项目 | 冻结值 |
|---|---|
| 数据 / split | 修复后 Phoenix-2014T dev |
| 样本 / 清洗后参考 gloss | 519 / 3,747 |
| 视频 ZIP SHA-256 | `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457` |
| checkpoint | `Online/CSLR/results/phoenix-2014t_ISLR/ckpts/best.ckpt`，epoch 92 |
| checkpoint SHA-256 | `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b` |
| 模型 | Online/CSLR Two-Stream S3D block5，RGB + HRNet WholeBody keypoint heatmap |
| 窗口 | 16 帧 |
| 协议 | `Online/CSLR/configs/experiments/phoenix_adaptive_baselines_v1.yaml` |
| 协议 SHA-256 | `3d583b1430def1ae0b517c79bf66abc8dae9528118b325c58620fd83e22f9f65` |
| 修复后 manifest SHA-256 | `2e6889ec14777d938a3c7ac9265ab97f87118523e9258f5f93beba4b3e854b8f` |

A0 是因果关键点运动调度：stride 1--3、EMA 0.4、分位数 0.2/0.7、历史 48 帧、warmup 16 帧、关键点阈值 0.2、至少 4 个有效点；解码为 triangular span-15、minimum weight 0.05。它只改变在线推理调度，不改变模型权重。

### 4.2 对照

| ID | 采样 | 解码 | 角色 |
|---|---|---|---|
| B0 | fixed stride 1 | window-greedy-7 | 原工程完整预算基线 |
| B1 | fixed stride 1，复用 B0 logits | span-weighted-15 | 隔离解码器变化 |
| B2 | uniform rate `1.4827861225574903` | span-weighted-15 | A0 等窗口预算基线 |
| B3 | fixed stride 2 | span-weighted-15 | 固定效率点 |
| B4 | fixed stride 3 | span-weighted-15 | 固定效率点 |
| A0 | causal adaptive stride 1--3 | span-weighted-15 | motion-only 冻结方法 |

后续新方法至少保留 B0、B1、B2、A0。改变数据 hash、checkpoint、窗口、解码、清洗规则或 A0 参数时必须建立新版本和 manifest，不能覆盖 v1。

## 5. 修复后权威结果

### 5.1 Dev 正确性

| ID | WER | DEL / INS / SUB（错误数） | errors / ref | clips |
|---|---:|---:|---:|---:|
| B0 | 22.231118% | 385 / 130 / 318 | 833 / 3,747 | 55,775 |
| B1 | 22.604750% | 422 / 124 / 301 | 847 / 3,747 | 55,775 |
| B2 | 22.551374% | 425 / 121 / 299 | 845 / 3,747 | 37,706 |
| B3 | 22.818255% | 425 / 128 / 302 | 855 / 3,747 | 28,014 |
| B4 | 22.925007% | 437 / 121 / 301 | 859 / 3,747 | 18,766 |
| A0 | 22.417934% | 434 / 104 / 302 | 840 / 3,747 | 37,615 |

A0 相对 B0：WER `+0.186816 pp`，clips `-32.559%`。A0 相对 B2：WER `-0.133440 pp`，只净少 5 个错误；paired bootstrap 95% CI `[-0.668713,+0.418291] pp`，跨 0，不能宣称显著优于 uniform。

### 5.2 三重复 runtime

同一健康 RTX 3090（PCI `81:00.0`）交替运行三次：

| ID | wall median | wall CV | forward median | 备注 |
|---|---:|---:|---:|---|
| B0 | 1642.57 s | 0.525% | 1036.95 s | 三重复 |
| B2 | 1133.10 s | 0.114% | 702.72 s | 三重复 |
| A0 | 1158.29 s | 0.771% | 701.09 s | 三重复 |

A0 相对 B0：wall `-29.483%`、`1.418x`；model forward `-32.390%`、`1.479x`。A0 相对 B2：模型前向近似相同（A0 `-0.232%`），但 wall 多 `2.224%`，所以当前调度没有端到端速度优势。B1 复用 B0；B3/B4 只有单次时间，不能与三重复统计混称。

### 5.3 权威机器结果

- 正确性：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/aggregate/dev_summary.json`
- 三重复 runtime：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_runtime_49faacc3/aggregate/dev_summary.json`
- `/tmp` 中的完整 runtime 产物不是长期权威存储；新对话不要依赖其仍然存在。

## 6. 数据恢复与历史边界

原 ZIP 缺 61 帧（train 54、dev 7、test 0），已从原始 release tar 全部恢复。修复后 947,756 张预期 PNG 全部存在，缺失/额外/重复/零字节为 0，全量 CRC 通过。

- 修复前 hash：`81629b2f3879a189613d87dafcbe04fa5053315ea9f9cfdb2f855d7a35007e30`
- 修复后 hash：`49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`
- 恢复日志：`code_agent_logs/2026-09-08/phoenix_video_frame_recovery.md`
- 修复前结果索引：`Online/CSLR/results/_archive/phoenix_pre_repair_81629b2f/`

修复前 13 个完整结果、4 个空/中断目录和 16 个日志已通过 inventory 与相对符号链接标记；原文件没有移动或复制。

历史 test（fixed 22.0005%、A0 23.0571%）绑定修复前资产。虽然 test split 缺帧为 0，但 test 已被查看，只能作为 **historical retrospective control**。尚未用修复后 manifest 运行新的 test。不得用 test 选择方法、阈值或预算，也不得把旧 test 改称修复后正式结果。

train keypoints 仍有 9 个样本、13 个非有限 x/y 标量。如果下一阶段重新训练 ISLR，必须先冻结清洗、重提取或 fail-fast 方案。

## 7. A0 与等预算 B2 为什么接近

报告：`code_agent_logs/2026-09-10/adaptive_vs_equal_budget_wer_analysis.md`

- 393/519 句（75.7%）的 hypothesis 完全相同；A0 胜 45 句，B2 胜 39 句；
- A0 改善 48 个错误、恶化 43 个，净改善只有 5 个；
- 6--15 词句子占 81% 参考词且同样持平，不能简单归因于句子短；
- A0/B2 中心精确 Jaccard 约 0.52，但 A0 中心距最近 B2 中心平均仅 0.314 帧、最大 1 帧；
- 相差 1 帧的 16 帧窗口共享 15/16 输入，再经 span-15 投票平滑；
- A0 只利用运动量，没有显式优化边界、blank 转换、不确定性或跨模态可靠性。

结论不是“WER 不受步长影响”，而是当前 motion-only A0 没有稳定地把等量计算放到比 uniform 更有价值的位置。

## 8. 新对话的下一步实验

### P0：无训练、只用 dev 的诊断

1. 分解 A0 相对 B2 多出的 2.224% wall time：运动计算、调度、加载和后处理；
2. 增加等预算 random 多 seed，判断差异是否低于普通采样方差；
3. 构造仅用于诊断的 boundary oracle 和 prediction-change oracle，与 B2/A0 严格等预算；
4. 计算窗口到 gloss boundary、blank/non-blank 转换和高损失区域的距离与命中率；
5. 输出决策：存在可利用 oracle 上界，或当前窗口/解码下优化空间很小。

不要先增加新的启发式步长。若 oracle 无收益，先重审窗口、解码或任务定义；若 oracle 明显优于 B2，再进入 P1。

### P1--P3

- P1：人工审计一批 alignment，优先定义可复现的 `signness + event end` 或 center-offset target；先报告 boundary F1/endpoint error，再接 scheduler。
- P2：可靠性必须在 train/calibration split 校准。Top-800 R1 中 Keypoint accuracy 66.28%，未校准 direct-confidence 选流仅 63.86%，禁止直接“谁 confidence 高选谁”。
- P3：最终比较 B0/B1/B2/A0、boundary-only、reliability-only 和完整组合；所有选择只用 dev。

参考研究文档：`code_agent_logs/2026-05-29/future_research_directions.md` 与 `code_agent_logs/2026-07-21/reliability_boundary_experiment_plan.md`。

## 9. GPU 与存储约束

禁止使用：

| PCI | UUID |
|---|---|
| `01:00.0` | `GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a` |
| `25:00.0` | `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83` |

此前健康卡为 PCI `81:00.0` / `GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631`，但每次仍需现场验证。始终用单个 UUID 绑定，不使用逻辑 index。

本文建立时 NFS 使用率约 99%、剩余约 139 GiB；`/tmp` 剩余约 320 GiB。大型 logits/中间结果写 `/tmp`，最终只把 manifest、resolved config、runtime、aggregate、必要预测和日志持久化到 NFS；不能把唯一结果留在 `/tmp`。

## 10. 新对话启动清单

先执行：

```bash
cd /mnt/workspace/projects/haojun/SLRT
git status --short --branch
git diff --check
python3 -m json.tool docs/results/phoenix_adaptive_baseline_v1_frozen.json >/dev/null
df -h /mnt/workspace/projects/haojun /tmp
```

然后按顺序阅读：

1. `docs/NEXT_RESEARCH_HANDOFF.md`
2. `docs/ADAPTIVE_BASELINE_V1.md`
3. `code_agent_logs/2026-09-10/adaptive_vs_equal_budget_wer_analysis.md`
4. `code_agent_logs/2026-05-29/future_research_directions.md`
5. `code_agent_logs/2026-07-21/reliability_boundary_experiment_plan.md`
6. `docs/REPRODUCIBILITY.md`

给新对话的首条任务建议：

> 阅读 `docs/NEXT_RESEARCH_HANDOFF.md` 并核对工作树。仅使用修复后 Phoenix dev 的既有结果和冻结 A0/B2 协议，设计并实施 P0 等预算 oracle/随机采样/边界命中诊断；不得读取 test、不得改变 A0 v1、不得覆盖现有结果。先给可复现计划和输出目录，CPU 可完成的分析优先，确需新前向时才申请健康单卡 GPU。

## 11. 声明边界

可以说：A0 在 dev 上相对 B0 减少 32.56% 窗口和 29.48% wall time，同时 WER 增加 0.1868 pp。
不可以说：A0 显著优于等预算 uniform、数据恢复提高了 WER、已有修复后 test、或 Top-800 已验证自适应 WER。

下一阶段只有在等预算 B2 上取得可复现、配对统计支持的收益，或在相近 WER 下获得额外端到端成本下降，才能升级当前结论。
