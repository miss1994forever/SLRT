# Code Agent Logs

本目录保存与仓库提交对应的实现、诊断和实验记录。大体积 checkpoint、logits、视频和 `results/` 不进入 Git；记录中保留配置、命令、指标和结果路径。

## 研究路线

- `2026-05-29/future_research_directions.md`：Top-800 Online CSLR 后续研究方向；当前主线是 Reliability- and Boundary-Aware Online CSLR。
- `2026-07-21/reliability_boundary_experiment_plan.md`：组合方向下一阶段实验入口与数据隔离规则。
- `2026-07-26/reliability_r1_results.md`：Top-800 isolated dev 的完整 R1 结果与 R2 入口约束。

## 自适应步长冻结基线

修复后 dev 的唯一冻结口径以 `docs/ADAPTIVE_BASELINE_V1.md` 为准；`docs/RESULTS.md` 同时保留历史 test 叙事。以下文件保留完整审计过程。

- `2026-07-15/adaptive_stride_implementation.md`：实现与验证记录。
- `2026-07-16/adaptive_stride_dev_tuning.md`：只使用 dev 的预注册调参记录。
- `2026-07-16/adaptive_stride_full_comparison.md`：调参前 span-13 历史完整对照。
- `2026-07-16/adaptive_stride_frozen_final_evaluation.md`：dev 冻结 span-15 后的唯一最终 test 评估。

冻结基线使用 Git tag `adaptive-stride-v1`。后续研究从 `research/reliability-boundary-online-cslr` 分支开展，不重写冻结基线历史。

稳定复现入口已统一到 `docs/REPRODUCIBILITY.md`；本目录保留按日期形成的原始审计链，不再承担 README 快速开始功能。

## 仓库维护

- `2026-09-02/repository_cleanup.md`：目录、配置、README、复现脚本和 `main` 发布整理记录。
- `2026-09-02/adaptive_stride_result_consistency_audit.md`：冻结结果、历史版本、机器产物与 runtime 口径核对。
- `2026-09-02/adaptive_baseline_framework_implementation.md`：B0--B4/A0 协议、统一采样/评估框架和离线验收记录。
- `2026-09-03/phoenix_adaptive_baseline_matrix_dev.md`：B0--B4/A0完整Phoenix dev正确性矩阵、等预算对照和置信区间。
- `2026-09-07/phoenix_data_integrity_audit.md`：Phoenix-2014T metadata、词表、全量视频 CRC/PNG 与 keypoints 完整性审计。
- `2026-09-08/phoenix_video_frame_recovery.md`：从原始 release tar 恢复 61 个缺失 RGB 帧、校验修复 ZIP 并记录新旧资产 hash。
- `2026-09-08/phoenix_repaired_dev_matrix.md`：使用修复后视频资产复跑完整 Phoenix dev B0--B4/A0 正确性矩阵，核对异常样本 logits、WER 与 ENOSPC 处置。
- `2026-09-08/phoenix_repaired_runtime_repeats.md`：在同一健康 GPU 上对修复后 Phoenix dev 的 B0/B2/A0 执行交替三重复 runtime，报告 wall time、模型前向、CV 与等预算加速。
- `2026-09-09/phoenix_result_structure_and_pre_repair_archive.md`：按资产 hash、manifest 和运行日志审计 Phoenix 结果 provenance，并建立修复前结果的非破坏性符号链接索引。
- `2026-09-10/adaptive_vs_equal_budget_wer_analysis.md`：逐样本、句长、视频长度和 schedule 对比 A0 与等预算 B2，解释 WER 接近的证据、机制与下一步最小实验。
- `2026-09-10/adaptive_baseline_freeze_audit.md`：核对配置、manifest、代码与修复后结果，并冻结 A0 v1 和 B0--B4 对照身份。
- `2026-09-10/next_research_handoff_consolidation.md`：将冻结基线、数据恢复、结果归档和下一阶段 P0 实验收束为新工作对话 hand-off。
