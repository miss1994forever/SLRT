# 机器可读结果索引

本目录只保存 compact JSON 索引和兼容摘要。大型预测、logits、pickle、checkpoint 与逐样本
运行产物保留在本地 `Online/CSLR/results/` 或 `data/`，不进入 Git。

| 文件 | 身份 | 对应叙事文档 |
|---|---|---|
| [`phoenix_adaptive_baseline_v1_frozen.json`](phoenix_adaptive_baseline_v1_frozen.json) | 修复后 A0/B0--B4 冻结身份 | [`../ADAPTIVE_BASELINE_V1.md`](../ADAPTIVE_BASELINE_V1.md) |
| [`phoenix_p0_schedule_diagnostics_v1_frozen.json`](phoenix_p0_schedule_diagnostics_v1_frozen.json) | P0 调度诊断冻结索引 | [`../P0_SCHEDULE_DIAGNOSTICS_V1.md`](../P0_SCHEDULE_DIAGNOSTICS_V1.md) |
| [`phoenix_p1_causal_center_predictor_v1_frozen.json`](phoenix_p1_causal_center_predictor_v1_frozen.json) | P1 predictor No-Go 冻结索引 | [`../P1_CAUSAL_CENTER_PREDICTOR_V1.md`](../P1_CAUSAL_CENTER_PREDICTOR_V1.md) |
| [`phoenix_p2_causal_center_tcn_v1_frozen.json`](phoenix_p2_causal_center_tcn_v1_frozen.json) | P2 causal TCN No-Go 冻结索引 | [`../P2_CAUSAL_CENTER_TCN_V1.md`](../P2_CAUSAL_CENTER_TCN_V1.md) |
| [`phoenix_adaptive_stride_summary.json`](phoenix_adaptive_stride_summary.json) | 兼容历史摘要，包含修复前 retrospective test 叙事 | [`../RESULTS.md`](../RESULTS.md) |

出现数字冲突时，优先级为：对应阶段的 `*_frozen.json` 与冻结文档、protocol manifest、
`RESULTS.md`，最后才是兼容历史摘要。JSON 中的本地路径只用于 provenance，不表示数据会随
Git 分发。
