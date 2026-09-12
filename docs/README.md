# 文档索引

## 使用与复现

- [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md)：目录职责、命名和产物管理规则；
- [RESULTS.md](RESULTS.md)：WER、窗口减少、runtime、版本配置和权威数据位置；
- [ADAPTIVE_BASELINE_V1.md](ADAPTIVE_BASELINE_V1.md)：修复后 Phoenix dev 的冻结 A0 与 B0--B4 单一权威定义；
- [P0_SCHEDULE_DIAGNOSTICS_V1.md](P0_SCHEDULE_DIAGNOSTICS_V1.md)：等预算 replay、random、prediction-change、boundary 与 sign-center oracle 的冻结诊断结论；
- [P1_CAUSAL_CENTER_PREDICTOR_V1.md](P1_CAUSAL_CENTER_PREDICTOR_V1.md)：train-only 因果 center predictor 的数据隔离、检测结果与 No-Go 决策；
- [NEXT_RESEARCH_HANDOFF.md](NEXT_RESEARCH_HANDOFF.md)：新工作对话入口、已冻结事实、历史结果边界和下一阶段实验清单；
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)：环境、数据、checkpoint、GPU 和完整实验命令；
- [reproduction/csl_daily_top800_pipeline.md](reproduction/csl_daily_top800_pipeline.md)：Top-800 三阶段历史流程说明。

## 实验设计

- [experiments/adaptive_stride/implementation_plan.md](experiments/adaptive_stride/implementation_plan.md)：自适应步长实施方案；
- [experiments/adaptive_stride/baseline_evaluation_plan.md](experiments/adaptive_stride/baseline_evaluation_plan.md)：基础对照、数据隔离、训练/推理登记和统一评估脚本实施计划；
- [../code_agent_logs/README.md](../code_agent_logs/README.md)：带日期的实现、调参和最终结果记录。

## 上游资料

- [UPSTREAM_README.md](UPSTREAM_README.md)：FangyunWei/SLRT 原始 README、论文及引用信息；
- 各上游子项目仍保留自己的 README，例如 `TwoStreamNetwork/`、`Online/`、`NLA-SLR/`、`CiCo/` 和 `Spoken2Sign/`。
