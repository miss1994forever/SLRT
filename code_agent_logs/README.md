# Code Agent Logs

本目录保存与仓库提交对应的实现、诊断和实验记录。大体积 checkpoint、logits、视频和 `results/` 不进入 Git；记录中保留配置、命令、指标和结果路径。

## 研究路线

- `2026-05-29/future_research_directions.md`：Top-800 Online CSLR 后续研究方向；当前主线是 Reliability- and Boundary-Aware Online CSLR。
- `2026-07-21/reliability_boundary_experiment_plan.md`：组合方向下一阶段实验入口与数据隔离规则。

## 自适应步长冻结基线

- `2026-07-15/adaptive_stride_implementation.md`：实现与验证记录。
- `2026-07-16/adaptive_stride_dev_tuning.md`：只使用 dev 的预注册调参记录。
- `2026-07-16/adaptive_stride_full_comparison.md`：调参前 span-13 历史完整对照。
- `2026-07-16/adaptive_stride_frozen_final_evaluation.md`：dev 冻结 span-15 后的唯一最终 test 评估。

冻结基线使用 Git tag `adaptive-stride-v1`。后续研究从 `research/reliability-boundary-online-cslr` 分支开展，不重写冻结基线历史。
