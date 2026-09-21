# 文档导航

从本页判断文档身份。原则是：**冻结结论不移动，当前 handoff 保持简短，历史过程进入日期
日志或 archive，机器 JSON 不承担叙事。**

## 当前入口

| 文件 | 作用 | 更新规则 |
|---|---|---|
| [`NEXT_RESEARCH_HANDOFF.md`](NEXT_RESEARCH_HANDOFF.md) | 当前状态、约束和唯一下一步入口 | 每完成一个阶段后重写，不累积旧计划 |
| [`RESULTS.md`](RESULTS.md) | 跨阶段统一结果与声明口径 | 只记录已验证数字 |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | 环境、资产和复现命令 | 命令或资产身份改变时更新 |
| [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) | 目录职责与维护规范 | 结构规则改变时更新 |
| [`ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md`](ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md) | P0--P3 完整事实账本与产物索引 | 每条实验链冻结后追加，不删除失败结果 |

## 已冻结的 Phoenix 研究阶段

| 阶段 | 权威文档 | 决策 |
|---|---|---|
| A0 | [`ADAPTIVE_BASELINE_V1.md`](ADAPTIVE_BASELINE_V1.md) | 降低窗口量；未显著优于等预算 uniform |
| P0 | [`P0_SCHEDULE_DIAGNOSTICS_V1.md`](P0_SCHEDULE_DIAGNOSTICS_V1.md) | sign-center offline Strong-Go；其余主要调度信号 No-Go |
| P1 | [`P1_CAUSAL_CENTER_PREDICTOR_V1.md`](P1_CAUSAL_CENTER_PREDICTOR_V1.md) | 最小因果 center predictor No-Go |
| P2 | [`P2_CAUSAL_CENTER_TCN_V1.md`](P2_CAUSAL_CENTER_TCN_V1.md) | 增强关键点 causal TCN No-Go；未打开 dev/test |
| P3 | [`ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md`](ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md) | robust oracle 上界强；全部可部署 OOF predictor 未过 gate，停止当前 per-window utility target |

对应的 compact、机器可读冻结索引统一列在 [`results/README.md`](results/README.md)。P3 的
逐实验配置、metrics 和本地产物路径由完整实验账本索引。阶段冻结文档一旦提交不重写结论；
新实验建立新版本并由 `RESULTS.md` 说明关系。

## 实验设计与专项复现

- [`experiments/adaptive_stride/implementation_plan.md`](experiments/adaptive_stride/implementation_plan.md)：
  自适应步长最初实施方案，属于历史设计；
- [`experiments/adaptive_stride/baseline_evaluation_plan.md`](experiments/adaptive_stride/baseline_evaluation_plan.md)：
  基础对照矩阵实施计划，属于历史设计；
- [`reproduction/csl_daily_top800_pipeline.md`](reproduction/csl_daily_top800_pipeline.md)：
  CSL-Daily Top-800 三阶段历史流程。

执行过程、审计证据和失败记录保留在 [`../code_agent_logs/README.md`](../code_agent_logs/README.md)，
不复制进稳定文档。

## 历史与上游

- [`archive/README.md`](archive/README.md)：已被替代的交接与计划索引；
- [`UPSTREAM_README.md`](UPSTREAM_README.md)：FangyunWei/SLRT 原始 README、论文与引用；
- `CiCo/`、`NLA-SLR/`、`Online/`、`Spoken2Sign/` 和 `TwoStreamNetwork/` 内的 README 属于各
  上游子项目，保持原位。

## 新文档放置规则

- 面向使用者且长期有效的说明放 `docs/`；
- 阶段冻结文档使用 `<STAGE>_<TOPIC>_V<N>.md`，机器索引放 `docs/results/`；
- 尚未执行的实验设计放 `docs/experiments/<topic>/`；
- 专项复现说明放 `docs/reproduction/`；
- 实施过程放 `code_agent_logs/YYYY-MM-DD/`；
- 被替代但仍需显式入口的计划摘要放 `docs/archive/`；
- 不为同一结论再创建一份新的总结文档，优先更新索引。
