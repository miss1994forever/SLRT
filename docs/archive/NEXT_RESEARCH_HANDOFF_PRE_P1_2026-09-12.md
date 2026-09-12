# P0/P1 前期 handoff 历史索引

状态：已被当前 [`../NEXT_RESEARCH_HANDOFF.md`](../NEXT_RESEARCH_HANDOFF.md) 取代。

历史版本冻结提交：`a57825d`

历史路径：`docs/NEXT_RESEARCH_HANDOFF.md`

该版本形成于 A0 基线冻结之后，主要内容包括：

- 修复后 Phoenix-2014T A0/B0--B4 身份、结果和 runtime；
- 数据恢复、修复前 test 历史边界和关键点异常；
- A0 与等预算 B2 接近的逐样本解释；
- P0 random、boundary、prediction-change 和 oracle 诊断启动计划；
- P1 train-only sign-center predictor 的启动约束。

这些计划现在已经执行完毕。结果分别冻结在：

- [`../ADAPTIVE_BASELINE_V1.md`](../ADAPTIVE_BASELINE_V1.md)；
- [`../P0_SCHEDULE_DIAGNOSTICS_V1.md`](../P0_SCHEDULE_DIAGNOSTICS_V1.md)；
- [`../P1_CAUSAL_CENTER_PREDICTOR_V1.md`](../P1_CAUSAL_CENTER_PREDICTOR_V1.md)。

需要审计旧 handoff 原文时，使用 Git 中的不可变版本：

```bash
git show a57825d:docs/NEXT_RESEARCH_HANDOFF.md
```

旧文本中的“下一步执行 P0/P1”等句子均为历史计划，不再表示当前待办。
