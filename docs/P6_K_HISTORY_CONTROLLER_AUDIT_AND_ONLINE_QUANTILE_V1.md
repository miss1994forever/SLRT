# P6 K/history控制器诊断与在线分位策略 v1

日期：2026-09-23。实验只复用train fit及已经消费的train-calibration；未读取dev、test或held-out，
全程CPU。factorial audit为诊断身份；online quantile在读取其policy outcome前另行冻结。

## Controller/state factorial audit

在同一718-sample calibration上补齐teacher-forced历史+固定阈值/token，并与已有两个格子比较：

| 历史 | 控制 | Errors | WER |
|---|---|---:|---:|
| Teacher-forced | known-length top-K | 283 | 5.0817% |
| Teacher-forced | fixed threshold/token | 312 | 5.6024% |
| Self-history | fixed threshold/token | 311 | 5.5845% |

从top-K换成token控制增加29 errors，CI `[+.2223,+.8410]` pp；再换self-history为-1 error，CI
`[-.2362,+.1959]`。沿固定teacher action schedule，teacher/self分数Pearson `.9739`、平均绝对变化
`.1613`、跨阈值侧比例7.35%；两个真实闭环动作agreement 90.32%、Jaccard `.8128`。

因此退化明确由controller/pacing主导，而不是self-history状态漂移。该分解是有反馈交互的描述性
诊断，不能当作严格可加因果效应。

## Causal within-video quantile controller

新控制器取消全局绝对阈值。每个source折的ridge仅由另四折训练；每个视频在当前block只用严格
过去分数的精确中位数作为阈值。token仍为初始0、每完整block积累0.5、容量2、无EOS/top-up；
状态只使用策略实际执行的posterior。

| 策略 | Windows | Errors | WER |
|---|---:|---:|---:|
| Online quantile K | **273,882** | **2,543** | **5.1257%** |
| Online uniform token | 278,307 | 2,758 | 5.5590% |
| Matched-count uniform | **273,882** | 2,773 | 5.5893% |
| Full center | 371,594 | 2,325 | 4.6863% |

Quantile K相对online uniform少215 errors、少4,425 windows，WER差`-.4334` pp，CI
`[-.5316,-.3344]`；相对完全等窗口uniform少230 errors、`-.4636` pp，CI
`[-.5839,-.3412]`。五项预注册门槛全部通过。

它相对full center少97,712 windows（26.3%），但多218 errors、`.4394` pp，CI
`[+.3280,+.5500]`，仍是明确的精度—效率折中，不是对full center的精度改进。

## 冻结结论

这是当前第一个同时满足unknown-EOS、no-topup、self-history、source-disjoint OOF，并显著优于
在线uniform和完全等窗口uniform的learned controller。它解锁的是“请求一次最终held-out规划”，
不是自动读取held-out。由于路线经历多次post-hoc诊断，最终检验必须完整冻结、明确多重选择风险，
且不得再用held-out调threshold、quantile或token。

证据目录：

- `p6_k_history_controller_state_factorial_audit_v1_49faacc3/`；
- `p6_k_history_online_quantile_oof_v1_49faacc3/`。
