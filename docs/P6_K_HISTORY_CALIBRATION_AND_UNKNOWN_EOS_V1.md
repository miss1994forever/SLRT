# P6 K/history独立calibration与unknown-EOS闭环 v1

日期：2026-09-23。用户在获知将读取此前未用于P6的calibration后继续实验。两个阶段均先冻结协议；
只使用train的718个calibration样本、65个source，与fit的6,378样本、578个source严格隔离。
没有读取dev、test或held-out，全程CPU。

## 独立已知长度确认

模型为在全部fit K/history行上训练的ridge alpha=100；calibration不参与归一化、拟合或调参。
每个样本用预测分数选择`ceil(B/2)`个offset2 bonus，与uniform逐样本完全等预算。

| 策略 | Errors | WER |
|---|---:|---:|
| Uniform sparse | 331 | 5.9436% |
| K/history | **283** | **5.0817%** |
| Full center | 287 | 5.1535% |

calibration local-JS R2为`.25851`，超过`.20`门槛。K相对uniform少48 errors、WER下降`.8619` pp，
source-bootstrap CI `[-1.2802,-.4564]`；三项门槛全部通过。K比full center少4 errors不应解释为
“少算必然更准”，更可能是相邻冗余窗口对当前span15解码存在轻微噪声。

该阶段独立确认了K的排序信号，但仍使用已知长度全局top-K和teacher-forced full-center历史，因此
只解锁unknown-EOS自身历史模拟，不解锁held-out。

## Unknown-EOS自身历史模拟

阈值固定为fit source-OOF K分数中位数`-3.802705`，没有使用calibration调阈值。每个完整block
积累0.5 token，初始0、容量2；余额不足1不能执行offset2，达到2强制执行，其余情况由固定阈值
决定。末尾不补token；K状态只包含策略实际执行的后验。online uniform使用相同token规则并在
eligible时立即执行，matched uniform离线匹配K的实际窗口数。

| 策略 | Windows | Errors | WER |
|---|---:|---:|---:|
| K unknown-EOS/self-history | 31,085 | 311 | 5.5845% |
| Online uniform token bucket | 31,532 | 325 | 5.8359% |
| Matched-count uniform | 31,085 | 330 | 5.9257% |
| Full center | 42,098 | 287 | 5.1535% |

K比online uniform少14 errors并少447 windows，但CI `[-.5298,+.0188]` pp跨0；比完全等窗口数的
matched uniform少19 errors，CI `[-.7126,+.0174]`也跨0。五项门槛通过3项，整体失败。

已知长度teacher-forced K为283 errors，unknown-EOS self-history为311，说明强离线结果没有完整
转移。差异同时包含全局top-K变成固定阈值/token pacing和teacher-forced变成on-policy history，
当前身份不能进一步归因；也不能在calibration结果后调整阈值、token容量或初始余额。

## 冻结结论

K/history的local-JS信号在独立source上复现，但部署控制器只呈现方向正确、统计不足的改善。
按预注册规则停止K路线，不读取held-out，不调控制器。证据目录：

- `p6_k_history_independent_calibration_v1_49faacc3/`；
- `p6_k_history_unknown_eos_self_history_v1_49faacc3/`。
