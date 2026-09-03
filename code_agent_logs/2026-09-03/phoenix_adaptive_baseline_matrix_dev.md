# Phoenix 自适应步长基础对照矩阵：完整 Dev 正确性运行

日期：2026-09-03

## 实验身份

- 数据：Phoenix-2014T dev，519个连续样本，3,747个参考gloss；
- 模型：Online/CSLR Two-Stream S3D，RGB + keypoint heatmap，冻结 checkpoint epoch 92；
- Git commit：`2393728100214cac4784ff046a32f16746911ffa`；
- 协议：`configs/experiments/phoenix_adaptive_baselines_v1.yaml`；
- manifest SHA-256：`aaa0b1e4e780ee4b4e949e7f3e514119ee80b0fb032bf4e082a5bd1a6dfdd5ed`；
- GPU：仅 PCI `81:00.0`，UUID `GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631`；
- test：未读取、未运行；
- 结果目录：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1/`。

## 完整结果

| ID | 采样 | 解码 | WER | DEL | INS | SUB | Clips | Wall time | 模型前向 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | fixed stride=1 | window-greedy-7 | 22.2311% | 10.2749% | 3.4694% | 8.4868% | 55,775 | 1680.61s | 1033.02s |
| B1 | fixed stride=1 | span-weighted-15 | 22.6048% | 11.2623% | 3.3093% | 8.0331% | 55,775 | 复用B0 | 复用B0 |
| B2 | uniform rate=1.482786... | span-weighted-15 | 22.5514% | 11.3424% | 3.2293% | 7.9797% | 37,706 | 1183.30s | 700.51s |
| B3 | fixed stride=2 | span-weighted-15 | 22.8183% | 11.3424% | 3.4161% | 8.0598% | 28,014 | 904.25s | 520.49s |
| B4 | fixed stride=3 | span-weighted-15 | 22.9250% | 11.6627% | 3.2293% | 8.0331% | 18,766 | 641.32s | 349.88s |
| A0 | adaptive stride=1--3 | span-weighted-15 | 22.4179% | 11.5826% | 2.7756% | 8.0598% | 37,615 | 1191.51s | 697.98s |

所有变体峰值 PyTorch allocated memory 约7.29--7.30 GiB。B1与B0复用同一次前向，wall time不能重复相加。

## 主要对照

| 对照 | WER变化 | Clips变化 | 单次wall time变化 | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| A0 vs B0 原始工程基线 | +0.1868 pp | -32.559% | -29.103% | [-0.5255, +0.8527] pp |
| A0 vs B1 同解码基线 | -0.1868 pp | -32.559% | -29.103% | [-0.7008, +0.3160] pp |
| A0 vs B2 等预算均匀采样 | -0.1334 pp | -0.241% | +0.693% | [-0.6687, +0.4183] pp |

B2共37,706 clips，A0共37,615 clips，绝对预算差异0.242%，通过预注册的2%上限。A0的stride分布为：stride 1共25,563次、stride 2共5,597次、stride 3共6,455次。

## 结论边界

1. B0完全复现历史冻结dev结果：519样本、3,747参考词、55,775 clips、WER 22.231118%。
2. 与B1相比，A0在减少32.56%窗口的同时WER低0.1868 pp；与近等预算B2相比，A0 WER低0.1334 pp。
3. 上述paired bootstrap区间均跨0，不能宣称A0相对B1或B2存在统计显著的WER改善。当前证据支持“A0保持相近精度并减少约三分之一窗口”，不支持“显著提升识别精度”。
4. 本轮是单次正确性运行。wall time受当时I/O和系统负载影响，只能作为初步数据；最终runtime结论仍需同卡交替重复至少3次。
5. test没有参与本轮，也不能根据这些结果修改A0后再把既有test当作全新盲测。

## 验收

- 5个实际前向变体均有`complete.json`且返回码为0；
- 统一评估器重新计算WER并核对样本顺序、参考文本、resolved config和decoder；
- B2/A0预算约束通过；
- 运行结束后8张GPU均为0 MiB、0%，无残留计算进程；
- 聚合结果：`aggregate/dev_summary.json`、`dev_summary.csv`、`dev_summary.md`。

下一步：在新的runtime输出根目录、相同健康GPU上交替重复3次；不运行test，不调整A0。
