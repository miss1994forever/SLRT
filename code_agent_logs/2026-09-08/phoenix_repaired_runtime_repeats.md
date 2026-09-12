# Phoenix-2014T 修复后 Dev Runtime 三重复

日期：2026-09-08

## 实验身份

- 数据：Phoenix-2014T dev，519 个样本、3,747 个清洗后参考 gloss；未运行 test。
- 视频：修复后 `PHOENIX2014T_videos.zip`，SHA-256 `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`。
- 模型：冻结的 Two-Stream S3D checkpoint epoch 92；没有训练或调参。
- 协议：`configs/experiments/phoenix_adaptive_baselines_v1.yaml`，manifest SHA-256 `2e6889ec14777d938a3c7ac9265ab97f87118523e9258f5f93beba4b3e854b8f`。
- Git commit：`65c88e5d9686fe338f151890dc07878809abd94b`。
- GPU：仅 PCI `81:00.0`、UUID `GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631`；已知故障的 `01:00.0` 和 `25:00.0` 未使用。
- 核心变体：B0、B2、A0 各完整运行 3 次；B1 复用 B0 前向，不单独计时。

## Preflight 与存储处置

宿主侧确认 8 张卡均为 0 MiB、0%，目标 `81:00.0` 为 41 °C。随后以 UUID 隔离执行轻量 CUDA 分配，PyTorch 只枚举到 1 张 RTX 3090，分配与读回成功。

启动前项目 NFS 为 100%、可用 0，而 `/tmp` 可用约 322 GiB、inode 使用约 2%。根据正确性 run 估算 9 次产物约 1.70 GiB，因此正式运行只写独立本地根：

`/tmp/slrt_phoenix_repaired_runtime_49faacc3/`

视频和 checkpoint 仍从原 NFS 位置只读。运行结束后 `/tmp` 产物约 1.8 GiB、剩余约 320 GiB。NFS 后续恢复约 139 GiB 可用，遂将不含大型 `dev_logits.pkl` 的 72 MiB 审计副本持久化到：

`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_runtime_49faacc3/`

该副本保留 manifest、resolved config、command、stdout、每次 `runtime.json`/`complete.json`、runtime profile、评估结果、无 logits 的逐样本结果和 aggregate，可由统一评估器复核；九份约 1.75 GiB 的 logits 仅留在 `/tmp`，避免重复占用 NFS。

## 交替顺序

为降低温度、缓存和共享 I/O 随时间漂移造成的固定顺序偏差，按以下顺序串行执行：

1. R1：B0 → B2 → A0；
2. R2：B2 → A0 → B0；
3. R3：A0 → B0 → B2。

时间戳显示所有九次严格串行，无重叠。第一次从 15:07:21 UTC 开始，最后一次于 18:23:57 UTC 完成。

## Runtime 结果

Wall time 为 runner 包围完整 `prediction_slide.py` 命令的时间；model forward 为 PyTorch 模型前向累计时间。

| 变体 | Wall R1 / R2 / R3 (s) | Wall mean (s) | median (s) | std (s) | CV | Forward R1 / R2 / R3 (s) | Forward median (s) | Forward CV | 峰值 CUDA allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1642.57 / 1651.48 / 1634.23 | 1642.76 | 1642.57 | 8.62 | 0.525% | 1036.77 / 1037.65 / 1036.95 | 1036.95 | 0.045% | 7.297 GiB |
| B2 | 1133.10 / 1132.77 / 1135.16 | 1133.68 | 1133.10 | 1.29 | 0.114% | 702.72 / 703.12 / 701.17 | 702.72 | 0.146% | 7.295 GiB |
| A0 | 1158.29 / 1144.42 / 1161.04 | 1154.58 | 1158.29 | 8.91 | 0.771% | 701.09 / 700.56 / 701.09 | 701.09 | 0.044% | 7.295 GiB |

三个变体的 wall-time CV 均低于预注册的 5% 上限。以三次中位数比较：

| 对照 | Clips 变化 | Wall time 变化 | Wall speedup | Model-forward 变化 | Forward speedup |
|---|---:|---:|---:|---:|---:|
| B2 vs B0 | -32.396% | -31.017% | 1.450× | -32.233% | 1.476× |
| A0 vs B0 | -32.559% | -29.483% | 1.418× | -32.390% | 1.479× |
| A0 vs B2 | -0.241% | +2.224% | 0.978× | -0.232% | 1.002× |

A0 和 B2 的模型前向时间几乎相同，符合两者近等 clips 预算；A0 的完整命令中位数比 B2 慢约 25.20 秒，说明自适应 schedule、数据处理、解码和共享 I/O 等非模型部分存在额外开销。相对 B0，A0 减少 32.56% 窗口，对应 29.48% wall-time 减少和 1.418× 加速。

## 正确性与预算复核

统一评估器逐次核对了样本顺序、参考文本、最终 hypothesis 和采样 schedule；同一变体的三次结果完全一致。

| 变体 | WER | Errors / Ref | Clips |
|---|---:|---:|---:|
| B0 | 22.231118% | 833 / 3,747 | 55,775 |
| B2 | 22.551374% | 845 / 3,747 | 37,706 |
| A0 | 22.417934% | 840 / 3,747 | 37,615 |

B2 与 A0 相差 91 clips，即相对 A0 为 0.241925%，通过预注册的 2% 等预算上限。A0 相对 B0 的 WER 为 +0.186816 pp；相对同解码、等预算 B2 为 -0.133440 pp。该 runtime 实验没有改变既有准确率结论。

## 运行异常和边界

- 九次模型运行均 return code 0，并生成 `complete.json`；没有 CUDA、模型或数据读取失败。
- NFS 初始为满盘。前两个早期 run 的 `stdout.log` 各记录一次项目附属 logger 写既有 NFS 日志位置时的 `No space left on device`，但 runner 主 stdout 和所有结果均写 `/tmp`，推理继续并完整成功。其余七次无该警告。
- B0 三次 wall-time CV 仅 0.525%，B2 为 0.114%，说明上述一次性日志警告没有造成明显计时离群。
- runner 没有逐次采样全机 load average；preflight 时主机负载低，结束时 load average 为 0.17 / 0.75 / 1.01。这是共享服务器 wall-time 的固有限制，论文应同时报告更稳定的 model-forward 指标。
- 本轮只测 dev、没有读取 test、没有调整任何参数，也没有重新训练模型。

## 验收与数据位置

统一评估器已在 `/tmp` 原始结果和 NFS 审计副本上各运行一次并通过。入口文件：

- `.../protocol_manifest.json`
- `.../aggregate/dev_summary.json`
- `.../aggregate/dev_summary.csv`
- `.../aggregate/dev_summary.md`
- `.../dev/{B0_fixed1_window7,B2_uniform_rate_span15,A0_adaptive_span15}/run_01..03/`

本实验完成了预注册的三重复 runtime 测量。下一步若需要论文级最终 test，只能复用已经冻结的 dev 配置，并将 test 明确标为 retrospective control；本轮结果本身不授权或触发 test。
