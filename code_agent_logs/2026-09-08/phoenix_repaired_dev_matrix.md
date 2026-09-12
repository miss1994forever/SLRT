# Phoenix-2014T 修复后 Dev 正确性矩阵

日期：2026-09-08

## 实验身份

- 数据：Phoenix-2014T dev，519 个样本、3,747 个清洗后参考 gloss；未运行 test。
- 视频资产：修复后的 `data/phoenix_2014t/PHOENIX2014T_videos.zip`，41,723,658,945 bytes，SHA-256 `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`。
- 模型：Online/CSLR Two-Stream S3D，RGB + keypoint heatmap，冻结 checkpoint epoch 92；checkpoint SHA-256 `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`。
- 协议：`configs/experiments/phoenix_adaptive_baselines_v1.yaml`，未修改参数、未调参；B1 复用 B0 前向。
- Git commit：`65c88e5d9686fe338f151890dc07878809abd94b`。
- GPU：仅 PCI `81:00.0`、UUID `GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631`。正式运行前为 0 MiB、0%、41 °C；已知故障的 `01:00.0` 和 `25:00.0` 未使用。
- manifest SHA-256：`2e6889ec14777d938a3c7ac9265ab97f87118523e9258f5f93beba4b3e854b8f`。
- 归档结果目录：`Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/`。

manifest 记录了当时工作树中的既有文档改动：`code_agent_logs/README.md` 以及未跟踪的 2026-09-07/08 审计日志。运行前已核对，模型代码、协议、配置和运行/评估脚本没有未提交改动；因此以 `--allow-dirty` 启动，同时由 manifest 对所有登记代码、配置和资产保存精确 hash。

## Preflight

1. `slrt_legacy` 环境 Python 3.9.25、PyTorch 1.10.2+cu113。
2. 全部 25 项单元测试通过。
3. 基础矩阵 dry-run 通过。
4. 在同一 UUID 上执行 B4 单样本真实 GPU smoke：return code 0，14 clips，模型前向 7.51 秒，峰值 CUDA allocated 6,931,376,128 bytes。
5. 完整 manifest 中视频 hash 与修复日志登记的 `49faacc3...d457` 一致。

## 完整结果

| ID | 采样/解码 | WER | DEL | INS | SUB | Clips | Wall time | 模型前向 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | fixed stride=1 / window-greedy-7 | 22.231118% | 10.274887% | 3.469442% | 8.486789% | 55,775 | 1675.88s | 1035.89s |
| B1 | fixed stride=1 / span-weighted-15 | 22.604750% | 11.262343% | 3.309314% | 8.033093% | 55,775 | 复用 B0 | 复用 B0 |
| B2 | uniform rate=1.482786... / span-weighted-15 | 22.551374% | 11.342407% | 3.229250% | 7.979717% | 37,706 | 1136.52s | 703.35s |
| B3 | fixed stride=2 / span-weighted-15 | 22.818255% | 11.342407% | 3.416066% | 8.059781% | 28,014 | 877.09s | 523.61s |
| B4 | fixed stride=3 / span-weighted-15 | 22.925007% | 11.662663% | 3.229250% | 8.033093% | 18,766 | 628.88s | 351.33s |
| A0 | adaptive stride=1--3 / span-weighted-15 | 22.417934% | 11.582599% | 2.775554% | 8.059781% | 37,615 | 1179.60s | 701.16s |

所有变体均有 `complete.json`、return code 0。统一评估器从保存结果重新计算 WER，并通过样本数、参考词数、样本顺序、resolved config、decoder 和预算约束检查。B2 与 A0 相差 91 clips，即 0.241925%，低于预注册的 2% 上限。

主要对照：

| 对照 | A0 WER 变化 | A0 clips 减少 | 单次 wall time 减少 | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| A0 vs B0 | +0.186816 pp | 32.559390% | 29.613257% | [-0.525470, +0.852723] pp |
| A0 vs B1 | -0.186816 pp | 32.559390% | 29.613257% | [-0.700818, +0.315969] pp |
| A0 vs B2 | -0.133440 pp | 0.241341% | -3.790448% | [-0.668713, +0.418291] pp |

这是一次正确性复跑，不是三重复 runtime benchmark。wall time 只能作为本轮运行记录，不能替代预注册的同卡交替三重复性能实验。

## 修复前后核对

与修复前 `baseline_matrix_v1` 相比，六个报告变体的精确 WER、错误数、clips 和采样 schedule 全部不变。原因不是恢复帧未被读取：逐样本比较 logits 后，只有原缺帧样本发生变化，其余 518 个样本完全一致。

原异常样本：`dev/11December_2009_Friday_tagesschau-3509`，参考为 `ALPEN LANG SCHNEE`。

| 变体 | 修复前报告 hypothesis | 修复后报告 hypothesis | Clips | 修复前后 logits 最大绝对差 |
|---|---|---|---:|---:|
| B0 | 空串 | 空串 | 16 | 4.930732 |
| B1 | 空串 | 空串 | 16（复用 B0） | 4.930732（复用 B0） |
| B2 | 空串 | 空串 | 11 | 4.935555 |
| B3 | 空串 | 空串 | 8 | 2.434158 |
| B4 | 空串 | 空串 | 6 | 4.934299 |
| A0 | 空串 | 空串 | 16 | 4.930732 |

恢复的真实 RGB 帧改变了该样本的模型输出 logits，但仍不足以令最终 decoder 产生 gloss，因此它在各变体中仍贡献 3 个 deletion，聚合 WER 恰好与修复前一致。A0 的 schedule 由完整且未改动的 keypoints 决定，所以恢复 RGB 不改变 clips。

## ENOSPC 中断与处置

首次直接写标准 NFS 结果目录时，B0 运行到约 352/519 后由 runner 报 `OSError: [Errno 28] No space left on device`。当时项目 NFS 显示 7.0 TB 全满、可用 0；新结果目录仅约 40 KB，因此不是本实验写满。`/tmp` 当时仍有约 323 GB 可用。

处置遵循以下约束：

1. 未删除旧 ZIP、旧矩阵或用户数据。
2. 失败目录保留为 `dev/B0_fixed1_window7/run_01_failed_enospc_20260908/`。
3. 将完全相同的 manifest 复制到 `/tmp/phoenix_baseline_matrix_v1_repaired_49faacc3/`，在同一 GPU、同一 checkpoint 和同一参数下从头执行完整矩阵。
4. `/tmp` 运行完成并统一评估通过后，NFS 已恢复约 104 GB 可用；将约 792 MB 完整结果复制回标准新目录，再从标准目录重新运行评估器并通过。
5. `/tmp` 与标准目录的 manifest SHA-256 均为 `2e6889ec...54b8f`。

本轮单次 wall time 包含使用本地 `/tmp` 作为结果落盘位置的条件；模型输入视频仍从同一 NFS 修复 ZIP 读取。这进一步说明正式 runtime 结论必须另做三重复，而不能把本轮单次时间当作最终性能数据。

## 结论与下一步

- 修复后完整 dev 正确性矩阵通过，旧矩阵未覆盖，test 未运行。
- 数据恢复确实改变了异常样本 logits，但没有改变最终解码、WER、clips、版本排序或原有结论。
- 当前证据仍支持“相对 B1，A0 减少约 32.56% 窗口并保持相近 WER”；置信区间跨 0，不能声称 WER 显著改善。
- 下一项独立实验才是新输出根目录上的同卡交替三重复 runtime；本轮没有启动该实验。

机器结果入口：

- `results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/protocol_manifest.json`
- `results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/aggregate/dev_summary.json`
- `results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/aggregate/dev_summary.csv`
- `results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3/aggregate/dev_summary.md`
