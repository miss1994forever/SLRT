# P4 HRNet positive-mass uncertainty pilot v1

日期：2026-09-21。实验为 train-only、outcome-blind 数值审计，没有读取 WER、utility、reference、
dev/test 或 held-out outcome。

## 冻结定义

复用 `p4_hrnet_heatmap_uncertainty_pilot_v1` 完全相同的50个视频和4,498帧。对每个 HRNet raw
heatmap 使用：

\[
p_{yx}=\frac{\max(H_{yx},0)}{\sum_{uv}\max(H_{uv},0)}.
\]

均值和协方差按 heatmap 宽72、高96归一化。预注册配置为
`Online/CSLR/configs/p4_hrnet_positive_mass_uncertainty_pilot_v1_preregistered.json`。
GPU 实验在沙盒外运行，只使用 PCI `61:00.0`、`81:00.0`、`A1:00.0`、`C1:00.0`、
`E1:00.0`；未使用宿主机 GPU0、`25:00.0` 或 `41:00.0`。

## 结果

- 4,498/4,498帧成功，598,234/598,234个关键点的正质量有效；
- 协方差全部 finite，最小归一化特征值为0.000946，通过 PSD gate；
- 双手归一化 covariance trace 中位0.00862，IQR 0.00515；entropy 中位5.474，IQR
  0.272，均不是常数；
- trace 与 raw peak 的 Spearman 相关为 -0.392，未退化成 confidence 的重复量；
- 每帧双手 trace 中位数与 bbox 面积的 Spearman 相关为0.046，未被人物框大小主导；
- positive-mass 统计的 CPU 实现中位耗时15.26 ms/帧，超过预注册的5 ms门槛。

数值有效性、变化性和非冗余性7项门控全部通过，只有实现延时门控失败。因此本轮结论是：
**positive-mass 定义在数值上可用，但当前 CPU 实现未通过部署成本冻结条件；停止在小样本阶段，
不启动完整 train 提取，也不进入下游 target/utility 检验。** 下一步若继续，应建立新的实现优化身份，
优先在 GPU 上于 heatmap 离开设备前聚合统计，或验证更快的 CPU kernel；不得事后修改本实验门槛。

## 实现优化后续

v2 将同一定义改为 CUDA float32 聚合，统计中位延时降到1.65 ms/帧，但相对 v1 的最大 moment
差为 `3.05e-5`，略高于预注册的 `2e-5`，因此 v2 仍按门控停止，没有放宽阈值。

v3 保持输入和 `x log x` 乘积为 float32，并使用 CUDA float64 做 reduction 和 moment 算术：

- 同一4,498帧全部成功；
- 统计中位1.88 ms/帧，P95 3.03 ms，低于5 ms门槛；
- 相对 v1 的 moment 最大绝对差 `2.91e-11`、entropy 最大差 `4.77e-7`、positive mass
  最大相对差0；
- 数值、变化性、非冗余、bbox独立性、延时和三项 parity 共11项门控全部通过；
- 5卡墙钟147.86秒，30.42 FPS；完整827,354帧线性投影约7.55小时。

因此 v3 已解锁完整 train 提取。全量作业已于2026-09-21 13:17 UTC以5卡 detached supervisor
启动，覆盖7,096个train视频、827,354帧；按视频原子检查点，可在 Codex 断线时继续并支持中断续跑。
当前兼容实现仍会先由旧 MMPose 把 raw
heatmap 拷到 CPU，再传回 GPU 聚合；正式实时集成应在 heatmap 离开 GPU 前聚合，只回传
`133×11` 个统计值。

全量提取于2026-09-21 20:44 UTC完成：7,096/7,096视频、827,354/827,354帧有效，5卡墙钟
7小时26分钟，输出4.573 GB，全部完成门控通过且没有读取 outcome。

随后运行了在全量结果前冻结的 CSC-L8-v2 heatmap-U identifiability probe。逐帧96维 U 档案覆盖
6,378个fit视频和743,147帧；对181,815 blocks、578 sources使用同一五折。结果为：K R2
`.02786`/accuracy `.41810`，U R2 `.02520`/accuracy `.41792`，K+U R2 `.03508`/accuracy
`.41794`。K+U 相对 K 只增加 `.00722` R2，accuracy下降 `.000154`。五折R2均为正，
但另外四项预注册门控失败，因此停止 heatmap-U CSC-L8 分支，不读取 policy outcome、闭环或 held-out。

独立的新身份随后只检验CSC target本身与WER的oracle relevance：在逐样本完全等预算下，完美CSC
winner将完整train fit从2,325降至2,224 errors（WER -0.2036 pp，source-cluster 95% CI
`[-0.3100,-0.0965]`），但101-error收益低于预注册249-error门槛。因此heatmap-U档案本身仍是
有效训练资产，但不应继续用于当前CSC scheduler；若复用，必须提出新的target并先做WER上界检验。

证据：

- `p4_hrnet_positive_mass_uncertainty_pilot_v1_49faacc3/metrics.json`；
- `p4_hrnet_positive_mass_uncertainty_pilot_v1_49faacc3/positive_mass_diagnostics.json`。
- `p4_hrnet_positive_mass_gpu_stats_pilot_v2_49faacc3/positive_mass_gpu_diagnostics.json`；
- `p4_hrnet_positive_mass_gpu_mixed_precision_pilot_v3_49faacc3/positive_mass_gpu_mixed_precision_diagnostics.json`。
- `p4_hrnet_positive_mass_fulltrain_v1_49faacc3/metrics.json`；
- `p4_csc_l8_heatmap_u_identifiability_v2_49faacc3/metrics.json`。
- `p4_csc_l8_oracle_relevance_v1_49faacc3/metrics.json`。
