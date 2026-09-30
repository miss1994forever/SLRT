# P6 online-quantile 视频流式接入与成本测量

日期：2026-09-27 UTC。范围：录制的 Phoenix-2014T **train-fit** 视频工程验证。未读取 dev/test 的视频、标注、logits、预测或指标；没有重跑 dev。这里的准确率不构成未见视频泛化证据。

## 冻结输入与版本

正式结果产生前写入 [`plan.json`](plan.json)：3 个明确的 train-fit 样本（`train/25October_2010_Monday_tagesschau-14`、`train/25October_2010_Monday_tagesschau-23`、`train/05May_2011_Thursday_tagesschau-30`），按 train 元数据顺序选取首批 72–96 帧 fit 成员；25 FPS 录制视频回放；PCI 61:00.0 / GPU `GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed`；每策略每样本重复 2 次。split 白名单来自 `fit_calibration_split.json`，SHA-256 `plan.json` 已记录。ZIP 成员本身没有 split 前缀，适配器只在核对 train-fit 名称后映射并读取该名称对应的逐帧成员。Keypoint 文件为仅 train 的文件。

控制器快照 SHA-256 `53a17e01641cea239379504f0f46fa079c8e858dc882713596364ef34d1425d5`，两轮正式运行结束后仍一致；五折模型、规则配置、适配器脚本哈希均在 `plan.json`。实际 ISLR checkpoint SHA-256 `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`，与冻结交接一致。视频 ZIP 的冻结哈希沿用交接记录，未对 39 GB ZIP 再全量哈希。没有修改控制器、算法、checkpoint 或 decoder。

## 时间语义与接入

窗口长 16 帧，复现 dense sliding 的左 7/右 8 边界复制；窗口 `s` 在帧 `min(T-1,s+8)` 到达后才可执行。每个 block 的 offset0 在可用时实际执行、取得 posterior 后登记到控制器。offset2 的原始帧保留在 25 帧缓存里；完整 block 的 offset3 到达后才运行 quantile/token 决策，若选中才组装并执行 offset2。在线 uniform 同样使用 coverage、完整 block、每块 +0.5 token、余额满 1 执行一次 bonus；它在运行时不使用 quantile 的事后预算。此处没有使用 matched uniform。每个策略只对选中的窗口调用 ISLR checkpoint，非 dense-logits 回放。

单调时钟逐帧记录到达、逐窗口记录 posterior 完成与 gloss 提交、逐 block 记录决策及 offset2 缓存等待。墙钟包括逐帧 ZIP 解码、选中窗口组装、设备传输、模型、控制器和 25 FPS pacing；不含模型载入及一次性 train keypoint pickle 载入。缓存等待是选中 offset2 的输入可用至决策的时间，**与总墙钟重叠，不能再相加**。首识别输出定义为首个付费窗口 posterior 可用；首次稳定 gloss 提交定义为 15 帧三角投票中不再受未来窗口影响的首个非 blank gloss，至少落后最新已付费窗口 7.5 个中心单位。终端 gloss 使用同一 span-15 投票、去重和 blank 过滤。输出仅覆盖识别/gloss 阶段。

## 正式测量

原始记录：[`measurements.jsonl`](measurements.jsonl)，SHA-256 `b36507deea436c238d7956731c213d5f78f0a9674a9f18a5763587bf41a9ad20`。机器汇总与逐样本窗口序列：[`summary.json`](summary.json)。以下 P50/P95 为 6 次视频运行分布；帧到达积压与单次缓存等待为事件分布。样本少，分位数仅描述这组工程运行。

| 指标 | Quantile | 在线 uniform |
|---|---:|---:|
| 窗口 / bonus（6 次合计） | 182 / 56 | 188 / 62 |
| gloss edit errors / reference tokens | 0 / 32 | 0 / 32 |
| 墙钟 P50 / P95 | 3.372 / 3.971 s | 3.373 / 3.974 s |
| 首识别输出 P50 / P95 | 0.383 / 0.421 s | 0.383 / 0.384 s |
| 首稳定 gloss P50 / P95 | 0.847 / 0.848 s | 0.847 / 0.848 s |
| 控制器累计 P50 / P95 | 4.38 / 5.40 ms/视频 | 0.020 / 0.025 ms/视频 |
| 模型累计 P50 / P95 | 1.187 / 1.425 s/视频 | 1.240 / 1.442 s/视频 |
| 前处理累计 P50 / P95 | 0.259 / 0.300 s/视频 | 0.258 / 0.311 s/视频 |
| 选中 bonus 缓存等待 P50 / P95 | 40.20 / 40.53 ms | 39.98 / 40.28 ms |
| 帧到达积压 P50 / P95 | 2.22 / 12.84 ms | 2.27 / 13.46 ms |
| 墙钟有效帧吞吐 | 24.14 FPS | 24.14 FPS |

每个视频每轮的 quantile/uniform 窗口数分别为 `35/36`、`26/27`、`30/31`；两次重复的 quantile 动作序列逐项相同。两策略 6 对最终 gloss 完全相同，且这 3 个 train-fit 样本均为 0 edit error。quantile 少 6/188 = 3.19% 的窗口，平均每视频模型时间少约 44 ms，但回放墙钟均约 3.452 s。有限样本显示计算减少而未见总耗时加速；固定 25 FPS pacing 占约 1.9 s/视频，且 offset2 有约一帧的缓存等待。

## 兼容 wait-2 文本阶段

项目存在 Phoenix 德语 wait-2 G2T checkpoint，实际 SHA-256 `f618ea4117e0bbbff7cbcf2771afcff4b452991243da430db43ef1d4aea30671`。固定补充计划 [`text_plan_v2.json`](text_plan_v2.json) 使用正式识别轨迹的第一次重复，3 视频 × 2 策略；[`text_measurements_v2.jsonl`](text_measurements_v2.jsonl) 保留两个稳定 gloss 可用时间、首词与整句生成耗时和文本。v1 短生成长度只得到空输出，已由 v2 的非空首词测量取代。

| 文本阶段（3 次/策略） | Quantile P50/P95 | 在线 uniform P50/P95 |
|---|---:|---:|
| 两个稳定 gloss 可用 | 1.329 / 1.762 s | 1.329 / 1.759 s |
| 首词阶段计算（含分词、嵌入、生成） | 28.8 / 36.5 ms | 27.8 / 27.9 ms |
| 整句文本阶段计算 | 211.1 / 240.9 ms | 213.7 / 236.2 ms |

这是在保存的 gloss 时间戳之后**顺序执行的文本阶段诊断**。`first_text_if_uncontended_s` 只是 gloss ready 加文本计算的无争用估计，不是同进程并发视频→文本服务实测；未测摄像头采集、网络、文本队列或最终稳定文本提交，因此不能报为系统总延迟。两个策略在每个视频的最终德语文本相同。

## 验证、瓶颈和剩余范围

一次小规模烟雾验证先通过，再运行冻结的 12 次正式视频测量。正式日志含 1,000 个逐帧到达事件、370 个实际执行窗口事件和 248 个 block 决策事件；所有选中窗口都有真实 checkpoint 前向调用，所有最终序列都有引用与预测，量测脚本执行前检查 split、模型、规则、控制器和自身哈希。两次重复量化动作完全一致。控制器独立一致性补核由另一任务负责：本报告只验证上文记录的控制器哈希，尚未收到其最终验收结论；若补核改变核心版本，应固定其通过版本并重新运行本工程测量。

复现视频测量（在协议允许的沙盒外 GPU 执行，从 `Online/CSLR` 目录）：

```bash
CUDA_VISIBLE_DEVICES=GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed OMP_NUM_THREADS=1 \
  /mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  tools/measure_p6_streaming_video_20260928.py \
  --plan /mnt/workspace/projects/haojun/SLRT/outputs/streaming_integration_latency_20260928/plan.json \
  --output /tmp/p6_streaming_repeat.jsonl
```

文本补充测量从 `Online/SLT` 目录执行 `evaluation/measure_p6_wait2_text_20260928.py --plan /mnt/workspace/projects/haojun/SLRT/outputs/streaming_integration_latency_20260928/text_plan_v2.json --output /tmp/p6_wait2_repeat.jsonl`，同样设置上述 GPU UUID 与环境。现有结果文件采用拒绝覆盖写入。
