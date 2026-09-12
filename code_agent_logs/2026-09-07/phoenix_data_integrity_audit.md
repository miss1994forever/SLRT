# Phoenix-2014T 数据完整性审计

日期：2026-09-07

> 后续状态：本文记录的是修复前审计。61 个缺失 RGB 帧已于 2026-09-08 从原始 release tar 恢复，标准 ZIP 当前缺失数为 0；恢复证据和新资产 hash 见 `code_agent_logs/2026-09-08/phoenix_video_frame_recovery.md`。train keypoints 的 13 个非有限坐标仍未处理。

## 结论

本轮完成了 metadata、词表、视频 ZIP 成员、ZIP CRC、全量 PNG 解码以及 isolated keypoints 的一致性检查。

- 现存的 947,695 张 PNG 全部通过 ZIP CRC 和 Pillow `verify()`，尺寸均为 `210 x 260`、模式均为 RGB；没有损坏、重复、零字节或额外 PNG。
- metadata 共声明 947,756 帧，因此归档仍缺少 61 个预期帧，涉及 train 10 个样本、dev 1 个样本；test 不缺帧。
- metadata 的字段、样本名、split、帧数、alignment 长度和 split 间隔离均通过检查。
- 1,116 项词表无重复，`<blank>` 位于索引 0；gloss 小写规范化后，train/dev/test 均无 OOV。
- keypoints 覆盖全部 8,257 个样本，时间长度与 metadata 完全一致，形状均为 `[T, 133, 3]`、dtype 均为 `float16`。dev/test 全部为有限值；train 有 9 个样本、共 13 个 x/y 坐标为正负无穷。

因此，这份本地数据可用于继续诊断和内部成对对照，但不能表述为“严格完整的官方资产”。在修复或预注册排除规则之前，后续正式 benchmark 应明确标记为本地归档结果。

## 资产身份

| 资产 | 字节数 | SHA-256 |
|---|---:|---|
| `phoenix14t.train` | 1,720,110 | `bc1708a98a84ed9d2983afed4fc561350e94f3f3a00c9e76252935e8b18b73ea` |
| `phoenix14t.dev` | 122,825 | `94f4fd4dbf0f4eb7f13d04e24ca08f971d6226a6c8498d656fae333311a901e8` |
| `phoenix14t.test` | 133,232 | `22edf2b3196e250b6d0fc9faf8cacdfeca6e77c1ea80300b2524e04892544197` |
| `phoenix_iso_with_blank.vocab` | 12,551 | `d79c8ef31d8ec4345b511d641a832a67f1d8045020bee35521edee3b773fcc34` |
| `keypoints_hrnet_dark_coco_wholebody_iso.pkl` | 756,978,313 | `69c56c1902b6a528f644d7be6ce14bb902a86fa83dd4fbcaa0d78a6ef58dc5ff` |
| `PHOENIX2014T_videos.zip` | 41,720,845,149 | `81629b2f3879a189613d87dafcbe04fa5053315ea9f9cfdb2f855d7a35007e30` |

这些 hash 与 `baseline_matrix_v1/protocol_manifest.json` 中正式 dev 对照运行登记的值一致。

## Metadata 与词表

| Split | 样本数 | 声明帧数 | 原始 gloss 数 | 样本名重复 | 小写规范化后 OOV |
|---|---:|---:|---:|---:|---:|
| train | 7,096 | 827,354 | 55,247 | 0 | 0 |
| dev | 519 | 55,775 | 3,748 | 0 | 0 |
| test | 642 | 64,627 | 4,264 | 0 | 0 |
| 合计 | 8,257 | 947,756 | 63,259 | 0 | 0 |

检查项包括 `name`、`signer`、`gloss`、`text`、`num_frames` 和 `alignments` 必需字段，正帧数、非空文本、PAMI alignment token 数等于 `num_frames`，以及 train/dev/test 样本 basename 互不重叠。错误数为 0。

实验评估清洗后的 dev/test 参考 gloss 数分别为 3,747/4,259；它们与上表原始 metadata token 数不同是清洗规则造成的，不是数据缺失。

## 视频归档

### 汇总

| Split | 预期帧 | 缺帧 | 缺帧比例 | 受影响样本 | 样本比例 |
|---|---:|---:|---:|---:|---:|
| train | 827,354 | 54 | 0.006527% | 10 / 7,096 | 0.140924% |
| dev | 55,775 | 7 | 0.012550% | 1 / 519 | 0.192678% |
| test | 64,627 | 0 | 0% | 0 / 642 | 0% |
| 合计 | 947,756 | 61 | 0.006436% | 11 / 8,257 | 0.133220% |

ZIP 中共有 947,695 个唯一 PNG 成员，总解压字节数 41,528,096,567；单图大小范围为 21,902--60,472 字节。归档实际采用 `images/<sample>/imagesNNNN.png`，loader 会从带 split 的逻辑路径回退到该结构。

`unzip -tq` 对整个 41.72 GB 归档进行 CRC 检查，结果为 `No errors detected in compressed data`，wall time 225.67 秒，最大 RSS 4,544 KB。随后对全部 PNG 执行 Pillow `Image.open(...).verify()`，947,695/947,695 通过，耗时 195.11 秒。

CRC/解码通过只证明“归档中已有的成员未损坏”，不能证明 metadata 要求的成员全部存在；因此它与 61 帧缺失并不矛盾。

### 缺帧明细

| Split | 样本 | 期望帧数 | 缺失帧编号（从 1 开始） | 数量 |
|---|---|---:|---|---:|
| dev | `11December_2009_Friday_tagesschau-3509` | 16 | 9, 10, 11, 13, 14, 15, 16 | 7 |
| train | `01April_2010_Thursday_heute-6710` | 16 | 6, 8--16 | 10 |
| train | `01April_2011_Friday_tagesschau-3379` | 16 | 16 | 1 |
| train | `05July_2011_Tuesday_tagesschau-6135` | 15 | 13--15 | 3 |
| train | `06December_2010_Monday_heute-6213` | 15 | 13--15 | 3 |
| train | `06September_2010_Monday_tagesschau-1144` | 16 | 8, 9, 11--16 | 8 |
| train | `08February_2010_Monday_heute-1505` | 16 | 6, 8--16 | 10 |
| train | `15October_2009_Thursday_tagesschau-7721` | 16 | 4, 6--16 | 12 |
| train | `17February_2010_Wednesday_heute-1486` | 16 | 16 | 1 |
| train | `18June_2010_Friday_tagesschau-4826` | 16 | 11--13, 15, 16 | 5 |
| train | `25August_2009_Tuesday_heute-3301` | 16 | 16 | 1 |

loader 目前对同一样本只打印第一次缺帧警告，并以最近一张已读取图像（没有参考帧时为黑图）占位。因此旧日志只显示 dev 的 `images0009.png`，不代表该样本只缺一帧。

## 对现有 dev 对照的影响

缺帧 dev 样本参考为 `ALPEN LANG SCHNEE`。B0/B1/B2/B3/B4/A0 在各自正式评估 decoder 下均输出空串，即都产生 3 个 deletion。它不会偏向其中某个变体，但会进入所有变体的绝对 WER。

仅作为敏感性审计，若事后排除该样本（不作为新的正式结果），结果为：

| 变体 | 原 519 样本 WER | 排除后 518 样本 WER |
|---|---:|---:|
| B0 | 22.231118% | 22.168803% |
| B1 | 22.604750% | 22.542735% |
| B2 | 22.551374% | 22.489316% |
| B3 | 22.818255% | 22.756410% |
| B4 | 22.925007% | 22.863248% |
| A0 | 22.417934% | 22.355769% |

所有绝对 WER 约下降 0.062 个百分点，排序不变；A0 与 B0 的差异基本不变。这支持现有成对比较在内部仍可解释，但不替代修复资产后的正式复跑。

## Keypoints

所有 keypoint 条目均能与 metadata 样本一一对应，时间长度完全一致。dev/test 无 NaN 或 infinity。在 126,051,548 个 keypoint（每个含 x/y/score 三项）中，train 发现 13 个非有限 x/y 标量，分布在以下 9 个样本：

| train 样本 | 非有限坐标数 |
|---|---:|
| `11September_2010_Saturday_tagesschau-5000` | 1 |
| `14October_2010_Thursday_tagesschau-288` | 2 |
| `10March_2011_Thursday_heute-50` | 1 |
| `06October_2011_Thursday_tagesschau-822` | 2 |
| `07December_2010_Tuesday_tagesschau-4151` | 1 |
| `01September_2010_Wednesday_tagesschau-5033` | 2 |
| `29August_2009_Saturday_tagesschau-5026` | 1 |
| `26November_2011_Saturday_tagesschau-5850` | 2 |
| `09August_2010_Monday_heute-5894` | 1 |

这不影响当前只使用 dev 的自适应步长对照，但未来重新训练 ISLR 前应将非有限值显式修复、重提取或 fail-fast。检测到 score 范围约为 0.00605--1.8496；HRNet DARK heatmap score 不必是 `[0,1]` 概率，因此本审计不把大于 1 单独判为损坏。

## 读取 42 GB 对机器的影响

数据位于 NFS4 挂载 `10.214.242.14:/volume2/workspace/projects/haojun`，检查时文件系统约 7.0 TB、已用 6.7 TB、剩余 311 GB、使用率 96%。本轮只读、未解压落盘：

- 不会额外永久占用约 42 GB 磁盘，也不会产生同等规模的本地 SSD 写入磨损；
- 主要消耗共享 NFS/网络吞吐，检查期间可能使同挂载上的数据加载变慢；
- Linux 会使用空闲内存作页缓存，应用需要内存时通常可回收，并非 42 GB 被永久占用；
- CRC/PNG 解码带来有限 CPU 开销；本轮使用 `nice -n 10` 和 `ionice -c3`，但 `ionice` 对 NFS 网络流量的约束有限；
- 不使用 GPU，不影响显存；为避免 I/O 干扰，检查期间不应同时采集正式 runtime。

## 后续处理

1. 优先从可信来源重新取得或核对 `PHOENIX2014T_videos.zip`，不要用插值帧伪装成原始数据。
2. 修复后以同一审计重新确认预期/实际成员差为 0，再复跑 dev 基础矩阵和 runtime。
3. 若短期无法恢复，需在运行前预注册固定排除清单，并让所有变体使用完全相同的样本；不能根据模型输出临时决定排除。
4. 为正式入口增加一次性资产 manifest 检查和严格模式，避免 loader 的占位行为让缺帧静默进入 benchmark。
5. 未来重新训练前处理 train 的 54 个缺失 RGB 帧和 13 个非有限 keypoint 坐标。
