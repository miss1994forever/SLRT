# Phoenix-2014T 缺失视频帧恢复

开始日期：2026-09-07
完成日期：2026-09-08

## 结果

已从本地原始 Phoenix-2014-T release tar 恢复完整性审计发现的 61 个缺失 RGB 帧，并将通过校验的修复版切换到项目标准路径：

`data/phoenix_2014t/PHOENIX2014T_videos.zip`

修复后的 metadata/ZIP 精确集合检查结果：

| 项目 | 结果 |
|---|---:|
| metadata 预期 PNG | 947,756 |
| ZIP 实际 PNG | 947,756 |
| 缺失 | 0 |
| 额外 | 0 |
| 重复路径 | 0 |
| 零字节成员 | 0 |
| ZIP 全量 CRC | 通过，`ZipFile.testzip() is None` |

train/dev/test 的缺失数现在均为 0。原先缺失的 7 个 dev 帧全部通过 PNG 解码，项目 `ZipReader` 也能通过运行时使用的 `images/dev/<sample>/...` 逻辑路径读取恢复帧。

## 资产身份与版本关系

| 角色 | 文件 | 字节数 | SHA-256 |
|---|---|---:|---|
| 恢复来源 | `phoenix-2014-T.v3.tar.gz` | 41,699,758,035 | `e47fec2d460ed2ea5d19198df46a231781f2abc7a8aa14132184af4a8166f90d` |
| 修复前 ZIP | `PHOENIX2014T_videos.incomplete_81629b2f.zip` | 41,720,845,149 | `81629b2f3879a189613d87dafcbe04fa5053315ea9f9cfdb2f855d7a35007e30` |
| 修复后标准 ZIP | `PHOENIX2014T_videos.zip` | 41,723,658,945 | `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457` |

旧的 `baseline_matrix_v1/protocol_manifest.json` 绑定修复前 hash `81629b2f...07e30`，因此既有 WER/runtime 仍属于修复前资产。修复后实验必须生成新 manifest，并记录 `49faacc3...d457`；不能把旧实验结果重新标记为修复后结果。

修复前 ZIP 暂时保留用于旧实验审计，至少保留到修复后 dev 矩阵和 runtime 复跑验收完成。它额外占用约 41.7 GB；检查完成时 NFS 剩余约 273 GB、使用率 97%。

## 恢复过程

1. 在不修改现有 ZIP 的前提下，从原始 tar 一次性提取 11 个受影响样本至 `data/phoenix_2014t/restore_stage_20260907/`。
2. 原始 tar 提供了这些样本的全部 176 张 PNG；此前缺少的 61 张全部存在。
3. 176 张暂存 PNG 均通过 Pillow `verify()`，尺寸均为 `210 x 260`、模式均为 RGB。
4. 暂存数据与原 ZIP 重叠的 115 张逐字节一致，差异只来自原 ZIP 不存在的 61 张。
5. 创建原 ZIP 的独立候选副本。副本追加前 SHA-256 为 `81629b2f...07e30`，与原 ZIP 完全一致。
6. 仅向候选追加 61 个不存在的 `images/<sample>/imagesNNNN.png` 成员，不覆盖任何已有成员。
7. 对候选执行全量 CRC、metadata 精确集合、重复路径、零字节和恢复帧解码检查。
8. 验收通过后，将原 ZIP 重命名为带原 hash 标识的备份，并把候选切换到标准路径。

原始 tar 顺序提取约耗时 8.5 分钟，41.7 GB 安全复制约耗时 517 秒，修复候选 SHA-256 约耗时 237 秒。大文件操作均使用低 CPU/I/O 优先级，不使用 GPU。

## PNG 校验链

修复前全量审计已对原 ZIP 的 947,695 张 PNG 逐张执行 Pillow `verify()`；候选追加前 hash 与原 ZIP 一致。恢复时又验证了原始 tar 中全部 176 张受影响样本图片，其中 115 张与原 ZIP 逐字节一致、61 张为新增帧。因此，修复后 947,756 张的并集均有 PNG 解码校验证据；同时修复后 ZIP 全量 CRC 通过。

## 尚未解决的问题

本次只修复 RGB 视频帧。完整性审计发现的 train keypoints 中 9 个样本、13 个非有限 x/y 标量尚未处理。它们不影响当前 Phoenix dev/test 自适应步长推理，但如果重新训练 ISLR，必须先确定可复现的清洗或重提取策略。

## 下一步

1. 使用修复后标准 ZIP 重新运行完整 dev B0--B4/A0 正确性矩阵，生成新 manifest。
2. 核对原缺帧 dev 样本 `11December_2009_Friday_tagesschau-3509` 的各变体输出及新 WER。
3. 在同一健康 GPU 上交替重复 runtime；修复复跑期间仍不使用 test 调参。
4. 修复后实验验收并备份 manifest 后，再决定是否删除 41.7 GB 的修复前 ZIP 和 5.4 MB 暂存目录。
