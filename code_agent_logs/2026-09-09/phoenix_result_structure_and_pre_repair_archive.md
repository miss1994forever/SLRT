# Phoenix 结果结构与修复前数据归档索引

日期：2026-09-09

## 目的与边界

本次对 `Online/CSLR/results/` 做只读 provenance 审计，并为使用修复前 Phoenix 视频资产产生的结果建立集中索引。没有移动、删除或复制任何实验结果、checkpoint 或数据文件；没有重新计算 41.7 GB 视频 ZIP 的 hash，也没有运行 GPU。

索引目录：

`Online/CSLR/results/_archive/phoenix_pre_repair_81629b2f/`

该目录仅包含 README、JSON/CSV inventory 和相对符号链接。原结果路径保持不变，链接本身不会增加大文件存储占用。

## 数据版本

| 版本 | SHA-256 | 状态 |
|---|---|---|
| 修复前 | `81629b2f3879a189613d87dafcbe04fa5053315ea9f9cfdb2f855d7a35007e30` | 缺少 61 帧；保留为历史审计资产 |
| 修复后 | `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457` | metadata 与 ZIP 精确匹配；当前标准资产 |

旧 `baseline_matrix_v1/protocol_manifest.json` 直接记录了完整修复前 hash。修复后正确性矩阵和 runtime manifests 直接记录完整修复后 hash。早期 legacy 结果没有资产 hash manifest，因此使用运行结果/日志时间、Phoenix 配置路径、恢复审计中的资产切换记录和相关实验日志交叉确认；没有只依据名称分类。

## 分类结果

### 确认使用修复前数据的完整结果

共 13 个目录，合计 2,791,228,054 bytes（约 2.60 GiB）：

- `baseline_matrix_v1`（manifest 直接 hash 证据）和 `baseline_matrix_v1_smoke_5`；
- 早期 `prediction_slide` / `origin_recheck` / `current_fixed`；
- fixed、span-13、冻结 span-15 的 test 与 smoke 结果；
- D0/D1/D2 dev tuning 的 `_s16` 完整结果。

完整逐目录清单、字节数和证据类型见归档的 `inventory.csv`。上述目录均通过 `links/results/` 下的相对符号链接集中展示。

### 修复前空目录或中断尝试

共 4 个、0 bytes：`prediction_slide_current_adaptive`、`prediction_slide_dev_tune_d0_fixed`、`prediction_slide_dev_tune_d1_max2` 和 `prediction_slide_dev_tune_d1_max2_retry`。它们没有完整机器结果，不能计作完成实验；索引和 inventory 明确标为 `confirmed_pre_repair_incomplete`。

另有 16 个修复前根日志，共 35,581 bytes，链接在 `links/logs/`，用于保存成功或失败运行的时间与命令证据。

### 确认使用修复后数据的结果

以下 3 个目录没有放入修复前结果链接区：

| 目录 | 字节数 | 证据 |
|---|---:|---|
| `baseline_matrix_v1_repaired_49faacc3` | 829,798,777 | manifest 完整 hash |
| `baseline_matrix_v1_repaired_49faacc3_smoke_1` | 81,666 | 恢复日志、运行时间和修复后 smoke 记录 |
| `baseline_matrix_v1_repaired_runtime_49faacc3` | 84,774,808 | manifest 完整 hash |

合计 914,655,251 bytes。正确性矩阵与三重复 runtime 是当前修复后权威机器结果。修复后异常样本的 logits 与修复前不同，但最终 hypothesis 仍为空，所以聚合 WER/clips 恰好不变；结果相同不代表数据版本相同。

### 未验证或非实验结果

以下项目没有混入 confirmed 清单：

- `ckpts/`：模型资产，与视频数据版本无关；
- `prediction_slide.py`：结果目录内的源代码快照，不是实验结果；
- `sign2text_service.log`：没有证据证明是 Phoenix benchmark 结果。

详情见 `unverified.csv`。

## 其他 results 根检查

检查了 `Online/CSLR/results/` 下的 CSL-Daily、Top-800 和 online_slr_csl 根目录。五份 CSL-Daily `VideoLoader.py` 快照仅在源码注释中包含 `PHOENIX2014T_videos.zip` 示例路径；未发现 protocol manifest、resolved config、命令或结果输出实际绑定 Phoenix 视频资产。因此没有从其他数据集结果根归档任何项目。

## Git 与可维护性

`.gitignore` 仍忽略所有常规模型输出，只对白名单目录 `_archive/phoenix_pre_repair_81629b2f/` 开放轻量索引。符号链接全部使用相对路径并已检查可解析。原始结果仍在被忽略的位置，不会因索引而进入 Git。

后续若删除或迁移旧资产，应先更新 inventory 与符号链接；在此之前不通过归档索引修改目标内容。
