# Phoenix 自适应步长基础对照实验实施计划

## 1. 目标

在不修改模型权重和已冻结自适应参数的前提下，建立一套可复现的基础对照协议，回答三个不同问题：

1. 相比原始固定 stride=1 工程路径，自适应方案的精度—效率变化是多少；
2. 在使用相同 span-15 解码器时，收益有多少来自采样、多少来自解码；
3. 在相近计算预算下，关键点运动自适应是否优于普通均匀降采样。

本计划只针对 Phoenix-2014T 连续识别 WER。CSL-Daily Top-800 R1 是 isolated reliability 诊断，不进入本实验矩阵。

## 2. 冻结项与非目标

以下内容在执行前冻结：

- 模型结构：SLRT Online Two-Stream S3D；
- 输入：RGB + HRNet WholeBody keypoints；
- checkpoint：`Online/CSLR/results/phoenix-2014t_ISLR/ckpts/best.ckpt`，历史记录 epoch 92；
- 自适应参数：stride 1--3、EMA 0.4、分位数 0.2/0.7、history 48、warmup 16、关键点阈值 0.2、最少有效点 4；
- 自适应解码：triangular span-15，minimum weight 0.05；
- 参数选择只使用 dev；执行本矩阵后不根据 test 修改参数。

本轮不重新训练 ISLR，不实现 R2，不调整 span、运动阈值或 EMA，也不声称改善 WER。目标是消除基础对照缺口并量化归因。

## 3. Baseline 定义

### 3.1 最小必做矩阵

| ID | 采样策略 | 解码 | 作用 | 是否已有完整结果 |
|---|---|---|---|---|
| B0 | fixed stride=1 | window-greedy-7 | 原始工程兼容基线 | 是 |
| B1 | fixed stride=1 | span-weighted-15 | 匹配解码器，隔离解码影响 | 否 |
| B2 | uniform-rate matched | span-weighted-15 | 匹配计算预算，隔离自适应选择价值 | 否 |
| B3 | fixed stride=2 | span-weighted-15 | 常规均匀降采样 | 否 |
| B4 | fixed stride=3 | span-weighted-15 | 激进均匀降采样 | 否 |
| A0 | adaptive stride=1--3 | span-weighted-15 | 当前冻结方案 | 是，runtime 需重复测量 |

B0 与 B1 应复用同一次 fixed-stride=1 前向，只改变保存 logits 后的解码。B1 是公平解码对照；B2 是判断“自适应是否优于相同预算普通采样”的主对照。

### 3.2 B2 的确定性定义

B2 不使用标签、模型置信度或未来关键点。先在 dev 用冻结 A0 的窗口预算计算目标平均步长：

```text
target_mean_stride = fixed_dev_clips / adaptive_dev_clips
                   = 55,775 / 37,615
                   = 1.4827...
```

对每个视频使用累积取整生成均匀起点：

```python
start_n = round(n * target_mean_stride)
```

去重并保证起点严格递增。该常数在读取 test 前冻结，test 不按 A0 的实际窗口数重新匹配。B2 是因果、确定性且不依赖样本标签的 rate-matched baseline。要求其实际 clip 总数与 A0 相差不超过 2%；超出时只报告差异，不根据 test 回调常数。

## 4. 数据集与隔离

| 内容 | Phoenix-2014T train | dev | test |
|---|---|---|---|
| 用途 | 产生既有 ISLR checkpoint | 实现验证、baseline矩阵、冻结报告 | 一次批量最终对照 |
| 连续样本数 | 按现有 metadata 登记 | 519 | 642 |
| 参考 gloss 数 | 执行时登记 | 3,747 | 4,259 |
| 是否允许调参 | 训练阶段历史行为 | 允许，但本轮参数已冻结 | 禁止 |

执行前计算并记录以下文件的 SHA-256、字节数和样本数：

- `phoenix14t.train/dev/test`；
- `phoenix_iso_with_blank.vocab`；
- isolated keypoint PKL；
- 视频 zip；
- checkpoint；
- 基础 YAML 和执行协议 YAML。

Dev 已知缺失帧样本 `11December_2009_Friday_tagesschau-3509` 继续使用现有统一占位帧路径，并在所有结果中登记。由于历史上 test 已运行过 span-13 和 span-15，新增加的 test baseline 必须标为 retrospective control；若用于正式论文的全新主张，应另设未触碰 holdout 或预定义交叉验证。

## 5. 训练设置登记

本轮 `training_required: false`，但必须生成 checkpoint provenance：

| 字段 | 要求 |
|---|---|
| training config | `configs/phoenix-2014t_ISLR.yaml` 的 SHA-256 |
| checkpoint | 路径、SHA-256、字节数、epoch、global step |
| model | Two-Stream S3D，block数、fusion、visual head |
| initialization | 两路 S3D K400 预训练资产哈希 |
| data | train metadata/vocab/keypoint资产哈希 |
| optimizer | Adam、学习率、weight decay、scheduler |
| seed | 321 |
| environment | Python、PyTorch、CUDA、cuDNN、GPU型号 |

如果无法证明当前 checkpoint 来自记录的 training config，应写成“checkpoint provenance partially verified”，不能推断未保存在 checkpoint/日志中的训练细节。

## 6. 统一推理设置

所有变体只允许改变采样器和表中指定的解码器：

| 字段 | 固定值 |
|---|---|
| base config | `configs/slide_phoenix-2014t.yaml` |
| checkpoint | 同一 SHA-256 的 `best.ckpt` |
| prediction source | ensemble |
| clip window | 16 frames |
| dataset batch | 1 video |
| inference split size | 16 clips |
| probability threshold | -1 |
| blank threshold | 0.5 |
| span kernel / weight | triangular / 0.05 |
| random seed | 321 |
| save features | false |

配置文件仍默认关闭自适应。每个运行的最终生效配置必须展开保存到 `resolved_config.yaml`，不得只保存“基础 YAML + 未记录 CLI”。

## 7. 需要实现的代码

### 7.1 实验协议配置

新增：

`Online/CSLR/configs/experiments/phoenix_adaptive_baselines_v1.yaml`

该文件不替代模型配置，而是引用基础配置并声明：数据隔离、checkpoint、共同推理字段、B0--B4/A0 变体、runtime重复次数、结果目录和允许的 split。

### 7.2 统一采样接口

新增 `Online/CSLR/utils/window_sampling.py`，将起点生成统一为：

```python
generate_window_starts(mode, total_frames, keypoints, config)
```

支持：

- `fixed`：stride 1/2/3；
- `uniform_rate`：冻结的 fractional mean stride；
- `adaptive_motion`：调用现有 `adaptive_window_starts`。

`prediction_slide.py` 新增非破坏性 CLI：

```text
--sampling_mode fixed|uniform_rate|adaptive_motion
--fixed_stride N
--uniform_mean_stride X
```

旧 `--adaptive_stride` 参数继续兼容，并映射到新接口。所有模式保存相同结构的 `adaptive_stride_metadata`，另增加 `sampling_mode` 字段。

### 7.3 实验矩阵执行脚本

新增：

`Online/CSLR/tools/run_adaptive_baseline_matrix.py`

职责：

1. 读取实验协议并校验所有冻结字段；
2. 默认只允许 dev；test 必须同时传入 `--allow-test` 和冻结 manifest hash；
3. 使用 GPU UUID 单卡串行执行，拒绝 PCI `01:00.0`、`25:00.0` 对应 UUID；
4. 为每个变体保存 resolved config、完整命令、Git commit、环境和资产哈希；
5. 捕获退出码、起止时间和 `/usr/bin/time` 数据；
6. 不覆盖已有完整运行，除非显式指定新的 run ID。

### 7.4 统一评估脚本

新增：

`Online/CSLR/tools/evaluate_adaptive_baseline_matrix.py`

只读读取各变体的 PKL/JSON，执行：

- 验证样本名、参考文本、checkpoint hash和数据hash完全一致；
- 统一调用 Phoenix cleanup 与 `wer_list`；
- 输出 WER、DEL/INS/SUB、reference length、error count；
- 汇总 clips、stride分布、平均/P50/P95 clips；
- 计算相对 B0/B1/B2 的 clip reduction、WER pp和speedup；
- 对逐句错误做固定 seed=321 的 paired bootstrap 置信区间；
- 输出 `summary.json`、`summary.csv` 和 `summary.md`。

测试集评估脚本不能搜索 span或阈值，只读取协议中冻结的 decoder。

### 7.5 Runtime benchmark

正确性运行和性能测量分开登记，但使用同一 resolved config。每个变体：

1. 先做一个不计入统计的 smoke/warmup；
2. 至少重复3次完整 dev 推理；
3. 在同一健康GPU、同一CPU worker数、无并发任务条件下交替执行各变体；
4. 记录完整命令 wall time、模型前向累计时间、峰值显存；
5. 报告 mean/std/median；若 wall-time CV > 5%，增加到5次；
6. 额外报告 clips/s，不把完整命令时间称为在线P95延迟。

若要测App延迟，应另建会话级流式benchmark，不与本轮服务器离线wall time混写。

## 8. 输出目录

```text
Online/CSLR/results/phoenix-2014t_ISLR/baseline_matrix_v1/
├── protocol_manifest.json
├── dev/
│   ├── B0_fixed1_window7/
│   ├── B1_fixed1_span15/
│   ├── B2_uniform_rate_span15/
│   ├── B3_fixed2_span15/
│   ├── B4_fixed3_span15/
│   └── A0_adaptive_span15/
├── test/                       # 冻结后才创建
└── aggregate/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

PKL、logits和runtime原始记录留在忽略目录。最终把去除机器绝对路径的摘要复制到 `docs/results/phoenix_adaptive_baselines_v1.json`，实验审计写入新的日期日志。

## 9. 执行顺序

### Phase 0：无GPU审计

- 实现 manifest 生成和资产哈希；
- 冻结协议 YAML；
- 确认 B0--B4/A0 的唯一变化字段；
- 增加采样和评估单元测试。

### Phase 1：CPU/小样本验证

- 用合成序列验证 fixed/uniform/adaptive 起点；
- 使用同一保存 logits 验证 B0/B1 仅解码不同；
- 对每个变体运行最多5个 dev 样本，不用于报告精度。

### Phase 2：完整 dev 正确性矩阵

- 串行运行全部变体；
- 首先要求 B0 复现历史 dev WER 22.2311%、55,775 clips；
- B0 不一致时停止，不运行后续正式矩阵；
- 生成 dev aggregate，不修改 A0 参数。

### Phase 3：Dev runtime重复测量

- 同卡交替完成3--5次；
- 锁定最终manifest与结果汇总脚本版本；
- 根据结果只判断结论，不调A0。

### Phase 4：Test处理

- 若只做仓库工程验证，可以不新增test运行，保持现有P2最终结果；
- 若必须补齐论文对照，一次批量运行全部冻结baseline，标注retrospective；
- 运行后禁止改变采样、decoder或报告选择规则。

## 10. 验收与决策规则

实施完成必须满足：

- B0 的样本数、参考长度、clips和WER与历史值完全一致或给出可定位差异；
- 所有变体使用相同样本顺序、参考文本、checkpoint和基础配置hash；
- B1 与B0复用相同fixed-stride=1 logits；
- B2 与A0 clips差异不超过2%，否则只作为近似预算对照；
- 所有指标可由统一评估脚本从原始产物重新生成；
- runtime 至少3次、报告离散程度，并避开两张故障卡；
- test 不参与参数选择。

算法结论按以下规则：

1. A0 vs B1：说明减少窗口的总精度代价；
2. A0 vs B2：在相近预算下，如果 A0 WER 更低，才支持“运动自适应优于均匀采样”；
3. A0 vs B3/B4：判断自适应是否位于固定stride的精度—效率Pareto前沿；
4. 若A0不优于B2，则当前收益主要来自减少窗口，不能归因于自适应决策；
5. 无论结果如何，保留 fixed stride=1 回退路径。

## 11. 预计计算量

Dev 一次完整运行历史约18--24分钟。B0/B1共享前向后，正确性矩阵约需5组前向；3次runtime重复约需15组前向，预计单卡约5--7小时。Test若执行，同量级约6--8小时。实际运行前重新确认健康GPU和节点负载，不使用逻辑编号选择卡。
