# Phoenix 自适应步长基线冻结审计

日期：2026-09-10

## 结论

先前的自适应步长实现已足以唯一确定为 **A0 v1**，本次将其修复后 dev 身份冻结到 `docs/ADAPTIVE_BASELINE_V1.md`，并增加机器可读索引 `docs/results/phoenix_adaptive_baseline_v1_frozen.json`。本次未运行 GPU、未读取或运行 test、未调参、未训练模型，也未移动或删除大结果。

冻结依据是视频 hash `49faacc3...d457`、checkpoint hash `b3390f0d...1767b`、协议 hash `3d583b14...f65` 和修复后 manifest hash `2e6889ec...4b8f`。当前 `prediction_slide.py`、`utils/adaptive_stride.py`、`utils/window_sampling.py`、矩阵运行/评估脚本以及两个核心 YAML 的 SHA-256 均与 manifest 记录一致。

## 已确定

- 数据：Phoenix-2014T dev，519 样本、3,747 个清洗后参考 gloss，修复后视频 ZIP；
- 模型：冻结的 epoch-92 Two-Stream S3D checkpoint，RGB + HRNet WholeBody keypoints；
- A0：因果关键点运动调度，stride 1--3，EMA 0.4，分位数 0.2/0.7，历史 48、warmup 16，关键点阈值 0.2、最少 4 点；
- 解码：triangular span-15，minimum weight 0.05；
- 对照：B0/B1/B2/B3/B4/A0 的采样、解码器和预算关系；
- 修复后完整 dev WER/DEL/INS/SUB/clips；
- B0/B2/A0 同卡交替三重复 runtime 及 CV；
- A0 与 B0/B1/B2 的 paired bootstrap CI。

YAML 中 `adaptive_stride.enabled: false` 和 span voting 默认关闭，是旧命令兼容策略；矩阵 runner 的 resolved config 对 A0 显式启用对应项。协议、resolved config 与实现值一致，不属于冲突。

## 修正的口径不一致

1. `docs/RESULTS.md` 与旧 `phoenix_adaptive_stride_summary.json` 的 dev runtime 来自修复前单次历史运行；当前权威 runtime 改为修复后三重复中位数 B0 `1642.57 s`、B2 `1133.10 s`、A0 `1158.29 s`。
2. 旧文档把历史 test 称作唯一正式最终评估。该 test 绑定修复前资产；即使 test split 没有缺帧，也不能改称修复后 test，现统一标为 historical retrospective control。
3. `docs/RESULTS.md` 仍称下一步要做三重复 runtime，但该实验已于 2026-09-08 完成，现已更新。
4. B1 复用 B0 前向，不能单独报告运行耗时；B3/B4 只有单次正确性运行时间，不能与 B0/B2/A0 的三重复统计并列成论文级 runtime。
5. 请求中提及的 `Online/ADAPTIVE_STRIDE_IMPLEMENTATION_PLAN.md` 当前仓库路径不存在；其历史设计内容不作为冻结身份依据。冻结以可执行协议、manifest、resolved config、源码 hash 和机器结果为准。

## 尚未确定

- 下一代 reliability/boundary-aware 调度信号、公式、参数与预算层级；
- oracle、边界命中、随机采样等诊断实验；
- 真正流式端到端延迟和 iOS 设备性能；
- 跨数据集泛化；
- 修复后 retrospective test（尚未运行）。

后续实验必须从 A0 继承，并至少保留 B0、B1 和等预算 B2。若改变 checkpoint、数据 hash、窗口、解码、评估清洗或冻结参数，应新建版本和 manifest，不能覆盖 v1。

## 验证

对修复后正确性与 runtime aggregate 做了 CPU 只读字段核对；核心文件 hash 与 manifest 一致。JSON 可解析，文档引用目标存在，`git diff --check` 通过。本次保留了工作树中既有的归档、日志与文档修改，没有提交或推送。
