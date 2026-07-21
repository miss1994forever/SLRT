# Reliability- and Boundary-Aware Online CSLR 实验入口

## 目标

基于 `2026-05-29/future_research_directions.md` 的组合方向，在保留现有 ISLR、滑窗和自适应步长回退路径的前提下，引入：

1. 逐窗口可靠性感知的 RGB/keypoint/fuse 融合；
2. signness、中心偏移和持续时间预测；
3. 因果在线的事件合并、跳过和下一步 stride 控制；
4. 最后再评估与现有自适应步长的组合收益。

## 冻结基线

- Git tag：`adaptive-stride-v1`；
- 自适应实现保持在 `Online/CSLR/utils/adaptive_stride.py`；
- Phoenix 冻结配置：min/max stride 1/3、span 15、split size 16；
- 默认功能开关继续保持关闭；
- 历史 test 结果只用于最终报告，不用于后续选择参数。

## 代码边界

- `adaptive_stride.py`：仅负责模型前、基于关键点的因果窗口采样；
- `reliability_fusion.py`：负责模型输出后的逐窗口流可靠性估计与融合；
- `boundary_head.py`：负责可训练的 signness/center/duration 输出；
- `boundary_decoder.py`：负责因果事件合并、抑制和下一窗口控制；
- 新功能使用独立配置节和开关，保证固定 stride 与 adaptive-stride-v1 都可复现。

## 实验顺序

### R0：冻结基线复核

只核对已有结果、配置和产物完整性，不重新用 test 调参。

### R1：零训练可靠性诊断

只读取 dev 保存的 RGB/keypoint/fuse logits，统计熵、最大概率、blank 概率、流间分歧、关键点质量与正确率的关系。先验证可靠性信号是否有效，不改训练。

### R2：启发式可靠性融合

在 dev 上预注册少量融合候选，与 equal ensemble 和固定权重对照。指标包括 WER、DEL/INS/SUB、重复率、blank 率和额外解码耗时。

### B1：边界标签审计

分析 `start/end/base_start/base_end/temp_idx` 的覆盖率、噪声和持续时间分布，先确定监督目标是否可信。

### B2：轻量边界头

增加 signness、center offset、duration heads；先单独验证边界指标和 WER，不立即让它控制 stride。

### C1：因果组合

当前窗口先由关键点运动选择，模型前向后再用可靠性与边界结果决定输出、合并以及下一窗口 stride。禁止使用未来窗口信息。

## 数据隔离

- train：训练模型和估计训练语料统计；
- dev：候选选择、阈值、权重和消融；
- test：配置完全冻结后只运行一次最终评估；
- 已查看过的 Phoenix test 不再用于该方向的参数选择；若主实验转向 Top-800 CSL-Daily，仍须在首次 test 前预注册选择规则。

## 第一项执行任务

先实现 R1 的只读 dev 诊断工具和结果文档，不修改模型权重，也不把可靠性逻辑直接写入 `adaptive_stride.py`。R1 证明信号有效后，再决定 R2 的最小候选集合。
