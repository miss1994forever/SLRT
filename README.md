# Online Sign Language Recognition for iOS

本仓库基于 [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT)，用于实时手语视频到文本的
iOS 应用研究。当前开发集中在 `Online/CSLR`：Two-Stream S3D 滑窗连续识别、等预算时间
调度，以及因果 sign-interior 预测。

原始 SLRT 的论文、引用和子项目入口保存在
[`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md)。本项目改动与实验结论不代表原作者的
官方结果。

## 当前状态

Phoenix-2014T 是当前在线 CSLR 主线，指标为 WER。CSL-Daily Top-800 是 isolated reliability
诊断，指标为 accuracy/AUROC；两条数据线不能混用。

| 阶段 | 任务 | 冻结结论 |
|---|---|---|
| A0 | motion-only 自适应步长 | Dev 相对 B0 减少 32.56% clips；未显著优于等预算 uniform |
| P0 | 等预算调度诊断 | boundary/prediction-change No-Go；label-derived sign-center 是 offline Strong-Go |
| P1 | train-only 最小因果 center predictor | Calibration gate 失败，No-Go；未运行 scheduler，没有新 WER |
| P2 | 211 维关键点 + 31 帧因果 TCN | Detection 明显提升但 gate 失败，No-Go；未打开 dev/test |

P0 的 sign-center 结果是 alignment-derived offline oracle，不是可部署模型。P1 的 11 维廉价
特征最多命中 17.83% calibration centers；P2 增强到左右手/手形表征和 causal TCN 后提高到
38.89%，但仍远低于预注册的 75% gate。P2 因此没有打开 dev、没有 scheduler WER，也没有
读取或运行 test-only 文件。

## 从这里开始

- [文档导航](docs/README.md)：区分当前入口、冻结结果、历史设计和机器索引；
- [下一阶段 handoff](docs/NEXT_RESEARCH_HANDOFF.md)：交给新对话的唯一当前任务入口；
- [统一结果口径](docs/RESULTS.md)：跨阶段数字与允许声明；
- [环境与复现](docs/REPRODUCIBILITY.md)：数据、checkpoint、GPU 和复现命令；
- [仓库结构规范](docs/REPOSITORY_LAYOUT.md)：文件应该放在哪里；
- [实验日志索引](code_agent_logs/README.md)：按日期保存的实施与审计链。

阶段冻结文档：

- [A0 自适应步长 v1](docs/ADAPTIVE_BASELINE_V1.md)
- [P0 调度诊断 v1](docs/P0_SCHEDULE_DIAGNOSTICS_V1.md)
- [P1 因果 center predictor v1](docs/P1_CAUSAL_CENTER_PREDICTOR_V1.md)
- [P2 因果 center TCN v1](docs/P2_CAUSAL_CENTER_TCN_V1.md)

## 本地路径

本节点推荐使用真实路径：

```text
/mnt/workspace/projects/haojun/SLRT
```

`/home/haojun/projects` 是指向 `/mnt/workspace/projects/haojun` 的符号链接，因此
`/home/haojun/projects/SLRT` 与上述路径是同一份仓库，不是两个副本。VS Code Remote 应直接
打开 `SLRT` 目录，不要把 `/home/haojun` 或外层 `/mnt/workspace/projects/haojun` 当作项目根，
否则 Explorer 会同时显示用户配置和仓库外的历史脚本。

## 最小验证

从仓库根目录执行：

```bash
cd /mnt/workspace/projects/haojun/SLRT/Online/CSLR
python -m unittest discover -s tests -p 'test_*.py' -v
```

需要 GPU 的实验必须先检查设备健康状态，再用单个 UUID 显式绑定。已知故障卡 PCI
`01:00.0` 与 `25:00.0` 禁止使用，不能依赖会变化的逻辑 GPU 编号。

## 数据与产物边界

- `data/`、视频、拆分关键点 pickle、checkpoint、logits 和大型逐样本结果只保存在本地；
- 阶段冻结时，仅通过 `.gitignore` 精确白名单提交 compact config、manifest、aggregate、
  必要小模型和机器索引；
- train 用于拟合和内部 calibration；dev 只在训练侧方案冻结后评估；当前阶段禁止 test；
- 新实验必须使用新输出目录，不能覆盖 A0、P0、P1 或 P2。

## 目录概览

```text
SLRT/
├── Online/CSLR/       # 当前主代码、配置、工具、测试和本地结果
├── docs/              # 当前 handoff、冻结结论、复现与机器索引
├── scripts/reproduce/ # 稳定复现脚本
├── code_agent_logs/   # 按日期保存的实施和实验审计
├── data/              # 本地数据，默认不进入 Git
└── CiCo、NLA-SLR、Spoken2Sign、TwoStreamNetwork 等上游子项目
```
