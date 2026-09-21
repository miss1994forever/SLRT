# Online Sign Language Recognition for iOS

本仓库基于 [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT)，用于实时手语视频到文本的
iOS 应用研究。当前开发集中在 `Online/CSLR`：Two-Stream S3D 滑窗连续识别、等预算时间
调度，以及 decoder-aware 窗口选择的可部署性研究。

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
| P3 | robust decoder-utility oracle 与可部署 predictor | 完整 train oracle 上界强，但 skeleton、三类 RGB、decoder-conditioned value 和信息阶梯均未达到 OOF gate；停止当前 per-window utility target |

P0 的 sign-center 结果是 alignment-derived offline oracle，不是可部署模型。P1/P2 均未通过
center gate。P3 在完整 train fit 上确认 robust continuation oracle 可减少 637 个 errors，但
366,802 个候选行中只有 769 个 beneficial；冻结的 skeleton、RGB-DCT、finger RGB、face RGB、
decoder-conditioned value 和逐级信息增强都未接近 249-error gate。当前结论不是“视觉无用”，
而是现有 future/reference/EOS-aware 的逐窗口反事实 utility target 与部署时可见信息耦合过弱。
因此不运行闭环、不在同一 OOF 上追调参，下一阶段先重新设计更稳定且可观测的 target/决策分解。

## 从这里开始

- [文档导航](docs/README.md)：区分当前入口、冻结结果、历史设计和机器索引；
- [下一阶段 handoff](docs/NEXT_RESEARCH_HANDOFF.md)：交给新对话的唯一当前任务入口；
- [统一结果口径](docs/RESULTS.md)：跨阶段数字与允许声明；
- [完整实验账本](docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md)：P0--P3 的事实、风险和机器产物索引；
- [环境与复现](docs/REPRODUCIBILITY.md)：数据、checkpoint、GPU 和复现命令；
- [仓库结构规范](docs/REPOSITORY_LAYOUT.md)：文件应该放在哪里；
- [实验日志索引](code_agent_logs/README.md)：按日期保存的实施与审计链。

阶段冻结文档：

- [A0 自适应步长 v1](docs/ADAPTIVE_BASELINE_V1.md)
- [P0 调度诊断 v1](docs/P0_SCHEDULE_DIAGNOSTICS_V1.md)
- [P1 因果 center predictor v1](docs/P1_CAUSAL_CENTER_PREDICTOR_V1.md)
- [P2 因果 center TCN v1](docs/P2_CAUSAL_CENTER_TCN_V1.md)
- [P3 完整实验账本](docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md)

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

所有 `nvidia-smi`、CUDA 检查和 GPU 实验都必须在沙盒外运行；沙盒内 CUDA 不可见不能判定
驱动故障。必须按 PCI/UUID 核验并显式绑定，避开 PCI `25:00.0`、`41:00.0` 和宿主机 GPU0
（当前为 `01:00.0`），不能依赖会变化的逻辑 GPU 编号。

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
