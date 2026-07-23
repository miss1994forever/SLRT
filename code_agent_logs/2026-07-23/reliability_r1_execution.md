# R1 零训练可靠性诊断执行记录

## 状态

R1 工具、预注册指标、单元测试和 CPU smoke 已完成；完整 13,077 样本 isolated dev 前向尚未启动，原因是节点 CUDA runtime 被 PCI `C1:00.0` 的 `Unknown Error` 阻塞。该状态不是模型、数据或新工具错误。

## 数据隔离

- 工具固定构建 `dev` dataloader，不提供 test 参数；
- checkpoint 只读，不训练、不更新权重；
- 不修改 `adaptive_stride.py`；
- 每样本只保存可靠性标量和预测 ID，不保存大体积三路 logits；
- 指标在查看完整结果前已写入 `2026-07-21/reliability_boundary_experiment_plan.md`。

## 新增实现

- `Online/CSLR/tools/reliability_dev_diagnostic.py`
- `Online/CSLR/utils/reliability_analysis.py`
- `Online/CSLR/tests/test_reliability_analysis.py`

输出：

- `reliability_records.csv`：三流预测、置信度、margin、归一化熵、blank 概率、JS divergence 和关键点质量；
- `reliability_summary.json`：AUROC、ECE、选择性准确率、流间一致/分歧、oracle/选流上限和关键点质量分箱。

## 验证

- 纯分析与关键点 dtype 回归测试：6/6 通过；
- Python 语法检查：通过；
- CPU smoke：1/1 dev 样本完整通过，三路输出和两个结果文件成功生成；
- CPU 单样本实测约 18.31 秒，推算完整 dev 约 66 小时，因此不采用 CPU 全量运行。

Smoke 产物：

`Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/reliability_r1_dev_cpu_smoke/`

## GPU 阻塞证据

2026-07-23 检查结果：

- 已知故障卡 PCI `25:00.0` 继续排除；
- PCI `C1:00.0` 返回 `Unable to determine the device handle ... Unknown Error`；
- 对空闲 PCI `81:00.0` 和 `A1:00.0` 分别用 UUID 屏蔽后，PyTorch 仍报告 `CUDA initialization: CUDA unknown error`，`torch.cuda.is_available()` 为 false；
- 因此当前节点上的所有 CUDA 进程初始化都不可用，不能通过改逻辑编号解决；
- 未尝试重置 GPU，以免影响正在 `25:00.0` 和 `61:00.0` 上运行的其他任务。

## CUDA 恢复后的唯一正式命令

先确认故障卡恢复或已由管理员隔离，再选择一张空闲健康卡，并用 UUID 运行：

```bash
CUDA_VISIBLE_DEVICES=<HEALTHY_GPU_UUID> \
/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  Online/CSLR/tools/reliability_dev_diagnostic.py \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 2 \
  --output-dir Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/reliability_r1_dev
```

正式运行后不改变预注册指标。完整结果只决定是否进入 R2，不触碰 test。
