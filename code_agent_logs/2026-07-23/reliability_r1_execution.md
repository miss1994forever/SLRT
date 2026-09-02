# R1 零训练可靠性诊断执行记录

## 状态

R1 已于 2026-07-26 在完整 13,077 样本 isolated dev 上完成。此前记录的 CUDA 阻塞已解除；正式运行只绑定 PCI `81:00.0` 对应的 UUID `GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631`，没有使用已知故障卡 PCI `01:00.0` 和 `25:00.0`。详细结果见 `../2026-07-26/reliability_r1_results.md`。

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

## 正式运行命令

命令从 `Online/CSLR` 目录执行。必须使用 UUID 绑定，以免 CUDA 逻辑编号变化导致误用故障卡：

```bash
CUDA_VISIBLE_DEVICES=GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631 \
/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  tools/reliability_dev_diagnostic.py \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 2 \
  --output-dir results/csl-daily-top-800_ISLR_full_stable/diagnostics/reliability_r1_dev
```

完整运行处理 3,270 个 batch、13,077 个样本，耗时 936.42 秒（15 分 36 秒），无 CUDA 或 OOM 错误。运行期间显存约 3,981 MiB，温度 51--53°C，利用率约 66--78%。没有训练、没有更新 checkpoint，也没有读取 test。

注意：首次从仓库根目录启动时，旧版 `two_stream.py` 的相对预训练模型路径解析失败，尚未进入 GPU 前向。切换到 `Online/CSLR` 后按上面的同一配置重新运行成功；该问题不影响实验结果。
