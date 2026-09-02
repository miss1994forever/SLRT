# 实验复现说明

## 1. 环境和工作目录

原始项目依赖见 `Online/requirements.txt`。本服务器历史实验使用 `slrt_legacy` 环境。所有 Online CSLR 命令从组件目录运行：

```bash
export SLRT_ROOT="$(git rev-parse --show-toplevel)"
cd "$SLRT_ROOT/Online/CSLR"
export OMP_NUM_THREADS=1
```

推荐配置中的相对路径均以该目录为基准。

## 2. 本地资产布局

以下目录被 Git 忽略，需要自行准备：

```text
data/
├── phoenix_2014t/
├── csl-daily/
└── csl-daily-top-800-all/
pretrained_models/s3ds_actioncls_ckpt/
artifacts/checkpoints/online_slrt/cslr_best.ckpt
Online/CSLR/results/<experiment>/ckpts/best.ckpt
```

Phoenix 自适应步长复现需要：

- `data/phoenix_2014t/` 的 train/dev/test metadata、视频压缩包和 HRNet WholeBody isolated keypoints；
- `Online/CSLR/results/phoenix-2014t_ISLR/ckpts/best.ckpt`（历史记录为 epoch 92）。

Top-800 R1 需要：

- `data/csl-daily/` 视频帧资产和 isolated keypoints；
- `data/csl-daily-top-800-all/` 的 split、vocab 和 gloss 映射；
- `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt`。

## 3. GPU 安全约束

不要使用 CUDA 逻辑编号选择卡。先由管理员或 `nvidia-smi --query-gpu=pci.bus_id,uuid,...` 确认健康状态，再传入单卡 UUID。

已知禁用卡：

| PCI | UUID | 状态 |
|---|---|---|
| `01:00.0` | `GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a` | 历史故障，禁用 |
| `25:00.0` | `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83` | 历史多次故障，禁用 |

复现脚本要求 `CUDA_VISIBLE_DEVICES` 是单个 UUID，并拒绝上述 UUID。

## 4. 单元测试

```bash
cd Online/CSLR
python -m unittest discover -s tests -p 'test_*.py' -v
```

预期覆盖自适应采样、低质量回退、真实时间跨度投票和 R1 可靠性统计。

## 5. Phoenix-2014T 自适应步长

参数只在 Phoenix-2014T dev 选择：窗口 16 帧、stride 1--3、EMA 0.4、历史 48 帧、warmup 16 帧、分位数 0.2/0.7、triangular span 15。

先在 dev 对照：

```bash
CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
  bash "$SLRT_ROOT/scripts/reproduce/phoenix_adaptive_stride.sh" fixed dev

CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
  bash "$SLRT_ROOT/scripts/reproduce/phoenix_adaptive_stride.sh" adaptive dev
```

脚本默认不允许 test。只有确认配置已冻结时才运行：

```bash
ALLOW_TEST=1 CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
  bash "$SLRT_ROOT/scripts/reproduce/phoenix_adaptive_stride.sh" adaptive test
```

冻结结果：

| 划分 | 固定 stride=1 WER | 自适应 WER | clips 变化 |
|---|---:|---:|---:|
| dev | 22.23% | 22.42% | -32.56% |
| test | 22.0005% | 23.0571% | -32.02% |

完整审计见 `code_agent_logs/2026-07-16/adaptive_stride_dev_tuning.md` 和 `adaptive_stride_frozen_final_evaluation.md`。test 结果不能用于回调 span 或阈值。

## 6. CSL-Daily Top-800 R1

R1 是 isolated dev 的零训练可靠性诊断，不是自适应步长 WER 实验：

```bash
CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
  bash "$SLRT_ROOT/scripts/reproduce/reliability_r1_dev.sh"
```

它固定读取 dev，不提供 test 选择。正式运行共 13,077 个样本，主要结果为 RGB 52.48%、Keypoint 66.28%、Fuse 62.11%。完整指标见 `code_agent_logs/2026-07-26/reliability_r1_results.md`。

## 7. 数据隔离和结果登记

每次正式实验记录：commit、配置、checkpoint、数据划分、GPU UUID、命令、样本数、耗时和指标。train 用于训练或统计，dev 用于候选选择，test 仅在配置冻结后运行一次。机器结果留在 `results/`，摘要写入新的日期日志。
