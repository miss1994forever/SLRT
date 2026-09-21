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

当前完整视频 ZIP 的 SHA-256 必须为 `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`。旧 hash `81629b2f...07e30` 缺少 metadata 声明的 61 帧，只用于历史结果审计；对应结果索引位于 `Online/CSLR/results/_archive/phoenix_pre_repair_81629b2f/`。

Top-800 R1 需要：

- `data/csl-daily/` 视频帧资产和 isolated keypoints；
- `data/csl-daily-top-800-all/` 的 split、vocab 和 gloss 映射；
- `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt`。

## 3. GPU 安全约束

所有 `nvidia-smi`、CUDA 检查和 GPU 实验必须在沙盒外运行；沙盒内 CUDA 不可见不能用来判定
驱动故障。不要使用宿主机 CUDA 逻辑编号选择卡。先用
`nvidia-smi --query-gpu=pci.bus_id,uuid,...` 核对 PCI/UUID，再传入明确的 UUID。

已知禁用卡：

| PCI | UUID | 状态 |
|---|---|---|
| `01:00.0` | `GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a` | 历史故障，禁用 |
| `25:00.0` | `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83` | 历史多次故障，禁用 |
| `41:00.0` | `GPU-2c6a50cb-f770-c785-dc1d-7f8cc7b7b9aa` | 当前研究协议禁用 |

宿主机 GPU0（当前为 PCI `01:00.0`）同样禁止。复现脚本应使用 UUID 白名单，并在启动前拒绝
上述 PCI/UUID；进程内的 `cuda:0` 只允许表示经过 `CUDA_VISIBLE_DEVICES` 隔离后的首张安全卡。

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

修复后 dev 冻结结果（runtime 为同卡三重复中位数）：

| 划分 | 固定 stride=1 WER | 自适应 WER | clips 变化 | command wall time 变化 |
|---|---:|---:|---:|---:|
| dev | 22.2311% | 22.4179% | -32.56% | -29.48% |
| historical test（修复前资产） | 22.0005% | 23.0571% | -32.02% | -26.25% |

完整冻结身份、参数、对照和机器数据位置见 [ADAPTIVE_BASELINE_V1.md](ADAPTIVE_BASELINE_V1.md)，历史叙事见 [RESULTS.md](RESULTS.md)。旧 test 结果不能用于回调 span 或阈值，也不能视为修复后资产的新 test。

### 5.1 注册基础对照矩阵（v1）

新增基础对照使用 `configs/experiments/phoenix_adaptive_baselines_v1.yaml`，固定同一 Two-Stream S3D checkpoint，并注册 B0/B1/B2/B3/B4/A0。它不重新训练模型。B1 复用 B0 前向；B2 是与 A0 近似等窗口预算的确定性 uniform-rate 对照。

先运行无 CUDA 测试和命令生成检查：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v

python tools/run_adaptive_baseline_matrix.py \
  --split dev \
  --variants B0_fixed1_window7 B2_uniform_rate_span15 A0_adaptive_span15 \
  --max-samples 5 \
  --output-root /tmp/phoenix_baseline_matrix_smoke \
  --dry-run --skip-asset-hashes
```

正式 dev 前必须提交代码，使 manifest 记录干净 commit。首次命令生成所有注册资产的 hash，并将解析后的配置、完整命令和结果写入独立 run 目录：

```bash
CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
python tools/run_adaptive_baseline_matrix.py \
  --split dev --gpu-uuid GPU-<healthy-uuid> --repetitions 1

python tools/evaluate_adaptive_baseline_matrix.py --split dev
```

性能重复测量已按同一协议完成；复跑时仍应使用新的输出根目录执行三次，避免与正确性 run 混写：

```bash
CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
python tools/run_adaptive_baseline_matrix.py \
  --split dev --gpu-uuid GPU-<healthy-uuid> --repetitions 3 \
  --output-root results/phoenix-2014t_ISLR/baseline_matrix_v1_runtime
```

运行器拒绝脏工作树、缺失资产和已登记故障卡。`prediction_slide.py` 同时记录完整命令 wall time、模型前向累计时间与 PyTorch 峰值显存。统一评估器重新计算 WER，核对样本/参考、resolved config、decoder、预算约束，并输出 JSON/CSV/Markdown。

test 只能使用 dev 阶段冻结的 manifest hash：

```bash
MANIFEST_SHA256=<sha256-of-protocol_manifest.json>
CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
python tools/run_adaptive_baseline_matrix.py \
  --split test --gpu-uuid GPU-<healthy-uuid> \
  --allow-test --frozen-manifest-sha256 "$MANIFEST_SHA256"

python tools/evaluate_adaptive_baseline_matrix.py \
  --split test --allow-test --frozen-manifest-sha256 "$MANIFEST_SHA256"
```

由于本项目历史上已经查看过 Phoenix test，该输出必须标为 retrospective control。完整设计和验收规则见 [基础对照实施计划](experiments/adaptive_stride/baseline_evaluation_plan.md)。

## 6. CSL-Daily Top-800 R1

R1 是 isolated dev 的零训练可靠性诊断，不是自适应步长 WER 实验：

```bash
CUDA_VISIBLE_DEVICES=GPU-<healthy-uuid> \
  bash "$SLRT_ROOT/scripts/reproduce/reliability_r1_dev.sh"
```

它固定读取 dev，不提供 test 选择。正式运行共 13,077 个样本，主要结果为 RGB 52.48%、Keypoint 66.28%、Fuse 62.11%。完整指标见 `code_agent_logs/2026-07-26/reliability_r1_results.md`。

## 7. 数据隔离和结果登记

每次正式实验记录：commit、配置、checkpoint、数据划分、GPU UUID、命令、样本数、耗时和指标。train 用于训练或统计，dev 用于候选选择，test 仅在配置冻结后运行一次。机器结果留在 `results/`，摘要写入新的日期日志。
