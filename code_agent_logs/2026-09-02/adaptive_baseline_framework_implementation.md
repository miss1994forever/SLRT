# Phoenix 基础对照框架实施记录

日期：2026-09-02

## 实施范围

本次完成 `docs/experiments/adaptive_stride/baseline_evaluation_plan.md` 的 Phase 0 和代码层 Phase 1，没有启动 CUDA，也没有新增读取 test 结果。完整 dev GPU 矩阵和 runtime 重复测量留待代码提交、健康卡确认后执行。

## 代码与协议

- 新增 `utils/window_sampling.py`：统一 `fixed`、`uniform_rate`、`adaptive_motion` 三类确定性窗口起点和 metadata；旧 `adaptive_stride` 配置继续兼容。
- 修改 `prediction_slide.py`：接入统一采样器，增加 sampling CLI 和精确输出目录；可保存模型前向累计时间、调用次数、clip 数及 PyTorch 峰值显存。
- 新增 `configs/experiments/phoenix_adaptive_baselines_v1.yaml`：冻结 Two-Stream S3D checkpoint、数据、共同推理配置以及 B0/B1/B2/B3/B4/A0 矩阵。
- 新增 `tools/run_adaptive_baseline_matrix.py`：生成资产/代码 hash manifest、resolved config、完整命令和 runtime；要求单卡 UUID，拒绝两张已知故障卡，正式运行默认拒绝脏工作树；test 要求冻结 manifest hash。
- 新增 `tools/evaluate_adaptive_baseline_matrix.py`：重新计算并校验 WER，检查样本/参考/配置，汇总 clips、stride、runtime、paired bootstrap 和预算约束，输出 JSON/CSV/Markdown。
- 新增采样、协议、评估器单元测试，并更新 Online CSLR README、配置索引和复现说明。

## 离线验证

使用环境：`/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python`

```bash
cd Online/CSLR
python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile prediction_slide.py utils/window_sampling.py \
  tools/run_adaptive_baseline_matrix.py tools/evaluate_adaptive_baseline_matrix.py
```

结果：24 个测试全部通过；4 个修改/新增入口均通过语法编译。

矩阵 dry-run 使用 `/tmp/slrt_baseline_matrix_smoke_20260902`，成功为 B0/B2/A0 生成 manifest、resolved config 和命令，未初始化 CUDA、未执行模型。

## Dev 等预算核对

从历史 fixed dev 产物逐样本读取窗口数作为帧长，使用冻结常数：

```text
uniform_mean_stride = 55,775 / 37,615 = 1.4827861225574903
```

统一采样器得到：

| 策略 | dev clips | 相对 A0 |
|---|---:|---:|
| B2 uniform-rate（预计） | 37,706 | +0.242% |
| A0 adaptive（历史冻结） | 37,615 | 0 |

差异低于协议预注册的 2% 上限。这里只验证窗口预算，没有计算 B2 WER，不能作为新增实验结果。

## 下一执行点

1. 提交本次代码，使正式 manifest 对应干净 commit；
2. 确认 PCI `81:00.0` 对应 UUID 当前健康且空闲；
3. 先用最多 5 个 dev 样本执行真实 GPU smoke；
4. 完整执行 dev 正确性矩阵，首先核对 B0 是否复现 519 样本、3,747 reference gloss、55,775 clips 和 22.2311% WER；
5. B0 通过后执行三次独立 runtime 矩阵并统一聚合；
6. 不根据 test 调参，test 是否补跑另行决定并标注 retrospective control。
