# 2026-07-16 Phoenix 自适应步长完整对照测试

## 测试结论

Phoenix-2014T test 全部 642 个样本的真实模型对照测试已完成。自适应步长配合真实时间跨度投票，在 WER 增加 0.73 个百分点的情况下，将 clip 数减少 32.02%，端到端运行时间减少 28.84%，等效吞吐提升约 1.41 倍。

## GPU 安全约束

- 明确排除故障卡：PCI `25:00.0`，UUID `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83`；
- 两组均只使用：PCI `61:00.0`，UUID `GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed`；
- 通过 UUID 设置 `CUDA_VISIBLE_DEVICES`，没有依赖可能变化的逻辑编号；
- 固定组和自适应组串行运行，避免 GPU 竞争干扰耗时；
- 两组均完成 642/642，包括旧实验发生 CUDA 故障的 574 附近；
- 运行结束后 PCI `61:00.0` 温度 24°C、显存 0 MiB、利用率 0%。

## 公共配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | Phoenix-2014T test |
| 样本数 | 642 |
| checkpoint | `results/phoenix-2014t_ISLR/ckpts/best.ckpt` |
| checkpoint epoch | 92 |
| 模型输入 | RGB + HRNet keypoints |
| `pred_src` | ensemble |
| clip 窗口长度 | 16 帧 |
| `prob_thr` | -1 |
| 固定基础 stride | 1 |

自适应组配置：

```yaml
adaptive_stride:
  enabled: true
  min_stride: 1
  max_stride: 3
  keypoint_confidence_threshold: 0.2
  ema_decay: 0.4
  quantile_low: 0.2
  quantile_high: 0.7
  calibration_window_frames: 48
  warmup_frames: 16
  min_valid_keypoints: 4

span_weighted_voting:
  enabled: true
  vote_span_frames: 13
  min_weight: 0.05
```

## 核心对照结果

| 指标 | 固定 stride=1 | 自适应 + span-weighted-13 | 变化 |
| --- | ---: | ---: | ---: |
| 样本数 | 642 | 642 | — |
| 总 clip 数 | 64,627 | 43,932 | -20,695 |
| clip 减少比例 | — | — | **32.02%** |
| 平均 clip/样本 | 100.67 | 68.43 | -32.24 |
| 最少 clip/样本 | 17 | 17 | 0 |
| 最多 clip/样本 | 242 | 161 | -81 |
| 最佳 WER | **22.00** | **22.73** | **+0.73** |
| 最佳解码 | `window_greedy_7` | `span_weighted_13` | — |
| DEL | 9.25 | 9.20 | -0.05 |
| INS | 3.50 | 4.27 | +0.77 |
| SUB | 9.25 | 9.25 | 0.00 |
| 真实运行时间 | 1547.29 秒 | 1101.12 秒 | **-28.84%** |
| 运行时间 | 25分47秒 | 18分21秒 | -7分26秒 |
| 等效吞吐提升 | 1.00× | **1.41×** | +40.52% |

WER 的精确值为固定 22.0005、自适应 22.7283，差值 0.7278 个百分点。相对 WER 增幅约 3.31%。

## 动态步长分布

| 步长 | 使用次数 | 占自适应 clip 比例 |
| --- | ---: | ---: |
| stride=1 | 30,110 | 68.54% |
| stride=2 | 6,505 | 14.81% |
| stride=3 | 7,317 | 16.66% |
| 合计 | 43,932 | 100% |

相对于固定窗口总数，自适应方案的等效平均步长约为 `64,627 / 43,932 = 1.471`。

## 所有解码方法 WER

| 解码方法 | 固定 stride=1 | 自适应 |
| --- | ---: | ---: |
| `window_greedy_3` | 29.09 | 24.86 |
| `window_greedy_5` | 23.15 | 25.38 |
| `window_greedy_7` | **22.00** | 32.71 |
| `window_greedy_9` | 23.17 | 41.35 |
| `window_greedy_11` | 25.90 | 50.95 |
| `window_greedy_13` | 31.51 | 59.87 |
| `span_weighted_13` | 未运行 | **22.73** |
| `naive_greedy` | 57.27 | 44.35 |

自适应模式下，普通 `window_greedy_N` 随 N 增大快速恶化，原因是它把非等间隔 clip 当作等间隔索引，导致投票覆盖的真实时间跨度过长、删除错误增加。`span_weighted_13` 使用真实中心帧距离后，WER 从普通 `window_greedy_13` 的 59.87 降至 22.73。这证明时间跨度投票不是可选优化，而是自适应采样正确工作的必要配套。

## 结果产物

固定组：

- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_full642/test/test_results.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_full642/test/test_evaluation_results.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_full642/test/test_logits.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_full642.log`

自适应组：

- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_full642/test/test_results.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_full642/test/test_evaluation_results.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_full642/test/test_logits.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_full642.log`

## 判断与建议

1. 新方案已经达到明确的效率收益：clip -32.02%，实际耗时 -28.84%。
2. WER 代价为 +0.73 个百分点，明显小于旧成果记录中约 +2.32 个百分点的差距。
3. 是否作为生产默认值取决于 App 对精度与延迟的权重；目前适合进入灰度测试，不建议立即删除固定 stride=1 回退路径。
4. 下一轮参数搜索应只使用 dev 集，不再根据 test 集调整参数。可以重点比较 `max_stride=2`、不同 vote span，以及更保守的分位数，以尝试将 WER 差值压到 0.5 以内。
5. 当前结果说明没有必要重新训练模型；进一步收益仍可优先从推理参数和增量缓存获得。
