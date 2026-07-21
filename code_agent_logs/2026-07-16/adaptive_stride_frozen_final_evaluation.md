# 2026-07-16 自适应步长冻结配置最终评估

## 结论

基于 dev 预注册规则选出的冻结配置已在 Phoenix-2014T test 上完成唯一一次全量最终评估（642/642）。最终 WER 为 **23.0571%**，总 clip 数为 **43,932**，实际运行时间为 **1,141.08 秒**。

与此前固定 stride=1 的工程基线相比，冻结方案减少 **32.02%** clips、减少 **26.25%** 实际时间，但 WER 增加 **1.0566 个百分点**。该 test 精度差值超过 dev 调参时设定的 0.75 个百分点目标，说明 dev 上观察到的精度保持没有完全泛化到 test。

本结果只用于最终报告。看到 test 结果后没有修改参数，也不再进行第二轮 test 调参。

## 冻结配置

| 项目 | 值 |
| --- | --- |
| 数据集 | Phoenix-2014T test |
| 样本 | 642 |
| checkpoint | `results/phoenix-2014t_ISLR/ckpts/best.ckpt`（epoch 92） |
| 输入 | RGB + HRNet keypoints |
| logits | ensemble |
| clip 窗口 | 16 帧 |
| adaptive min/max stride | 1 / 3 |
| EMA | 0.4 |
| motion quantile | 0.2 / 0.7 |
| calibration / warmup | 48 / 16 帧 |
| span voting | triangular，15 帧，min weight 0.05 |
| inference split size | 16 |

实际命令显式传入了 `--adaptive_stride 1 --adaptive_min_stride 1 --adaptive_max_stride 3 --span_weighted_voting 1 --vote_span_frames 15 --split_size 16`，避免依赖开关默认值。

## 最终指标

| 指标 | 冻结 adaptive + span-15 |
| --- | ---: |
| WER | **23.0571%** |
| DEL | 11.1294% |
| INS | 3.4280% |
| SUB | 8.4996% |
| 参考 gloss 数 | 4,259 |
| 编辑错误数 | 982 |
| clips | 43,932 |
| 平均 clips/样本 | 68.43 |
| 最少 / 最多 clips | 17 / 161 |
| 实际时间 | 1,141.08 秒（19分01秒） |

## 与既有固定基线比较

固定基线来自冻结前已经完成的同一 test 工程对照，不是本次调参新增运行。

| 指标 | 固定 stride=1 | 冻结 adaptive span-15 | 变化 |
| --- | ---: | ---: | ---: |
| WER | **22.0005%** | 23.0571% | **+1.0566 pp** |
| clips | 64,627 | 43,932 | **-32.02%** |
| 实际时间 | 1,547.29 秒 | 1,141.08 秒 | **-26.25%** |
| 等效吞吐 | 1.00× | **1.356×** | +35.60% |

相对 WER 增幅为 4.80%。历史 span-13 test WER 为 22.7283%，本次 span-15 高 0.3288 个百分点；该信息只能用于解释最终结果，不能据此回调冻结参数。

## 步长分布

| 步长 | 次数 | 比例 |
| --- | ---: | ---: |
| stride=1 | 30,110 | 68.54% |
| stride=2 | 6,505 | 14.81% |
| stride=3 | 7,317 | 16.66% |
| 合计 | 43,932 | 100% |

## 所有本次解码结果

| 解码方法 | WER |
| --- | ---: |
| window-greedy-3 | 24.89% |
| window-greedy-5 | 25.38% |
| window-greedy-7 | 32.68% |
| window-greedy-9 | 41.35% |
| window-greedy-11 | 50.93% |
| window-greedy-13 | 59.90% |
| **span-weighted-15** | **23.06%** |
| naive-greedy | 44.26% |

普通 window-greedy 仍不适合非等间隔自适应 clips；最终指标应使用真实时间中心的 span-weighted 解码。

## GPU 安全与运行完整性

- 明确排除故障卡 PCI `25:00.0` / UUID `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83`；
- 本次只绑定 PCI `61:00.0` / UUID `GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed`；
- 完成 642/642，无 CUDA、OOM 或数据加载错误；
- 结束后该卡显存 0 MiB、温度 23°C、利用率 0%。

## 结果文件

- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_dev_tuned_span15_final_test/test/test_results.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_dev_tuned_span15_final_test/test/test_evaluation_results.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_dev_tuned_span15_final_test/test/test_logits.pkl`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_dev_tuned_span15_final_test.log`

## 最终判断

冻结方案的效率收益得到重复确认，但 test WER 代价超过预期，因此不建议直接替代固定 stride=1 作为唯一生产路径。严格实验结论应保留当前冻结结果，不根据 test 改参；若继续研究，应重新设计只依赖 train/dev 的稳健选择方法（例如多个预定义 dev 子集或交叉验证），形成新的实验协议后再评估，而不是继续查看同一 test 选择 span。
