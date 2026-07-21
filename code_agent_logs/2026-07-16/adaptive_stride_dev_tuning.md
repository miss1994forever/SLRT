# 2026-07-16 自适应步长 dev 调参记录

## 数据隔离规则

- 本轮只读取 Phoenix-2014T dev 输入和 dev 参考 gloss；
- 不读取 test evaluation/results 来选择参数；
- test 完整对照结果视为 v2 历史工程评估，不作为本轮候选排序依据；
- dev 选出唯一配置后冻结参数；是否重新运行最终 test 由后续独立步骤决定。

## 预注册候选

所有候选使用同一个 `best.ckpt`、ensemble logits、16 帧窗口、EMA 0.4、分位数 0.2/0.7、48 帧标定历史、16 帧 warmup、关键点阈值 0.2。

| ID | 采样策略 | 真实时间投票候选 |
| --- | --- | --- |
| D0 | 固定 stride=1 | 原始 window greedy 3/5/7/9/11/13 |
| D1 | adaptive，min=1，max=2 | span 7/9/11/13/15 |
| D2 | adaptive，min=1，max=3 | span 7/9/11/13/15 |

跨度候选从同一次模型前向保存的 dev logits 和窗口中心离线计算，不重复运行模型。

## 预注册选择规则

1. 先在 D0 中选择 dev WER 最低的固定解码，作为 dev baseline；
2. 每个自适应采样组只保留其 dev WER 最低的 span；
3. 若自适应方案相对固定 baseline 的 WER 增幅不超过 **0.75 个百分点**，在满足条件的方案中选择 clip 减少比例最高者；
4. 若没有方案满足 0.75 约束，选择 dev WER 最低的自适应方案；
5. WER 相同时依次比较 clip 数、INS、运行时间；
6. 看到 dev 结果后不改变上述门槛或候选集合。

## GPU 安全规则

- 禁止使用 PCI `25:00.0` / UUID `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83`；
- 每个进程使用 GPU UUID 绑定；
- 启动前记录 PCI、显存和温度；
- 单卡只运行一个评估进程。

## 运行结果

首次按配置默认 `split_size=36` 启动时，D0/D1 都在 dev 第一个超长样本生成关键点热图时 OOM（单次连续分配约 7.38 GiB）。这属于推理微批过大，不是 PCI `25:00.0` 故障，也未生成候选结果。

处理决定：三个候选统一使用 `split_size=16` 重跑。该参数只决定一次送入模型的 clip 数，不改变窗口、logits 数学结果、解码或预注册候选；三组保持相同设置，耗时仍可横向比较。

## 完整 dev 结果

三组均完成 Phoenix-2014T dev 的 519/519 个样本，参考 gloss 总长度为 3,747。运行中同一个样本
`dev/11December_2009_Friday_tagesschau-3509` 缺少 `images0009.png`，数据加载器统一使用占位帧；该现象在三组中完全相同，因此不造成组间偏差，但绝对 WER 应保留这一数据质量说明。

### 固定步长基线

| 解码方法 | WER (%) | DEL (%) | INS (%) | SUB (%) |
| --- | ---: | ---: | ---: | ---: |
| window-greedy-3 | 27.73 | — | — | — |
| window-greedy-5 | 22.98 | — | — | — |
| **window-greedy-7** | **22.23** | **10.27** | **3.47** | **8.49** |
| window-greedy-9 | 23.41 | — | — | — |
| window-greedy-11 | 26.13 | — | — | — |
| window-greedy-13 | 31.71 | — | — | — |

按预注册规则，固定 baseline 为 `window_greedy_7`，WER 22.23%。固定组共有 55,775 个 clips。

备注：事后复核工具也计算出了固定 stride 下 `span_weighted_13` 的 WER 22.15%，但它不在预注册的 D0 候选集合内，因此没有用它改写 baseline 或选择规则。

### 自适应 span 扫描

| 组 | max stride | span | WER (%) | DEL (%) | INS (%) | SUB (%) | 错误数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 2 | 7 | 24.79 | 5.95 | 9.21 | 9.63 | 929 |
| D1 | 2 | 9 | 23.11 | 7.18 | 6.67 | 9.26 | 866 |
| D1 | 2 | 11 | 22.42 | 8.25 | 5.15 | 9.02 | 840 |
| **D1** | **2** | **13** | **22.39** | **9.93** | **3.90** | **8.57** | **839** |
| D1 | 2 | 15 | 22.58 | 11.53 | 3.02 | 8.03 | 846 |
| D2 | 3 | 7 | 24.95 | 5.71 | 9.23 | 10.01 | 935 |
| D2 | 3 | 9 | 23.22 | 7.15 | 6.35 | 9.71 | 870 |
| D2 | 3 | 11 | 22.74 | 8.51 | 4.83 | 9.39 | 852 |
| D2 | 3 | 13 | 22.42 | 10.11 | 3.66 | 8.65 | 840 |
| **D2** | **3** | **15** | **22.42** | **11.58** | **2.78** | **8.06** | **840** |

D2 的 span=13 与 span=15 总 WER 和错误数完全相同。依据预注册的同分顺序比较 INS，保留 INS 更低的 span=15。

### 效率比较与最终选择

| 指标 | D0 固定 | D1 max=2/span=13 | **D2 max=3/span=15** |
| --- | ---: | ---: | ---: |
| dev WER (%) | **22.23** | 22.39 | 22.42 |
| 相对固定 WER（百分点） | — | +0.16 | **+0.19** |
| clips | 55,775 | 43,349 | **37,615** |
| clip 减少率 | — | 22.28% | **32.56%** |
| 实际时间（秒） | 1,450.60 | 1,231.22 | **1,084.18** |
| 实际时间减少率 | — | 15.12% | **25.26%** |

D1 和 D2 都满足 WER 增幅不超过 0.75 个百分点的约束。按预注册规则，在满足约束的方案中选择 clip 减少率最高者，因此最终冻结：

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
  vote_span_frames: 15
  min_weight: 0.05
```

D2 步长分布为 stride=1：25,563（67.96%）、stride=2：5,597（14.88%）、stride=3：6,455（17.16%）。

## 参数冻结与产物

- 已将 `configs/slide_phoenix-2014t.yaml` 的调优值更新为 `max_stride=3`、`vote_span_frames=15`、`split_size=16`；功能开关仍保持 `enabled: false`，避免未经明确选择就改变旧命令行为；
- 实际启用时同时传入 `--adaptive_stride 1 --span_weighted_voting 1`；
- 离线复核脚本：`SLRT/Online/CSLR/tools/evaluate_adaptive_span_sweep.py`；
- D1/D2 的完整机器可读结果分别位于各自 dev 结果目录的 `dev_span_sweep.json`；
- 本轮候选计算与排序未使用、也未重新运行 test；历史 test 报告不参与选择。冻结配置后的 test 只能作为一次最终报告，不得再用于回调参数。
