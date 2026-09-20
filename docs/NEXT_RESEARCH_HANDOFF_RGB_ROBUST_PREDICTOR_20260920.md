# 下一对话交接：完整数据上的 RGB vs 骨骼 robust-utility predictor

更新时间：2026-09-20
仓库：`/mnt/workspace/projects/haojun/SLRT`

本文件是新对话的唯一入口。它接续已经完成的完整 train robust 主链，下一目标不是继续调当前
skeleton predictor，而是检验：**在完全相同的反事实标签、source folds、模型协议和 top-K gate 下，
廉价、因果可见的 RGB/手形外观表征能否比骨骼表征更早识别正 decoder utility。**

如果本文件与口头记忆冲突，以冻结配置、机器可读 metrics 和实验账本为准。

## 0. 可直接交给新对话的任务

> 阅读 `docs/NEXT_RESEARCH_HANDOFF_RGB_ROBUST_PREDICTOR_20260920.md` 和
> `docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md`。先核验完整 robust 标签、冻结 source folds、
> skeleton OOF 基线及 RGB 视频/缓存的身份。然后只在 train fit 6,378 样本上建立一个新的、独立
> 预注册的 RGB-vs-skeleton OOF 表征实验。禁止读取 calibration/dev/test outcome，禁止调当前
> skeleton predictor，禁止在未达到 signed top-K utility 249 errors 的门槛时运行闭环。先审计
> RGB 表征是否真正因果可见、是否需要逐帧执行及其成本；不得把被跳过窗口的昂贵 ISLR logits
> 或昂贵 backbone 特征伪装成廉价 scheduler 输入。

## 1. 当前冻结结论

### 1.1 Robust oracle 上界成立

完整 train fit 范围为 6,378 个样本、578 个 source、183,401 个三选一 blocks：

- 等预算 uniform：2,325 errors，WER 4.6862717433%；
- robust continuation oracle：1,688 errors，WER 3.4023340657%；
- 改善 637 errors / 1.2839376776 pp；
- source-cluster bootstrap 95% CI：`[-1.3912910568, -1.1797072002]`；
- strong-go 成立，但这是读取 reference、future 和 EOS 的不可部署上界。

证据：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_robust_continuation_oracle_fit6378_v1_49faacc3/metrics.json
```

### 1.2 Robust 标签仍依赖 continuation

- center-future positive sides：1,219；
- 在 late-future 下仍为正：769；
- survival：0.6308449549，source-bootstrap CI `[0.6065302542, 0.6571674021]`；
- center positive 直接反转为负仅 3 个；
- 但 union-nonzero side 的符号一致率只有 0.4624369525。

因此 769 个 robust-positive 交集标签是必要的，不能退回单一 continuation target。

证据：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_robust_label_stability_fit6378_v1_49faacc3/metrics.json
```

### 1.3 当前 skeleton/prefix predictor 明确 no-go

完整 counterfactual 数据有 366,802 个 side candidates：

- beneficial：769；
- harmful：2,015；
- neutral：364,018；
- 五折为确定性的 source-disjoint `robust-oof-v1` 映射；同一 source 不得跨 fold。

冻结输入：

- `B0`：candidate bookkeeping + past-paid-window decoder prefix；
- `K`：B0 + 冻结的 17 个 P2 temporal features 的 31-step causal history，使用
  last/mean/std/delta 汇总；
- candidate expensive ISLR logits、reference、future total length 和 EOS 均不是 predictor 输入。

主要 signed 三分类结果：

- B0 PR-AUC 0.0030240744；top-769 signed utility `-9`；
- K PR-AUC 0.0025492400；recall@769 0.0104031209；
- K 的 top-769 捕获 8 个正效用，同时含 10 个 harmful，signed utility `-2`；
- K-B0 source-bootstrap CI `[-5, 19]`；
- 进入闭环的冻结门槛为 `ceil(0.005 × 49,613) = 249 errors`。

所以当前分支已经停止：不得调 threshold、扩大同一 skeleton 特征、改网络后反复查看同一 OOF，
也不得因为 GPU 已恢复就运行闭环。

证据：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_robust_predictor_oof_v1_49faacc3/resolved_config_preregistered.json
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_robust_predictor_oof_v1_49faacc3/metrics.json
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_robust_predictor_oof_v1_49faacc3/oof_scores.npz
```

`oof_scores.npz` 为 5,410,985 bytes，SHA-256：

```text
9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a
```

该数组按结果忽略规则只保留本地；Git 中提交其预注册配置、metrics 和此完整性哈希。

## 2. 已冻结、必须复用的输入

### 2.1 数据与模型身份

```text
Phoenix video identity SHA-256:
49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457

ISLR checkpoint SHA-256:
b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b
```

checkpoint 文件名曾被修改后恢复。不得只凭文件名认定身份，必须重新核 SHA-256。

### 2.2 Dense replay

```text
Online/CSLR/results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3/
```

完整度：56/56 shards、7,096 samples、827,354 windows。RGB predictor 阶段不应重新生成 dense
ISLR logits。

### 2.3 Robust counterfactual labels

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3/
```

`dataset_manifest.json` 必须满足：6,378 samples、578 sources、183,401 blocks、366,802 side rows、
769 beneficial、2,015 harmful，并精确复现 oracle 的 2,325→1,688 errors。禁止重新定义 utility。

### 2.4 骨骼/pose/hand/motion archive

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_causal_pose_hand_motion_v1_49faacc3/
  train_causal_pose_hand_motion_sequences.npz
```

身份：7,096 samples、827,354 frames、211 features；archive SHA-256：

```text
ab659a6e5f1f9f545fe705318ff9be5d7bc2d843acad8930d446b4f83e4dbcb3
```

### 2.5 实验账本

```text
docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md
docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.json
```

## 3. 新实验唯一主问题

在相同 366,802 candidates、相同 reward、相同 source folds、相同模型容量、训练 epoch、随机种子
数量和 top-K=769 下，比较：

| 名称 | 输入 | 地位 |
|---|---|---|
| B0 | bookkeeping + decoder prefix | 冻结低层基线，直接复用已保存 OOF |
| K | B0 + skeleton temporal features | 冻结骨骼基线，直接复用已保存 OOF |
| R | B0 + 新的 RGB/手形外观 temporal features | 新实验主候选 |
| KR | B0 + skeleton + RGB | 仅在训练前明确预注册时才可作为次要互补诊断 |

主检验是 `R` 对 `K`，不是 RGB 对空基线。`KR` 不能在看到 R 失败后临时加入并作为确认性结果。

## 4. RGB 表征的因果与成本边界

新对话必须先写 `resolved_config_preregistered.json`，再生成特征或训练。配置至少冻结：

1. RGB 数据源、视频/帧 sample-id 对齐方式及 SHA-256；
2. 人体/双手 crop 定义、无效 crop 处理、resize 和归一化；
3. encoder 身份与权重哈希；
4. 每个 decision 可见的最后一帧；
5. 历史长度、pooling、feature dimension；
6. 训练折、标准化、网络、epoch、seed、candidate priority 和 gate；
7. 是否运行 `KR` 及其地位；
8. 输出新目录名，禁止覆盖现有结果。

当前 skeleton OOF 的 decision 声明为“offset3 到达后”，属于已有 bounded candidate lookahead，
不是严格 0 帧。为了表示公平，R 首先必须使用与 K 完全相同的可见时间截止点；若另做严格 0
lookahead，必须作为单独实验报告，不能把两种时间协议混在 RGB-vs-K 比较中。

以下输入禁止作为部署型 R：

- 被 scheduler 跳过候选窗口的 ISLR logits；
- 为每个候选执行完整昂贵 ISLR backbone 后取得的 embedding；
- future frames、reference、真实 EOS、完整视频长度；
- 使用 fold-evaluation 样本拟合的标准化、PCA、码本或阈值。

如果只能取得昂贵 RGB backbone 特征，可以做“表征可预测性上界诊断”，但必须明确标记为
nondeployable，并且不能据此声称减少计算。真正可部署的 RGB 分支必须报告：逐帧执行频率、
FLOPs/真实时间、显存、CPU–GPU 搬运和是否抵消跳窗收益。

## 5. 不要重复的 RGB 尝试

已有 partial32/frozen512 的同帧双手 crop HOG-preview 实验：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_partial32_hog_preview_fit512_oof_v1_49faacc3/metrics.json
```

它使用 648 维 same-frame HOG、59,108 frames，最终 gate 失败并冻结为
`stop HOG-preview route`。这不等价于完整 robust-label RGB 实验，但说明**原样放大全量 HOG**
价值很低。新 R 应提出实质不同的外观表示，例如冻结的轻量手形 embedding 与因果历史汇总；
差异和新增成本必须预先说明。

## 6. 冻结训练与评价协议

除 RGB 输入维度外，优先复用完整 OOF 的协议：

- 5 deterministic source-disjoint folds；必须复用 `oof_scores.npz` 中的 fold/source mapping；
- fold 内训练标准化，evaluation fold 不参与任何拟合；
- signed harmful/neutral/beneficial 三分类为主要 objective；
- binary beneficial-vs-rest 只作兼容诊断；
- MLP 32-16、20 fixed epochs、AdamW lr 0.002、三 seed ensemble；
- top-K 的 K 固定为 769；
- 报告 PR-AUC、recall@K、precision@K、positive utility captured、harmful actions、signed utility；
- 报告 R-K 的 source-cluster bootstrap CI；
- 训练过程不得读取 calibration、dev、test 或 closed-loop outcome。

如果 RGB dimension 使原 MLP 输入不现实，应先以无标签准则固定 PCA/投影维度，并在每个 fold 的
train 部分单独拟合。不得根据 OOF utility 挑维度。

## 7. 决策门槛

### Strong go

只有预注册优先级中的候选满足：

```text
signed top-K robust utility >= 249 errors
```

才进入 unknown-EOS closed loop。249 来自完整 reference 长度 49,613 的 0.5 pp headroom，不能因
结果不理想而下降。

### 有信号但不解锁闭环

若 R 明显优于 K、bootstrap CI 支持改善，但累计 utility 仍小于 249，只能声称 RGB 表征提高了
离线 ranking；不得运行最终 WER/wall-time 主实验。应冻结为下一轮独立假设或停止。

### No-go

若 R 的 PR-AUC/recall@K 没有可靠优于 K，或 top-K signed utility 非正，则停止当前 RGB 表征；
不要继续调 crop、历史长度、网络和 threshold。若 KR 已预注册，可按既定次序报告，但不能事后
改为主检验。

## 8. 只有 strong-go 后才解锁的工作

1. unknown-EOS token-bucket 闭环；不使用尾部 top-up，不提前知道 T；
2. 与逐样本等预算 uniform coverage skeleton 比最终 WER；
3. 0/4/8 帧 lookahead，分别报告算法延迟；
4. 真实端到端 wall time、controller cost、CPU–GPU 搬运、队列积压；
5. P50/P95 稳定提交延迟与正确前缀形成时间；
6. 只有主方法成立后，再做 AdaBrowse-inspired 同协议比较和 held-out confirmatory evaluation。

## 9. 工程与 GPU 注意事项

- 工作区很脏，许多改动属于用户或既有研究链；禁止 reset/checkout/清理无关文件。
- 未经用户明确要求，不做 Git commit。
- 每个新结果使用新目录，禁止覆盖 oracle、dataset、feature archive 或 skeleton OOF。
- GPU PCI `25:00.0` 和 `41:00.0` 曾故障，必须避开；GPU0 也曾被标记为不稳定。
- 历史上稳定的是 GPU 3/4/5/6，但运行前仍应检查实时状态和 PCI/UUID，不能只相信 ordinal。
- predictor 训练可用一张稳定 GPU；RGB archive 提取是否适合多卡取决于实现，但必须先做小规模
  sample-id、frame-count、causality 和数值 smoke test，再扩到全量。
- 如果 decoder/JSON 处理在 CPU 更快，不要为了“使用 GPU”而搬到 GPU。

## 10. 新对话应交付的最小产物

1. RGB 可见性与成本审计；
2. outcome 前写入的 `resolved_config_preregistered.json`；
3. 完整 RGB feature manifest：样本数、帧数、invalid crop、feature shape、输入/输出 SHA-256；
4. source-disjoint OOF scores 与 metrics；
5. B0/K/R（以及预注册时才有的 KR）统一比较；
6. 明确的 strong-go / ranking-only / no-go 决定；
7. 把结果追加到实验账本，不删除或重写失败结果。

当前最重要的科学边界：**完整 oracle 已证明窗口选择存在上界，但当前因果 skeleton/prefix 无法
预测该上界。RGB 实验必须证明新增外观证据提高了“选出正累计 utility”的能力，而不只是提高
平均动作分类准确率。**
