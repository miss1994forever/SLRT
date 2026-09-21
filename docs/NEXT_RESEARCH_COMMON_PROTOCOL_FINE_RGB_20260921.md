# 细粒度 RGB robust-utility predictor：共同冻结协议

更新时间：2026-09-21
仓库：`/mnt/workspace/projects/haojun/SLRT`

本协议在任何 finger-CNN 或 face-CNN OOF outcome 产生前同时冻结两个独立 rescue 假设。两个分支
可以由不同对话执行，但不得读取对方 outcome、修改对方配置或共享可变结果目录。若口头说明与
本协议、各分支 `resolved_config_preregistered.json`、机器可读 metrics 冲突，以后者为准。

## 1. 已完成事实与禁止重复

- 完整 fit：6,378 samples、578 sources、183,401 blocks、366,802 side rows。
- robust labels：769 beneficial、2,015 harmful、364,018 neutral。
- B0 signed top-769 utility `-9`；K utility `-2`。
- 低频双手 RGB-DCT R 已 no-go：PR-AUC `0.0020419769563`、utility `-15`；KR `-10`。
- strong-go 门槛固定为 `249 errors`，未达到前禁止 closed loop。
- 不得重跑 HOG、低频 DCT，或在已查看的 RGB-DCT OOF 上调 crop/history/network。

证据目录：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_rgb_dct_robust_predictor_oof_v1_49faacc3/
```

## 2. 两个同时预注册的独立假设

固定 family 顺序：

1. `FINGER_RGB`：关键点对齐的局部双手 tiny CNN，保留手指构形、接触与遮挡细节；
2. `FACE_RGB`：眼-口对齐的局部面部 tiny CNN，保留 mouthing 和非手部表情线索。

两个 OOF 都必须完成后才能考虑下一阶段。即使第一个失败，也按预注册继续第二个；若两者都达到
249，则按上述顺序选择 `FINGER_RGB`，不能事后选 utility 更高者。两个分支的 K+RGB 组合只作
预注册次要互补诊断，不能解锁闭环。

## 3. 共同数据身份

```text
Phoenix video SHA-256:
49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457

HRNet whole-body train keypoints SHA-256:
18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763

ISLR checkpoint SHA-256:
b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b

pose archive SHA-256:
ab659a6e5f1f9f545fe705318ff9be5d7bc2d843acad8930d446b4f83e4dbcb3

frozen skeleton OOF SHA-256:
9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a
```

必须在分支审计中重新核验；不得只凭文件名。禁止读取 calibration/dev/test outcome。

## 4. 共同 tiny CNN 与自监督协议

两个分支都不下载外部数据或权重。每个 outer OOF fold 单独在另外四折 sources 上自监督训练
encoder；evaluation-fold 图像不得参与 encoder、归一化、码本、阈值或 checkpoint 选择。

### 4.1 Encoder

输入 RGB 固定归一化到 `[-1,1]`。网络固定为：

```text
Conv3x3, stride2, 3->16, bias=False + BatchNorm + SiLU
DepthwiseConv3x3 stride2, 16 + PointwiseConv 16->24 + BatchNorm + SiLU
DepthwiseConv3x3 stride2, 24 + PointwiseConv 24->32 + BatchNorm + SiLU
DepthwiseConv3x3 stride1, 32 + PointwiseConv 32->32 + BatchNorm + SiLU
GlobalAveragePool -> 32-D embedding
```

projection head 仅用于自监督：`Linear(32,64)-SiLU-Linear(64,32)`；predictor 使用 projection 前的
L2-normalized 32-D embedding。必须记录逐 fold checkpoint SHA-256。

### 4.2 自监督训练

- objective：同一 crop 的两种增强视图，NT-Xent，temperature `0.2`；
- 每 fold：按 SHA-256 稳定排序，从 outer-train sources 的有效 crop 取前 200,000；
- 增强：平移 x/y 各 `[-2,2]` 像素、scale `[0.90,1.00]`、brightness/contrast/saturation
  各 `[0.8,1.2]`；禁止水平翻转、未来帧配对和 outcome-aware sampling；
- optimizer：AdamW，lr `0.001`、weight decay `0.0001`；
- epochs：固定10；batch 512；无 early stopping、无 scheduler；
- finger seed `261040`；face seed `261041`；随机增强使用 `seed + fold`。

训练不读取 robust reward、reference、future、EOS、decoder logits 或 decoder prefix。

## 5. 滑窗与因果可见性

encoder 对每个到达帧、每个有效 ROI 运行一次，embedding 写入环形缓存；重叠窗口复用缓存，不能
为每个 candidate 重跑 CNN。每个 side candidate 使用自己的16帧物理窗口：

```text
candidate_start + [-7, -6, ..., +8]
```

开头重复第一真实帧；只有尾部真实 EOS 已到达后才可重复最后真实帧，但 EOS flag 不进入 predictor。
决策截止点为 `min(last real frame, decision_arrival + 8)`，与 K 的 bounded-lookahead deadline 相同，
不得称为严格 zero-lookahead。16帧分成连续四段、每段4帧，对每段取 embedding 和 validity 均值。

## 6. 共同 predictor 协议

- 精确复用 `robust-oof-v1` 的5个 source-disjoint folds，并断言 folds/rewards/source index 与旧
  `oof_scores.npz` 相等；
- B0/K 分数直接复用，不重新训练或调骨骼 predictor；
- 主模型：B0 + 对应 RGB candidate-window feature；
- 次要模型：B0 + K + RGB；预注册但不具 gate 资格；
- fold 内标准化只用 outer-train rows；
- MLP `32-16`、20 epochs、AdamW lr `0.002`、weight decay `0.0001`、batch 8192；
- binary seeds `[261026,261027,261028]`，仅诊断；
- signed seeds `[261029,261030,261031]`，三分类主目标；
- top-K固定769；报告 PR-AUC、recall/precision@K、positive utility、harmful count、signed utility；
- 报告主模型减 K 的 source-cluster bootstrap 95% CI；
- 主模型 signed top-769 utility `<249` 时必须 no-go，不运行 closed loop。

## 7. GPU、成本与并发

所有 `nvidia-smi`、CUDA 检查和 GPU 实验必须在沙盒外运行。禁止 PCI `25:00.0`、`41:00.0`
和历史不稳定 GPU0；运行前按 PCI/UUID 重新核验。先用32个 train-fit 窗口做 crop、encoder、缓存、
sample-id/frame-count/causality smoke test，再扩全量。

两个对话不得同时扫39 GB ZIP；特征提取应错峰。每个分支必须报告逐帧 P50/P95、encoder MACs、
参数量、显存、CPU-GPU 搬运、五折离线训练成本和单一部署 encoder 成本。

## 8. 结果隔离与合并

```text
Finger:
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_finger_rgb_tinycnn_robust_predictor_oof_v1_49faacc3/

Face:
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_face_rgb_tinycnn_robust_predictor_oof_v1_49faacc3/
```

各分支只写自己的目录和实现文件；执行对话不得直接改总账本。待两个分支完成后，由整合对话一次性
追加 `docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.{md,json}`，不得删除失败结果。
