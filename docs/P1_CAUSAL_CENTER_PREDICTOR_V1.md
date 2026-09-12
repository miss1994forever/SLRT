# Phoenix P1 因果手语中心预测器 v1（冻结）

更新时间：2026-09-12

本文冻结 P0 sign-center offline oracle 之后的首个 train-only 可部署性实验。机器索引为
`docs/results/phoenix_p1_causal_center_predictor_v1_frozen.json`，完整轻量证据位于
`Online/CSLR/results/phoenix-2014t_ISLR/p1_train_center_audit_v1/`。

## 1. 问题与结论

P0 证明：在 50% dense budget、span-15 下，把窗口放到 alignment-derived sign midpoint
附近可将 dev WER 从 `23.058447%` 降到 `21.537230%`。P1 问的是：只看当前和历史关键点，
能否因果地预测这些 midpoint proxy，并达到连接 scheduler 所需的定位质量？

冻结结论为 **No-Go**。固定逻辑回归在 train calibration 上未达到预注册 gate，因此没有选择
阈值、没有运行 scheduler replay，也没有产生新的 WER。该结论只否定当前 11 维廉价特征和
固定事件策略，不否定更强的时序视觉表征。

## 2. 数据隔离与 provenance

原关键点 pickle 物理混合 train/dev/test。受控拆分保留原文件不变，生成三个互斥文件；三者
key 并集严格等于源 8,257 个视频，逐数组 shape、dtype 和数值与源一致：

| split | 视频 | 帧 | SHA-256 |
|---|---:|---:|---|
| train | 7,096 | 827,354 | `18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763` |
| dev | 519 | 55,775 | `85d1f0e2ca5943786450c82a23bceb0254b779379d3776e4d394b6cf17b80811` |
| test | 642 | 64,627 | `5cb767b9ec1fc0dbc331245132e0383ee333d30adba1e9e4447667acddaa09d1` |

源文件 SHA-256 为
`69c56c1902b6a528f644d7be6ce14bb902a86fa83dd4fbcaa0d78a6ef58dc5ff`；本地拆分
manifest SHA-256 为
`44376a9795f3ced9450cbb4386d9128e4fcca08e78a712c7a6bd74e31b14f1c7`。
大型 pickle 继续由 Git 忽略。P1 训练只读取 train-only；模型、归一化、阈值网格、事件策略
和 gate 冻结后，dev-only 只打开一次做检测评估；test-only 从未在 P1 中打开或运行。

## 3. Target、划分与特征

- target 是 train alignment-derived nonblank segment midpoint `±1` 帧的并集，不是人工逐帧
  ground truth；blank 不产生正例；
- 排除唯一 reference/segment 顺序不一致且含重复片段的视频后，共 827,338 行；
- 按 source video 的确定性哈希划分：fit 5,655 视频、660,552 行；calibration 1,440 视频、
  166,786 行；
- 11 个严格 past-only 特征：有效关键点比例、当前运动、过去 4/16/48 帧运动 mean/max、
  运动差分、当前人体框宽高；
- 非有限或低置信关键点先显式 mask，再中和坐标；统计量忽略无效点；
- 模型是 fit-only 标准化、class-balanced、固定 `C=1.0` 的逻辑回归；CPU L-BFGS 23 次迭代
  正常收敛；
- 事件由阈值向上穿越产生，并使用固定 4 帧 refractory period。

## 4. Train calibration gate

预注册要求：center recall within `±1` 至少 75%，且匹配事件的 positive-delay fraction 不超过
25%。

| 指标 | 结果 |
|---|---:|
| frame AUROC | 0.545158 |
| frame AUPRC / prevalence | 0.220340 / 0.201588 |
| 任意阈值最大 recall | 17.8307% |
| 该阈值 precision / positive delay | 21.5519% / 33.4167% |
| 满足 delay ≤25% 时最大 recall | 2.3727% |

没有阈值同时满足两项要求，所以 `selected_threshold=null`，gate 为 false。这是表征能力失败，
不是优化器、CPU 或程序失败。

## 5. 一次性 dev 检测

pre-dev freeze 后对 dev-only 做了一次不依赖阈值的检测评估：

| 指标 | 结果 |
|---|---:|
| 视频 / 帧 / proxy centers | 519 / 55,775 / 3,748 |
| frame AUROC | 0.552188 |
| frame AUPRC / prevalence | 0.224221 / 0.201542 |
| Brier / ECE-15 | 0.248797 / 0.297301 |

没有用 dev 发明新阈值或修改模型。由于 calibration gate 已失败，scheduler replay 按协议跳过；
历史 uniform `23.058447% / 864 errors` 和 perfect-center oracle
`21.537230% / 807 errors` 仅作为参考，P1 没有新 WER。

## 6. 冻结决策与下一步

不要继续在 dev 上微调当前逻辑回归阈值，也不要把 threshold-free AUROC/AUPRC 写成 scheduler
有效。下一步若继续本方向，应建立新版本并增强因果输入，例如左右手分离的姿态/速度、手形
变化、相对身体坐标和短历史序列 head；仍需只用 train 拟合与校准，并在通过训练 gate 后才
连接 scheduler。另一条合理方向是回到 window/decoder 或模态计算门控。

冻结时 86 项完整测试通过，21 个 provenance 文件哈希匹配，`git diff --check` 通过；P0
文件未修改，GPU 未使用。
