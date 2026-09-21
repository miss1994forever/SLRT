# 下一阶段研究交接：重设计可部署调度 target

更新时间：2026-09-21

仓库：`/mnt/workspace/projects/haojun/SLRT`

本文件只描述当前冻结状态和下一步。完整数字、协议冲突与产物路径以
[`ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md`](ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md)
及对应机器可读 metrics 为准。

## 0. 交给新对话的任务

> 核对完整 train robust oracle、skeleton/RGB/DCV predictor 和 utility information ladder 的
> 冻结结果。不要继续调当前 per-window counterfactual utility classifier。先提出并预注册一个
> 决策时可观测、标签更稳定的新 target 或序列级决策分解，明确因果可见性、source-disjoint OOF、
> 等预算 gate 和停止条件；在 train OOF 达到冻结 headroom 前不得运行闭环或读取新的 held-out
> outcome。

开始前确认工作区干净、资产哈希一致且没有冲突实验。所有 `nvidia-smi`、CUDA 检查和 GPU
实验必须在沙盒外运行，避开 PCI `25:00.0`、`41:00.0` 和宿主机 GPU0。沙盒内 CUDA 不可见
不能判定驱动故障。

## 1. 当前冻结结论

| 阶段 | 结论 |
|---|---|
| A0 等预算基线 | 减少约三分之一窗口，但没有证据显著优于等预算 uniform |
| P0 sign-center oracle | alignment-derived offline 上界强；不可部署 |
| P1/P2 center predictor | 因果特征改善 detection，但都未通过 gate |
| P3 robust continuation oracle | 完整 train fit 上 errors `2325→1688`，少637；窗口选择上界成立 |
| P3 robust labels | 366,802 rows：769 beneficial、2,015 harmful、364,018 neutral |
| P3 deployable predictors | K、RGB-DCT、finger RGB、face RGB、DCV 均为非正 top-K utility |
| P3 information ladder | 最丰富 L3 utility `+5`，远低于冻结的 `249` gate |

因此，失败的不是“视觉信息完全无效”，而是视觉/decoder 可见线索与当前 robust counterfactual
utility 标签之间的映射过弱且高度稀疏。oracle 使用 future continuation、reference 和真实 EOS
生成标签；部署 predictor 不能看到这些变量。同一 action 的 utility 还会随 continuation 和 decoder
设置变化，所以继续堆叠同类特征或调阈值不太可能解决根因。

## 2. 已停止的方向

- 不再调整当前 K/R/HF/RF/DCV/L1--L3 的 crop、history、network、seed 或 top-K threshold；
- 不把 K+RGB 等次要组合升级成新的主检验；
- 不在同一 OOF outcome 上进行 post-hoc model selection；
- 不运行 unknown-EOS closed loop、held-out WER 或 test；
- 不把 oracle improvement 描述成可部署 scheduler improvement。

这些约束不禁止复用已经验证的因果 skeleton/RGB encoder 作为新 target 的输入，但必须建立新版本、
新预注册配置和新输出目录。

## 3. 下一步设计要求

下一项实验必须先回答“训练目标是否在部署时可辨识”，再回答“模型是否足够大”。候选方案可以是：

1. 把单步精确 utility 改成更稳定的 coarse target，例如 future-independent 的信息增益、稳定边界区间、
   commit risk 或预算压力；
2. 把独立候选分类改成序列级 credit assignment，让模型优化一段轨迹的累计预算收益；
3. 把动作拆成 coverage 保底与少量 bonus allocation，只学习可验证的相对偏好或 abstention；
4. 在不读取 outcome 的前提下先做 label-stability/identifiability audit，若可部署观测对新 target 仍无
   信息量，则在训练前停止。

无论选择哪一项，都必须冻结：

- train-only 身份、578-source grouping 和 source-disjoint folds；
- predictor 可见信息、lookahead（0/4/8 必须分开命名）与 unknown-EOS 行为；
- uniform coverage skeleton、逐样本预算和 action deadline；
- 主指标、249-error headroom、cluster bootstrap CI 与停止规则；
- controller 固定成本、逐帧执行频率、缓存复用和真实部署成本口径。

## 4. 解锁顺序

1. 先完成 train-side target 稳定性和可辨识性审计；
2. 再运行冻结的 source-disjoint OOF；
3. 只有主指标达到249-error headroom，才运行 unknown-EOS token-bucket 闭环；
4. 闭环必须先与逐样本等预算 uniform 比较，并分别报告0/4/8 lookahead；
5. 闭环通过后，才允许读取冻结 held-out outcome，并测量 wall time、controller cost、队列积压和
   P95稳定提交延迟；
6. test 和论文级多重比较校正留到方法完全冻结之后。

## 5. 不可变身份

- Phoenix 视频 ZIP SHA-256：
  `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`；
- HRNet whole-body train keypoints SHA-256：
  `18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763`；
- ISLR checkpoint SHA-256：
  `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`；
- frozen skeleton OOF SHA-256：
  `9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a`；
- 当前窗口为16帧，现有 utility 链使用 triangular span-weighted-15 和 bounded `+8` candidate
  lookahead，不能称为严格 zero-lookahead。

改变数据、checkpoint、decoder、窗口、预算、action space 或 target 时必须建立新实验身份，不能
覆盖现有结果目录。

## 6. 允许与禁止的声明

可以说：完整 train 上存在强 robust oracle headroom；现有可部署特征族无法可靠预测逐窗口
counterfactual utility；细粒度 RGB 相比 RGB-DCT 有局部改善，但没有超过 skeleton K 或达到 gate。

不能说：视觉对在线 CSLR 没有价值、decoder-aware 调度不存在、当前 predictor 已经接受独立
held-out 否证，或沙盒内 CUDA 不可见表示驱动故障。
