# 下一对话交接：细粒度手指 RGB robust-utility predictor

更新时间：2026-09-21
仓库：`/mnt/workspace/projects/haojun/SLRT`

## 0. 可直接交给新对话的任务

> 完整阅读本文件、`docs/NEXT_RESEARCH_COMMON_PROTOCOL_FINE_RGB_20260921.md`、当前实验账本和
> `NEXT_RESEARCH_HANDOFF_RGB_ROBUST_PREDICTOR_20260920.md`。严格执行共同协议中的
> `FINGER_RGB` 分支：先做身份/因果/成本 smoke audit，再在任何 feature 或 outcome 产生前写入
> `resolved_config_preregistered.json`；随后以五个 outer-fold-specific 自监督 encoder 提取
> candidate-local 手指 RGB 表征，并运行与旧 K 相同的 source-disjoint predictor OOF。
> 禁止读取 FACE_RGB outcome、calibration/dev/test、closed-loop outcome，禁止修改 K 或在看到
> 本分支 OOF 后调 crop/CNN/增强/时间汇总。utility 未达到249时立即 no-go。

## 1. 分支唯一假设

低频 DCT/HOG 丢失了手指构形细节；关键点对齐的 tiny CNN 若保留指尖、指间接触、遮挡和局部
轮廓，可能提供 K 之外的 decoder-utility 信号。主检验是 `HF` 对 K，不是对 B0。

| 名称 | 输入 | 地位 |
|---|---|---|
| B0 | 冻结 bookkeeping + decoder prefix OOF | 复用 |
| K | 冻结 B0+skeleton OOF | 主要对照 |
| HF | B0 + candidate-local finger RGB | 主候选、唯一 gate-eligible |
| KHF | B0 + K + finger RGB | 次要互补诊断，不可解锁闭环 |

## 2. 冻结手部 ROI

- whole-body keypoints：左手全局索引 `91:112`，右手 `112:133`；confidence `>=0.2`；
- 每手必须有 wrist(0)、index MCP(5)、pinky MCP(17) 三个有效 anchor，否则该手 crop 无效；
- 三点仿射映射到 `48x48` canonical 坐标：

```text
wrist -> (24,42)
index MCP -> (12,27)
pinky MCP -> (36,27)
```

- `cv2.warpAffine`，bilinear，越界 constant zero；不做未来/历史补 crop；
- 左右手按解剖 anchor 映射到同一 canonical orientation，共享 encoder；
- RGB `[0,255] -> [0,1] -> [-1,1]`；无效手为32-D zero embedding + validity 0；
- 每帧两个32-D embedding + 两个 validity，共 `66-D/frame`；
- 每个 candidate 的16帧分四段均值，得到 `264-D/candidate`；
- HF 总输入宽度 `49+264=313`，KHF 为 `49+68+264=381`。

不得改为整手 bbox、HOG、DCT、全身 RGB 或昂贵 ISLR embedding。若三点仿射 smoke 显示系统性
截断，必须在预注册前报告并停止请求方向；不得自行看 outcome 后修改 crop。

## 3. Encoder 与 OOF

严格使用共同协议的 tiny CNN、NT-Xent、200,000 crop/fold、10 epochs、seed `261040`。每个 outer
fold encoder 只能使用另外四折 sources；保存五个 checkpoint、训练 manifest、crop sample hash 和
SHA-256。不得按自监督 loss 挑 epoch，固定取 epoch 10。

用 fold-specific encoder 生成该 fold 所需的 train/eval candidate features，再训练 predictor。HF 与
KHF 使用共同协议的 MLP、epochs、seeds。输出必须包含逐 fold encoder/source 隔离断言和旧 OOF
row parity。

## 4. 成本解释

部署时每个到达帧运行左右手共享 CNN 各一次，重叠窗口复用 embedding；不能按三个 candidate
重跑 CNN。pose detector 成本虽然与 K 共用，仍需单列“共享但非零”。invalid ROI、仿射越界率、
P50/P95、MACs、参数量、显存和 CPU-GPU 搬运必须进入 manifest。

## 5. 输出与停止条件

唯一输出目录：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_finger_rgb_tinycnn_robust_predictor_oof_v1_49faacc3/
```

至少交付：

```text
visibility_cost_audit.json
resolved_config_preregistered.json
encoder_manifest.json
feature_manifest.json
metrics.json
oof_scores.npz
```

`oof_scores.npz` 与大 feature/checkpoint 缓存保留本地并记录 SHA-256；Git 只保留小型配置、manifest、
metrics 和实现。禁止 Git commit，除非用户另行要求。

决策：

- HF signed top-769 utility `>=249`：只报告 branch strong-go；仍不得运行闭环，等待 FACE_RGB 完成
  和整合对话按 family 顺序裁决；
- HF 明显优于 K 但 `<249`：ranking-only；不闭环；
- HF 不优于 K 或 utility 非正：no-go，停止，不调参；
- 无论 HF 结果如何，KHF 只报告次要诊断。
