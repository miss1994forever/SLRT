# 下一对话交接：面部表情 RGB robust-utility predictor

更新时间：2026-09-21
仓库：`/mnt/workspace/projects/haojun/SLRT`

## 0. 可直接交给新对话的任务

> 完整阅读本文件、`docs/NEXT_RESEARCH_COMMON_PROTOCOL_FINE_RGB_20260921.md`、当前实验账本和
> `NEXT_RESEARCH_HANDOFF_RGB_ROBUST_PREDICTOR_20260920.md`。严格执行共同协议中的
> `FACE_RGB` 分支：先做身份/因果/成本 smoke audit，再在任何 feature 或 outcome 产生前写入
> `resolved_config_preregistered.json`；随后以五个 outer-fold-specific 自监督 encoder 提取
> candidate-local 面部 RGB 表征，并运行与旧 K 相同的 source-disjoint predictor OOF。
> 禁止读取 FINGER_RGB outcome、calibration/dev/test、closed-loop outcome，禁止修改 K 或在看到
> 本分支 OOF 后调 crop/CNN/增强/时间汇总。utility 未达到249时立即 no-go。

## 1. 分支唯一假设

面部 mouthing、眉眼和头部非手部标记可能提供 K 的手部/pose 几何没有表达的语言线索，从而更早
判断某个候选窗口是否会改善 decoder。主检验是 `RF` 对 K，不是对 B0。

| 名称 | 输入 | 地位 |
|---|---|---|
| B0 | 冻结 bookkeeping + decoder prefix OOF | 复用 |
| K | 冻结 B0+skeleton OOF | 主要对照 |
| RF | B0 + candidate-local face RGB | 主候选、第二 family gate 候选 |
| KRF | B0 + K + face RGB | 次要互补诊断，不可解锁闭环 |

## 2. 冻结面部 ROI

- COCO WholeBody face landmarks 使用全局索引 `23:91`，confidence `>=0.2`；
- 以68点局部编号计算：右眼 `36:42`、左眼 `42:48`、外唇 `48:60`；每组至少一半点有效；
- anchor 为左右眼有效点质心和外唇有效点质心；三点仿射映射到 `64x64` canonical 坐标：

```text
right-eye center -> (20,22)
left-eye center -> (44,22)
outer-lip center -> (32,45)
```

- `cv2.warpAffine`，bilinear，越界 constant zero；不做未来/历史补 crop；
- RGB `[0,255] -> [0,1] -> [-1,1]`；无效 face 为32-D zero embedding + validity 0；
- 每帧 `33-D`；每个 candidate 的16帧分四段均值，得到 `132-D/candidate`；
- RF 总输入宽度 `49+132=181`，KRF 为 `49+68+132=249`。

不得扩成上半身、全身或完整画面 RGB，也不得使用昂贵 ISLR embedding。该分支只回答局部面部
表情/口型是否提供增量信息。若 landmark anchor smoke 大量失效，必须在预注册前报告并请求方向。

## 3. Encoder 与 OOF

严格使用共同协议的 tiny CNN、NT-Xent、200,000 crop/fold、10 epochs、seed `261041`。输入尺寸
为64x64，其他架构相同。每个 outer fold encoder 只能使用另外四折 sources；保存五个 checkpoint、
训练 manifest、crop sample hash 和 SHA-256。固定取 epoch 10，不按自监督 loss 或 utility 选择。

用 fold-specific encoder 生成该 fold 所需的 train/eval candidate features，再训练 predictor。RF 与
KRF 使用共同协议的 MLP、epochs、seeds。输出必须包含逐 fold encoder/source 隔离断言和旧 OOF
row parity。

## 4. 成本解释

部署时每个到达帧只运行一次 face CNN，重叠窗口复用 embedding。必须报告 face anchor 有效率、
仿射越界率、P50/P95、MACs、参数量、显存和 CPU-GPU 搬运。不能因为离线 OOF 有五个 encoder
而把部署成本写成五倍；也不能忽略逐帧 face crop/CNN 的固定控制成本。

## 5. 输出与停止条件

唯一输出目录：

```text
Online/CSLR/results/phoenix-2014t_ISLR/
  p3_fulltrain_face_rgb_tinycnn_robust_predictor_oof_v1_49faacc3/
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

- RF signed top-769 utility `>=249`：只报告 branch strong-go；不得自行闭环，等待 FINGER_RGB 完成
  和整合对话按 family 顺序裁决；
- RF 明显优于 K 但 `<249`：ranking-only；不闭环；
- RF 不优于 K 或 utility 非正：no-go，停止，不调参；
- 无论 RF 结果如何，KRF 只报告次要诊断。
