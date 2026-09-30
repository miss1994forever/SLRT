# P6 online-quantile 可复现性与实现验收（2026-09-25）

## 冻结范围与已有工作

先读取 `NEXT_RESEARCH_HANDOFF.md`、`P6_K_HISTORY_CONTROLLER_AUDIT_AND_ONLINE_QUANTILE_V1.md`、
`P6_K_HISTORY_CALIBRATION_AND_UNKNOWN_EOS_V1.md`、
`P6_FINAL_EVALUATION_READINESS_AUDIT.md` 和预注册 JSON。完整 train-fit 结果已于
2026-09-23 落盘，本次没有重新运行策略、训练新算法或读取 dev/test/final held-out。
原始预注册配置、模拟脚本、identity、metrics、per-sample 文件均未修改。

## 本次交付

- `Online/CSLR/utils/online_quantile_controller.py`：offset0 到达时登记 coverage、
  完整四窗口 block 到达后执行分位/token 决策；当前 offset0 加历史已付费窗口构成 K 状态。
- `Online/CSLR/configs/frozen_models/p6_online_quantile_fold_models_v1.json`：冻结五折 ridge
  模型系数、标准化量和来源哈希，SHA-256 为
  `116996aef99b4177fb6159086eba6a9ca90f03df6e61e8974a50a760922221e0`。
- `Online/CSLR/tools/export_p6_online_quantile_models_portable.py` 和
  `tools/p6_ridge_fit_portable.cpp`：仅用标准库与本地 C++ 编译器，从已冻结的 fit K/target/fold
  统计量重建同一 ridge 公式；已保存的 K OOF 预测只用于数值验收，不参与拟合。
- `Online/CSLR/tools/audit_p6_online_quantile_release.py`：只读检查冻结资产、产物、
  source-fold 身份、逐样本预算及汇总数字。
- `Online/CSLR/tools/audit_p6_online_quantile_action_parity.py`：从冻结 K 行恢复已付费窗口
  的 36 维摘要，在 train-fit 轨迹上逐 block 比对流式动作和冻结规则。
- `Online/CSLR/tools/verify_p6_online_quantile_release.py`：组合以上检查和单元测试。
- `Online/CSLR/tests/test_p6_online_quantile_controller.py`：流式控制与冻结规则的
  长度/分数轨迹对照，检查未知终点、历史可见性、尾部处理和非法输入。

## 验收结果

只读检查报告：
`Online/CSLR/results/phoenix-2014t_ISLR/p6_online_quantile_release_audit_20260925.json`。
9/9 检查通过，覆盖 6,378 个 train-fit 样本、578 个 source。原冻结结果仍为：
quantile/online uniform/matched uniform/full center errors =
2,543/2,758/2,773/2,325；窗口数 =
273,882/278,307/273,882/371,594。原五项 gate 全通过。
流式规则测试 4/4 通过。模型重建的 183,401 行 OOF K 预测与原冻结预测的最大绝对差
`2.38814e-7`、RMSE `1.01892e-7`。在 6,378 个样本、183,401 个完整 block 上，
流式实现与冻结规则的 action 逐 block 一致；强制/自主动作及剩余 token 逐样本一致。
由冻结 K 行恢复 36 维摘要的 teacher-forced K 特征最大绝对差为 `2.68817e-5`；
这个误差未导致动作分歧。综合验收报告为
`docs/evidence/P6_ONLINE_QUANTILE_RELEASE_ACCEPTANCE_20260925.json`。

冻结配置 SHA-256：`48f2a6f58ee3e5ee321baeffed922c14416fa4ec01f2c28625e9f4560c40ef33`。
原模拟脚本 SHA-256：`f08241e5e60bd841eca41a06a15754ff89c29bea62adc124331a167298b1242a`。
fit 数据集 SHA-256：`daeefd11c1fa7ac554a3c39e6ef932bf39c4bed8a89e187c6c1b3db4130e105d`。
这些与预注册 identity、manifest 和固定资产清单一致。

## 实现对齐要点

1. 每个 block 的 offset0 在到达时执行并记入状态；达到 token 余额 1 时才可选 offset2。
   每个完整 block 加 0.5 token、初始 0、容量 2、余额达到 2 强制执行。
2. 分数只能用该策略已执行窗口的 posterior 加当前 offset0。阈值只用本视频严格过去
   block 分数的精确中位数；不读取全局阈值、未来分数或视频总长度。
3. 流式接口在四个位置均已到达后确认 block 完整并作 offset2 决策；不完整尾块已有
   offset0 coverage，不增加 token、不补做 bonus。该实现需要缓存当前 block 的 offset2
   输入并在 block 完整时决定是否执行，因此存在至少一个位置的决定延迟；真实端到端
   延迟尚未测量。
4. matched uniform 仅为离线等预算诊断，按 quantile 的每样本实际 bonus 数布置，
   不作为可在线部署策略。此比较的总窗口数逐样本相等。

## 复现步骤

从 `Online/CSLR` 目录运行，不使用 GPU：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 tools/verify_p6_online_quantile_release.py
```

若需独立重建已交付模型，从同一目录运行以下 CPU 命令，并比对输出 SHA-256 与上文模型哈希：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 tools/export_p6_online_quantile_models_portable.py \
  --output /tmp/p6_online_quantile_fold_models_check.json
sha256sum /tmp/p6_online_quantile_fold_models_check.json
```

导出脚本先核对固定资产哈希，且拒绝覆盖已有输出。每折由其余四个 source fold 的
K/target 行训练，ridge alpha=100，均值和标准差仅由训练折计算。运行时按 source 的
冻结 fold 身份选择对应模型。不要重新运行原
`analyze_phoenix_p6_k_online_quantile_oof.py --run`，其已有 outcome 文件且脚本会拒绝覆盖。

## 剩余限制与所需输入

模型和动作对齐已在冻结的 train-fit K 行上验证；这里没有重新执行原 NumPy 特征提取或
视频前端。36 维 bonus 摘要通过后续 teacher-forced K 行的均值反推，仅用于本次验收的
输入重建；在线决策本身只接收已付费窗口。现有 `prediction_slide.py` 没有接入这个 P6
控制器，本次验收不等同于端到端视频推理或运行时延验收。若要交付实时视频服务，还需
接入该前端并测量缓存 offset2 所产生的延迟与控制器成本。

final readiness audit 已确认本地 train fit/calibration 均已消费，dev 和 test 有历史 outcome；
不存在可称为未见 final held-out 的本地 split。本次不运行新评价。真正外部确认需要新的
封存连续手语数据；若改做修复后 dev/test 回顾性评价，必须先明确标为非 confirmatory。

本次没有 CUDA/GPU 检查或实验；遵守禁用 PCI `01:00.0`、`25:00.0`、`41:00.0`
以及宿主机 GPU0 的约束。
