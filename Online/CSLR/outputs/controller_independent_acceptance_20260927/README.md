# P6 online-quantile 独立一致性验收（2026-09-27）

## 结论

在冻结 train-fit 名单的 6,378 个样本、183,401 个完整 block 上，原冻结
`analyze_phoenix_p6_k_online_quantile_oof.py:schedules()` 与发布控制器的逐窗口
coverage/bonus 选择完全一致；动作、token 与原 `per_sample` 计数无差异。
原脚本未修改。发布控制器也无需修复，其 SHA-256 为
`53a17e01641cea239379504f0f46fa079c8e858dc882713596364ef34d1425d5`。

两侧各自使用原来的 K 特征和 ridge 评分代码。输入直接从原始 dense logits
计算 posterior，再按各自动作传入已付费窗口，不从后续 teacher-forced K 行反推
offset2 特征。逐 block 最大分数绝对差为 `2.01148e-6`，K 特征最大绝对差为
`8.49366e-7`；以 `1e-4` 分数容差验收，动作要求精确一致。阈值为各侧严格过去
分数的中位数；完整轨迹见 `trace_full_enriched.jsonl.gz`，其中逐行记录样本、block、
coverage/bonus 窗口索引、双方分数、双方阈值、token 和动作。

发布模型 SHA-256 为
`116996aef99b4177fb6159086eba6a9ca90f03df6e61e8974a50a760922221e0`，
与 2026-09-25 冻结审计报告和本验收代码中的固定身份一致。验收现场用当前五折系数
重新计算全部 183,401 行 K OOF 预测，与原冻结 OOF 数组相比最大绝对误差
`2.38814e-7`、RMSE `1.01892e-7`，并核对 fit dataset 和原 OOF 文件哈希。
此前验收只信任模型 JSON 内自报的误差；本次已补足现场计算。

## 范围与输入隔离

先核对 `fit_calibration_split.json` 中 `fit` 的 6,378 个名字与冻结 fit dataset
的 `sample_names` 完全一致。dense `train_logits.pkl` 是 train 的混存字典，
提取器只按该名单读取 fit 样本的 float32 logits payload；其他成员的 payload
通过字节长度直接跳过，不构造数组、不计算 posterior。脚本拒绝未见过的 pickle
opcode、重复或缺失样本。没有访问 dev/test 视频、标注、logits、预测或 outcome，
没有重跑 dev，也没有 GPU 推理。

`full_report.json`、`model_validation.json`、`final_acceptance.json` 是机器可读
验收记录；后者含脚本、轨迹、冻结模拟、控制器、模型及 split 的 SHA-256。

## 复现

从 `SLRT/Online/CSLR` 运行，使用已有 legacy Python 环境；输出文件使用独立目录。
重跑前将本目录中本次生成的 JSON/JSONL 产物移到新的空目录，或在新副本中运行，
因为脚本拒绝覆盖原始轨迹。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  outputs/controller_independent_acceptance_20260927/independent_acceptance.py --limit 4
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  outputs/controller_independent_acceptance_20260927/independent_acceptance.py
python3 outputs/controller_independent_acceptance_20260927/finalize_acceptance.py
```

## 剩余限制

本结果是已消费 train-fit 样本的工程一致性验证，不是未见视频泛化证据；
没有评价识别 WER、前端视频接入或端到端延迟。发布模型仍是原冻结五折模型，
没有调参或进行新算法搜索。另一任务负责视频入口与延迟验证；本任务未修改其代码或产物。
