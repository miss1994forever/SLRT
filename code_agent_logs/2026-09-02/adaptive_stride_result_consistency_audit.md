# 自适应步长结果口径核对

## 结论

正式对外结果统一为 Phoenix-2014T 的 dev 冻结 span-15 配置：test WER 23.0571%，相对固定 stride=1 的 22.0005% 增加 1.0566 pp；clips 从 64,627 降至 43,932（-32.02%）；单次完整命令 wall time 从 1,547.29 秒降至 1,141.08 秒（-26.25%）。

调参前 span-13 的 test WER 22.7283%、wall time -28.84% 只作为历史工程对照，不再与 span-15 指标拼接。实施前材料中的 21.86/24.18/39.11% 没有纳入当前机器结果。

## 核对来源

- 固定 test：`prediction_slide_adaptive_v2_fixed_full642/test/`；
- 冻结 test：`prediction_slide_adaptive_dev_tuned_span15_final_test/test/`；
- 历史 span-13 test：`prediction_slide_adaptive_v2_span13_full642/test/`；
- Dev D0/D1/D2：对应 `prediction_slide_dev_tune_*_s16/dev/` 目录；
- 当前配置：`Online/CSLR/configs/slide_phoenix-2014t.yaml`；
- 正式启用参数：`scripts/reproduce/phoenix_adaptive_stride.sh`。

## 验证方法

- 从 `*_evaluation_results.pkl` 读取固定与冻结 test 的 WER 分项；
- 从 `*_results.pkl` 汇总逐样本 `adaptive_stride_metadata`，复算 clip 总数和 stride 1/2/3 计数；
- 从 D2 `dev_span_sweep.json` 读取离线预注册的 span-15 dev 指标；
- 复算 clips 减少率、wall-time 减少率、speedup 和 WER pp；
- 对照 YAML 的 max stride、EMA、分位数、warmup、span 和默认关闭开关；
- 校验统一 JSON 可解析、文档链接存在且 `git diff --check` 通过。

Runtime 不存在于 PKL 中，取自当次 `/usr/bin/time real` 审计记录；模型日志时间戳只能近似佐证。因此只称为单次完整命令 wall time，不称平均延迟或稳定 benchmark。

统一结果入口：`docs/RESULTS.md`；机器可读摘要：`docs/results/phoenix_adaptive_stride_summary.json`。
