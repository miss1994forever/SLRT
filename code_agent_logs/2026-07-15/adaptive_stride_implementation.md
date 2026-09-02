# 2026-07-15 Online CSLR 自适应步长实现日志

## 目标

依据 `docs/experiments/adaptive_stride/implementation_plan.md` 实现推理侧自适应步长优化。实现不修改模型结构、训练逻辑或 checkpoint；所有新增功能默认关闭，确保现有固定 stride=1 部署不会自动改变。

## 实现选择

没有照搬成果报告中的旧实现，也没有继续使用当前原型的 `alpha / velocity` 反比公式。最终采用：

- 关键点坐标按当前相邻帧的空间范围归一化；
- 使用有效关键点位移的中位数抑制孤立跳点；
- 低置信度、NaN 或有效关键点不足时保守使用 `min_stride`；
- 使用因果 EMA，不读取未来帧；
- 使用最近一段历史的因果滚动分位数做视频内速度标定；
- 将归一化运动量反向映射到整数 stride 1～3；
- 非等间隔窗口使用真实中心帧距离和三角核对类别概率加权；
- 固定步长和旧配置字段保持兼容。

## 修改文件

### SLRT

- `SLRT/Online/CSLR/utils/adaptive_stride.py`
  - 新增 `AdaptiveStrideConfig` 和配置校验；
  - 新增鲁棒关键点速度估计；
  - 新增因果 EMA 与滚动分位数映射；
  - 新增动态窗口起点及诊断 metadata；
  - 新增真实时间跨度三角核概率投票。
- `SLRT/Online/CSLR/prediction_slide.py`
  - `sliding_windows` 接入动态起点并可返回 metadata；
  - 阈值过滤时同步过滤窗口中心时间；
  - 增加 `span_weighted_N` 解码结果与 WER 评估；
  - 保存每个样本的步长、速度、运动量和关键点质量；
  - 配置文件字段和 CLI 参数均可控制新功能；
  - 修复高阈值下没有任何窗口通过筛选时的安全回退；
  - 默认关闭新功能，固定步长路径保持不变。
- `SLRT/Online/CSLR/configs/slide_phoenix-2014t.yaml`
- `SLRT/Online/CSLR/configs/slide_csl-daily.yaml`
  - 增加默认关闭的 `adaptive_stride` 和 `span_weighted_voting` 配置。
- `SLRT/Online/CSLR/tests/test_adaptive_stride.py`
  - 增加 5 个自适应采样与跨度投票单元测试。

### Sign2Text 后端

- `sign2text.app/backend/sign2text-ml/app/runtime.py`
  - 实时 tensor 推理接入同一自适应滑窗模块；
  - 解码阶段接入窗口中心时间和跨度加权投票；
  - 未启用跨度投票时默认使用现有测试中更稳的 `window_greedy_13`；
  - 返回 `adaptiveStrideEnabled`、`clipCount` 和 `effectiveMeanStride` 诊断字段。
- `sign2text.app/backend/sign2text-ml/configs/slide_csl-daily_runtime.yaml`
  - 增加默认关闭的运行时配置。

## 推荐参数

```yaml
data:
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

postprocess:
  span_weighted_voting:
    enabled: true
    vote_span_frames: 13
    kernel: triangular
    min_weight: 0.05
```

CLI 临时启用方式：

```bash
python prediction_slide.py \
  --config configs/slide_phoenix-2014t.yaml \
  --adaptive_stride 1 \
  --span_weighted_voting 1 \
  --vote_span_frames 13
```

## 验证结果

### 单元测试

运行：

```bash
/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  -m unittest discover -s tests -p 'test_adaptive_stride.py' -v
```

结果：5/5 通过。

覆盖项：

- 关闭自适应时生成 stride=1 起点；
- 静止动作在 warmup 后采用较大 stride；
- 低置信度关键点保守回退 stride=1；
- 跨度投票按真实时间距离选邻居；
- 非法分位数配置启动即报错。

### 静态检查

- `prediction_slide.py`、`adaptive_stride.py`、后端 `runtime.py`：`py_compile` 通过；
- 三个修改后的 YAML：解析通过；
- `prediction_slide.py --help`：启动和新增 CLI 参数解析通过；
- 合成 tensor 固定滑窗兼容 smoke：通过，10 帧输入仍生成 10 个 stride=1 窗口。

仓库全量 `git diff --check` 仍报告 `Online/CSLR/training.py:411` 的既有行尾空格；该文件不是本次修改范围，未擅自改动。

### Phoenix 真实关键点采样 smoke

仅运行自适应采样器，不加载模型和 GPU。输入：Phoenix test 全部 642 个样本。

| 指标 | 结果 |
| --- | ---: |
| 固定 stride=1 窗口 | 64,627 |
| 新自适应窗口 | 42,301 |
| 预计窗口减少 | 34.55% |
| 等效平均步长 | 1.528 |
| stride=1 使用次数 | 27,746 |
| stride=2 使用次数 | 6,386 |
| stride=3 使用次数 | 8,169 |

固定窗口总数与历史材料中的 64,627 完全一致。新算法已确认会在真实关键点上动态使用 1/2/3，而不是像旧反比原型一样几乎始终使用 stride=1。

上述 34.55% 是 clip 数代理指标，不是完整 GPU 性能结果。尚未运行完整模型，因此本日志不声称 WER、wall time 或显存已经改善。

## 安全与兼容性

- 配置默认 `enabled: false`，上线不会自动改变当前结果；
- 关闭开关后仍使用固定步长原逻辑；
- checkpoint 只读，不会被修改或重新保存；
- 兼容旧字典字段 `enable`、`conf_thr`、`ema_beta`；
- 异常或关键点质量不足时偏向 stride=1，而不是跳过更多帧；
- 未覆盖工作区原有未提交改动。

## 尚未完成

- Phoenix/CSL-Daily 完整模型固定与自适应 WER 对照；
- GPU wall time、P95 延迟、显存和真实吞吐测量；
- `7 -> 13` 置信度门控回退；
- 后端跨 `/infer` 请求的会话级增量特征缓存；
- iOS 实机在遮挡、暗光和丢帧情况下的联调。

在完整 dev 集评估前保持默认关闭。建议下一步先运行 Phoenix dev 的 A0 固定基线和 A2 自适应 + span-weighted-13，再决定是否调整分位数或设为生产默认值。

## 真实模型 GPU smoke（追加）

### GPU 安全选择

运行前通过 PCI Bus ID 核对全部 GPU：

- 故障卡：逻辑 GPU 1，UUID `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83`，PCI `25:00.0`；全程排除；
- 本次测试卡：逻辑 GPU 3，UUID `GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed`，PCI `61:00.0`；
- 使用 UUID 设置 `CUDA_VISIBLE_DEVICES`，避免逻辑编号变化导致误用故障卡；
- 运行前完成 CUDA 2048×2048 矩阵乘法和同步检查；
- 运行结束后 PCI `61:00.0` 温度 27°C、显存已释放为 0 MiB。

### 测试范围

- 数据：Phoenix-2014T test 前 5 个样本；
- 模型：`results/phoenix-2014t_ISLR/ckpts/best.ckpt`，epoch 92；
- 固定组：stride=1，原始窗口投票；
- 自适应组：stride 1～3，因果分位数采样，`span_weighted_13`；
- 两组均执行真实 RGB + keypoint 模型前向，并保存 result/evaluation/logits 文件。

### 结果

| 指标 | 固定 stride=1 | 自适应 + span-weighted-13 |
| --- | ---: | ---: |
| 样本数 | 5 | 5 |
| clip 数 | 770 | 521 |
| 平均 clip/样本 | 154.0 | 104.2 |
| clip 减少 | — | 32.34% |
| stride=1 次数 | 770 | 353 |
| stride=2 次数 | 0 | 81 |
| stride=3 次数 | 0 | 87 |
| 本组最佳 WER | 12.50 (`window_greedy_11`) | 12.50 (`span_weighted_13`) |
| span-weighted-13 DEL | — | 6.25 |
| span-weighted-13 INS | — | 0.00 |
| span-weighted-13 SUB | — | 6.25 |

固定组 `window_greedy_13` WER 为 18.75；固定组最佳是 `window_greedy_11` 的 12.50。自适应条件下继续使用普通索引 `window_greedy_13` 会恶化到 52.08，而按真实时间跨度计算的 `span_weighted_13` 恢复到 12.50，验证了非等间隔采样必须配套时间跨度投票这一设计。

结果文件：

- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_fixed_smoke5/test/`
- `SLRT/Online/CSLR/results/phoenix-2014t_ISLR/prediction_slide_adaptive_v2_span13_smoke5/test/`

该 smoke 只覆盖 5 个样本，不能用于宣称完整测试集 WER 不下降。它证明了 GPU 选择策略、checkpoint 加载、动态滑窗、真实时间投票、WER 计算和结果保存链路均可正常运行。下一步应在同一健康 GPU 上运行完整 dev 对照，再运行一次最终 test 配置。
