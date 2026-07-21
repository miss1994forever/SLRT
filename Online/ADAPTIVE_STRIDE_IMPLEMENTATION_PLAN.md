# Online CSLR 自适应步长实施方案

> 实施状态（2026-07-16）：核心自适应采样、时间跨度加权投票、离线脚本与实时后端接口已经完成；默认保持关闭。调参前 span-13 历史完整对照中，Phoenix test 的 clip 减少 32.02%、运行时间减少 28.84%、WER 从 22.00 变为 22.73；该结果见 `code_agent_logs/2026-07-16/adaptive_stride_full_comparison.md`。随后只用 dev 冻结的 span-15 最终 test WER 为 23.0571，见 `code_agent_logs/2026-07-16/adaptive_stride_frozen_final_evaluation.md`。实现记录见 `code_agent_logs/2026-07-15/adaptive_stride_implementation.md`。会话级增量缓存仍待执行。

## 1. 目标与范围

本方案用于在当前工作区重新实现 Online CSLR 的自适应步长改进，并将其接入 Sign2Text iOS App 的服务端推理链路。

本轮仅给出实施设计，不修改推理、服务端或 iOS 业务代码。后续实现应满足：

- 在手势快速运动时使用较小步长，保留动作边界；
- 在停顿或长持动作时使用较大步长，减少重复 clip；
- 自适应采样后，解码投票按真实帧/时间跨度进行，避免把不等间隔的 clip 当成等间隔序列；
- 低置信度时可回退到更保守的解码结果；
- 固定步长仍是可随时启用的基线和故障回退路径；
- 训练权重保持不变，改动限定在推理采样、后处理、配置和可观测性。

## 2. 资料依据与目标基线

参考资料：

- 工作区根目录 `srtp结项答辩.pptx`；
- 工作区根目录 `项目成果.docx`；
- `SLRT/Online/CSLR/prediction_slide.py`；
- `sign2text.app/backend/sign2text-ml/app/runtime.py` 及会话推理链路。

成果材料记录的 Phoenix 测试集结果如下：

| 项目 | 记录值 |
| --- | ---: |
| 测试样本数 | 642 |
| 固定步长（stride=1）clip 数 | 64,627 |
| 自适应步长 clip 数 | 39,353 |
| clip 数节省 | 39.11% |
| 吞吐提升（以 clip 数估算） | 约 1.64 倍 |
| 自适应步长 + 跨度加权投票，`window_greedy_13` WER | 24.18 |
| 固定 stride=1 基线 WER | 约 21.86 |

上述数字应作为复现目标，而不是未经复测直接视为当前仓库的验收结果。尤其是成果报告末尾“最优精度配置”一句存在表述歧义：24.18 是自适应方案结果，而固定基线记录为 21.86，后续报告须分别列出，不能混写。

成果材料记录的目标配置为：

```yaml
data:
  stride: 1
  adaptive_stride:
    enabled: true
    min_stride: 1
    max_stride: 3
    confidence_threshold: 0.2
    ema_decay: 0.4
    quantile_low: 0.2
    quantile_high: 0.7

postprocess:
  span_weighted_voting:
    enabled: true
    kernel: triangular
    min_weight: 0.05
```

最佳固定解码窗口记录为 `window_greedy_13`；动态回退实验的最佳候选记录为 `7 -> 13`，但未超过固定使用 13 的精度。

## 3. 当前代码审计

### 3.1 原始固定步长链路

`prediction_slide.py` 的核心流程是：

1. 按 `win_size` 和固定 `stride` 生成 clip；
2. 分批运行 ISLR；
3. 按置信度阈值筛选预测；
4. 使用 `window_greedy_{3,5,7,9,11,13}` 按预测数组索引做多数投票；
5. 去重、去 blank 后计算 WER。

`runtime.py` 在实时请求中读取同样的 `win_size` 和 `stride`，每次对会话缓冲区重建 tensor 并重新滑窗。它当前仍调用固定步长接口，默认返回 `window_greedy_7`。

### 3.2 已存在但未完成的自适应原型

当前 `SLRT/Online/CSLR/prediction_slide.py` 已有一版工作区未提交改动：

- `_compute_frame_velocity`：按关键点相邻帧二维位移估计速度；
- `_ema_1d`：对速度做 EMA；
- `_adaptive_window_starts`：使用 `round(alpha / (velocity + epsilon))` 并裁剪到最小/最大步长；
- `sliding_windows(..., adaptive_cfg=...)`：可以接受非等间隔起点；
- 命令行中已有 `adaptive_*` 参数。

这部分可复用，但尚不能视为成果报告中方案的完整实现：

- 参数名和算法不一致：代码使用 `enable/alpha/epsilon/ema_beta/conf_thr`，材料使用 `enabled/ema_decay/quantile_low/quantile_high/confidence_threshold`；
- 当前反比公式对不同视频坐标尺度敏感，且没有分位数归一化；
- 速度默认聚合所有关键点，没有明确限制到手部与手臂；
- 窗口起点没有随预测结果保留到后处理阶段；
- 解码仍按“相邻数组元素”投票，没有按真实时间跨度加权；
- 没有实现置信度门控的 `7 -> 13` 回退；
- 实时 `runtime.py` 未传入自适应配置；
- API 响应只报告单一 `stride`，无法反映实际 clip 起点和节省量；
- 会话推理会重复处理整个缓冲区，自适应采样减少单次 clip 数，但尚未消除跨请求重复计算。

## 4. 算法设计

### 4.1 输入与时间基准

离线评估以帧序号为时间基准；实时服务优先使用上传帧的 `timestampMs`，缺失或异常时再按帧序号和约定 FPS 回退。这样可处理摄像头实际帧率抖动。

关键点速度只使用模型配置中可用的手部和上肢点。建议优先级为：

1. 左右手关键点；
2. 左右腕、肘、肩；
3. 若以上集合不可用，回退到当前输入的所有有效关键点；
4. 有效点不足时不做激进跳步，强制 `min_stride`。

坐标应先按画面宽高或躯干尺度归一化，再计算位移，防止分辨率和人物远近改变速度量级。

### 4.2 运动分数

对帧 `t` 定义有效关键点集合 `V_t`，仅保留前后两帧置信度均不低于 `confidence_threshold` 的点：

```text
raw_speed[t] = robust_mean(
    distance(normalized_xy[t, k], normalized_xy[t-1, k]) / delta_time,
    k in V_t
)
```

`robust_mean` 建议使用截尾均值或中位数，降低 MediaPipe 单点跳变、遮挡和零填充的影响。之后使用 EMA：

```text
speed_ema[t] = ema_decay * speed_ema[t-1]
             + (1 - ema_decay) * raw_speed[t]
```

注意：需通过小型单元测试确认旧服务器中 `ema_decay=0.4` 表示“历史项权重”还是“新观测权重”。当前方案默认它表示历史项权重。

### 4.3 分位数标定与步长映射

对每个视频（离线）或会话滚动历史（实时）计算：

```text
v_low  = quantile(speed_ema, quantile_low)
v_high = quantile(speed_ema, quantile_high)
motion = clip((speed_ema - v_low) / max(v_high - v_low, eps), 0, 1)
stride = round(max_stride - motion * (max_stride - min_stride))
```

因此：

- `speed <= v_low` 时使用 `max_stride`；
- `speed >= v_high` 时使用 `min_stride`；
- 中间区域线性、反向映射；
- 关键点质量不足、时间戳倒退、速度为 NaN 或标定区间退化时使用 `min_stride`。

实时冷启动时不应使用未来帧计算全局分位数。建议维护最近 1～3 秒运动分数的滚动分位数；历史不足时使用 `min_stride`，达到最小样本数后才启用自适应。离线复现也应增加“因果模式”，确保评估不偷看未来，并分别报告非因果全视频标定与因果滚动标定结果。

生成起点时必须保证：

- 起点严格递增且无重复；
- 首个起点覆盖序列开头；
- 尾部至少生成一个能够覆盖最后真实帧的窗口；
- padding 逻辑不因非等间隔起点而移动窗口的真实时间语义；
- 返回每个 clip 的 `start_frame/end_frame/center_time/stride/mean_speed/keypoint_quality` 元数据。

### 4.4 按真实时间跨度投票

固定索引窗口（例如前后各 6 个预测）在自适应采样下代表的时间长度不固定。后处理应改为：以当前 clip 中心时刻为中心，选择固定帧跨度或毫秒跨度内的邻居，并使用三角核加权。

```text
distance = abs(neighbor_center - current_center)
weight   = max(min_weight, 1 - distance / vote_radius)
score[c] = sum(weight_i * probability_i[c])
```

优先对类别概率求加权和，再取 `argmax`；只有为严格复现旧实验时才提供“对 top-1 标签加权计数”的兼容模式。边界处不应通过复制首尾预测制造虚假时间覆盖，可直接对实际邻居重新归一化权重。

为了与 `window_greedy_13` 对齐，配置层应明确 `13` 的含义：

- 固定步长基线：13 个 clip；
- 自适应模式：建议定义为 stride=1 时 13 个 clip 对应的等价帧跨度，并命名为 `vote_span_frames: 13`，避免误解为 13 个非等间隔 clip。

### 4.5 置信度门控回退

保留材料中的 `7 -> 13` 实验路径，但默认生产方案仍使用已记录更稳定的 13：

1. 先计算较小跨度 7 的结果；
2. 若加权后最大类别概率低于阈值、top-1/top-2 margin 过小，或有效邻居不足，则用跨度 13 重算；
3. 若跨度 13 仍不可靠，则输出 blank/保持上一稳定 partial，不强行提交新 gloss；
4. 回退阈值与关键点置信度阈值分开命名，例如 `decode_confidence_threshold` 与 `keypoint_confidence_threshold`。

不要复用当前 `prob_thr <= 0.2` 的特殊分支语义作为回退判据；应将 clip 筛选和解码回退拆成两个可单测的步骤。

## 5. 推荐配置结构

统一离线脚本与实时服务的字段名，并支持显式兼容旧名称：

```yaml
data:
  win_size: 16
  stride: 1                  # 关闭 adaptive 时的基线值
  adaptive_stride:
    enabled: false           # 默认关闭，完成复测后再切生产默认值
    min_stride: 1
    max_stride: 3
    keypoint_confidence_threshold: 0.2
    ema_decay: 0.4
    quantile_low: 0.2
    quantile_high: 0.7
    calibration_mode: causal_rolling
    calibration_window_frames: 48
    warmup_frames: 16
    insufficient_keypoints_fallback: min_stride

postprocess:
  span_weighted_voting:
    enabled: true
    vote_span_frames: 13
    kernel: triangular
    min_weight: 0.05
    use_probabilities: true
  confidence_fallback:
    enabled: false
    primary_span_frames: 7
    fallback_span_frames: 13
    decode_confidence_threshold: null  # 在 dev 集标定，不能沿用关键点阈值
```

启动时应校验：步长为正整数、`min <= max`、分位数满足 `0 <= low < high <= 1`、EMA 在 `[0,1)`、权重为正，并记录最终生效配置。

## 6. 实施步骤

### 阶段 A：冻结基线与补齐测试夹具

1. 保存当前工作区状态，避免覆盖已有未提交修改；实现时只改自适应相关文件。
2. 使用 Phoenix 测试集和固定 checkpoint 复跑 stride=1 基线，记录 commit、配置、环境和随机种子。
3. 固化若干短 tensor 作为测试夹具：静止、匀速、速度突变、遮挡、零置信度、时间戳抖动、短于窗口。
4. 记录 baseline clip 数、WER 分项、GPU 时间、峰值显存和端到端延迟。

产物：可重复的 baseline 命令、结果文件和测试夹具说明。

### 阶段 B：抽离自适应采样模块

1. 将速度估计、EMA、滚动分位数、步长映射和窗口起点生成从 `prediction_slide.py` 抽到独立模块，例如 `utils/adaptive_stride.py`。
2. 使用 dataclass/字典 schema 统一参数，并兼容读取当前原型的旧字段，输出弃用警告。
3. 让 `sliding_windows` 同时返回窗口 tensor 和时间元数据；固定步长也走同一元数据结构。
4. 保证关闭开关后结果与原始固定步长逐元素一致。

产物：纯函数模块、单元测试、固定模式兼容性测试。

### 阶段 C：实现跨度加权投票与回退

1. 把现有重复的解码逻辑抽成独立后处理模块。
2. 增加基于 clip 中心帧/时间戳的邻居选择和三角核概率加权。
3. 增加可选的 `7 -> 13` 置信度回退，但默认关闭，先做消融实验。
4. 所有阈值筛选后同步过滤 clip 元数据，杜绝 logits 与时间轴错位。
5. 结果文件中保存采样起点、动态步长、运动分数、权重和回退原因。

产物：解码单元测试、固定投票与跨度投票对比结果。

### 阶段 D：离线复现与参数标定

按以下矩阵逐步实验，不同时打开多个新变量：

| 实验 | 采样 | 投票 | 回退 |
| --- | --- | --- | --- |
| A0 | stride=1 | 原始索引投票 13 | 关 |
| A1 | adaptive 1～3 | 原始索引投票 13 | 关 |
| A2 | adaptive 1～3 | 跨度加权 13 | 关 |
| A3 | adaptive 1～3 | 跨度加权 7 | `7 -> 13` |
| A4 | stride=2/3 | 对应固定基线 | 关 |

先在 dev 集选择参数，再对 test 集只运行一次最终配置。至少报告 WER、DEL/INS/SUB、clip 数、clip 节省率、wall time、GPU time、峰值显存、平均和 P95 延迟。固定 stride=2/3 基线很重要，它能判断收益来自“自适应”还是仅来自总体降采样。

### 阶段 E：接入实时后端

1. `runtime.py` 传入自适应配置并消费窗口元数据；最佳解码方法由配置决定，不再硬编码为 7。
2. `SessionState` 增加每会话的采样状态：上次处理帧、EMA、滚动速度历史、下一候选窗口起点、已提交 gloss。
3. 首版可以保留“每次重跑整个缓冲区”以验证正确性；第二步改为只推理新产生的 clip，真正兑现实时算力节省。
4. 会话 deque 淘汰旧帧时同步清理关键点和采样状态，防止内存持续增长及帧索引错位。
5. API metadata 增加 `adaptiveStrideEnabled`、`clipCount`、`fixedClipCountEstimate`、`effectiveMeanStride`、`fallbackCount` 和延迟统计。
6. 异常时自动退回固定 stride=1，并在响应 notes/日志中说明原因。

产物：离线/在线同输入一致性测试、会话增量测试、API schema 测试。

### 阶段 F：iOS 联调与灰度启用

iOS 端原则上无需执行步长算法，但必须稳定上传单调 `frameIndex`、`timestampMs`、画面尺寸及关键点置信度。联调时覆盖正常速度、停顿、快速手势、双手遮挡、背光和丢帧场景。

先通过服务端环境变量或配置开关灰度启用；当关键点质量、WER 和延迟监控达到验收条件后，再考虑设为默认。

## 7. 测试计划

### 7.1 单元测试

- 静止序列趋向 `max_stride`；高速序列趋向 `min_stride`；
- 速度单调增加时步长不应反向增大（允许整数取整平台）；
- 全零/NaN/低置信度关键点安全回退到 `min_stride`；
- 不同分辨率下相同归一化运动得到近似相同步长；
- 起点严格递增、首尾覆盖正确、clip tensor 长度恒为 `win_size`；
- 等间隔且权重均匀时，新投票与旧投票兼容；
- 非等间隔合成样例能验证真实跨度选邻居；
- 置信度低时触发 7 到 13，高时不触发；
- 关闭全部新开关时固定基线输出不变。

### 7.2 集成与回归测试

- `prediction_slide.py` 在 Phoenix dev/test 上完整运行并生成原有结果文件；
- 同一 tensor 经离线脚本与 `OnlineCSLRRuntime` 输出一致；
- 多次 `/infer` 只新增必要 clip，不重复提交相同 gloss；
- 缓冲区达到 96 帧并淘汰旧帧后仍正确运行；
- MediaPipe 关键点短时缺失时不产生超大步长；
- 自适应模块异常时固定 stride=1 回退可用。

## 8. 验收标准

建议分为“算法复现”和“App 可用”两级：

### 算法复现

- 固定模式与改造前的 WER、预测序列和 clip 数一致；
- Phoenix 642 个测试样本的自适应 clip 数接近历史 39,353，并解释允许偏差的来源；
- clip 节省目标约 39.11%，同时完整报告 WER 代价；
- 能复现或合理解释历史自适应 `window_greedy_13` WER 24.18；
- 所有配置和中间元数据可追溯。

### App 可用

- 快速动作不因错误跳步产生明显边界丢失；
- 停顿/长持阶段 clip 生成频率显著下降；
- 在目标服务器上 P95 推理延迟和 GPU 使用量相对 stride=1 有量化改善；
- 遮挡、暗光、关键点缺失时优先保守采样或回退，不崩溃；
- 可通过一个配置开关立即恢复固定 stride=1。

## 9. 风险与待确认项

1. **历史实现细节不完整。** 材料给出了参数和结果，但没有给出分位数映射、置信度含义、时间跨度半径的精确定义。若能从另一服务器取回旧 commit、diff、配置和结果 pickle，应在编码前做一次对照；否则按本方案实现并明确标注为“依据材料重建版”。
2. **当前文件含未提交改动。** `prediction_slide.py` 以及若干 SLRT 文件已经被修改，实施时必须保留这些工作，不做 reset/checkout。
3. **关键点尺度和抖动。** 未归一化位移会受分辨率、人物远近影响；遮挡跳变可能被误判为快速手势，导致算力增加。鲁棒聚合和质量门控不可省略。
4. **离线与实时分位数差异。** 全视频分位数使用未来信息，不能直接代表实时行为；必须优先验收因果滚动版本。
5. **当前服务不是严格增量推理。** 若每次请求都重跑整个 96 帧缓冲区，单次 clip 节省不等于真实端到端吞吐提升；增量缓存应作为上线前任务。
6. **精度—效率权衡。** 历史结果表明节省 39.11% clip 的同时 WER 上升约 2.32 个百分点。是否可接受应由产品目标决定，并通过固定 stride=2/3 基线判断自适应机制的净收益。

## 10. 推荐落地顺序

建议按以下顺序执行：

1. 冻结并复跑 stride=1 基线；
2. 抽离并测试自适应起点生成器；
3. 保存时间元数据并实现跨度加权投票；
4. 在 Phoenix dev 上标定，在 test 上复现历史结果；
5. 接入后端但先保持非增量模式，验证离线/在线一致；
6. 实现会话级增量推理和可观测指标；
7. 最后联调 iOS 并灰度开启；
8. `7 -> 13` 动态回退作为独立消融项，不阻塞主方案上线。

这一顺序能先验证算法结论，再处理实时系统状态，出现精度或性能回归时更容易定位责任层。
