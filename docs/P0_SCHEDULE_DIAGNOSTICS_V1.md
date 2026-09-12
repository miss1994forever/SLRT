# Phoenix P0 调度诊断 v1（冻结）

更新时间：2026-09-12

本文冻结 `NEXT_RESEARCH_HANDOFF.md` 规划的、只使用修复后 Phoenix-2014T dev 的 P0
调度诊断。机器索引为
`docs/results/phoenix_p0_schedule_diagnostics_v1_frozen.json`。若后续日志与本文冲突，以
本文、机器索引、各实验 `protocol_manifest.json` 和 `aggregate/dev_summary.json` 为准。

## 1. 冻结范围与声明边界

- 数据：修复后 Phoenix-2014T dev，519 个样本、3,747 个清洗后 reference gloss；
- 视频 ZIP SHA-256：`49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`；
- checkpoint SHA-256：`b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`；
- 基线 manifest SHA-256：`2e6889ec14777d938a3c7ac9265ab97f87118523e9258f5f93beba4b3e854b8f`；
- 模型输出：复用修复后 B0 stride-1 dense logits；P0 没有重新训练或运行模型；
- test：未读取、未运行，也未用于任何选择；
- 主坐标：固定 B0 padded-array coordinate，constant left padding 7；
- 主预算：逐样本严格匹配指定 uniform/A0 预算；
- 默认窗口/解码：centered 16-frame、triangular span-weighted-15、minimum weight 0.05。

P0 中的 alignment、boundary 和 center 都是 **label/alignment-derived offline proxy**，不是
人工 ground truth，也不是可部署预测。P0 的 oracle WER 只能用来判断上界，不能写成新方法
结果。

## 2. Frozen baseline reference

| ID | WER | errors | clips | 角色 |
|---|---:|---:|---:|---|
| B1 fixed1 span15 | 22.604750% | 847 | 55,775 | dense 同解码参考 |
| B2 uniform span15 | 22.551374% | 845 | 37,706 | 冻结等预算参考 |
| A0 adaptive span15 | 22.417934% | 840 | 37,615 | 冻结 motion-only 方法 |

A0 相对 B2 只少 5 errors，paired bootstrap CI 跨 0；P0 不改变这一冻结结论。

## 3. Dense replay 与坐标审计

P0 首先证明 B0 dense logits 可以重放不同 schedule：

| Replay | WER | errors | hypotheses 与原运行一致 |
|---|---:|---:|---:|
| B1 | 22.604750% | 847 | 519 / 519 |
| B2 | 22.551374% | 845 | 519 / 519 |
| A0 | 22.417934% | 840 | 519 / 519 |

发现的关键陷阱是：旧 `sliding_windows()` 根据每个 schedule 的最后一个 start 计算 centered
padding。A0 有 82/519 个样本相对 B0 的 left padding 少 1 帧；必须按物理输入映射到 B0
dense row。直接按同名 numeric start 抽取会错误得到 `22.337870%`，形成虚假改善。

因此后续 P0 统一采用不依赖最终 schedule 的固定 B0 坐标。该修正不把当前管线升级为严格
streaming：centered window 仍含 lookahead，span-15 仍使用未来邻域，已知长度 pacing 仍不是
未知 EOS 的顺序执行。

## 4. Random sampling：No-Go

主协议为每个样本严格匹配 A0 clip 数，保留时间轴两端，30 个预声明 seeds：

- WER mean/std：`23.212348 ± 0.275634%`；
- range：`22.658127--23.779023%`；
- errors mean/range：`869.77` / `849--891`；
- 30/30 seeds 全部差于 B2 和 A0；
- 最好的 random 仍比 B2 多 4 errors，比 A0 多 9 errors。

仅匹配全 dev 总预算的次协议更差：WER `23.273730 ± 0.270076%`。结论是时间覆盖结构
重要，但 random 结果不能证明 A0 显著优于 uniform。

## 5. Prediction-change：No-Go

固定 half-ceiling uniform skeleton，把其余预算分配给 top-1 change、blank transition、JS
divergence、entropy increase 或 margin drop。分别测试完整序列排序和 past-only score + known
length pacing。

- 同预算固定坐标 uniform：`22.631438%`，848 errors；
- 最优 past-only top-1 change：`22.631438%`，848 errors；
- 最优 offline JS divergence：`22.738191%`，852 errors；
- 10 个预声明策略均未显示相对 uniform 的可利用上界。

该 No-Go 仅适用于已注册的 skeleton/event 设计与当前 decoder，不外推为“预测变化永远
无用”。

## 6. Gloss boundary proxy：No-Go

使用 center-label 与 PAMI0/1/2 派生的 start、end、start+end，合计 12 个策略。主
center-label 覆盖 519/519，但上游 provenance 不完整；52 个视频存在相邻 nonblank segment
重叠。PAMI 三套边界数量差异很大，语义未被本地文件证明。

- 同预算 fixed-coordinate uniform：`22.631438%`，848 errors；
- 事后最优 boundary proxy：`22.578062%`，846 errors；
- 改善仅 `0.053376 pp` / 2 errors，Bonferroni CI 跨 0；
- 所有候选仍差于 B2 和 A0；
- 几乎所有策略的 ±2/±4 帧边界 recall 已饱和到 1.0；
- 多个 end-only 策略显著恶化。

结论：在当前预算、centered-16 和 span-15 下，边界加密没有显示值得训练 boundary head 的
上界。

## 7. Sign-center proxy：offline Strong-Go

由于 ISLR 使用 isolated/sign-centered 样本训练，P0 进一步把 bonus 分配到 nonblank segment
midpoint 或中央 50% 区域，并联合检查预算与 decoder。

最重要的预声明主结果：

| 50% dense budget + span15 | WER | DEL / INS / SUB | errors | clips |
|---|---:|---:|---:|---:|
| Uniform | 23.058447% | 11.769416 / 3.442754 / 7.846277% | 864 | 28,014 |
| Center midpoint proxy | 21.537230% | 9.874566 / 2.882306 / 8.780358% | 807 | 28,014 |
| Difference | **-1.521217 pp** | — | **-57** | 0 |

paired sentence bootstrap 10,000 次、Bonferroni-9 CI 为
`[-2.416242,-0.626285] pp`。PAMI2 同方向，PAMI0/PAMI1 在该 cell 反向。另一个通过
Strong-Go 的主 cell 是 33% dense + span7，relative gain `-2.161729 pp`，但其 absolute WER
为 `24.472912%`，不优于 50% + span15。

这证明存在“把计算放到 sign interior”产生的 label-aware offline 上界，不证明已有可部署
scheduler，也不允许把 `21.537230%` 当作正式 dev 成绩。

## 8. Noisy center：进入训练前的门槛

固定 50% dense + span15 主 cell，对 center proxy 加定位偏移、随机 jitter、漏检和误检：

- 固定 offset：`-2/-1/0` 帧保持 Strong-Go；`+1` 帧起不再满足；
- 随机 jitter：只有 `±1` 帧稳定满足 Strong-Go，`±2` 的校正 CI 刚跨 0；
- recall：`75%` 仍满足，`50%` 不再稳定；
- false centers：额外 `100%` 的跨 seed 平均仍边缘满足，但校正上界仅 `-0.0099 pp`；
- 组合压力 `±4 + 75% recall + 50% FP`：WER `23.331554%`，比 uniform 差
  `0.273107 pp`，No-Go。

系统对单一漏检/误检有容忍度，但对正延迟和组合误差脆弱。未来 predictor 的研究目标是：

1. 在当前/历史视觉中预测“高辨识度 sign interior”，而不是等 segment 结束后计算 midpoint；
2. 主要定位误差约 `±1` 帧；
3. center recall 至少约 75%；
4. 避免系统性正延迟；
5. 先报告 center detection，再连接 scheduler。

## 9. 冻结决策

| 方向 | P0 决策 | 下一步 |
|---|---|---|
| Random scheduling | No-Go | 不再扩 seed 或调随机规则 |
| Motion-only 微调 | No-Go | A0 v1 保持冻结 |
| Prediction-change scheduling | No-Go | 不训练对应 scheduler |
| Boundary-only scheduling | No-Go | 不训练 boundary head |
| Sign-center / interior | Offline Strong-Go | 进入 train-only predictor 可行性 |

下一阶段不得直接用 dev proxy 训练，也不得运行 test。先审计 train center-label 的生成、覆盖和
非有限关键点影响；建立 train-only label/feature 管线；冻结 calibration/evaluation 规则；再在
dev 上比较 detection 与等预算 scheduler。若 learned predictor 无法接近 P0 的定位/recall 门槛，
停止该方向并转向 decoder/window 或模态计算门控。

## 10. 权威产物

六组实验均位于：

`Online/CSLR/results/phoenix-2014t_ISLR/p0_<name>_v1_49faacc3/`

提交只保留每组的 `resolved_config.json`、`protocol_manifest.json` 和
`aggregate/dev_summary.json`。逐样本 JSONL 与 dense logits 继续由 `.gitignore` 排除，不能作为
唯一结论来源。

对应工具与测试：

- `tools/analyze_phoenix_schedule_replay.py` / `tests/test_schedule_replay.py`；
- `tools/analyze_phoenix_random_schedule.py` / `tests/test_random_schedule_replay.py`；
- `tools/analyze_phoenix_structured_oracles.py` / `tests/test_structured_oracles.py`；
- `tools/analyze_phoenix_boundary_proxy_oracles.py` / `tests/test_boundary_proxy_oracles.py`；
- `tools/analyze_phoenix_sign_center_oracles.py` / `tests/test_sign_center_oracles.py`；
- `tools/analyze_phoenix_noisy_sign_center.py` / `tests/test_noisy_sign_center.py`。

冻结时 43 项相关 CPU tests 通过，所有 JSON 有效，manifest 输入 hash 匹配，`git diff --check`
通过。P0 没有使用 GPU、没有读取 test、没有覆盖 A0 v1。
