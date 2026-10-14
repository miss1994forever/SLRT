# P7 ISLR 时间输入方案实验报告

日期：2026-10-12
状态：2026-10-12 的逐卡预检一度恢复，A/B/C 等预算短训随后启动。A 所用的 `41:00.0` 在首轮约
500 步时掉线；C 所用的 `e1:00.0` 随后在约 1305 步时同样掉线。A/C 均未产生 checkpoint，已判
无效并分别归档。只有 B（`61:00.0+81:00.0`）利用故障前已经建立的 CUDA 上下文继续训练。B 完成、
宿主机恢复新 CUDA 进程初始化、A/C 在健康双卡上从头重跑后，才会进行预注册的对比评估。

## 当前结论

现阶段不支持直接把当前 16 帧输入替换成“词内帧重采样到 16 帧”，也没有证据支持立即实现真正
variable-length。冻结 recognizer 对词内重采样存在严重分布偏移，而受控上下文显著改善它在训练
视频上的识别。最稳妥的候选仍是改进的 fixed-16：保留必要上下文，同时修正短片段 padding 与
foreground 坐标，并验证 foreground 聚合能否改善 source-disjoint 与连续 WER。

这不是最终推荐。最终选择仍要求 A/B/C 等预算训练、train-only 连续句子 WER、冻结 dev 一次性确认
及 DEL/INS/SUB 分解。官方 test 始终禁止读取。

## 为什么原项目采用 16 帧

原论文把固定窗口看作平均手语词持续时间附近的折中，并实际比较了 8、16、32 帧；报告的
dev/test WER 分别为 `25.9/25.6`、`22.2/22.1`、`23.0/23.2`，因此选择 16。固定长度还让
双流 S3D、batch、时间下采样和在线滑窗保持简单一致。

当前证据说明这个考虑仍有两点成立：上下文对旧 recognizer 很有用，且 16 帧比 24/32 帧包含更高
的目标词比例。失效之处是：5--6 帧代理词在 16 帧连续窗口中只占约三分之一，且训练代码的前景
坐标没有随随机 padding 移动，foreground/saliency supervision 并未始终监督真实前景位置。

活跃的双流路径不是把原始 16 帧直接交给 head。`S3D_two_stream_v2` 的实测时间 token 数为：
8/12 帧各 1 个、16 帧 2 个、24 帧 3 个、32 帧 4 个。因而 8/12 帧可以运行，16 帧不是代码硬性
最小值；但 8/12 帧都只保留一个时间 token，可能损失动作阶段信息。固定 16 帧的 foreground 分支
把 2 个 token 插值为 8 个 bin，所以默认 `temp_idx_ratio=2` 与当前 16 帧活跃路径相符。

## 实现审计

### 数据构造

- train 有 85,379 个 bag，其中 55,247 个非空白、30,132 个空白。
- 每个 epoch 约抽取 512,274 个 clip；非空白基础切片仅占期望非空白 clip 的 11.8%，其余主要是
  连续句子中的 16 帧上下文窗口。
- 39.14% 的非空白 bag 的代理长度不超过 6 帧。
- 对这些短 bag，连续增强窗口的目标词原始帧占比均值约 32.62%。

### foreground/saliency

`split_nonblk` 是辅助前景分类分支，并不是主 logits 的硬 mask。主 logits 仍聚合整个时间序列。
原训练对短基础切片随机选择左右 padding，但 `temp_idx` 按居中 padding 计算。以 5 帧目标为例，
旧 `temp_idx=[2,4]` 时，各时间 bin 真正落在前景中的平均概率只有 27.27%。因此论文意图存在，
但当前实现没有在所有短样本上按预期对齐。

### 已修正的实现问题

1. 增加确定性 `center_pad`，并让 `temp_idx` 可按实际 padding/重采样方式对齐。
2. 修复 `even` 重采样后 RGB 使用句中绝对帧、关键点却错误读取句首帧的问题。
3. 增加自定义 train 来源校准集的 raw-length 元数据映射。
4. 增加显式关闭外部实验上报的配置开关。
5. RGB/关键点绝对帧一致性与 `temp_scale` 缩短路径的两个回归测试均通过。

这些修复默认不改变原基线；只有实验配置显式启用时才改变行为。

## 已冻结的 WER 原因基线

当前 P6 dev 为 3,747 个 reference、860 个错误，WER 22.9517%；DEL 402、INS 133、SUB 325。
DEL+SUB 占全部错误 84.5%。主要证据排序如下：

1. 480/860 个错误在真实代理区间的全部 dense 窗口中从未出现正确 top-1。
2. 163/860 个错误已有正确 selected top-1，但序列聚合后仍错。
3. 84/860 个错误属于区间内有正确窗口而 predictor 没选到。
4. 43/860 个错误可定位为词间过渡位置产生 insertion。

因此 predictor 漏帧和过渡动作误发射都真实存在，但不是当前 WER 主体。

### train 与 dev 的差异

| split / 代理长度 | reference | DEL+SUB | 错误率 | 中点 top-1 |
|---|---:|---:|---:|---:|
| train-fit 5--6 | 19,382 | 1,228 | 6.34% | 95.11% |
| dev 5--6 | 1,729 | 625 | 36.15% | 62.58% |
| dev 7--9 | 1,227 | 77 | 6.28% | 92.01% |
| dev 10+ | 791 | 25 | 3.16% | 90.77% |

控制为相同 gloss、相同短长度后，151 个 gloss 的 dev DEL+SUB 为 30.80%，train-fit 为 3.94%，
差 7.66 倍。最强结论是短代理词的 train--dev 泛化失败；它与训练切片方式高度吻合，但冻结观察
本身不能证明因果。

## 冻结 checkpoint 输入敏感性

以下结果来自 2,048 个按长度平衡的 train 非空白样本。recognizer 已看过这些训练视频，因此只代表
输入敏感性，不是 held-out 泛化率。

| 输入 | overall Top-1 | 5--6 帧 Top-1 | 5--6 帧 Top-5 | 5--6 帧 blank rate |
|---|---:|---:|---:|---:|
| base + endpoint pad 到 16 | 91.50% | 79.49% | 96.29% | 9.57% |
| 词内均匀复制到 16 | 79.69% | 52.93% | 90.04% | 32.23% |
| 受控连续上下文 16 | 96.29% | 96.48% | 待汇总 | 0.59% |

在 5--6 帧样本上，受控上下文相对 base 提高 16.99 percentage points，source-cluster bootstrap
95% CI `[13.67, 20.51]`；词内重采样降低 26.56 points，CI `[-30.47, -22.46]`。
foreground 聚合相对主聚合只提高 0.98 point，CI `[0.20, 1.95]`。

词频是独立放大因素。在加权主头上：

| 输入 | 常见词 >=51 | 中频 11--50 | 稀有 1--10 |
|---|---:|---:|---:|
| base-16 | 93.0% | 88.0% | 34.7% |
| context-16 | 97.2% | 98.3% | 59.2% |
| inside-resampled-16 | 81.9% | 65.8% | 28.6% |

inside-resampled 的下降跨越所有频率档，不能只归因于稀有词。

### 代表性单词 REF / HYP

下表均为 5 帧样本、加权主头；`context` 恢复正确词，但这仍是训练视频上的敏感性证据。

| REF | base-16 HYP | context-16 HYP | inside-resampled-16 HYP |
|---|---|---|---|
| NORD | `<blank>` | NORD | `<blank>` |
| BESONDERS | KUEHL | BESONDERS | KUEHL |
| SCHNEIEN | `<blank>` | SCHNEIEN | REGEN |
| SUEDOST | REGION | SUEDOST | `<blank>` |
| HOCHWASSER | `<blank>` | HOCHWASSER | `<blank>` |
| INFORMIEREN | poss-EUCH | INFORMIEREN | DURCHGEHEND |
| SONNE | DONNERSTAG | SONNE | `<blank>` |
| HAUPTSAECHLICH | MEISTENS | HAUPTSAECHLICH | MEISTENS |

## D/E 候选的输入几何

| 输入长度 | 5--6 帧词平均前景占比 | 居中在线 lookahead | 名义帧计算量 / 16 |
|---:|---:|---:|---:|
| 8 | 62.5% | 4 帧 | 0.50 |
| 12 | 41.7% | 6 帧 | 0.75 |
| 16 | 31.2% | 8 帧 | 1.00 |
| 24 | 20.8% | 12 帧 | 1.50 |
| 32 | 15.7% | 16 帧 | 2.00 |

这是输入几何和近似成本，不是模型实测吞吐。D 的 `margin 0/2/4/6` 与 E 的
`8/12/16/24/32` 配置已经冻结并哈希，尚待 GPU 推理。冻结网格评估器已加入同步的纯 model-forward
计时、items/s、batch latency、峰值 allocated/reserved 显存与名义 lookahead 记录；CUDA 恢复后会
在相同 batch size 下生成实测成本，避免只用帧数比例代替运行时结果。

## 单词、2--3 词与整句尺度

- 单词：5--6 帧只占 dev reference 的 46.1%，却贡献 625/727，即 86.0% 的 DEL+SUB。
- 2-gram：两个词都短时，至少一个 DEL/SUB 的比例为 55.66%。
- 3-gram：三个词都短时，该比例为 70.70%。
- dev 中有 126 个长度至少 2 的连续 DEL/SUB run；全部包含短词，93 个 run 全由短词组成。
- 句内短词比例从 0 增加到 `(2/3,1]` 时，dev WER 从 7.58% 增加到 32.69%。

## 尚未完成的因果实验

等预算短训使用相同 K400 初始化、相同 16,384 个 source-disjoint fit bag、相同 optimizer、seed、
步数和独立的 1,280 条 train calibration：

- A：原始 fixed-16；
- B：fixed-16 + 居中 padding + 对齐 foreground 坐标；
- C：只用词内帧并复制到 16。

恢复前的机器可读公平性审计已经通过全部 12 项检查：A/B/C 的模型、optimizer、scheduler、
`t_max=100`、seed、batch、epoch 和 calibration 完全一致，配置没有非预注册差异；A/C 的 16,384
个 base bag 身份逐项完全相同，fit 的 5,230 个源视频与 calibration 的 817 个源视频没有交集，
且三个配置的 test 路径均为不可访问哨兵。审计结果位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_abc_fairness_audit_v1/summary.json`。

### 2026-10-12 正式短训与新硬件故障

沙盒外逐卡预检确认六张候选卡能够完成 PyTorch 分配、kernel 与同步，随后以互不重叠的双卡启动
A/B/C。禁用卡 `25:00.0` 与此前不稳定的 `c1:00.0` 均未使用。A 在 epoch 0 首轮约 500/2048 步
由 local rank 1 报 `CUDA error: unspecified launch failure`；紧接着宿主机 `nvidia-smi` 对
`41:00.0` 报 `Unable to determine the device handle ... Unknown Error`。A 没有产生 checkpoint，
因此整个中断运行已排除并移动到独立故障归档，不能用于候选比较。

同一时刻 B 已在 `61:00.0+81:00.0` 上超过 649 步，C 已在 `a1:00.0+e1:00.0` 上超过 587 步，
两者仍在正常推进。这也构成 `61:00.0` 当时可安全执行该训练负载的实证。故障后只暴露仍被
`nvidia-smi` 列出的 `01:00.0`，连续三次新 PyTorch 进程仍得到 `is_available=false` 和
`device_count=0`；所以现有证据只能确认 `41:00.0` 掉线，并不能判定空闲的 `01:00.0` 自身故障。
机器可读事故记录及日志哈希位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_gpu_failure_20261012_41/summary.json`。

A/C 将在驱动恢复后使用完整通过复检的健康双卡、相同双卡 world size、相同初始化、seed、数据与预算
从头重跑。B 的既有进程只在完整结束并通过产物校验后保留；若中途受影响，同样整次作废重跑。

同日 07:17 UTC，C 在 epoch 0 约 1305/2048 步由 local rank 1 报出相同的
`CUDA error: unspecified launch failure`，对应的 `e1:00.0` 随即也被 `nvidia-smi` 报为
`Unknown Error`。C 同样没有 checkpoint，已整次作废并独立归档；配对的 `a1:00.0` 仍被 NVML
正常列出且显存为 0，不能据此判定 a1 故障。此时 B 已超过 1457 步且 61/81 均为 100% 利用率。
第二次设备掉线说明节点存在持续的多卡/PCIe 或驱动不稳定风险，而不是某个输入候选导致训练失败。
机器可读记录位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_gpu_failure_20261012_e1/summary.json`。

第一次 pilot 错误地把 cosine `t_max` 缩为 3，导致三轮学习率为 `0.00045/0.00015/0`。第二个
有效 epoch 后，B 的 ensemble Top-1 为 20.00%（等于校准集空白比例），C 为 20.31%；该 pilot
已独立归档并明确标为不可用于候选选择。构建器已经恢复基线 `t_max=100`。

A 在第 250 步因 GPU 1 硬件掉线终止。B/C pilot 退出并释放其他 GPU 后，新进程即使只暴露健康的
GPU 2，PyTorch 仍返回 `CUDA unknown error`、`device_count=0`，证明这是宿主机驱动级阻塞。
失败步数和无效 scheduler pilot 都不会计入比较。只有 CUDA 恢复、A/B/C 从相同初始化按修正配置
重跑，并完成 source-disjoint isolated 与连续 WER，才允许晋级到完整训练和一次性冻结 dev 确认。

2026-10-06 再检查时，`nvidia-smi` 已把故障卡从逻辑设备列表排除，并把其余七张卡重新编号为
0--6；这说明 NVML 管理接口部分恢复。可是只暴露健康卡 UUID
`GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a` 时，PyTorch 仍报告 `device_count=0`，直接调用
`libcuda.so.1` 的 `cuInit(0)` 也返回 `CUDA_ERROR_UNKNOWN (999)`。所以 `nvidia-smi` 成功不能作为
训练恢复的判据。

宿主机只读检查进一步确认 PCI `0000:25:00.0` 为 `rev ff`、`Unknown header type 7f`，但仍绑定
`nvidia` 驱动，仍存在于 `/proc/driver/nvidia/gpus`，并且其 `/dev/nvidia2` 节点仍在；驱动记录还
显示 `GPU Excluded: No`。因此最符合全部观察的解释是：失联设备仍残留在内核驱动枚举中，污染了
全局 CUDA 初始化。该结论是基于设备状态与 `cuInit` 行为的诊断推断，不等同于硬件厂商级故障判定。

2026-10-08 的新检查显示故障位置发生迁移：`0000:25:00.0` 恢复为 `rev a1` 并再次被 NVML 列出，
但仍按用户的长期约束永久禁用；与此同时 `0000:c1:00.0` 变为 `rev ff / Unknown header type 7f`，
从 NVML 列表消失却仍绑定 `nvidia`，且 `GPU Excluded: No`。改用另外六张可见、非禁用 UUID 后，
`cuInit(0)` 仍返回 999。这个时间变化强化了“宿主机多卡/PCIe 或驱动状态不稳定”的解释，并排除
了仅靠固定逻辑编号或 `CUDA_VISIBLE_DEVICES` 绕过故障的方案。新探针位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_cuda_recovery_probe_20261008/summary.json`。

为排除 Codex 沙盒和继承环境变量，第三次检查在获准的沙盒外进程中使用 `env -i` 清空环境，只保留
`HOME`、`PATH`、`CUDA_DEVICE_ORDER` 和一张已验证健康卡的 UUID，再直接调用系统
`libcuda.so.1`；结果仍为 `cuInit(0)=999`。因此沙盒只能解释沙盒内 `nvidia-smi` 的可见性差异，
不能解释宿主机外干净进程的 CUDA 初始化失败。

解除条件是宿主机侧恢复 CUDA driver API，使一个仅暴露健康 GPU UUID 的新进程满足
`cuInit(0) == CUDA_SUCCESS` 且 `torch.cuda.is_available() == True`、`device_count == 1`。通常需要
完整重启宿主机或由管理员重新加载 NVIDIA 驱动并清除当前掉线设备的残留状态。不需要额外数据权限，
也不需要开放 test。更新后的 blocking report 位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_blocking_report_v3/blocking_report.json`，恢复命令位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_blocking_report_v2/resume_commands.md`；恢复探针位于
`Online/CSLR/results/phoenix-2014t_ISLR/p7_cuda_recovery_probe_20261006/probe.json`。
GPU 作业启动前必须先运行 `Online/CSLR/tools/check_p7_cuda_preflight.py`，按 UUID 选择设备；该门禁会
同时拒绝 PCI `0000:25:00.0`、UUID `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83`，并在 `cuInit`
或 PyTorch 可用性检查失败时以非零状态退出。2026-10-06 的实际验证按预期拒绝了启动。

## 变长方案 gate

真正 variable-length 需要修改 batch padding、双流时间 mask、主/辅助头聚合、在线窗口接口与导出
路径。只有 8/12/16/24/32 分桶或 masked 固定长度在 held-out WER 上胜过所有 fixed-16 候选，且
收益足以覆盖吞吐和实现复杂度时，才实施 F。当前证据没有通过这个 gate。

## 可复现入口

- 预注册：`Online/CSLR/configs/p7_islr_temporal_input_v1_preregistered.json`
- 审计：`Online/CSLR/tools/audit_phoenix_islr_temporal_inputs.py`
- 冻结敏感性：`Online/CSLR/tools/evaluate_phoenix_islr_input_sensitivity.py`
- paired bootstrap：`Online/CSLR/tools/analyze_phoenix_islr_input_sensitivity.py`
- 频率/signer 分层：`Online/CSLR/tools/stratify_phoenix_p7_input_sensitivity.py`
- A/B/C 资产：`Online/CSLR/tools/build_phoenix_p7_temporal_smoke.py`
- A/B/C 公平性审计：`Online/CSLR/tools/audit_phoenix_p7_abc_fairness.py`
- CUDA/PCI 启动门禁：`Online/CSLR/tools/check_p7_cuda_preflight.py`
- D/E 资产：`Online/CSLR/tools/build_phoenix_p7_temporal_grid.py`
- 输入几何：`Online/CSLR/tools/analyze_phoenix_p7_temporal_grid_geometry.py`
- 连续句子 screen：`Online/CSLR/tools/build_phoenix_p7_continuous_smoke.py`
