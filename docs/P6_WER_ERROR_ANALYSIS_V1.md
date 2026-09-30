# P6 online-quantile WER 多尺度错误分析 v1

日期：2026-09-29。结论范围是 Phoenix-2014T 当前冻结的 P6 online-quantile 控制器：
完整 train-fit 的 source-disjoint controller OOF 回放，以及已授权的 519 条修复后 dev 回顾性
回放。本文不读取 test，不重新运行识别器，不调整模型、分位数、token 或 decoder。

## 结论

如果问题是“哪类错误占当前 dev WER 最大”，答案是**删除**：402/860，占 46.7%。但删除并非
压倒性唯一来源，替换有 325 个、占 37.8%；两者合计解释 84.5% 的 dev 错误。插入只有 133 个、
占 15.5%。因此 dev 的实际问题应概括为“漏词为首、错词紧随，其共同构成识别泛化错误”，不能只
针对重复输出或插词优化。

目前证据最支持的机制是**底层识别器从训练样本到 dev 的泛化下降，低频 gloss 又明显放大了这个
下降**。调度器影响错误之间的取舍，但不是 dev 22.95% WER 的主要来源：把 P6 quantile 换为
full-center 后，dev 只从 860 变为 858 个错误，差 2 个；quantile 的 85.1% 精确错误事件仍存在于
full-center。所有 9 个 signer 都同时出现在 train 和 dev，而每个 signer 的 dev WER 都远高于其
train WER，所以差距也不能仅由某个未见 signer 解释。

低频词是一个强放大因素。按 gloss 在完整 train reference 中的出现次数分组，dev reference 上的
删除或替换率为：

| train 出现次数 | dev reference 数 | 删除或替换 | 错误率 |
|---:|---:|---:|---:|
| 0--9 | 118 | 94 | **79.7%** |
| 10--49 | 301 | 120 | **39.9%** |
| 50--199 | 818 | 159 | 19.4% |
| 200+ | 2,510 | 354 | 14.1% |

0--49 次的词只占 dev reference 的 11.2%，却贡献 29.4% 的全部删除和替换。不过，高频组仍有
354 个错误，所以词频不能独立解释全部问题；它是风险因素，不是完整原因。词频还可能与动作难度、
复合 gloss 和上下文多样性共同变化，本文不把这个相关关系写成因果效应。

## 总体错误分布

对齐沿用项目原有加权编辑距离：DEL/INS 成本为 3，SUB 成本为 4。所有逐样本 error 和窗口数、
各策略总 error、reference 总数均与冻结产物完全一致。

| split | reference | WER | DEL | INS | SUB | 完全正确句 |
|---|---:|---:|---:|---:|---:|---:|
| train-fit | 49,613 | 5.1257% | 1,108（43.6%） | 1,087（42.7%） | 348（13.7%） | 4,524/6,378 |
| dev retrospective | 3,747 | 22.9517% | 402（46.7%） | 133（15.5%） | 325（37.8%） | 118/519 |

括号内为该类占全部错误的比例。按 source cluster bootstrap，train 的 DEL/INS/SUB WER 分量
95% CI 分别为 `[2.06, 2.42]`、`[2.03, 2.37]`、`[0.62, 0.78]`；dev 分别为
`[9.58, 11.89]`、`[2.87, 4.25]`、`[7.67, 9.70]`。从 train 到 dev，删除率增加约 4.80 倍，
替换率增加约 12.37 倍，插入率只增加约 1.62 倍。这解释了为什么 train 看起来是“删除/插入
各半”，dev 却变成“删除/替换为主”。

需要注意，train 的 OOF 只针对 controller：底层 recognizer 见过 train 视频。dev 是回顾性数据，
也不是新的 confirmatory held-out。因此这里能够确认强烈的 train--dev 泛化差距，不能从这组数据
单独估计未来外部数据的误差。

## 单词尺度

dev 最常被删除的词是 `IX` 25 次、`REGION` 21 次、`KOMMEN` 12 次、`BISSCHEN` 和 `DANN`
各 10 次。按删除加替换合计，`REGION` 和 `IX` 各错 34 次，`KOMMEN` 17 次，`DANN` 16 次。
但前两项合计只占全部 reference-consuming 错误的 9.4%，说明错误没有集中在少数几个词上。

替换对也很分散：最高的 `WEHEN → KOMMEN` 和 `SCHNEIEN → REGEN` 都只有 4 次；随后是
`OBER → HOCH`、`WIND → WEHEN` 等各 3 次。这些例子符合视觉或语义邻近混淆，但单个混淆对
数量太小，不能据此断言具体动作特征是根因。

高错误率词包括 `UND` 10/18、`MEISTENS` 7/14、`UEBERWIEGEND` 5/10、`DANN` 16/40、
`WEHEN` 12/32、`IX` 34/97。这里同时出现功能性过渡词、频率不高的天气表达以及指代词，支持
“弱/短动作和低频词都值得进一步检查”，但当前 alignment 证据不能区分两者。

## 连续 2--3 词尺度

train 中 90.2% 的错误事件位于单个 reference 词的错误 run 或纯插入 run，只有 9.8% 位于连续
两个及以上 reference 词的错误 run。dev 中对应比例是 67.2% 和 **32.8%**。因此 dev 不只是
“更多单词错了”，错误还更容易在一个局部连续扩散成两三个词。

dev 中经常包含错误的上下文包括：

| reference 上下文 | 含 DEL/SUB 次数 / 总次数 |
|---|---:|
| `JETZT WETTER` | 6/26 |
| `MAESSIG WEHEN` | 4/8 |
| `SCHNEE REGEN` | 4/7 |
| `JETZT WETTER WIE-AUSSEHEN` | 6/15 |
| `SCHOEN ABEND WUENSCHEN` | 3/4 |

这些数字表示 n-gram 中至少一个 reference 词被删除或替换，并不表示整个短语作为一个类别被
模型识错。真正连续的 2--3 reference 词错误 run 很分散：只有 `SOLL EINFLUSS` 出现两次，
其余高位模式大多只出现一次。当前证据更像广泛的局部解码崩塌，而不是一个固定短语词典缺陷。

## 句子尺度

dev 的 519 句中有 401 句至少错 1 个词（77.3%），239 句至少错 2 个词（46.1%）；train 分别为
29.1% 和 7.2%。dev 的 top 10% 高错误句只贡献 29.2% 错误，而 train 的 top 10% 贡献 52.2%。
因此 dev 问题广泛分布，删除少量异常句不能解决总体 WER。

短句更难：dev reference 长度 1--3 的 WER 为 30.72%，4--6 为 24.94%，7--9 为 21.29%，
10+ 为 22.66%。精确发生在首尾 reference 词或首尾插入边界的错误共有 238 个，占 dev 错误的
27.7%；边界是明显风险位置，但仍不足以解释多数错误。

视频长度也不是单一根因。dev 的四个帧数桶 WER 为 26.87%（≤64 帧）、22.66%（65--96）、
20.88%（97--128）、23.83%（129+）。短视频较差，但其余长度仍稳定在约 21--24%。

9 个 signer 均见于 train。样本量较充足的 signer 在 dev 上约为 20.1--28.1% WER，在 train 上
约为 4.5--7.0%；每个 signer 都出现显著方向一致的差距。小样本的 Signer02 为 12/31 errors，
不宜单独解释。

## 调度、decoder 与识别器的相对作用

P6 quantile 相对等预算 matched uniform 在 train 少 230 errors，其构成为 DEL `-329`、INS
`+115`、SUB `-16`；dev 少 12 errors，构成为 DEL `-45`、INS `+14`、SUB `+19`。因此控制器的
主要作用是提高召回、减少删除，同时会引入一部分插入和替换。这也解释了为什么只看总 WER 会掩盖
策略真实取舍。

与 full-center 对比更能判断调度是不是主因：

| split | quantile errors | full-center errors | quantile - full | quantile 错误事件在 full 中仍存在 |
|---|---:|---:|---:|---:|
| train-fit | 2,543 | 2,325 | +218 | 75.5% |
| dev | 860 | 858 | +2 | 85.1% |

dev 上 full-center 没有实质改善：它比 quantile 多 27 个删除，但少 6 个插入和 23 个替换，净差
只有 2。冻结的 source-cluster CI 为 `[-0.408, +0.524] pp`。这强烈反对“dev WER 主要由 P6
漏选窗口造成”的解释。

修复后 dev 的 dense 历史对照也显示 decoder 只改变较小一部分：window-greedy-7 为 833 errors
（22.231%），span-weighted-15 为 847（22.605%）；二者相差 14 errors。dense window-greedy-7
相对 P6 quantile 少 27 errors，但仍留下 833 errors。这个比较同时改变窗口集合与 decoder，不能
作为严格分解；它只说明可由当前采样/decoder 选择解释的量级远小于 860 个总错误。

综上，证据强度排序是：

1. **强证据：recognizer 的 train--dev 泛化问题主导 dev WER。** full-center 无法降低总体错误，
   85% 的 quantile 错误事件保持不变，差距跨所有 signer 存在。
2. **强相关证据：低频或 train 未出现的 gloss 显著放大删除/替换。** 但常见词仍贡献大量错误。
3. **中等证据：dev 错误更容易形成局部 2--3 词错误段，短句和句子边界风险较高。** 这些因素
   解释局部分布，不能单独解释总量。
4. **有限证据：train 的部分大段插入来自相邻 segment 边界泄漏。** 67 个多词插入 run 中有
   27 个与前后 reference 存在至少 2 词连续重合，保守覆盖 75/1,087 个 train 插入（6.9%）；
   dev 的 4 个多词插入 run 没有这种重合。它能解释 train 最醒目的极端错例，但不是总体主因。

一个明确的 train 边界例子是
`train/02November_2010_Tuesday_heute-2746`：预测开头额外插入 10 个词，其中 9 个连续词与前一
segment 的 reference 相同。相对地，dev 最坏样本
`dev/02December_2010_Thursday_heute-8445` 有 1 DEL、2 INS、8 SUB，更符合跨词汇的泛化混淆。

## 限制与下一步

- alignment 在多个等价最短路径中采用项目现有的加权 tie-breaking；总 WER 不受影响，但个别
  DEL/SUB/INS 归类可能随对齐规则变化。主要比例差异很大，不依赖单个例子。
- n-gram “受影响率”是描述性上下文统计，不能把相关短语直接当作识别单元或因果因素。
- full-center 和 dense decoder 对照能够排除“调度是主要来源”，但不能进一步区分视觉 encoder、
  CTC/posterior、span decoder 和标注边界各自的贡献。
- 如果继续做机制定位，价值最高的只读分析是：按 train 频率匹配后比较动作持续时间/关键点质量，
  以及对 dev 的 D/S 词做 posterior margin 和相邻词边界检查。任何新模型或参数选择都应建立新
  实验身份；本文结果不授权读取 test。

## 可复现产物

- 分析脚本：`Online/CSLR/tools/analyze_p6_online_quantile_wer_errors.py`
- 逐样本 reference、四策略 hypothesis、alignment 和错误事件：
  `Online/CSLR/results/phoenix-2014t_ISLR/p6_online_quantile_wer_error_analysis_v1/per_sample_predictions.jsonl.gz`
- 完整统计：同目录 `summary.json`
- 输入/输出哈希与验收：同目录 `manifest.json`

只核验现有产物：

```bash
cd Online/CSLR
CUDA_VISIBLE_DEVICES= \
  /mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  tools/analyze_p6_online_quantile_wer_errors.py --verify
```

从冻结 logits 独立重建到临时目录：

```bash
cd Online/CSLR
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  tools/analyze_p6_online_quantile_wer_errors.py --output /tmp/p6-wer-analysis-check
```
