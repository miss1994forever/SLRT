# P6 WER 主要成因分析 v2

日期：2026-09-29

## 一句话结论

当前 dev WER 的首要问题不是 predictor 漏选关键帧，也不是词间动作被识别为词，而是**短时
gloss 上的识别器泛化失败**：在 727 个 deletion/substitution 中，有 480 个（66.0%，source
cluster bootstrap 95% CI `[62.4%, 69.6%]`）即使检查真实 gloss 代理区间内的全部 stride-1
窗口，也从未把正确 gloss 排到 top-1。长度 5--6 帧的代理片段只占 dev reference 的 46.1%，
却贡献 625/727（86.0%）个 deletion/substitution。

这一现象与当前 ISLR 切片训练方式高度吻合，但现有结果只能给出强机制证据，不能完成严格的
训练切片因果归因。长度 5 帧的连续训练窗口使用 16 帧输入，其中目标代理区间最多只占 5/16，
其余来自相邻上下文；基础切片则通过重复端点补到 16 帧。需要一次保持网络和 decoder 不变的
切片/上下文干预实验，才能区分“短动作本身难”“代理边界不准”和“上下文被错误地当成目标特征”。

## 冻结基准与错误总量

分析对象是当前冻结的 Phoenix-2014T P6 online-quantile 方案。train 使用 source-disjoint
controller OOF 回放，但底层 ISLR recognizer 见过 train 视频；dev 是 519 句已授权的回顾性
回放。持久化分析没有读取 test，也没有改模型、阈值或 decoder。

| split | reference | WER | DEL | INS | SUB | 总错误 |
|---|---:|---:|---:|---:|---:|---:|
| train-fit | 49,613 | 5.1257% | 1,108 | 1,087 | 348 | 2,543 |
| dev | 3,747 | 22.9517% | 402 | 133 | 325 | 860 |

dev 中 deletion 占 46.7%，substitution 占 37.8%，二者合计 84.5%。所以首先要解释的是为什么
真实词没有被正确识别，而不是先优化 insertion。

## 可加和的诊断分解

下表把 dev 的 860 个编辑错误分成互斥的诊断证据类别。reference-consuming 错误先检查真实
gloss 代理区间内是否存在正确 top-1，再检查 P6 是否选到了它；insertion 则定位该插入词在
已选窗口中的最高 posterior 所处区域。

| 诊断类别 | 错误数 | 占 860 | 当前解释 |
|---|---:|---:|---|
| 真实词区间内从未出现正确 top-1 | **480** | **55.8%** | recognizer/数据泛化主导；其中 314 个正确词曾在 top-2--5，166 个连 top-5 都没有 |
| 区间内存在正确 top-1，但 P6 未选到 | 84 | 9.8% | predictor/scheduler 的直接漏选候选 |
| 已选窗口存在正确 top-1，序列仍错 | 163 | 19.0% | 序列聚合、重复词折叠、局部竞争或对齐歧义候选 |
| 插入词峰值落在两个 gloss 之间 | 43 | 5.0% | 词间过渡动作误识别候选 |
| 插入词峰值落在另一个 gloss 内 | 53 | 6.2% | 相邻词混淆、重复或序列归属错误候选 |
| 插入词峰值落在同名 gloss 内 | 8 | 0.9% | 重复发射候选 |
| 插入词峰值落在句外背景 | 29 | 3.4% | 背景误发射候选 |

这是一种可复现的证据分区，不是完整的因果分解。代理边界由 CTC alignment 得到，并非人工帧级
真值；“已选 top-1 但序列错”也可能包含 reference alignment 的多解。表内数字可以相加，背后的
潜在原因可能重叠。

## 原因排序

### 1. 短时 gloss 的 recognizer 泛化失败：影响最大，证据强

| split / 代理长度 | reference | DEL+SUB | 错误率 | 中点 top-1 |
|---|---:|---:|---:|---:|
| train-fit 5--6 帧 | 19,382 | 1,228 | 6.34% | 95.11% |
| dev 5--6 帧 | 1,729 | 625 | **36.15%** | **62.58%** |
| dev 7--9 帧 | 1,227 | 77 | 6.28% | 92.01% |
| dev 10+ 帧 | 791 | 25 | 3.16% | 90.77% |

短片段风险不是由 dev 恰好包含更多难词就能解释。限制到相同 gloss、长度都不超过 6 帧，且每个
gloss 至少有 3 个 dev 和 10 个 train-fit 样本后，共得到 151 个 gloss：dev 为 441/1,432
（30.80%），train-fit 为 609/15,445（3.94%）。用每个 gloss 的 train 错误率预测 dev，只预期
57.6 个错误，实际为 441 个，是 7.66 倍。该控制同时固定了 gloss identity 和短长度，但 train
recognizer 是 in-sample，因此它证明的是强烈的 train--dev 泛化崩塌，不是泛化误差的无偏估计。

训练构造提供了合理机制：recognizer 输入固定为 16 帧，bag size 为 6。每个代理片段包含一个
基础 crop，并为片段内每个中心帧生成一个 16 帧连续窗口。对于 5 帧片段，恰好是 1 个端点重复
补齐的基础 crop 加 5 个连续窗口；连续窗口中目标区间最多占 31.25%，至少 11/16 帧来自片段外。
模型可能学习相邻词或边界环境，也可能被不精确的代理分段放大干扰。

低频 gloss 是第二个放大因素，但不是独立主因。train 出现 0--9 次的词在 dev 上 DEL+SUB 为
94/118（79.7%），10--49 次为 120/301（39.9%）；然而 200+ 次高频词仍贡献 354 个错误。

**结论强度：**“短代理区间与 recognizer 泛化失败主导错误”是强证据；“根因就是 chopped
ISLR 训练造成上下文污染”是目前最强假设，仍需切片干预实验确认。

### 2. 序列聚合/decoder 相关错误：第二位，中等证据

163 个 dev 错误（19.0%）对应的真实词，在 P6 已选窗口中至少一次成为 top-1，但最终序列仍将
它删除或替换。train 的相同比例为 696/1,456（47.8%）；train 更常见的表现是 recognizer 局部
证据正确但序列输出没有保留，而 dev 更常见的是局部识别证据本身就错。

这个类别包含多个机制：相邻窗口竞争、重复 gloss 被折叠、同一词多次发射后的编辑对齐选择，以及
现有 span decoder 的聚合规则。它提示 decoder 有可改善空间，但不能把 163 个错误全部当作可由
换 decoder 消除。冻结 dense 对照中，window-greedy-7 仍有 833 个错误（22.231%），相对 P6
只少 27 个，也说明单独替换当前采样/聚合方式不会解决大多数 WER。

### 3. predictor 漏掉关键窗口：存在，但不是主要原因

84 个 dev 错误（9.8% 总错误；占 DEL+SUB 的 11.6%，95% CI `[9.3%, 14.0%]`）满足“真实区间
内有正确 top-1，但 P6 没选到”。这是对“predictor 跳过重要帧”最直接的支持。

但每个代理 gloss 都至少有一个被选中心落在其区间内，且换成 full-center 后 dev 只从 860 变为
858 个错误；85.1% 的 P6 错误事件仍存在。另一个使用标签感知中心位置的 oracle，在固定预算下
最多减少 57 个错误（1.521 WER 点，约 6.6% 总错误）。这些对照共同削弱了“主要因为完全跳过
gloss”这一解释。

起点、中点、终点的 dev top-1 准确率分别为 68.05%、78.17%、35.68%。终点明显不可靠，说明
窗口放到代理尾部容易看到下一动作或回落阶段；但是只追边界的 oracle 改善只有 2 个错误且置信
区间跨 0，因此“把 predictor 改成找边界”也没有证据支持。

### 4. 词间动作被识别成词：现象真实，规模约为总错误的 5%

dev 的 133 个 insertion 中，43 个插入词的最高已选 posterior 位于两个 gloss 代理区间之间：
占 insertion 的 32.3%（95% CI `[24.4%, 40.8%]`），占全部 860 个错误的 5.0%。使用全部 dense
窗口定位时为 48 个，结论相近。

所以“motion between two words is recognized as a word”确实发生，但它不能解释当前 WER 的主体。
另外 53 个 insertion 的峰值位于其他 gloss 区间内，29 个在句外背景，8 个在同名 gloss 内。

### 5. 选到错误位置：是风险指标，不等于主因

dev 的 20,510 个已选窗口中，65.8% 落在 gloss 区间内，23.8% 位于 gloss 间，10.4% 位于句外
背景。这个比例说明 P6 经常选择非词内位置，但 full-center 没有降低总错误，说明“位置在区间外”
本身不是充分的错误原因。很多词间窗口可提供动作准备/收尾信息，只有当它引发具体错误 posterior
或替代了区间内正确窗口时，才能算机制证据。

## 2--3 个词尺度

短代理的风险会在局部上下文中累积：

| dev reference 片段 | 短词数量 | 至少一个 DEL/SUB | 片段内全部词都错 |
|---|---:|---:|---:|
| 2-gram | 0/2 | 86/906 = 9.49% | 0.22% |
| 2-gram | 1/2 | 583/1,544 = 37.76% | 2.20% |
| 2-gram | 2/2 | **433/778 = 55.66%** | **15.04%** |
| 3-gram | 0/3 | 54/410 = 13.17% | 0% |
| 3-gram | 1/3 | 395/1,013 = 38.99% | 0.20% |
| 3-gram | 2/3 | 555/931 = 59.61% | 0.86% |
| 3-gram | 3/3 | **251/355 = 70.70%** | **4.79%** |

dev 有 126 个长度至少为 2 的连续 DEL/SUB run；每一个 run 都至少包含一个短代理，93 个 run
全部由短代理构成。共 279 个错误词，其中 203 个位于全短 run。错误因此常表现为连续两三个词的
局部崩塌，而不只是互相独立的单词错误。

## 句子尺度

| 句内短代理比例 | dev 句数 | dev WER | 完全正确句率 | train-fit WER |
|---|---:|---:|---:|---:|
| 0 | 28 | 7.58% | 71.43% | 3.72% |
| (0, 1/3] | 151 | 16.36% | 29.80% | 3.78% |
| (1/3, 2/3] | 269 | 24.84% | 16.73% | 5.70% |
| (2/3, 1] | 71 | **32.69%** | **11.27%** | 9.09% |

句内短代理越多，WER 单调上升且完全正确率单调下降；train 和 dev 方向一致，dev 斜率和绝对误差
更大。这把单词层的信号连接到了整句结果，也反对“只有少数异常句造成总体 WER”的解释。

## REF 与 HYP 直接对比

### A. recognizer 在真实区间内连 top-5 都没有：删除/替换主因

样本 `dev/23August_2010_Monday_heute-5321`，`OST` 代理区间 `[0,5)`；区间内最佳排名为 980，
top-1 为 `VERSCHWINDEN`。

- REF: `OST LANGSAM ABER SUED WEITER REGEN GEWITTER`
- HYP: `VERSCHWINDEN ABER SUED WEITER REGEN GEWITTER`

这里同时出现 `OST → VERSCHWINDEN` 和 `LANGSAM` 删除。即使把该区间的每一个 dense 窗口都交给
当前 recognizer，`OST` 也没有形成可用候选，不能归因于 P6 没挑中某一帧。

样本 `dev/10August_2011_Wednesday_tagesschau-6303`，`REGION` 区间 `[42,47)`；正确词的区间内
最佳排名为 1086。

- REF: `IM-VERLAUF TEMPERATUR REGION`
- HYP: `IM-VERLAUF TEMPERATUR`

### B. 区间内有强正确证据，但 predictor 没选到

样本 `dev/03October_2012_Wednesday_tagesschau-7655`，`MEISTENS` 区间 `[24,29)`。dense 最佳窗口
把 `MEISTENS` 排 top-1，posterior 0.8768；已选窗口的最佳证据只有 0.1256，且 top-1 为
`KOENNEN`。

- REF: `SUEDOST MEISTENS TROCKEN`
- HYP: `SUEDOST KOENNEN TROCKEN`

这是 predictor/scheduler 选择错误的清晰正例，但全 dev 中同类只有 84 个。

### C. 已选窗口有强正确证据，序列仍删除

样本 `dev/10February_2010_Wednesday_tagesschau-2524`，`REGION` 区间 `[46,51)`；已选窗口中
`REGION` 已为 top-1，posterior 0.8857。

- REF: `SONNTAG SCHNEE WENIG NORD REGION MOEGLICH SONNE`
- HYP: `SONNTAG SCHNEE WENIG NORD MOEGLICH SONNE`

该错误更符合窗口证据到最终序列之间的聚合/竞争问题，而不是 recognizer 完全不认识该词。

### D. 词间过渡位置产生 insertion

样本 `dev/02October_2009_Friday_tagesschau-1354`，插入的 `REGION` 在中心帧 36 达到 posterior
0.8539，位置在 `NORD` 与 `WETTER` 两个代理区间之间。

- REF: `DAZU MORGEN NORD WETTER WECHSELHAFT SONNTAG AUCH MITTE`
- HYP: `DAZU MORGEN NORD REGION WETTER WECHSELHAFT SONNTAG AUCH MITTE`

样本 `dev/21November_2009_Saturday_tagesschau-4911` 中，`VIEL` 的 posterior 0.8461 峰值位于
`WECHSELHAFT` 与 `WOLKE` 之间，也是同一机制的例子。

## 对初始假设的判定

| 初始假设 | 当前判定 | 依据 |
|---|---|---|
| chopped ISLR 切片导致不准 | **强支持，但尚未因果确认** | 短代理贡献 86.0% DEL+SUB；同 gloss train/dev 差 7.66 倍；16 帧训练窗口对 5 帧目标含大量上下文 |
| 两词之间 motion 被认成词 | **证实为次要原因** | 43 个 dev insertion，占总错误 5.0% |
| predictor 跳过关键帧 | **存在但被削弱** | 84 个错误，占 9.8%；full-center 几乎无净改善 |
| predictor 选了两词之间/错误帧 | **位置风险存在，因果覆盖有限** | 34.2% 已选窗口不在词内，但只有已关联错误的子集可解释 WER |
| decoder/序列规则造成错误 | **中等支持，第二层问题** | 163 个错误已有 selected top-1 仍未输出，占 19.0% |
| 低频、视觉相似、signer shift | **低频为放大项；单一 signer shift 不支持** | 低频错误率高；所有 signer 均见于 train 且均有同向 dev 差距 |

## 最有价值的下一步实验

首选一个固定 recognizer 架构、固定 decoder、固定 dev 清单的四臂实验，仅改变 ISLR 输入构造：

1. `base-padded`：当前完整代理片段，短片段端点重复补到 16 帧；
2. `center-16`：当前以代理中心为中心的 16 帧连续窗口；
3. `inside-resampled-16`：只取代理区间内部并重采样到 16 帧，消除相邻词上下文；
4. `context-labelled`：显式加入左右上下文，但使用中心词监督并记录目标帧占比。

按代理长度（5--6、7--9、10+）、gloss train 频率和 signer 分层报告 base-only nonblank top-1、区间内
any-top1、最终 DEL/SUB 与 WER。若 `inside-resampled-16` 显著修复短 dev gloss，支持上下文污染；
若四臂都失败，更可能是动作本身信息不足、代理边界错误或 domain shift。这个实验比继续微调 scheduler
更可能降低主体误差。

第二优先级是在相同已选窗口上比较当前 span decoder、简单 CTC collapse 和重复保留策略，专门检查
163 个“selected top-1 已存在”错误。第三优先级才是针对 84 个 selection miss 训练覆盖率目标，并
对 43 个 gap insertion 增加过渡区 abstention。

## 限制与数据纪律

- gloss 区间来自 stride-4 CTC alignment；5、9、13 帧等量化长度可能同时编码真实持续时间和
  alignment 置信度。报告使用“代理区间”，不将其当成人工帧级真值。
- posterior 位置能定位错误证据出现在哪里，不能单独证明网络使用了哪一帧的视觉特征。
- train-fit 只对 controller 是 OOF，recognizer 对 train 视频是 in-sample；train/dev 差距适合
  诊断，不适合当作外部泛化误差估计。
- 本报告不使用 test。一次未纳入任何统计的 CPU 可行性 smoke run 调用了原始配置；该配置可能
  反序列化包含多个 split 的旧 keypoint 容器，但没有运行或统计 test 样本。该临时输出已清理，
  后续任何识别器实验都应显式使用 split-only keypoint 文件。

## 可复现产物

- 机制分析脚本：`Online/CSLR/tools/analyze_p6_frame_mechanisms.py`
- 机器可读汇总：
  `Online/CSLR/results/phoenix-2014t_ISLR/p6_frame_mechanism_analysis_v1/summary.json`
- 逐错误/逐区间记录：同目录 `per_event.jsonl.gz`
- 输入输出哈希与校验：同目录 `manifest.json`
- 总体 WER、词频和 full-center 对照：`docs/P6_WER_ERROR_ANALYSIS_V1.md`

校验现有产物：

```bash
cd /mnt/workspace/projects/haojun/SLRT
/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  Online/CSLR/tools/analyze_p6_frame_mechanisms.py --verify
```

从冻结 logits 重建（不调用 recognizer，不读取 test）：

```bash
cd /mnt/workspace/projects/haojun/SLRT
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python \
  Online/CSLR/tools/analyze_p6_frame_mechanisms.py \
  --output /tmp/p6-frame-mechanism-check
```
