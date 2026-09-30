# P5 H4 schedule-ranker source-disjoint OOF v1

日期：2026-09-23。配置、特征、ridge、fold、负对照和六项停止门槛均在任何predictor outcome产生前
冻结。实验只使用完整train fit，CPU运行；没有使用GPU，也没有读取calibration、dev、test、
held-out或closed-loop outcome。

## 身份

主分析只保留43,437个完整H=4 horizon，避免用尾部H<4泄漏真实EOS。每行目标是81种schedule在
fixed-center/fixed-late两种continuation下的最小signed reward。决策时点固定为horizon第一个block
的offset-0已经可见；输入不得读取horizon内部后续帧。

五折严格按578个source分组。四个固定特征组为：

- `K`：decoder/schedule因果历史；
- `KS`：K加31帧heatmap-U的last/mean/std/last-first汇总；
- `KO`：K加31帧heatmap-U的前8阶DCT保序表示；
- `KX`：K加逐行确定性时间打乱后的同一DCT表示，作为负对照。

模型是固定alpha=100的81-output closed-form ridge。center schedule分数固定为0，其余schedule按OOF
预测分数取argmax。该分析使用oracle过去动作形成teacher-forced状态，仍不是可部署闭环。

## 结果

H4标签本身只有824/43,437个horizon具有正oracle robust value，oracle总reward为844。81-way标签
共有3,518,397个候选，其中正25,014、负32,946、零3,460,437，极度稀疏且大量并列。

| 模型 | Signed robust reward | 95% source bootstrap | 改变center | 正/负reward horizon |
|---|---:|---:|---:|---:|
| K | -40 | [-93, 14] | 42,706 | 317 / 356 |
| KS | -24 | [-77, 30] | 42,856 | 344 / 363 |
| KO | **-51** | **[-105, 4]** | 42,928 | 321 / 364 |
| KX | -19 | [-71, 34] | 42,790 | 337 / 353 |

KO相对K为-11 reward，bootstrap `[-56,34]`；相对time-shuffled KX为-32，`[-78,14]`。KO要求
至少+249且两个增量至少+50并满足bootstrap下界>0；六项门槛全部失败。

结果的主要形态不是“选不到唯一winner”，而是ridge几乎总给某个非中心schedule微小正分：KO在
43,437行中只保留center 509次。大部分更改落在真实reward=0的并列区，但364次落入负reward，
超过321次正reward，累计变成-51。summary-U、ordered-U和time-shuffled-U均未形成可靠abstention
或正收益，ordered-U也没有超过打乱顺序的负对照。

`metrics.json`中的`winner_accuracy`是与`targets.argmax`的字典序首个最大值比较。由于42,613行
oracle最大值为0且存在大量并列，该数值不能解释为P5 tie-break winner准确率；本实验的主结论只用
预注册signed robust reward、bootstrap和增量门槛。

## 决策

本identity停止，不运行unknown-EOS closed loop或held-out。结果不否定H4 oracle上界，而说明当前
因果历史和heatmap-U无法可靠识别仅约1.9%的正horizon；保序heatmap没有显示超越时间打乱对照的
证据。若继续研究，只能以新身份先研究可校准的rare-event/abstention target，并明确标为在本OOF
结果之后提出，不能通过事后调score threshold把本轮改写为成功。

证据：`Online/CSLR/results/phoenix-2014t_ISLR/p5_h4_schedule_ranker_oof_v1_49faacc3/`。
