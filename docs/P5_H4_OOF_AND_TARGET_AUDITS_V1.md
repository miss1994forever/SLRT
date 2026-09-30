# P5 H4 OOF设计与target稳定性诊断 v1

日期：2026-09-23。两个诊断均在对应输出前冻结，只读取已有train-fit标签和特征，使用CPU；没有
读取calibration、dev、test、held-out或closed-loop outcome。结果只能定位失败，不解锁策略。

## OOF设计诊断

使用完全相同的class-weighted ridge和conditional ranker，比较训练内、row-random五折、
sample-disjoint五折和原source-disjoint五折，并加入label-shuffled负对照。

KO opportunity AP分别为：

| 切分 | AP |
|---|---:|
| 训练内resubstitution | .04832 |
| row-random OOF | .03015 |
| sample-disjoint OOF | .02799 |
| source-disjoint OOF | .02925 |
| source OOF + shuffled label | .01926 |

正例率为.01897。row-random与source-disjoint只差+.00091，sample-disjoint反而低-.00126；没有
证据表明source grouping造成主要性能崩溃。shuffled-label AP几乎等于prevalence，负对照正常。

824个正机会分布于418/578个source，单source最多9个，top-10 source只占7.04%；不是少数source
垄断。五个source folds分别有146/154/169/164/191个正例，覆盖充分。

如果事先只在真实正机会中做conditional ranking，KO可在训练内捕获89.6% oracle reward；
row-random、sample-disjoint、source-disjoint分别捕获46.1%、46.2%、44.8%。这说明schedule身份
并非完全不可学；主要瓶颈是从全部horizon中发现约1.9%的机会。

## 两continuation target稳定性

center-future和late-future分别产生921和907个正机会，交集824、并集1004，Jaccard=.821。两边
都有机会的824个horizon中，最佳schedule一致率94.8%。active candidate reward Pearson=.880，
sign agreement=78.6%，只有12个candidate出现正负号相反。robust target恰好保留两future机会
交集，并覆盖418个source。

因此，在当前两种冻结continuation之间，target并不表现出足以解释OOF失败的不稳定性。诊断把
问题缩小到：当前K/heatmap-U特征与线性ridge无法分离“是否存在机会”；ordered-U也未提供可靠
增量。

## 非线性容量诊断

随后冻结固定CPU MLP（409-64-16-1、10 epochs、无early stopping），只测试KO opportunity。

| 切分 | MLP AP | 同口径ridge AP |
|---|---:|---:|
| 训练内 | .27843 | .04832 |
| row-random OOF | .03247 | .03015 |
| sample-disjoint OOF | .02754 | .02799 |
| source-disjoint OOF | .02897 | .02925 |
| shuffled-label source OOF | .01997 | .01926 |

MLP能明显记忆训练行，但任何OOF都没有材料性提升；source相对ridge甚至-0.00028。row-source差仅
.00350，shuffled-label仍接近prevalence。由此可排除“只因线性ridge容量不足”的主要解释。
当前KO输入没有稳定、可跨样本复用的opportunity信号；继续换分类器或调阈值不值得。

证据：

- `p5_h4_oof_design_audit_v1_49faacc3/metrics.json`；
- `p5_h4_target_stability_audit_v1_49faacc3/metrics.json`；
- `p5_h4_nonlinear_capacity_audit_v1_49faacc3/metrics.json`。
