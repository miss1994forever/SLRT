# P6 heatmap可解释运动等预算WER replay v1

日期：2026-09-23。该实验在读取WER前冻结，使用前一诊断已经生成的source-disjoint OOF分数，
在完整train fit 6,378个样本上逐样本保留完全相同数量的窗口。它是post-diagnostic探索性
decision-relevance检验，不是独立确认；未读取calibration、dev、test或held-out，也未运行
unknown-EOS闭环。

| 策略 | Errors | WER | 相对uniform errors | 相对K errors |
|---|---:|---:|---:|---:|
| Uniform sparse | 2,761 | 5.5651% | — | +374 |
| K/history | 2,387 | 4.8112% | -374 | — |
| K+运动 | 2,391 | 4.8193% | -370 | +4 |
| K+质量 | 2,402 | 4.8415% | -359 | +15 |
| K+运动+质量 | 2,399 | 4.8354% | -362 | +12 |
| K+运动+运动² | **2,377** | **4.7911%** | -384 | -10 |
| K+time-shuffled运动 | 2,390 | 4.8173% | -371 | +3 |

K+运动+质量相对uniform少362 errors，CI `[-0.8650,-0.5946]` pp；但相对K多12 errors，
CI `[-0.0100,+0.0582]`，相对打乱运动多9 errors，CI `[-0.0181,+0.0547]`。因此六项主检查
只通过KMQ-vs-uniform两项，整体pattern失败。

K+运动平方数值上比K少10 errors（-0.0202 pp），但CI `[-0.0509,+0.0102]`跨0；它是预声明
次要比较，且改进远低于25-error门槛，不能升级为新主模型继续试验。

结论与旧ordered-U一致：heatmap信息可以提高JS回归指标，但没有产生超越K/history的可靠WER
收益。停止可解释heatmap scheduler分支，不解锁独立calibration、闭环或held-out。既然没有WER
增益，也不为该分支启动正式GPU端到端延时实验；保留已有positive-mass统计kernel的1.88 ms中位
成本作为工程可行性记录，但不作部署收益声明。

证据位于`Online/CSLR/results/phoenix-2014t_ISLR/p6_heatmap_interpretable_motion_wer_replay_v1_49faacc3/`。
