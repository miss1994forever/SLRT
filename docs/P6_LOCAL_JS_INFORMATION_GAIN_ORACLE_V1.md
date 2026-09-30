# P6 local posterior-JS information-gain oracle v1

日期：2026-09-23。配置在oracle outcome前冻结。实验只使用完整train fit和已有dense logits，CPU
运行；没有使用GPU，也没有读取calibration、dev、test、held-out或closed-loop outcome。

## 目标与协议

每个完整四窗口block固定选择offset0 skeleton。训练target是在offset0与offset2的ISLR softmax
posterior之间计算自然对数单位的symmetric Jensen-Shannon divergence。它不使用reference、decoder
future continuation、offset2之后的未来或真实EOS；但训练target需要观察候选offset2，因此部署时
仍需predictor在付费前估计。

每个样本有B个完整block，只保留`ceil(B/2)`个offset2 bonus。JS oracle选score最大的block，
equal-budget uniform在时间上均匀放置相同数量bonus。两者逐样本窗口数完全一致；full-center控制在
每个完整block都保留offset2，并要求复现2,325 errors。

## 结果

| 策略 | 选择窗口 | Errors | WER |
|---|---:|---:|---:|
| Full center | 371,594 | 2,325 | 4.6863% |
| Equal-budget uniform sparse | 281,480 | 2,761 | 5.5651% |
| Local-JS oracle | 281,480 | **2,562** | **5.1640%** |

JS oracle相对uniform sparse减少199 errors、`-0.4011 pp`，578-source cluster bootstrap 95% CI
`[-0.5340,-0.2642]`。预注册要求至少减少125 errors且CI上界<0，两项均通过，全部完整性检查也
通过。

JS oracle仍比full center多237 errors（约+0.4777 pp），所以当前结果不是“无损减少bonus”。它证明
的是：在只保留一半bonus的相同预算下，一个局部、无reference的信息增益target能显著优于均匀
分配。这与此前难以预测的终端utility不同，值得进入source-disjoint causal predictor检验。

下一步只解锁local-JS predictor：在offset0时用K/history、summary-U、ordered-U和time-shuffled-U
预测连续JS或sample内ranking。必须先通过train-fit source-disjoint OOF和等预算replay，才允许
设计unknown-EOS token bucket；本oracle本身不解锁闭环或held-out。

证据：`Online/CSLR/results/phoenix-2014t_ISLR/p6_local_js_information_gain_oracle_v1_49faacc3/`。
