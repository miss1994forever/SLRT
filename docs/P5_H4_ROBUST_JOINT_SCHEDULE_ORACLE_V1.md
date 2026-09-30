# P5 H4 robust joint-schedule oracle v1

日期：2026-09-22。配置在任何本实验oracle outcome产生前冻结。实验只使用完整train fit，CPU
重放已有dense logits；没有使用GPU，也没有读取calibration、dev、test或held-out。

## 问题

旧robust utility oracle逐block选择一个bonus窗口，无法表达多个连续block之间的替代、互补和
decoder状态联系。本实验把连续四个完整block组成一个horizon，对每个block仍固定选择offset 0
skeleton，再从offset 1/2/3选择一个bonus。H4共精确枚举`3^4=81`种schedule；尾部1--3个完整
block分别枚举`3^H`种。

每个schedule在fixed-center和fixed-late两种future continuation下重放终端edit error。候选相对
all-center schedule在两种future下的改善取最小值；只有该robust value严格大于0才允许改变
schedule。并列时依次选择：robust value更大、两种reward之和更大、非中心bonus更少、offset tuple
字典序更早。该oracle读取reference、future continuation和真实序列终点，不可部署。

预注册配置：
`Online/CSLR/configs/p5_h4_robust_joint_schedule_oracle_v1_preregistered.json`。

## 完整性

- 6,378 samples、578 sources、reference length 49,613；
- uniform精确复现2,325 errors；
- 旧blockwise robust oracle精确复现1,688 errors；
- 三种策略逐样本选择窗口数完全相同；总计371,594/743,147窗口；
- 56/56 shards完成，metrics与per-sample压缩档案哈希和manifest一致。

## 结果

| 策略 | Errors | WER |
|---|---:|---:|
| 等预算uniform offset-2 | 2,325 | 4.6863% |
| 旧blockwise robust oracle | 1,688 | 3.4023% |
| H4 joint robust oracle | **1,447** | **2.9166%** |

H4相对uniform：`-878 errors / -1.7697 pp`，578-source cluster bootstrap 95% CI
`[-1.8992,-1.6424]`。H4相对旧blockwise oracle：`-241 errors / -0.4858 pp`，CI
`[-0.5511,-0.4223]`。

48,229个horizon中只有857个改变了all-center schedule；共1,097个非中心bonus，累计robust
horizon reward为877。全部预注册门控通过：总headroom超过249 errors、相对blockwise新增收益
超过125 errors，且两组CI上界均小于0。

## 结论与边界

多窗口联系不是此前small-dev pair实验中的微弱偶然：在完整train fit、robust双continuation和
严格等预算条件下，四block联合选择显著优于逐blockutility oracle。这只证明schedule-level target
具有更强上界，不证明因果模型能够预测，也不证明真实online收益。

下一步只解锁一个新身份的source-disjoint H4 schedule predictor。它应直接给整段schedule排序，
并比较K/history、静态summary-U、保序temporal-U及time-shuffled U；未达到冻结249-error
可捕获收益前，不运行unknown-EOS闭环或held-out。

证据：

- `p5_h4_robust_joint_schedule_oracle_v1_49faacc3/metrics.json`；
- 同目录`manifest.json`和`per_sample.jsonl.gz`。
