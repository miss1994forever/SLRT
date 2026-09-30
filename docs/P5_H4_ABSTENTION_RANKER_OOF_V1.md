# P5 H4 rare-event abstention + conditional ranker OOF v1

日期：2026-09-23。本实验是在观察到direct ranker过度离开center之后提出的post-OOF探索新身份，
不是独立确认。配置在abstention outcome前冻结；只使用相同train-fit五折和CPU，没有读取
calibration、dev、test、held-out或closed-loop outcome。

## 方法

第一阶段用class-weighted ridge预测一个horizon是否存在任意`robust_value>0`的schedule。每个外折
固定选择数量为`round(eval行数 × 外折训练侧正例率)`，不调阈值。第二阶段只用外折训练侧正机会
拟合81-output conditional ranker，并且只在第一阶段选中的行执行；其余全部保留center。

四组输入仍为K、K+summary-U、K+ordered-U和K+time-shuffled-U。43,437行中正机会824，正例率
1.897%。

## 结果

| 模型 | Detector AP | 固定预算正例召回 | Signed reward | 95% source bootstrap |
|---|---:|---:|---:|---:|
| K | .03188 | 5.34% | -15 | [-31, 0] |
| KS | .03205 | 4.85% | -15 | [-29, -2] |
| KO | **.02925** | **4.98%** | **-11** | **[-28, 6]** |
| KX | .02764 | 3.88% | -14 | [-27, -2] |

KO相对K的AP为-0.00264、reward为+4，bootstrap `[-14,21]`；相对KX的AP为+0.00160、reward
为+3，bootstrap `[-14,20]`。KO在823个固定动作中只得到22个正reward、35个负reward；其余为0。
九项门槛全部失败。

## 结论

限制动作率显著降低了direct ranker的过度行动，但没有解决机会识别或schedule选择问题。heatmap-U
对正机会有极弱ranking lift，ordered-U没有超过K，且相对shuffled-U的差异很小、不确定。当前证据
支持停止这组H4 future/reference/EOS-derived utility标签的因果预测链；不读取calibration，不运行
闭环或held-out。

证据：`Online/CSLR/results/phoenix-2014t_ISLR/p5_h4_abstention_ranker_oof_v1_49faacc3/`。
