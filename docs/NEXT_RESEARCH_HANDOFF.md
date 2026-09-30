# 下一阶段研究交接：重设计可部署调度 target

更新时间：2026-09-23

仓库：`/mnt/workspace/projects/haojun/SLRT`

本文件只描述当前冻结状态和下一步。完整数字、协议冲突与产物路径以
[`ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md`](ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md)
及对应机器可读 metrics 为准。

## 0. 交给新对话的任务

> 核对完整 train robust oracle、skeleton/RGB/DCV predictor 和 utility information ladder 的
> 冻结结果。不要继续调当前 per-window counterfactual utility classifier。先提出并预注册一个
> 决策时可观测、标签更稳定的新 target 或序列级决策分解，明确因果可见性、source-disjoint OOF、
> 等预算 gate 和停止条件；在 train OOF 达到冻结 headroom 前不得运行闭环或读取新的 held-out
> outcome。

开始前确认工作区干净、资产哈希一致且没有冲突实验。所有 `nvidia-smi`、CUDA 检查和 GPU
实验必须在沙盒外运行，避开 PCI `25:00.0`、`41:00.0` 和宿主机 GPU0。沙盒内 CUDA 不可见
不能判定驱动故障。

## 1. 当前冻结结论

| 阶段 | 结论 |
|---|---|
| A0 等预算基线 | 减少约三分之一窗口，但没有证据显著优于等预算 uniform |
| P0 sign-center oracle | alignment-derived offline 上界强；不可部署 |
| P1/P2 center predictor | 因果特征改善 detection，但都未通过 gate |
| P3 robust continuation oracle | 完整 train fit 上 errors `2325→1688`，少637；窗口选择上界成立 |
| P3 robust labels | 366,802 rows：769 beneficial、2,015 harmful、364,018 neutral |
| P3 deployable predictors | K、RGB-DCT、finger RGB、face RGB、DCV 均为非正 top-K utility |
| P3 information ladder | 最丰富 L3 utility `+5`，远低于冻结的 `249` gate |
| P4 CSC-L8 target audit | 语义曲率 target 稠密且稳定；A1 全部 gate 通过 |
| P4 CSC-L8 identifiability | OOF R2 `.0478`、winner accuracy `.4164`；A2 gate 失败，未读 policy outcome |
| P4 HRNet heatmap pilot | 4,498 train帧5卡成功；28.23 FPS，完整 train 约8.14小时；raw heatmap 可用但 softmax T=1 退化近均匀 |
| P4 positive-mass/full train | v3全部11项门控通过；全量827,354帧100%有效，5卡7小时26分，4.573 GB |
| P4 heatmap-U predictor | K+U R2 `.03508`、accuracy `.41794`；相对K仅`+.00722` R2且accuracy下降；1/5主门控通过，停止 |
| P4 CSC→WER oracle relevance | 完美CSC winner使errors `2325→2224`、WER `-0.2036 pp`，CI `[-0.3100,-0.0965]`；关系显著但仅少101 errors，未达249 gate，停止CSC target |
| P5 H4 joint robust oracle | 完整train等预算：uniform `2325`、blockwise robust `1688`、H4 joint `1447` errors；H4相对blockwise再少241，CI `[-0.5511,-0.4223]` pp，全部gate通过 |
| P5 H4 schedule-ranker OOF | 43,437个H4 horizon、578-source五折；K/KS/KO/KX signed robust reward分别`-40/-24/-51/-19`；ordered-U弱于K和time-shuffled，6/6 gate失败 |
| P5 H4 abstention OOF | 固定约1.9%动作率；KO detector AP `.02925`、正机会召回4.98%、reward `-11`，相对K仅+4、相对shuffled仅+3；9/9 gate失败 |
| P5 OOF/target诊断 | KO AP：train `.0483`、row `.0302`、sample `.0280`、source `.0292`；shuffled-label `.0193≈prevalence`。row-source无gap；两future机会Jaccard `.821`、best schedule一致94.8% |
| P5 nonlinear capacity诊断 | 固定CPU MLP：train AP `.2784`，但row/sample/source仅`.0325/.0275/.0290`；source比ridge低`.00028`，shuffled-label `.0200`；只有记忆、无泛化增益 |
| P6 local-JS relevance oracle | 只保留一半center bonus：uniform sparse `2761`、JS oracle `2562` errors，少199/`-0.4011 pp`，CI `[-0.5340,-0.2642]`；全部gate通过 |
| P6 local-JS predictor OOF | K/KS/KO/KX errors=`2387/2399/2418/2386`，均优于uniform 2761；KO-vs-uniform少343，但KO比K/KX多31/32；ordered-U增量失败，总gate 2/6通过 |
| P6 heatmap运动诊断 | 显式部位运动M单独R2 `.0142`；K+M相对K `+.00474`，time-shuffled仅`+.00074`；K+M+质量为`+.01224`。存在弱时序信号，但不解锁policy/WER |
| P6 heatmap运动WER诊断 | K/KM/KQ/KMQ/KMM2/KX errors=`2387/2391/2402/2399/2377/2390`；KMQ比K多12且CI跨0；KMM2少10但CI跨0。停止heatmap scheduler分支 |
| P6 K独立calibration | 718样本/65新source：R2 `.2585`；uniform/K/full-center errors=`331/283/287`；K-vs-uniform -48，CI全负，3/3 gate通过 |
| P6 K unknown-EOS闭环 | self-history K/online-uniform/matched-uniform/full-center errors=`311/325/330/287`；K方向较好但两个CI上界略大于0，3/5 gate通过；停止K路线 |
| P6 K controller/state诊断 | TF-topK/TF-token/SH-token errors=`283/312/311`；controller shift +29、state shift -1；主要瓶颈是固定阈值/pacing，不是self-history |
| P6 K online quantile OOF | unknown-EOS self-history quantile/online-uniform/matched/full errors=`2543/2758/2773/2325`；少215/230 errors，CI全负，5/5 gate通过 |
| P6 final-evaluation readiness | train fit/cal均已消费，dev多次用于历史选择，test已有历史WER；本地无真正未见held-out，未运行新评价 |

因此，失败的不是“视觉信息完全无效”，而是视觉/decoder 可见线索与当前 robust counterfactual
utility 标签之间的映射过弱且高度稀疏。oracle 使用 future continuation、reference 和真实 EOS
生成标签；部署 predictor 不能看到这些变量。同一 action 的 utility 还会随 continuation 和 decoder
设置变化，所以继续堆叠同类特征或调阈值不太可能解决根因。

P5进一步确认决策单位也是瓶颈：把四个连续block联合枚举后，oracle相对逐block robust多减少
241 errors。但随后冻结的H4 schedule-ranker OOF仍失败：H4正headroom只出现在824/43,437个
horizon，四个ranker都得到负signed robust reward；ordered heatmap-U为-51，弱于K的-40和
time-shuffled U的-19。联合target有上界，但当前因果表示仍不能可靠识别稀疏正事件。

失败后以新身份测试了rare-event detector + conditional ranker。每折只允许按训练正例率选择约
1.9%的horizon，但KO仍只召回4.98%正机会并得到-11 reward；相对K和shuffled的reward增量仅+4
和+3且CI跨0。因而问题不只是direct ranker缺少abstention，当前输入也不能定位收益机会。

随后诊断确认OOF设计不是主要失败源：row-random、sample-disjoint、source-disjoint的KO AP均约
.028--.030，source切分没有额外崩溃；label-shuffled回到prevalence。两种future continuation的
机会与最佳schedule也高度一致。当前剩余关键分叉是“线性模型容量不足”还是“输入缺少机会信息”。

固定非线性CPU MLP解决了该分叉：训练内AP升到.2784，但三种OOF仍约.028--.032，source甚至
略低于ridge。说明网络能记忆训练行，却没有可泛化的KO opportunity规律。不得继续在当前
target/输入上更换分类器；后续必须换target，或引入信息内容实质不同且先做独立可观测性审计的输入。

新target P6取得正结果：以offset0/offset2 posterior JS作为局部信息增益，在逐样本完全等预算、
只保留一半bonus时，相对uniform sparse减少199 errors。score不读reference或终端future，oracle
relevance gate通过；这正式解锁source-disjoint local-JS predictor，但尚未解锁闭环或held-out。

P6 predictor随后完成。local-JS连续target有稳定OOF R2（K .256、KO .275），四组等预算replay都
大幅优于uniform sparse；K为2,387 errors，距full-center仅62。但ordered-U虽然R2最高，最终WER
反而比K和time-shuffled差31/32 errors。预注册要求KO同时证明heatmap时序增量，所以总gate失败，
不得直接运行闭环或held-out。local-JS/K方向保留为需要独立确认的post-OOF候选。

随后以纯诊断身份显式提取heatmap质心运动，修正了旧KS/KO没有直接计算相邻帧位移的问题。
body/face/左右手共32维运动M、36维质量Q在同一183,401行source-OOF上，M/Q/MQ的R2分别为
`.01419/.01869/.02953`；K+M、K+Q、K+M+Q相对K分别增加`.00474/.00763/.01224`，而
K+time-shuffled-M只增加`.00074`。右手最近一步运动的单变量相关最高（rho `.1064`），但分位
曲线总体是从静止上升后平台或回落，不支持“越大越重要”的单调规则。该实验没有读取WER或
reference，且预注册为no-unlock诊断；不能用它绕过P6 gate。

之后冻结并运行同一train-fit上的post-diagnostic等预算WER replay。K/KM/KQ/KMQ/KMM2/KX为
`2387/2391/2402/2399/2377/2390` errors；KMQ相对K多12、相对time-shuffled多9，两个CI均跨0，
主检查2/6通过。KMM2虽比K少10 errors，但CI跨0且是次要比较，不能post-hoc升级。显式运动再次
呈现“JS R2小幅提高但WER无可靠收益”，故停止heatmap scheduler分支，不运行其正式延时测量、
独立calibration、闭环或held-out。

用户随后明确继续K-only独立确认。此前未用于P6的718个calibration样本/65个独立source上，
fit-only ridge得到JS R2 `.25851`；逐样本等预算uniform/K/full-center为`331/283/287` errors，K相对
uniform少48、CI `[-1.2802,-.4564]` pp，3/3门槛通过。随后按解锁协议运行固定OOF中位阈值、
token初始0/每block+.5/容量2/no-topup的unknown-EOS自身历史模拟：K/online-uniform/
matched-uniform/full-center为`311/325/330/287` errors。K少14/19 errors但两个CI上界为
`+.0188/+.0174`，五项门槛仅3项通过。已知长度强信号未完整转成部署控制器，停止K路线，
不得调threshold/token或读取held-out。

为拆分28-error闭环退化，随后做固定factorial audit：teacher-forced top-K、teacher-forced token、
self-history token分别为`283/312/311` errors。controller shift为+29且CI全正，state shift为-1且CI
跨0；固定schedule上的两种分数Pearson `.9739`。因此主要失败来自全局top-K到固定绝对阈值/pacing
的转换，而非self-history。

据此建立新的fit-only因果分位控制器：每折模型只训另四个source fold，每个视频用严格过去分数
中位数代替全局阈值，保留unknown-EOS、token容量2和no-topup。完整fit上quantile/online-uniform/
matched-uniform/full-center为`2543/2758/2773/2325` errors；quantile少215/230 errors，CI分别
`[-.5316,-.3344]`和`[-.5839,-.3412]`，同时少于online uniform 4,425 windows。5/5 gate通过。
它仍比full center多218 errors，是精度—效率折中。当前只解锁一次单独冻结的final held-out规划；
在用户明确授权前不得读取held-out。

用户随后授权继续final held-out规划，但outcome读取前的readiness audit发现本地没有真正未见split：
train 7,096样本已全部进入fit/calibration；dev已被A0/P0/P1/P2等反复使用；test已有历史WER，不能因
修复后hash尚未重跑就重新称为held-out。因此没有运行dev/test。若继续，只能引入新的外部/未来
封存数据，或由用户另行授权一个明确标注为retrospective、非confirmatory的修复后dev/test评价。

## 2. 已停止的方向

- 不再调整当前 K/R/HF/RF/DCV/L1--L3 的 crop、history、network、seed 或 top-K threshold；
- 不把 K+RGB 等次要组合升级成新的主检验；
- 不在同一 OOF outcome 上进行 post-hoc model selection；
- 不运行 unknown-EOS closed loop、held-out WER 或 test；
- 不把 oracle improvement 描述成可部署 scheduler improvement。
- 不继续训练或调参 `CSC-L8-v1`；它不仅在 outcome-blind identifiability gate 失败，完美 winner
  的等预算 WER 上界也未达到冻结的249-error实用门槛。

这些约束不禁止复用已经验证的因果 skeleton/RGB encoder 作为新 target 的输入，但必须建立新版本、
新预注册配置和新输出目录。

## 3. 下一步设计要求

下一项实验必须先回答“训练目标是否在部署时可辨识”，再回答“模型是否足够大”。候选方案可以是：

1. 把单步精确 utility 改成更稳定的 coarse target，例如 future-independent 的信息增益、稳定边界区间、
   commit risk 或预算压力；
2. 把独立候选分类改成序列级 credit assignment，让模型优化一段轨迹的累计预算收益；
3. 把动作拆成 coverage 保底与少量 bonus allocation，只学习可验证的相对偏好或 abstention；
4. 在不读取 outcome 的前提下先做 label-stability/identifiability audit，若可部署观测对新 target 仍无
   信息量，则在训练前停止。

`CSC-L8-v1` 已严格走完第4项并停止：标签稳定不等于可辨识。下一个实验必须建立新的 target 或
观测身份，不能在已查看的 A2 OOF 上调整 CSC projection、ridge、feature pool 或 gate。

HRNet raw-heatmap uncertainty 已完成独立 train-only 可行性 pilot。若继续，必须先冻结
positive-mass covariance/entropy 定义及数值稳定性 gate；不能把 fixed-Gaussian SLRT heatmap 当作
检测器不确定性，也不能直接使用已经证明近均匀的 `softmax(T=1)` covariance。

positive-mass v1 已按上述要求在同一4,498帧上完成：无效质量为0、协方差全部 finite/PSD，
trace/entropy 非常数，且不与 raw peak 或 bbox 面积退化共线；CPU kernel 中位15.26 ms/帧，
未通过5 ms门槛。独立 v2 CUDA float32 实现达到1.65 ms，但严格 v1 parity 以
`3.05e-5 > 2e-5` 失败，没有放宽门槛。v3 CUDA mixed-precision 达到1.88 ms、P95 3.03 ms，
moment/entropy/mass parity 及其余数值门控全部通过，正式解锁完整 train 提取。完整827,354帧
5卡线性投影约7.55小时。全量任务实际于2026-09-21 20:44 UTC完成：7,096视频、827,354帧
全部有效，墙钟7小时26分钟，输出4.573 GB，completion gate通过。

后续 heatmap-U identifiability 在全量结果前冻结为
`p4_csc_l8_heatmap_u_identifiability_v2_preregistered.json`，并已完成。逐帧 U 档案覆盖6,378个
fit视频/743,147帧；181,815 blocks、578 sources、同一source-disjoint五折上，K/U/K+U的
pooled R2分别为`.02786/.02520/.03508`，winner accuracy为`.41810/.41792/.41794`。
K+U相对K仅`+.00722` R2，accuracy为`-.000154`；只有5/5折R2为正这一项通过，其余四项
主门控失败。必须停止该heatmap-U CSC-L8分支，不得运行policy outcome、闭环或held-out，也不得
在同一结果上调整U pooling、history、projection、ridge或gate。

为先验证 target 与最终识别目标的关系，另以新身份、在任何 CSC policy/WER outcome 产生前冻结了
`p4_csc_l8_oracle_relevance_v1_preregistered.json`。在完整train fit的6,378个样本上，以逐样本完全
等预算的uniform offset-2为基线，用真实CSC分数的argmax构造“完美winner”上界。基线为2,325
errors、WER 4.6863%；完美CSC为2,224 errors、WER 4.4827%，即少101 errors、下降0.2036 pp，
source-cluster bootstrap 95% CI为`[-0.3100,-0.0965]`。统计关系为正，但101低于预注册249-error
门槛，因此 relevance gate 失败。不得再训练CSC predictor、运行CSC闭环或读取held-out；这不是
“CSC与WER无关”，而是“即使预测完美，其等预算收益也不足以支持当前研究目标”。

随后预注册并完成P5 H4 robust joint-schedule oracle。完整train fit上，H4精确枚举连续四block的
81种bonus schedule，并在fixed-center/fixed-late两种future continuation下最大化最差终端收益。
它精确复现uniform 2,325和旧blockwise robust 1,688 errors，H4得到1,447 errors（WER 2.9166%）。
相对uniform少878 errors、`-1.7697 pp`，CI `[-1.8992,-1.6424]`；相对blockwise再少241 errors、
`-0.4858 pp`，CI `[-0.5511,-0.4223]`。全部冻结gate通过，解锁H4 schedule predictor和保序
heatmap-U比较，但仍未解锁闭环或held-out。

之后已完成标签集和预注册H4 schedule-ranker OOF。43,437个完整H4 horizon中只有824个具有正
oracle robust value。K、K+summary-U、K+ordered-U、K+time-shuffled-U的signed reward分别为
`-40/-24/-51/-19`；ordered-U相对K为-11、相对打乱对照为-32，六项gate全部失败。因此H4
predictor分支没有解锁闭环或held-out。当前关键问题是稀疏正事件的可靠abstention，而不是继续在
同一OOF上调ranker阈值。

无论选择哪一项，都必须冻结：

- train-only 身份、578-source grouping 和 source-disjoint folds；
- predictor 可见信息、lookahead（0/4/8 必须分开命名）与 unknown-EOS 行为；
- uniform coverage skeleton、逐样本预算和 action deadline；
- 主指标、249-error headroom、cluster bootstrap CI 与停止规则；
- controller 固定成本、逐帧执行频率、缓存复用和真实部署成本口径。

## 4. 解锁顺序

1. H4 joint oracle上界和schedule标签完整性已通过；
2. source-disjoint schedule-level OOF已运行，K/history、summary-U、ordered-U和time-shuffled-U全部失败；
3. direct ranker和post-OOF rare-event/abstention均失败，**不得运行** unknown-EOS token-bucket闭环、calibration或held-out；停止当前H4 utility预测链；
4. 非线性容量probe也已失败；停止当前H4 utility/opportunity target的模型迭代；
5. P6 local-JS relevance和K-based OOF均强，但ordered heatmap增量失败，预注册总gate未通过；
6. 固定绝对阈值闭环失败后，factorial audit定位为controller/pacing问题；新建的因果视频内分位控制器在fit source-OOF 5/5 gate通过；
7. final-evaluation readiness audit确认本地不存在真正未见held-out，因此未执行新评价；
8. 只有获得新封存数据，或用户明确授权“retrospective、非confirmatory”的修复后dev/test评价，才可继续；后者必须声明多次路线选择风险且只运行一次、不得调参；
9. 未来新target闭环必须先与逐样本等预算uniform比较，并分别报告0/4/8 lookahead；
10. 新target闭环通过后，才允许读取冻结 held-out outcome，并测量 wall time、controller cost、队列积压和
   P95稳定提交延迟；
11. test 和论文级多重比较校正留到方法完全冻结之后。

## 5. 不可变身份

- Phoenix 视频 ZIP SHA-256：
  `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`；
- HRNet whole-body train keypoints SHA-256：
  `18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763`；
- ISLR checkpoint SHA-256：
  `b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b`；
- frozen skeleton OOF SHA-256：
  `9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a`；
- 当前窗口为16帧，现有 utility 链使用 triangular span-weighted-15 和 bounded `+8` candidate
  lookahead，不能称为严格 zero-lookahead。

改变数据、checkpoint、decoder、窗口、预算、action space 或 target 时必须建立新实验身份，不能
覆盖现有结果目录。

## 6. 允许与禁止的声明

可以说：完整 train 上存在强 robust oracle headroom；现有可部署特征族无法可靠预测逐窗口
counterfactual utility；细粒度 RGB 相比 RGB-DCT 有局部改善，但没有超过 skeleton K 或达到 gate。

不能说：视觉对在线 CSLR 没有价值、decoder-aware 调度不存在、当前 predictor 已经接受独立
held-out 否证，或沙盒内 CUDA 不可见表示驱动故障。
