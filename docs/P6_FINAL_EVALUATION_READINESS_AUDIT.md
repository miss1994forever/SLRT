# P6 final evaluation readiness audit

日期：2026-09-24。用户授权继续一次性final held-out规划后，在读取任何新outcome前核对本地split
历史。结论是：当前仓库没有可以诚实称为“从未观察的final held-out”的Phoenix-2014T split，
因此没有执行新评价。

## 数据身份

- train共7,096样本、643个source，已经由冻结split全部分为fit 6,378/578 source和calibration
  718/65 source；两部分均已用于P6；
- dev 519样本已用于A0、P0、P1/P2及decoder-oracle等历史选择；
- test已有历史固定与自适应WER结果（22.0005%与23.0571%）。虽然修复后资产尚未重跑test，
  test样本、reference和历史outcome已经被研究过程观察，不能因资产hash更新重新命名为未见held-out；
- 本地没有第三个未消费的train source partition或外部同任务evaluation split。

## 冻结决定

不运行dev/test，不把修复后test称为独立confirmatory held-out。当前P6 causal online quantile结果
仍是完整train-fit source-disjoint OOF证据：相对online/matched uniform显著改善，但尚无真正外部
确认。

后续只有两种诚实选择：

1. 获得新的、未参与路线选择的连续手语数据或预先封存的新source/time split，再执行一次性确认；
2. 经用户明确授权，在修复后dev或test做**retrospective final evaluation**，明确披露split历史已被
   查看、多次方法选择和不可作为严格confirmatory evidence；结果无论好坏不得再调参。

本audit不产生WER、窗口数或显著性结果。
