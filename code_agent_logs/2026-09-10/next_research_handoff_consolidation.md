# 下一阶段研究 hand-off 收束

日期：2026-09-10

## 完成内容

将 `docs/NEXT_RESEARCH_HANDOFF.md` 从早期研究导向说明更新为新工作对话的单一交接入口，并结合本轮对话已完成的工作统一以下事实：

- 修复后 Phoenix dev A0 v1 的不可变数据、模型、协议和参数身份；
- B0--B4/A0 正确性矩阵和 B0/B2/A0 三重复 runtime；
- A0 与等预算 B2 的统计边界和逐样本机制诊断；
- 修复前 61 帧缺失数据及历史结果归档边界；
- 历史 test 只能作为 retrospective control，尚无修复后正式 test；
- Phoenix CSLR 与 CSL-Daily Top-800 ISLR 两条数据线不可混用；
- 已知故障 GPU、NFS 高占用和 `/tmp` 临时产物策略；
- 新对话应优先执行的 P0 oracle、随机采样和边界命中诊断。

同步在仓库根 README 和 docs 索引加入 hand-off 入口。未运行新实验，未删除或移动结果，未修改 A0 算法、冻结配置或机器结果，未提交或推送。

## 交接边界

新对话应先审阅并提交当前整组冻结、恢复、归档和 hand-off 修改，再开展下一代算法。后续实验不得覆盖 A0 v1，不得使用 test 调参；如果改变冻结项，必须新建版本、manifest 和独立结果目录。
