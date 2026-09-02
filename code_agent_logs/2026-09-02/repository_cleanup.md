# 仓库整理与 main 发布记录

## 目标

统一当前项目的目录职责、配置路径、README 和复现实验入口，同时保留原始 SLRT 子项目与已有实验审计链。

## 整理内容

- 根 README 改为本项目入口，明确 Phoenix 自适应步长与 CSL-Daily Top-800 R1 是两条不同实验线；
- 原始上游 README 移至 `docs/UPSTREAM_README.md`，保留论文与引用信息；
- 稳定文档集中到 `docs/`，历史实验过程继续保存在 `code_agent_logs/YYYY-MM-DD/`；
- 自适应步长方案移至 `docs/experiments/adaptive_stride/`；
- Top-800 流程移至 `docs/reproduction/`；
- 新增 Phoenix 自适应步长和 Top-800 R1 的 UUID 单卡复现脚本；
- 所有 Top-800 CSLR、CTC fusion 和 SLT 配置移除用户名相关绝对路径；
- 数据准备和后台训练脚本从脚本位置推导仓库目录，并支持环境变量覆盖；
- 新增 `artifacts/` 约定，外部 checkpoint 和凭据继续由 Git 忽略。

## 兼容性决定

没有大规模移动上游源码或历史配置。推荐入口通过配置索引固定，历史配置继续原路径保留，以免破坏旧命令、checkpoint 路径和实验日志。

## 验证

- 32 个 `Online/*/configs/*.yaml`：全部通过 YAML 解析；
- 6 个修改/新增 Shell 脚本：`bash -n` 通过；
- 根 README 与 `docs/` 本地 Markdown 链接：全部存在；
- `git diff --check`：通过；
- `Online/CSLR` 单元测试：11/11 通过。

未重新训练模型、未运行 test、未修改 checkpoint 或结果产物。

## Git 历史说明

整理前研究分支与远程 `main` 来自两个独立根提交。发布时使用保留双方父提交的合并提交连接历史，并保留当前整理后的工作树；这样 `origin/main` 可以正常快进，不需要强制推送，也不会丢失原远程主线历史。
