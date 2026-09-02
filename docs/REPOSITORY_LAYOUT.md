# 仓库结构与维护规范

## 目录职责

```text
SLRT/
├── Online/                    # 当前项目主代码：CSLR、CTC fusion、SLT
│   └── CSLR/
│       ├── configs/           # 组件配置；README 标明推荐与历史配置
│       ├── tools/             # 只读诊断和离线评估工具
│       ├── tests/             # 自适应步长与可靠性分析测试
│       └── results/           # 本地产物，Git 忽略
├── docs/                      # 面向使用者的稳定文档与复现入口
├── scripts/reproduce/         # 可复现实验脚本
├── code_agent_logs/YYYY-MM-DD # 按日期保存的实施和实验审计记录
├── data/                      # 本地数据，Git 忽略
├── pretrained_models/         # 本地预训练权重，Git 忽略
└── artifacts/checkpoints/     # 外部/迁移 checkpoint，Git 忽略
```

`CiCo/`、`NLA-SLR/`、`Spoken2Sign/` 和 `TwoStreamNetwork/` 是上游 SLRT 子项目。除非对应实验需要，不把当前 Online CSLR 的脚本或日志放进这些目录。

## 配置规则

1. 配置文件放在实际读取它的组件下，例如 `Online/CSLR/configs/`；
2. 命令从组件目录执行，配置中的仓库资产使用相对路径；
3. 输出统一写入组件的 `results/<experiment>/`；
4. 推荐配置、历史配置和 smoke 配置必须在同目录 README 中分类；
5. 训练用 train，参数选择用 dev，冻结后 test 只做最终报告；
6. 新实验不得覆盖旧 `model_dir` 或 `save_subdir`。

## 文件命名

- 稳定说明：放入 `docs/`，使用含义明确的英文文件名；
- 实验日志：`code_agent_logs/YYYY-MM-DD/<topic>.md`；
- 机器产物：放入 `results/`，不提交 CSV、PKL、logits、checkpoint 或运行日志；
- 本地密钥、下载链接和机器专用脚本：使用 `*.local.sh`，不提交。

## 兼容策略

现有源码和历史配置不做一次性大搬迁，以免破坏 checkpoint 加载和旧命令。整理采用“稳定入口集中、历史文件保留”的方式：README 指向唯一推荐配置，旧配置继续用于审计与复现历史结果。
