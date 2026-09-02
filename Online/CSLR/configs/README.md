# Online CSLR 配置索引

所有命令从 `Online/CSLR` 执行。配置中的相对路径也以该目录为基准。

## 推荐配置

| 配置 | 用途 | 数据集 |
|---|---|---|
| `phoenix-2014t_ISLR.yaml` | Phoenix isolated 模型训练 | Phoenix-2014T |
| `slide_phoenix-2014t.yaml` | 固定/自适应滑窗评估 | Phoenix-2014T |
| `csl-daily-top-800_ISLR_full_stable.yaml` | 当前 Top-800 ISLR 与 R1 checkpoint | CSL-Daily Top-800 |
| `slide_csl-daily-top-800_full_stable.yaml` | Top-800 连续滑窗入口 | CSL-Daily Top-800 |

## 历史和诊断配置

- `*_smoke.yaml`：小规模链路验证；
- `*_subset8192.yaml`、`*_subset32768.yaml`：历史子集训练；
- `*_first_fault.yaml`：故障诊断保留，不作为新实验起点；
- `*_full_lr1e-4.yaml`：历史续训候选；
- 无 `full_stable` 后缀的 Top-800 配置：早期流程兼容入口。

新实验应从推荐配置复制到新的文件名，并修改 `training.model_dir`，不能覆盖既有结果目录。数据路径使用 `../../data/...`，预训练模型使用 `../../pretrained_models/...`，外部迁移 checkpoint 使用 `../../artifacts/checkpoints/...`。

自适应步长默认关闭。复现冻结实验时使用命令行显式传入 `--adaptive_stride 1 --span_weighted_voting 1 --vote_span_frames 15`，避免配置默认值被误认为已经启用。
