# Online Sign Language Recognition for iOS

本仓库基于 [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT)，用于实时手语视频到文本的 iOS 应用研究。当前维护重点是 `Online/CSLR`：Two-Stream S3D 孤立词识别、滑窗在线连续识别、自适应步长，以及后续可靠性/边界感知实验。

> 原始 SLRT 的论文列表、引用信息和各子项目入口保存在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。本项目改动不代表原作者的官方结果。

## 当前实验主线

仓库中有两条不同的数据线，结果不能混用：

| 实验 | 数据集与划分 | 任务 | 当前结论 |
|---|---|---|---|
| 自适应步长冻结基线 | Phoenix-2014T dev/test | 在线 CSLR，指标为 WER | test：固定步长 22.0005%，自适应 23.0571%；clips 减少 32.02% |
| R1 可靠性诊断 | CSL-Daily Top-800 isolated dev | ISLR，指标为 accuracy/AUROC | Keypoint accuracy 66.28%；原始置信度直接选流无收益 |

因此，当前**尚未**得到 CSL-Daily Top-800 自适应步长 WER，也不能用 Top-800 R1 的分类准确率解释 Phoenix 的 WER。

## 推荐入口

- [仓库结构与目录规范](docs/REPOSITORY_LAYOUT.md)
- [环境、资产和实验复现](docs/REPRODUCIBILITY.md)
- [文档索引](docs/README.md)
- [Online CSLR 代码说明](Online/CSLR/README.md)
- [配置索引](Online/CSLR/configs/README.md)
- [实验日志索引](code_agent_logs/README.md)

## 最小验证

以下命令从仓库根目录执行：

```bash
(cd Online/CSLR && python -m unittest discover -s tests -p 'test_*.py' -v)
```

需要 GPU 的复现实验必须用 UUID 显式绑定，不能依赖会变化的逻辑编号：

```bash
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  bash scripts/reproduce/reliability_r1_dev.sh
```

本节点已知故障卡 PCI `01:00.0` 与 `25:00.0` 禁止使用。脚本也会拒绝两张已登记故障卡的 UUID。

## 目录约定

- 源码：保持在原 SLRT 子项目目录，当前项目代码集中在 `Online/`；
- 配置：跟随对应组件放在 `<component>/configs/`；
- 可复现说明：统一放在 `docs/`；
- 历史实验记录：统一放在 `code_agent_logs/YYYY-MM-DD/`；
- 数据、checkpoint、logits 和 `results/`：只保存在本地，不进入 Git。

所有推荐命令都从对应组件目录运行，例如 `Online/CSLR`。推荐配置只使用仓库相对路径，克隆位置不需要固定为 `/home/haojun/projects/SLRT`。
