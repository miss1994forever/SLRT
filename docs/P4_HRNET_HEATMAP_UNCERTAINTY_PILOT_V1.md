# P4 HRNet raw-heatmap uncertainty pilot v1

日期：2026-09-21。实验在任何 WER、utility、reference、dev/test 或 held-out outcome 之外完成。

## 目的与冻结范围

检验原始 SLRT MMPose HRNet-W48 WholeBody 是否能直接返回 raw heatmap，测量多卡提取成本，并
判断 B-KinD 风格 confidence/covariance 是否值得形成新观测族。预注册配置：
`Online/CSLR/configs/p4_hrnet_heatmap_uncertainty_pilot_v1_preregistered.json`。

- train-only，按 SHA-256 固定选择50个视频；
- 每视频均匀选择最多100帧，共4,498帧；
- Faster R-CNN + HRNet-W48 384x288 DARK-plus，与原 SLRT 提取链一致；
- 5张 RTX 3090：PCI `61:00.0`、`81:00.0`、`A1:00.0`、`C1:00.0`、`E1:00.0`；
- 未使用宿主机 GPU0 或 PCI `25:00.0`、`41:00.0`；
- 不保存每帧完整 heatmap；保存133点各8项统计，另对每视频前2个采样帧保存双手 raw heatmap。

## 结果

- 4,498/4,498帧成功，人体检测失败0；
- 5卡推理墙钟159.35秒，aggregate 28.23 FPS；
- 单卡有效吞吐约5.96 FPS；
- 完整 train 827,354帧线性投影：单卡38.55小时，5卡8.14小时；
- 每帧中位耗时：decode 1.56 ms、detector 46.64 ms、pose+heatmap 102.03 ms、CPU统计
  10.34 ms、总计162.45 ms；
- 与冻结坐标比较（双方 confidence >.2）：face 中位0.275 px，hands 中位2.177 px、P95
  16.228 px；hand confidence 相关系数 .723。

## 归一化审计

直接把 HRNet score map 以 `softmax(T=1)` 归一化会近似均匀：理论均匀熵8.8410，实际中位
8.8380。它的双手 covariance trace 中位1192.34，不能直接冻结为有意义的不确定性。

在100个预注册 raw-hand heatmap 上，positive-mass normalization 的 covariance trace 中位71.69、
entropy 5.52、peak probability .0126；温度 softmax 对结果高度敏感。因此 pilot 证明 raw heatmap
可取得且成本可控，但没有证明某一 covariance 定义有效。下一步必须在不读 outcome 前单独预注册
positive-mass 主定义和温度敏感性，不得在下游 OOF 后选择归一化。

证据：

- `p4_hrnet_heatmap_uncertainty_pilot_v1_49faacc3/metrics.json`；
- `p4_hrnet_heatmap_uncertainty_pilot_v1_49faacc3/normalization_diagnostics.json`。
