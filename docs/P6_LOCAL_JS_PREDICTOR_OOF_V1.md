# P6 causal local-JS predictor source-disjoint OOF v1

日期：2026-09-23。连续target、四组特征、ridge、fold、逐样本预算、WER replay和六项门槛均在
predictor outcome前冻结。实验只使用完整train fit和CPU，没有读取calibration、dev、test、
held-out或closed-loop outcome。

183,401个完整block的target为`log(JS(offset0,offset2)+1e-8)`。输入截止在offset0：K为full-center
teacher-forced paid-logit history，KS加入summary-U，KO加入ordered DCT-U，KX加入time-shuffled-U。
五折按578个source严格分组。每个样本按OOF score保留`ceil(B/2)`个bonus，再用与P6 oracle相同的
span15解码；四组策略都与uniform sparse逐样本等预算。

| 模型 | OOF R2 | Errors | WER | vs uniform errors |
|---|---:|---:|---:|---:|
| Uniform sparse | — | 2,761 | 5.5651% | — |
| K | .25596 | **2,387** | **4.8112%** | **-374** |
| KS | .26714 | 2,399 | 4.8354% | -362 |
| KO | **.27474** | 2,418 | 4.8737% | -343 |
| KX | .25711 | **2,386** | **4.8092%** | **-375** |

KO相对uniform减少343 errors、-0.6914 pp，source bootstrap CI `[-0.8285,-0.5556]`，超过
125-error gate。但KO比K多31 errors，CI以WER pp计`[+0.0198,+0.1048]`；比KX多32，CI
`[+0.0204,+0.1078]`。因此六项门槛只通过KO-vs-uniform两项，整体gate失败。

结果表明local-JS是可预测且与预算分配相关的target；K/history的source-OOF策略在少约24%的选择
窗口时仅比full-center多62 errors。与此同时，heatmap-U没有增量：ordered-U提高连续R2却损害最终
WER，time-shuffled与K几乎相同。R2不能替代decision-level replay。

K/KX是预声明次要比较，但总预注册身份要求KO同时超过K与KX。不能在看到结果后把K改成主模型并
直接解锁闭环或held-out。若继续K-only路线，需要新的、明确标注为post-OOF选择且使用独立数据的
确认协议；当前结果只支持停止heatmap-U增量假设并保留local-JS/K作为候选方向。

证据：`Online/CSLR/results/phoenix-2014t_ISLR/p6_local_js_predictor_oof_v1_49faacc3/`。
