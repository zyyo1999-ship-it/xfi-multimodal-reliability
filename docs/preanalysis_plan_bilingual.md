# Formal Multimodal Pre-analysis Plan / 正式多模态预分析计划

Protocol version / 协议版本：`Track-B-v1.2`  
Frozen before real multimodal inference / 冻结时间早于真实多模态推理：2026-08-03  
Machine-readable source / 机器可读协议：`configs/multimodal_lower_limb_formal.yaml`

> This document specifies the confirmatory analysis before any real LiDAR +
> mmWave result is observed. Synthetic smoke-test outputs validate software only
> and are excluded from scientific evidence.
>
> 本文在查看任何真实 LiDAR + mmWave 结果前固定验证性分析。合成 smoke test 仅验证
> 软件链路，不能作为科研证据。

Amendment before real outcomes / 真实结果前修订：v1.1 retains the prespecified
2,000-repetition paired cluster bootstrap for the 95% confidence interval and
adds a 20,000-repetition paired cluster sign-flip test for p-values. No real
Track B metric had been observed when this amendment was made. / v1.1 保留预设的
2,000 次配对聚类 Bootstrap 置信区间，并增加 20,000 次配对簇符号翻转检验计算 p 值；
修订时尚未观察任何 Track B 真实指标。

Amendment before degradation outcomes / 退化结果前修订：v1.2 keeps the
subject-action-recording cluster as the primary unit and adds a stricter
subject-cluster sensitivity analysis with the same bootstrap, sign-flip, and
Holm procedures. This addresses correlation across multiple actions from one
person without changing the primary endpoint. No formal degradation or
calibration outcome had been observed when this amendment was made. / v1.2
保持“受试者-动作录制”为主要聚类单位，并增加更严格的“受试者聚类”敏感性分析，沿用
相同的 Bootstrap、符号翻转和 Holm 流程，用于检查同一受试者多个动作之间的相关性；
该修订不改变主要终点，修订时尚未观察正式退化或校准结果。

## 1. Frozen title and scope / 冻结题目与范围

**Confidence Calibration for Multimodal Indoor Lower-Limb Rehabilitation
Activity Recognition under Sensing-Quality Degradation**

**感知质量退化条件下室内下肢康复动作多模态识别的置信度校准研究**

The study uses synchronized MM-Fi recordings from healthy volunteers and a
frozen pretrained X-Fi MMFi-HAR model. It evaluates seven rehabilitation-related
lower-limb actions. It does not evaluate patients, diagnosis, treatment efficacy,
or clinical safety.

本研究使用 MM-Fi 健康志愿者的同步数据和冻结的预训练 X-Fi MMFi-HAR 模型，评估七类
与康复训练相关的下肢动作。研究不涉及患者、诊断、治疗效果或临床安全性。

## 2. Research questions / 研究问题

- **RQ1 Recognition / 识别：** How do asymmetric and joint LiDAR/mmWave
  degradations affect Accuracy and Macro-F1? / LiDAR 与 mmWave 的单侧及联合退化
  如何影响 Accuracy 与 Macro-F1？
- **RQ2 Geometry / 退化几何：** Is count-matched contiguous azimuth-sector
  loss more harmful than uniform random point loss? / 在保留点数相同时，连续方位
  扇区缺失是否比均匀随机删点更有破坏性？
- **RQ3 Negative fusion / 负融合：** At what observed severities is fusion with
  a degraded sensor no longer better than the surviving sensor alone? / 在什么
  退化区间，保留退化传感器的融合不再优于单独使用健康传感器？
- **RQ4 Calibration transfer / 校准迁移：** Does a temperature fitted only on
  clean data remain reliable under unseen degradation? / 仅在 clean 数据上拟合的
  温度能否迁移到退化条件？
- **RQ5 Quality-aware calibration / 质量感知校准：** Can observable quality
  features lower test NLL relative to pooled global temperature scaling without
  changing predicted classes? / 可观测质量特征能否在不改变预测类别的情况下，相比
  混合条件全局温度缩放降低测试 NLL？

## 3. Fixed hypotheses / 冻结假设

- **H1:** Clean LiDAR + mmWave is expected to outperform each clean unimodal
  control. / clean 双模态预计优于两个 clean 单模态对照。
- **H2:** Recognition and confidence reliability are expected to deteriorate as
  point loss increases. / 随着点丢失增加，识别性能和置信度可靠性预计下降。
- **H3:** Count-matched sector loss is expected to be more damaging than uniform
  loss because it removes spatially coherent evidence. / 同点数扇区缺失预计比随机
  缺失破坏更大，因为它删除连续空间证据。
- **H4:** Severe degradation may produce negative fusion: a corrupted but active
  sensor can be worse than explicitly masking it. / 严重退化可能导致负融合：保留退化
  传感器可能比显式移除它更差。
- **H5:** Quality-aware scalar temperature scaling is expected to reduce NLL
  relative to pooled global scaling on degraded fusion conditions while exactly
  preserving argmax predictions. / 质量感知标量温度缩放预计在退化融合条件下降低 NLL，
  并严格保持 argmax 预测类别不变。

These are hypotheses, not promised findings. Null or opposite results will be
retained and reported. / 以上是假设而非承诺结果；零结果和相反结果均须保留并汇报。

## 4. Data, key, and model / 数据、样本键与模型

- Dataset / 数据集：MM-Fi, CC BY-NC 4.0；
- Model / 模型：official frozen X-Fi MMFi-HAR checkpoint / 官方冻结权重；
- Model outputs / 模型输出：27 logits；
- Primary modalities / 主模态：LiDAR + mmWave；
- Target actions / 目标动作：`A06, A09, A10, A12, A15, A16, A26`；
- Unique synchronized key / 同步唯一键：

```text
(scene, subject, action, frame_index)
```

The data gate requires identical LiDAR/mmWave keys, finite values, readable
point-cloud shapes, consistent labels, recorded file counts, and file hashes.

数据关要求 LiDAR/mmWave 样本键完全一致、数值有限、点云形状可读取、标签一致，并记录
文件数量与哈希。

## 5. Frozen split and output spaces / 冻结划分与输出空间

The source samples follow the official X-Fi random validation split. A second,
post-hoc subject-disjoint partition divides those validation samples into
calibration and final test sets. No calibration model or hyperparameter may use
final-test labels.

源样本沿用官方 X-Fi 随机验证划分；随后再按受试者互斥地划分为后处理校准集和最终测试集。
任何校准模型或超参数都不得使用最终测试标签。

- **Primary / 主分析：** strict 27-class probabilities on seven-action samples.
- **Secondary / 次分析：** seven selected logits renormalized as a conditioned
  seven-class closed set; it must be labelled `conditioned7`.

The secondary result cannot replace an unfavorable strict-27 result. / 次分析不得
替代不理想的 strict-27 主结果。

## 6. Exact 323-condition matrix / 精确的 323 条件矩阵

Modality masks / 模态掩码：

```text
LiDAR + mmWave
LiDAR only
mmWave only
```

Severities / 删除率：`0, 0.25, 0.50, 0.75, 0.90`  
Geometries / 退化几何：`uniform`, `azimuth_sector`  
Seeds / 随机种子：`7, 21, 42, 84, 168`

For each geometry-seed pair / 每个“几何-种子”组合包含：

- 4 degraded LiDAR-only conditions / 4 个退化 LiDAR 单模态条件；
- 4 degraded mmWave-only conditions / 4 个退化 mmWave 单模态条件；
- 24 non-clean cells from the `5 x 5` fusion grid / 双模态 `5 x 5` 网格中除
  clean 外的 24 个条件。

Therefore / 因此：

```text
3 clean masks + 2 geometries x 5 seeds x (4 + 4 + 24)
= 3 + 320
= 323 unique inference conditions
```

Clean conditions are evaluated once rather than repeated under meaningless
corruption seeds. / clean 条件只运行一次，不在无意义的退化种子下重复计算。

## 7. Controlled corruptions / 受控退化

### Uniform nested point loss / 嵌套式均匀随机删点

A deterministic ordering is generated for each sample, modality, and seed.
Higher severities retain nested subsets of lower severities. / 每个样本、模态和种子
产生一个确定性点顺序；更高强度保留的点集是较低强度点集的子集。

### Count-matched azimuth-sector loss / 同点数方位扇区缺失

A contiguous azimuth region is removed while retaining the same point count as
the corresponding uniform condition. This isolates geometry from point count.

删除连续方位区域，同时让保留点数与对应随机条件一致，从而把“空间结构”效应与“点数”
效应分开。

Both are controlled software proxies, not claims about every physical sensor
failure. / 两种方法都是软件受控代理，不代表所有真实物理故障。

## 8. Calibration methods / 校准方法

1. `uncalibrated`: `softmax(z)` / 未校准；
2. `clean_global_ts`: one scalar fitted on clean calibration samples per mask /
   每个模态掩码在 clean 校准样本上拟合一个温度；
3. `pooled_global_ts`: one scalar fitted across all preregistered calibration
   conditions / 所有预注册校准条件共同拟合一个温度；
4. `severity_oracle_ts`: one scalar per known condition; reference upper bound,
   not a deployable primary method / 每个已知条件单独拟合，作为参考上界；
5. `quality_aware_ts`: a positive sample-wise scalar temperature based only on
   observable quality features / 仅根据可观测质量特征生成逐样本正标量温度。

```text
T_i = T_min + softplus(w^T q_i + b)
p_i = softmax(z_i / T_i)
```

`q_i` contains 11 prespecified features: log point counts, azimuth/range
occupancy for both sensors, both unimodal predictive entropies, Jensen-Shannon
disagreement, and two explicit missing-modality flags.

`q_i` 固定包含 11 个特征：两种传感器的对数点数、方位/距离占用率、两个单模态预测熵、
Jensen-Shannon 分歧，以及两个显式模态缺失标记。

L2 regularization is selected using subject-grouped cross-validation on
calibration subjects only. / L2 正则仅在校准人员上通过按受试者分组交叉验证选择。

## 9. Outcomes and primary estimand / 指标与主要估计量

### Primary metric / 主要指标

**Negative Log-Likelihood (NLL), lower is better. / 负对数似然，越低越好。**

### Primary confirmatory comparison / 主要验证性比较

For each corruption geometry, compare `quality_aware_ts - pooled_global_ts` NLL
over all non-clean LiDAR+mmWave conditions. Use a paired subject-action-recording
cluster bootstrap with 2,000 repetitions for the 95% confidence interval and a
20,000-repetition paired cluster sign-flip test for the p-value. Apply Holm
correction to the two geometry-stratified sign-flip p-values.

As a prespecified sensitivity analysis, repeat the overall and geometry-specific
comparisons after merging all recordings from the same subject into one
cluster. The recording-cluster result remains primary; the subject-cluster
result tests whether the interpretation survives a more conservative
dependence assumption.

对每种退化几何，在全部非 clean 的 LiDAR+mmWave 条件上比较
`quality_aware_ts - pooled_global_ts` 的 NLL。以“受试者-动作录像”为簇执行 2,000 次
配对 Bootstrap 计算 95% 置信区间，并执行 20,000 次配对簇符号翻转检验计算 p 值；
对两个按几何分层的符号翻转 p 值做 Holm 校正。

作为预先说明的敏感性分析，把同一受试者的所有动作录制合并为一个簇，重复总体和按几何
分层的比较。录像簇结果仍为主要结果；受试者簇结果用于检验在更保守的相关性假设下，
结论方向是否保持一致。

- Delta `< 0` and 95% CI entirely below 0 supports improvement / 差值小于 0 且
  95% CI 完全低于 0，支持改善；
- CI crossing 0 is inconclusive / 区间跨 0，证据不足；
- CI entirely above 0 indicates deterioration / 区间完全高于 0，表示恶化。

### Secondary metrics / 次要指标

Accuracy, Macro-F1, per-action recall, Brier score, fixed-bin ECE, adaptive ECE,
MCE, confidence-accuracy gap, AURC, confusion matrices, reliability diagrams,
and normalized robustness AUC.

These metrics support interpretation but cannot override an unfavorable primary
NLL result. / 次要指标用于辅助解释，不能推翻不理想的主要 NLL 结果。

## 10. Fixed interpretation rules / 固定解释规则

1. Positive scalar temperature scaling must preserve Accuracy, Macro-F1, and
   each sample's argmax. / 正标量温度缩放必须保持 Accuracy、Macro-F1 和每个样本的
   argmax 不变。
2. Calibration improvement means probabilities became more reliable; it does
   not mean the recognizer became more accurate. / 校准改善表示概率更可信，不表示
   识别器更准确。
3. If ECE improves but NLL worsens or is inconclusive, report a metric-dependent
   trade-off. / 若 ECE 改善但 NLL 恶化或证据不足，只能报告指标权衡。
4. If recognition collapses, calibration cannot restore missing class evidence.
   / 若识别性能崩溃，校准不能恢复已经丢失的类别信息。
5. If quality-aware TS fails, report that the prespecified observable features
   and scalar model were insufficient. / 若质量感知方法失败，应报告固定特征与标量模型
   不足，而不是更换主要结果。

## 11. Required gates before claims / 得出结论前的必需门槛

1. aligned full-validation LiDAR/mmWave extraction and data audit / 全验证集同步提取与审计；
2. official full checkpoint hash and exact state-dict load / 官方完整权重哈希及无误加载；
3. clean all-27 baseline reproduction within a justified tolerance / clean 27 类基线复现；
4. target-action baseline and 323-condition inference / 目标动作基线及 323 条件推理；
5. subject-disjoint calibration/test audit / 人员互斥审计；
6. both output-space analyses and artifact audit / 两套标签空间分析与产物审计；
7. every numerical claim traceable to CSV/NPZ/JSON evidence / 每个数值结论可追溯。

## 12. Claim boundaries / 结论边界

- healthy volunteers, not patients / 健康志愿者而非患者；
- seven rehabilitation-related actions, not a validated therapy protocol /
  七类相关动作而非临床康复方案；
- controlled point sparsification, not every real sensor defect / 受控点云稀疏化
  并非所有真实传感器故障；
- post-hoc subject separation prevents calibrator/test leakage but does not prove
  the frozen recognizer never saw those subjects in other actions / 后处理人员互斥防止
  校准泄漏，但不能证明冻结识别器在其他动作中从未见过这些人员；
- experiment completion does not guarantee EI acceptance / 完成实验不保证 EI 录用。

## 13. Evidence status at freeze / 冻结时证据状态

- Real mmWave-only Track A results exist but are not Track B evidence. / 已有毫米波
  单模态主线 A 结果，但不属于主线 B 证据。
- The 323-condition synthetic integration audit passes and proves software
  completeness only. / 323 条件合成集成审计通过，但只证明软件完整性。
- No real Track B multimodal metric has been observed or reported. / 冻结时尚未
  观察或汇报任何真实主线 B 多模态指标。
