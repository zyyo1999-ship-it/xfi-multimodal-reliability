# Research Story / 研究故事

## Motivation / 研究动机

Robotic perception must do more than output a label: it should expose when
degraded sensing makes that label unreliable. Experience with automation system
integration and ROS2 navigation motivated this evaluation of **recognition
robustness** and **confidence reliability** as two separate questions.

机器人感知不仅要输出类别，还应在传感器退化使预测不可靠时反映这种风险。自动化系统
集成与 ROS2 导航经历促使本项目把**识别鲁棒性**和**置信度可靠性**作为两个不同问题
进行评估。

## Project positioning / 项目文献定位

The project bibliography connects four lines of work. This positioning does not
imply that every cited paper was personally read in full:

1. MM-Fi provides synchronized multimodal human-sensing recordings.
2. X-Fi provides a released modality-invariant recognizer and modality masks.
3. Missing-modality studies show that fusion is not automatically robust.
4. Calibration research shows that accuracy and confidence reliability are
   different properties.

因此，核心问题是：当 LiDAR 与毫米波点观测质量下降时，一个冻结的 X-Fi 识别器如何
失效，以及由可观测输入质量驱动的标量温度能否比单一混合温度更好地校准其置信度？

The bibliography positions the study within related work and should not be
interpreted as a personal reading log. / 参考文献用于说明研究定位，不应被理解为
个人逐篇精读记录。

## Study design / 实验设计

- Reproduce the released 27-class clean checkpoint before extending it.
- Select seven lower-limb-related actions as a controlled benchmark subset.
- Keep the recognizer frozen to isolate input degradation and post-hoc
  calibration.
- Apply count-matched uniform and contiguous azimuth-sector point loss.
- Evaluate 323 clean, degraded, fused, and unimodal conditions.
- Separate post-hoc calibration and final-test subjects.
- Report Accuracy and Macro-F1 for recognition; NLL, Brier, ECE, and AURC for
  confidence; use clustered resampling for correlated recordings.

The frozen design is available in
[`preanalysis_plan_bilingual.md`](preanalysis_plan_bilingual.md), and the
implementation path is explained in
[`code_walkthrough_bilingual.md`](code_walkthrough_bilingual.md).

## Findings / 主要发现

The clean reproduction gate matched the released X-Fi reference within the
preregistered tolerance. Under controlled degradation, fused recognition was
not consistently better than the strongest matched unimodal branch. A
quality-aware scalar temperature did not change class predictions, but improved
probability reliability relative to one pooled global temperature:

| Method / 方法 | NLL | ECE |
|---|---:|---:|
| Pooled global temperature | 1.9683 | 0.1428 |
| Quality-aware temperature | **1.8455** | **0.0630** |

The recording-cluster difference in NLL was `-0.122792`, with a 95% interval of
`[-0.147971, -0.096277]`. These findings support quality-aware confidence
handling; they do not demonstrate a new recognition model.

## Contribution and boundary / 贡献与边界

The contribution is a reproducible evaluation layer: synchronized-data audits,
controlled corruptions, frozen-checkpoint inference, post-hoc calibration,
clustered statistics, figures, tests, and machine-readable result audits.

本项目使用健康志愿者数据、软件模拟点缺失和一个公开冻结权重。它不是临床验证，也未
测试真实硬件故障或跨数据集泛化。质量门控融合与拒识是由结果引出的下一步研究，而不是
已经实现的模块。

## Evidence entry points / 证据入口

- Protocol / 协议：[`configs/multimodal_lower_limb_formal.yaml`](../configs/multimodal_lower_limb_formal.yaml)
- Reproduction / 复现：[`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md)
- Code / 代码：[`src/`](../src)
- Tests / 测试：[`tests/`](../tests)
- Figures / 图表：[`results/figures/`](../results/figures)
- Tables and audit / 表格与审计：[`results/tables/`](../results/tables)
- References / 文献：[`references.bib`](references.bib)

## Development and verification / 开发与验证说明

AI-assisted development tools supported parts of implementation and
documentation. The repository retains executable tests, frozen protocols,
hashes, audited outputs, and explicit claim boundaries for independent
inspection. / AI 辅助开发工具参与了部分实现与文档工作；仓库保留可执行测试、
冻结协议、哈希、审计结果与明确的结论边界，以支持独立核验。
