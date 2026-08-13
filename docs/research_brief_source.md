# Research Brief Source

## Title

**Confidence Calibration for Multimodal Indoor Lower-Limb Activity Recognition
under Sensing Degradation**

Yiyang Zhang | MSc Smart Manufacturing, Nanyang Technological University

## Research question

When aligned LiDAR and mmWave point observations deteriorate, can observable
quality cues make a frozen X-Fi model's confidence more trustworthy than one
pooled temperature?

## Study design

- **Evidence gate:** 54,433 clean frames and all 27 X-Fi/MM-Fi HAR actions.
- **Target analysis:** 15,315 aligned frames, 7 lower-limb actions and 33
  healthy volunteers.
- **Post-hoc split:** 17 calibration subjects and 16 disjoint test subjects.
- **Controlled degradation:** uniform random and contiguous azimuth-sector
  point loss, each evaluated with 5 fixed corruption seeds.
- **Formal matrix:** 323 clean, degraded, fused and unimodal conditions.
- **Model:** one released X-Fi checkpoint, frozen during evaluation.

## Main evidence

1. **Reproduction gate passed.** On the complete clean validation cohort,
   reproduced Accuracy was 0.889 for LiDAR-mmWave fusion, 0.532 for LiDAR only
   and 0.860 for mmWave only. The published references were 0.887, 0.527 and
   0.857, respectively; all were within the frozen 0.03 tolerance.
2. **Recognition was vulnerable to degradation.** Fused Accuracy was 0.900 on
   clean target data and averaged 0.482 across 240 degraded fused conditions.
   Under severe asymmetric degradation, fixed fusion could underperform the
   stronger available unimodal branch.
3. **Quality cues improved confidence reliability, not classification.** Mean
   ECE decreased from 0.143 with pooled global temperature scaling to 0.063
   with quality-aware temperature scaling; mean NLL decreased from 1.968 to
   1.846. Predictions and Accuracy were unchanged.

## Interpretation

Observable sensing quality can improve how much a frozen multimodal model
should be trusted, but it cannot repair a wrong prediction. A focused next step
is quality-gated fusion or selective abstention, followed by validation under
physical sensor faults.

## Claim boundaries

The evidence comes from healthy volunteers, controlled software point loss and
one released checkpoint. It is not rehabilitation-patient, clinical,
physical-failure or real-time deployment validation.

## Sources

- MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless
  Sensing, NeurIPS 2023 Datasets and Benchmarks.
- X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing,
  ICLR 2025.
