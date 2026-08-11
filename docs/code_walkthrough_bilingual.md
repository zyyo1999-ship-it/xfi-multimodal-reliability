# Formal Multimodal Code Walkthrough / 正式多模态代码导读

## 1. Purpose / 文档目的

**EN.** This document explains the exact code used by the formal LiDAR+mmWave study. It follows the real call chain from official archive acquisition to audited statistical results. It does not describe the earlier mmWave-only contact pilot.

**中文。** 本文解释正式 LiDAR+mmWave 研究实际执行的代码，沿着“官方压缩包获取 -> 同步帧提取 -> X-Fi 推理 -> 置信度校准 -> 统计检验 -> 结果审计”的真实调用链展开，不讲前期单毫米波建联实验。

Remember / 记住一句话：**Code is the executable research protocol. / 代码就是可执行的科研协议。**

## 2. End-to-end call graph / 端到端调用图

```text
stage_all_mmfi_subjects_from_baidu.sh
  -> stage_mmfi_subject_from_baidu.sh
     -> BaiduPCS-Go download
     -> SHA-256 provenance record
     -> extract_and_audit_mmfi_subject_archive.sh
        -> extract_mmfi_point_modalities.py
        -> audit_extracted_point_modalities.py

wait_for_staging_and_launch_formal.sh
  -> launch_formal_multimodal_background.sh
     -> formal_multimodal_worker.sh
        -> run_formal_multimodal_study.sh
           -> full data audit
           -> run_multimodal_inference.py (clean gate)
           -> evaluate_multimodal_clean_baseline.py
           -> run_multimodal_inference.py (323 conditions)
           -> analyze_multimodal_calibration.py (strict27)
           -> analyze_multimodal_calibration.py (conditioned7)
           -> audit_multimodal_formal_results.py
           -> summarize_multimodal_formal_results.py
```

## 3. Data acquisition and provenance / 数据获取与来源记录

### 3.1 `scripts/stage_all_mmfi_subjects_from_baidu.sh`

**What / 是什么**

The script loops over `S01` to `S40`. A subject is skipped only when both its audit report and archive provenance record already exist.

脚本遍历 `S01` 到 `S40`。只有当某位受试者同时存在审计报告和压缩包来源记录时才跳过，因此中断后可以断点续传。

```bash
for number in $(seq -w 1 40); do
  subject="S${number}"
  if [[ -s "${audit}" && -s "${provenance}" ]]; then
    continue
  fi
done
```

`seq -w 1 40` is a GNU command that emits zero-padded numbers. `-s` is a Bash file test meaning “the file exists and is not empty.”

`seq -w 1 40` 生成带前导零的编号；Bash 的 `-s` 表示“文件存在且非空”。

### 3.2 `scripts/stage_mmfi_subject_from_baidu.sh`

The script performs five guarded operations:

1. validates the subject and cloud filename;
2. verifies BaiduPCS-Go authentication;
3. checks free disk space;
4. downloads one official subject archive;
5. records observed SHA-256 and size before extraction.

脚本执行五层保护：检查受试者编号、检查登录、检查磁盘、下载单个官方压缩包、在解压前记录 SHA-256 与文件大小。

```bash
ARCHIVE_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
```

- `sha256sum`: computes a cryptographic content fingerprint / 计算文件内容的密码学指纹。
- `awk '{print $1}'`: selects the first whitespace-separated field / 取输出的第一列。
- `$(...)`: command substitution; its output becomes a shell value / 命令替换，将输出作为变量值。

**Boundary / 边界：** The record proves which bytes were used in this run. Without a publisher-provided checksum it cannot independently prove that the provider never changed the archive.

记录可以证明本次实验使用了哪些字节，但如果发布方没有公开官方校验值，它不能独立证明云端文件从未被替换。

## 4. Exact synchronized extraction / 精确同步提取

### 4.1 `src/extract_mmfi_point_modalities.py`

The extractor does not unpack the entire 77 GB dataset permanently. It reads an archive member-by-member and writes only frame IDs that belong to the frozen official validation cohort.

提取器不会把约 77 GB 数据全部永久解压。它逐成员读取压缩包，只写出冻结官方验证队列中需要的帧。

The canonical sample key is:

```text
(environment, subject, action, frame_index)
```

规范样本键由“环境、受试者、动作、帧号”共同组成。LiDAR 与 mmWave 只有四个字段完全一致时才能构成一条多模态样本。

### 4.2 `src/audit_extracted_point_modalities.py`

Important libraries and methods / 关键库与方法：

| Code / 代码 | Library / 库 | Meaning / 含义 |
|---|---|---|
| `Path.glob()` | `pathlib` | Find matching files / 查找匹配文件 |
| `json.loads()` | `json` | Parse JSON text / 解析 JSON 文本 |
| `np.load(..., mmap_mode="r")` | NumPy | Read `.npy` without copying the whole array into RAM / 以内存映射读取 |
| `np.isfinite()` | NumPy | Reject NaN and infinity / 排除 NaN 与无穷值 |
| `filecmp.cmp(..., shallow=False)` | `filecmp` | Compare full file bytes / 比较完整文件字节 |
| `hashlib.sha256()` | `hashlib` | Incremental cryptographic digest / 增量计算 SHA-256 |

Binary layout checks use:

```python
row_bytes = np.dtype(np.float64).itemsize * width
if byte_count == 0 or byte_count % row_bytes:
    raise RuntimeError(...)
```

`float64` occupies 8 bytes. A LiDAR row has 3 values and an mmWave row has 5. Therefore valid file sizes must be positive multiples of `8*3` or `8*5`.

`float64` 占 8 字节。LiDAR 每点 3 列，mmWave 每点 5 列，所以合法文件大小必须分别是 `24` 或 `40` 的正整数倍。

The final audit additionally hashes every aligned pair in deterministic sample order. This binds inference to exact LiDAR and mmWave bytes, not just filenames and sizes.

最终审计会按确定性样本顺序对每组对齐数据做内容哈希，使推理结果绑定到具体 LiDAR/mmWave 字节，而不只是文件名和大小。

## 5. Frozen task protocol / 冻结任务协议

### 5.1 `src/lower_limb_protocol.py`

Seven rehabilitation-relevant actions are selected:

```python
LOWER_LIMB_ACTIONS = (
    "A06", "A09", "A10", "A12", "A15", "A16", "A26"
)
```

七类动作包括原地踏步、前弓步、侧弓步、深蹲和向上跳等“康复相关”动作。MM-Fi 参与者是健康受试者，因此不得称为真实患者康复数据。

Twenty subjects are frozen for post-hoc calibration and twenty for final testing. `partition_for_subject()` is called for every frame, so one subject cannot appear in both partitions.

20 名受试者用于后处理校准，另 20 名用于最终测试。每一帧都通过 `partition_for_subject()` 分区，避免同一人的帧同时进入校准和测试。

**Important scope / 重要边界：** This split prevents the calibrator from seeing test subjects. It does not retrain X-Fi and does not redefine X-Fi's original action-wise random training split.

该划分防止校准器看到测试受试者，但没有重新训练 X-Fi，也没有改变 X-Fi 原始按动作随机划分协议。

### 5.2 `src/multimodal_protocol.py`

`ExperimentCondition` is an immutable dataclass. One object defines:

- active modality mask;
- degradation geometry;
- random corruption seed;
- LiDAR drop rate;
- mmWave drop rate.

`ExperimentCondition` 是不可变数据类，每个对象完整定义一次实验条件。

The formal matrix is:

```text
3 clean masks
+ 2 geometries * 5 seeds * (4 LiDAR-only + 4 mmWave-only + 24 fused)
= 3 + 2 * 5 * 32
= 323 conditions
```

The unimodal controls are computed once per severity and reused when building quality features. This avoids redundant GPU inference without changing the experiment.

单模态对照在每个退化等级只计算一次，构造质量特征时复用，减少重复 GPU 推理而不改变实验设计。

## 6. Point-cloud loading and degradation / 点云读取与退化

### 6.1 `src/multimodal_dataset.py`

```python
points = np.fromfile(path, dtype=np.float64).reshape(-1, width)
```

- `np.fromfile`: reads raw binary values / 读取原始二进制数值。
- `reshape(-1, width)`: infer the number of points while fixing feature width / 自动推断点数并固定每点列数。
- LiDAR shape: `[N, 3]` / LiDAR 形状。
- mmWave shape: `[M, 5]` / mmWave 形状。

`PointCloudCache` is an LRU cache. It keeps recently used arrays in RAM because the same clean frame is reused under hundreds of controlled conditions.

`PointCloudCache` 是最近最少使用缓存。相同原始帧会在数百种退化条件中重复使用，缓存可以减少磁盘读取。

### 6.2 Deterministic sample seed / 确定性样本种子

```python
payload = f"{sample_id}|{modality}|{experiment_seed}".encode("utf-8")
digest = hashlib.blake2b(payload, digest_size=8).digest()
```

The random seed depends on sample ID, modality, and experiment seed. Therefore rerunning the same condition produces the same dropped points, while a different formal seed produces another valid corruption realization.

随机种子由样本 ID、模态和实验种子共同决定。相同条件重复运行会删除相同点，不同正式种子则形成新的退化实现。

### 6.3 `src/corruptions.py`

Uniform point loss / 均匀删点：

```python
kept_indices = np.sort(generator.permutation(points.shape[0])[:keep_count])
```

`generator.permutation(N)` gives a random ordering of `0..N-1`. Taking a prefix creates nested severity: points retained at 90% loss are a subset of those retained at 75% loss under the same seed.

`generator.permutation(N)` 随机排列所有点索引。取固定前缀形成嵌套退化：同一种子下，90% 丢失保留的点是 75% 丢失保留点的子集。

Azimuth-sector loss / 方位扇区缺失：

```python
azimuth = np.arctan2(points[:, 1], points[:, 0])
circular_distance = np.abs(np.arctan2(
    np.sin(azimuth - sector_centre),
    np.cos(azimuth - sector_centre),
))
```

`arctan2(y, x)` converts planar coordinates to direction. The sine/cosine expression computes wrapped angular distance across the `-pi/pi` boundary. The nearest directions are removed first, creating a count-matched contiguous occlusion proxy.

`arctan2(y,x)` 把平面坐标转换为方向角。正弦/余弦表达式处理 `-pi/pi` 环绕边界；优先删除靠近随机中心方向的点，形成与均匀删点数量匹配的连续遮挡代理。

This is a controlled proxy, not a physical sensor simulator.

这是受控代理，不是真实传感器故障物理模型。

### 6.4 `collate_aligned_points`

Point clouds have different lengths. PyTorch `pad_sequence(..., batch_first=True)` pads shorter samples with zeros so a batch becomes `[B, N_max, D]`.

不同帧点数不同。PyTorch 的 `pad_sequence` 用零补齐较短样本，使批次成为 `[批大小, 最大点数, 特征维度]`。

## 7. X-Fi model loading / X-Fi 模型加载

### 7.1 `src/xfi_runtime.py`

`build_xfi_for_checkpoint()` constructs the official module tree but skips redundant preliminary backbone checkpoint loading. The complete released X-Fi checkpoint then supplies every parameter.

`build_xfi_for_checkpoint()` 构建与官方一致的模块树，但跳过冗余的初始骨干权重加载，随后由完整 X-Fi checkpoint 覆盖全部参数。

Although `load_state_dict(..., strict=False)` is called internally to inspect missing keys, the wrapper rejects every missing or unexpected key for a full multimodal checkpoint. The practical behavior is strict loading.

内部使用 `strict=False` 是为了读取缺失键列表，但完整多模态权重只要有任意缺失或多余键就会报错，实际效果等同严格加载。

### 7.2 Modality mask / 模态掩码

X-Fi expects `[RGB, depth, mmWave, LiDAR]`:

```python
MODALITY_MASKS = {
    "lidar_mmwave": (False, False, True, True),
    "lidar_only":   (False, False, False, True),
    "mmwave_only":  (False, False, True, False),
}
```

RGB and depth placeholders satisfy the fixed function interface, but their mask values are false, so they do not contribute evidence.

RGB 和深度张量只为满足固定函数接口，其掩码为假，不参与证据融合。

## 8. Formal inference and resume safety / 正式推理与断点安全

### 8.1 `src/run_multimodal_inference.py`

For each condition, the runner:

1. discovers aligned frames;
2. verifies the full data audit identity;
3. loads frozen X-Fi in `eval()` mode;
4. creates a deterministic `DataLoader`;
5. runs `torch.inference_mode()`;
6. exports logits, labels, IDs, partitions, and quality features;
7. atomically writes a compressed `.npz` artifact;
8. records its SHA-256 in the manifest.

每个条件都执行同步发现、审计绑定、冻结模型、确定性批处理、无梯度推理、完整元数据导出、原子写入和哈希记录。

```python
with torch.inference_mode():
    logits = model(rgb, depth, mmwave, lidar, mask)
```

`inference_mode()` disables gradient tracking and version bookkeeping. This reduces memory and is correct because no training occurs.

`inference_mode()` 关闭梯度和张量版本追踪，因为本研究只推理、不训练，可降低显存占用。

### 8.2 Run signature / 运行签名

The resume signature includes:

- checkpoint SHA-256;
- source-code bundle SHA-256;
- aligned-data fingerprint;
- full content-audit SHA-256;
- archive provenance bundle SHA-256;
- condition IDs, seed, batch size, and sample count.

断点签名包含权重、代码、数据内容审计、来源记录、条件矩阵、种子、批大小和样本数。任何不可变身份不同都会拒绝混用旧结果。

`atomic_npz()` writes a temporary file and then calls `os.replace()`. A crash cannot leave a half-written file under the final artifact name.

`atomic_npz()` 先写临时文件，再用 `os.replace()` 原子替换，崩溃不会把半写文件伪装成正式结果。

## 9. Confidence calibration mathematics / 置信度校准数学

### 9.1 Softmax and logits / Softmax 与 logits

For class logit `z_k`:

```text
p_k = exp(z_k) / sum_j exp(z_j)
```

Logits are unnormalized model scores. Softmax converts them to a probability distribution.

Logits 是未归一化分数；Softmax 将其转换为总和为 1 的类别概率。

### 9.2 Temperature scaling / 温度缩放

```text
p_k(T) = exp(z_k / T) / sum_j exp(z_j / T), T > 0
```

- `T > 1`: softer, less confident / 概率更平、更不自信。
- `0 < T < 1`: sharper, more confident / 概率更尖、更自信。
- positive scalar `T` preserves argmax / 正温度不改变最大类别，因此不改变 Accuracy。

`fit_temperature()` minimizes calibration-set negative log-likelihood with SciPy `minimize_scalar` in log-temperature space.

`fit_temperature()` 使用 SciPy 在对数温度空间最小化校准集 NLL，自动保证温度为正。

### 9.3 Quality-aware temperature / 质量感知温度

`src/multimodal_calibration.py` builds eleven deployable features:

- Dimensions 1-2: log LiDAR and mmWave point counts / LiDAR 与 mmWave 点数的对数；
- Dimensions 3-4: LiDAR and mmWave azimuth occupancy / 两个模态的方位角占用率；
- Dimensions 5-6: LiDAR and mmWave range occupancy / 两个模态的距离占用率；
- Dimensions 7-8: LiDAR-only and mmWave-only predictive entropy / 两个单模态预测熵；
- Dimension 9: unimodal Jensen-Shannon disagreement / 两个单模态预测的 JS 分歧；
- Dimensions 10-11: LiDAR and mmWave missing-modality flags / 两个模态缺失标志。

质量感知模型使用 11 个部署时可获得的特征，不使用真实标签或人为设定的退化等级作为输入。

Entropy and JS require LiDAR-only and mmWave-only logits. The formal inference therefore computes two additional unimodal forward passes and stores their reusable outputs before building fused quality features. / 熵与 JS 需要 LiDAR-only 和 mmWave-only logits，因此正式推理会额外执行两次单模态前向传播，并先保存可复用输出，再构造融合质量特征。这是可部署的观测成本，不是零成本。

```text
x_std = (x - mean) / std
T(x) = T_min + softplus(w^T x_std + b)
```

`softplus(a)=log(1+exp(a))` guarantees positive temperature. L2 regularization strength is selected with subject-grouped cross-validation on calibration subjects only.

Softplus 保证温度为正；L2 正则强度只在校准受试者内部按受试者分组交叉验证选择。

### 9.4 Jensen-Shannon disagreement / JS 分歧

For two unimodal distributions `P` and `Q`:

```text
M = (P + Q) / 2
JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M)
```

Low JS means the sensors agree; high JS means their predictions conflict. Probabilities and the mixture are clipped before logarithms to prevent `log(0)` and `NaN` under extreme logits.

JS 小表示两个传感器预测一致，JS 大表示冲突。代码在取对数前限制概率下界，防止极端 logits 导致 `log(0)` 和 `NaN`。

## 10. Formal analysis / 正式分析

### 10.1 `src/analyze_multimodal_calibration.py`

Two output spaces are evaluated:

- `strict27`: primary; all 27 X-Fi classes compete / 主要分析，27 类共同竞争。
- `conditioned7`: secondary; select the seven relevant logits and renormalize / 次要分析，只在 7 个目标类别内重归一化。

The script compares five methods:

1. uncalibrated;
2. clean per-mask global temperature;
3. pooled global temperature;
4. per-condition oracle temperature;
5. quality-aware sample temperature.

The per-condition oracle uses knowledge unavailable in ordinary deployment and is an upper-reference baseline, not the proposed deployable method.

条件 oracle 依赖部署时通常未知的条件身份，只是参考上界，不是拟议部署方法。

### 10.2 Metrics / 指标

| Metric | Meaning / 含义 | Direction / 趋势 |
|---|---|---|
| Accuracy | Fraction correctly classified / 正确分类比例 | higher / 越高越好 |
| Macro-F1 | Equal-weight class F1 / 各类别等权 F1 | higher |
| NLL | Penalizes probability assigned to truth / 惩罚真实类别低概率 | lower |
| Brier | Squared probability error / 概率平方误差 | lower |
| ECE | Binned confidence-accuracy gap / 分箱置信度-准确率差 | lower |
| Adaptive ECE | Equal-count bin calibration gap / 等样本分箱误差 | lower |
| AURC | Selective prediction risk area / 选择性预测风险面积 | lower |

Primary confirmatory estimand / 主要验证量：

```text
Delta NLL = NLL_quality-aware - NLL_pooled-global
```

Negative `Delta NLL` favors quality-aware calibration.

负的 `Delta NLL` 表示质量感知校准更好。

### 10.3 Paired recording-cluster statistics / 配对记录级统计

Frames from the same subject-action recording are correlated. Therefore uncertainty is computed at the `subject/action` cluster level rather than pretending every frame is independent.

同一受试者同一动作的连续帧相关，不能把每一帧假设为独立样本，所以统计单位采用“受试者/动作记录簇”。

- 2,000 paired cluster bootstrap repetitions estimate a confidence interval.
- 20,000 paired sign-flip repetitions test whether the paired mean difference is zero.
- Holm correction controls family-wise error across two degradation geometries.

对应中文：2,000 次配对记录簇 Bootstrap 估计置信区间；20,000 次配对符号翻转检验平均差是否为零；Holm 校正在两种退化几何之间控制家族错误率。

### 10.4 Subject-cluster sensitivity / 受试者聚类敏感性分析

`analyze_multimodal_calibration.py` also aggregates all actions from the same
person under one subject key and repeats the bootstrap, sign-flip, and Holm
workflow. This is stricter because it does not treat two action recordings from
one person as independent clusters. / 该脚本还把同一人的全部动作按受试者键聚合，
重复 Bootstrap、符号翻转和 Holm 流程。它更保守，因为不会把同一个人的两段动作录制
当成独立簇。

The output is written to
`paired_subject_cluster_sensitivity_by_geometry.csv` and to the
`paired_subject_cluster_sensitivity` fields in
`calibration_models_and_statistics.json`. / 输出写入上述 CSV，并记录在统计 JSON
的受试者聚类敏感性字段中。

## 11. Final machine audit / 最终机器审计

`src/audit_multimodal_formal_results.py` rejects the package unless all of the following hold:

- full 27-class clean gate passes for three masks;
- all 323 condition IDs exist;
- artifact hashes match the manifest;
- logits are finite and `[samples, 27]`;
- sample ordering matches across conditions;
- exact point counts match requested degradation;
- calibration/test subjects are disjoint;
- strict27 and conditioned7 outputs are complete;
- temperature scaling preserves Accuracy and Macro-F1;
- recording- and subject-cluster bootstrap, sign-flip, and Holm outputs are valid.

最终审计是交付门槛，不通过就不能把结果表述为正式证据。

## 12. Result summarization and evidence guard / 结果摘要与证据保护

`src/summarize_multimodal_formal_results.py` writes:

```text
research_summary/key_findings.json
research_summary/key_findings_bilingual.md
```

It labels outputs as `formal_real_data_evidence` only when the formal audit passes. Synthetic smoke results are explicitly marked `software_smoke_only`.

只有正式审计通过才标记为真实数据证据；合成冒烟测试只能证明软件流程可运行。

## 13. Verification commands / 验证命令

```bash
# Full unit-test suite / 完整单元测试
.venv-server/bin/python -m unittest discover -s tests -p 'test_*.py'

# Shell syntax / Shell 语法
bash -n scripts/run_formal_multimodal_study.sh

# Runtime status / 运行状态
bash scripts/formal_multimodal_status.sh

# Final audit / 最终审计
.venv-server/bin/python src/audit_multimodal_formal_results.py \
  --baseline-dir results/formal_multimodal_lower_limb/clean_baseline_all27 \
  --inference-dir results/formal_multimodal_lower_limb/inference \
  --analysis-dir results/formal_multimodal_lower_limb/analysis
```

## 14. What you must be able to explain / 你必须能独立解释什么

1. Why frame-ID alignment is required / 为什么必须按帧号同步。
2. Why the original X-Fi split and post-hoc calibration split answer different questions / 为什么原模型划分与校准划分回答不同问题。
3. Why positive temperature changes confidence but not predicted class / 为什么正温度只改置信度、不改类别。
4. Why quality-aware features exclude labels and corruption severity / 为什么质量特征不能使用真实标签和人工退化等级。
5. Why subject-action is primary and subject clustering is a sensitivity analysis / 为什么主要统计按受试者-动作聚类，并用受试者聚类做敏感性分析。
6. Why controlled point loss is not equivalent to a physical sensor fault / 为什么受控删点不等于真实硬件故障。
7. Why a passed machine audit is necessary but not sufficient for publication / 为什么机器审计通过仍不等于论文必然录用。

Remember / 记住一句话：**Every claim must point to a verified artifact. / 每个结论都必须指向可验证产物。**

## 15. Derived decision questions / 派生决策问题

`src/derive_multimodal_formal_findings.py` does not rerun X-Fi. It reads the
audited condition-level metric table and answers three deployment-oriented
questions. / 该脚本不重新运行 X-Fi，而是从已审计的条件级指标表中回答三个部署问题。

1. **Fusion versus matched unimodal / 融合与匹配单模态：** compare fused
   Accuracy with the better unimodal branch under the same geometry, seed, and
   retained-point severity.
2. **Keep versus drop / 保留还是移除：** compare keeping a degraded sensor
   with explicitly masking that sensor while leaving the other stream clean.
3. **Quality-aware versus pooled TS / 质量感知与混合全局校准：** pair
   NLL values condition by condition so the comparison is not confounded by a
   different degradation mix.

The pairing keys are part of the scientific method. If geometry or seed differs,
the rows are not a controlled comparison. / 配对键属于科研方法的一部分；如果退化几何或随机种子不同，就不是受控对照。

Outputs / 输出：

```text
analysis/<space>/derived/fusion_vs_unimodal_by_condition.csv
analysis/<space>/derived/keep_degraded_vs_drop_by_seed.csv
analysis/<space>/derived/quality_vs_pooled_by_condition.csv
analysis/<space>/derived/derived_findings.json
analysis/<space>/derived/*.png
```

## 16. From audited numbers to documents / 从审计数值到文档

This stage uses four programs with separate responsibilities. / 该阶段由四个职责分离的程序组成。

**1. audit_multimodal_formal_results.py**

Input / 输入：inference archives, metric tables, and figures / 推理归档、指标表和图表。
Output / 输出：`formal_multimodal_audit.json`.
Safety rule / 安全规则：reject missing or inconsistent evidence / 拒绝缺失或不一致证据。

**2. summarize_multimodal_formal_results.py**

Input / 输入：passed audit and completed analysis / 已通过的审计和完整分析。
Output / 输出：`key_findings.json` and a bilingual summary / 机器可读发现和双语摘要。
Safety rule / 安全规则：grant `claim_permitted` only to the complete 323-condition run / 只对完整的 323 条件正式运行放行结论。

**3. render_multimodal_formal_results_report.py**

Input / 输入：machine-readable findings / 机器可读研究发现。
Output / 输出：a concise bilingual report / 精简双语报告。
Safety rule / 安全规则：interpret the primary endpoint from its confidence interval and p-value / 依据置信区间和 p 值解释主要终点。

**4. finalize_multimodal_formal_documents.py**

Input / 输入：paper and teaching templates plus audited findings / 论文、教学模板和已审计发现。
Output / 输出：final manuscript, teaching report, supplement, and manifest / 最终论文、教学报告、补充材料和交付清单。
Safety rule / 安全规则：refuse unresolved placeholders and unaudited evidence / 拒绝未替换占位符和未审计数据。

The interpretation function has three possible outcomes: supported improvement,
supported harm, or inconclusive. It does not rewrite an unfavorable result as a
success. / 结果解释只允许“支持改善”、“支持恶化”或“证据不足”，不会把不利结果改写成成功。

## 17. Secure delivery package / 安全交付包

`src/create_formal_multimodal_delivery.py` copies code, tests, configuration,
analysis tables, figures, manifests, environment records, and reports. It
deliberately excludes raw MM-Fi data, X-Fi weights, the 323 large condition
archives, cloud-drive sessions, and SSH credentials. / 打包器只收集代码、测试、配置、分析表图、清单、环境记录和报告；不包含原始数据、模型权重、323 个大型推理文件、网盘会话或 SSH 凭据。

Before archiving, `assert_package_is_safe()` rejects: / 压缩前的安全门会拒绝：

- private-key markers / 私钥内容标记；
- checkpoint and key suffixes / 权重与密钥后缀；
- raw-data or credential directories / 原始数据与凭据目录；
- long browser-session credential values / 长网页会话凭据。

Every included file receives SHA-256, and the archive itself receives a second
SHA-256 file. A hash proves byte identity, not scientific correctness; scientific
correctness is handled by the audit. / 包内文件和压缩包都有 SHA-256。哈希只证明字节一致，科学正确性由审计程序负责。

## 18. Final one-command workflow / 最终一键流程

After all 323 inference conditions complete: / 323 个推理条件完成后：

```bash
RESULT_ROOT=/data/xfi_mmfi_workspace/results/formal_multimodal_lower_limb \
  bash scripts/finalize_formal_multimodal_delivery.sh
```

This command recaptures the environment, reruns the audit, regenerates summaries,
finalizes the manuscript and teaching report, and builds a hashed compact archive.
/ 该命令会重新记录环境、重跑审计、重建摘要、完成论文和教学报告，最后生成带哈希的精简交付包。
