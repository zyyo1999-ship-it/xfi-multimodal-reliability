# Reproducibility workflow

The formal workflow is intentionally staged. Do not begin with the 323-condition
run: first prove that the data alignment and released checkpoint behave as
expected.

## 1. Install and test

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build matching the host CUDA version.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 2. Add official third-party source

```bash
mkdir -p third_party
git clone https://github.com/NTUMARS/X-Fi.git third_party/X-Fi
git clone https://github.com/ybhbingo/MMFi_dataset.git third_party/MMFi_dataset
```

Obtain MM-Fi data and the released X-Fi checkpoint through the official
channels described in `DATA_AND_WEIGHTS.md`.

## 3. Audit synchronized point data

```bash
python src/audit_extracted_point_modalities.py \
  --data-root data/aligned_points \
  --scope all \
  --hash-content \
  --output artifacts/data_audit.json
```

This verifies frame alignment, shape, finite values, expected recordings, and
content identity before inference.

## 4. Run a small smoke test

```bash
python src/run_multimodal_inference.py \
  --data-root data/aligned_points \
  --checkpoint assets/xfi_weights/MMFi_HAR/<released-full-checkpoint>.pt \
  --data-audit artifacts/data_audit.json \
  --output-dir artifacts/smoke \
  --plan smoke \
  --scope target \
  --batch-size 8 \
  --max-samples 128 \
  --max-conditions 2
```

Inspect `artifacts/smoke/inference_manifest.json`. The run signature binds the
checkpoint hash, source bundle, data identity, sample count, condition IDs, and
batch configuration. Resume is rejected if any immutable input changes.

## 5. Reproduce the clean checkpoint gate

```bash
python src/run_multimodal_inference.py \
  --data-root data/aligned_points \
  --checkpoint assets/xfi_weights/MMFi_HAR/<released-full-checkpoint>.pt \
  --data-audit artifacts/data_audit.json \
  --output-dir artifacts/clean_baseline \
  --plan baseline \
  --scope all \
  --batch-size 32 \
  --cache-gib 20

python src/evaluate_multimodal_clean_baseline.py \
  --inference-dir artifacts/clean_baseline \
  --output artifacts/clean_baseline_gate.json \
  --tolerance 0.03 \
  --expected-frame-count 54433
```

The full formal analysis should proceed only after LiDAR-only, mmWave-only, and
fused Accuracy pass the frozen tolerance.

## 6. Run the frozen formal matrix

```bash
python src/run_multimodal_inference.py \
  --data-root data/aligned_points \
  --checkpoint assets/xfi_weights/MMFi_HAR/<released-full-checkpoint>.pt \
  --data-audit artifacts/data_audit.json \
  --output-dir artifacts/formal_inference \
  --plan formal \
  --scope target \
  --batch-size 32 \
  --cache-gib 20
```

The runner writes one atomic compressed result per condition and can resume only
when the immutable run signature matches.

## 7. Fit calibrators and audit results

```bash
python src/analyze_multimodal_calibration.py \
  --inference-dir artifacts/formal_inference \
  --output-dir artifacts/formal_analysis \
  --output-space strict27 \
  --bootstrap-repetitions 2000 \
  --sign-flip-repetitions 20000

python src/audit_multimodal_formal_results.py \
  --baseline-dir artifacts/clean_baseline \
  --inference-dir artifacts/formal_inference \
  --analysis-dir artifacts/formal_analysis \
  --output artifacts/formal_audit.json
```

The selected aggregate outputs from the completed run are retained under
`results/`. Raw per-frame logits and predictions are excluded from Git.

## Determinism and remaining variation

The code fixes corruption seeds, model seed, deterministic PyTorch settings,
subject partitions, condition IDs, and source/data/checkpoint hashes. Exact
throughput depends on hardware. Minor floating-point differences can still
occur across CUDA, driver, PyTorch, and GPU versions.
