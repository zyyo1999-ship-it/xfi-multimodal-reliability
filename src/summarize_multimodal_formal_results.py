#!/usr/bin/env python3
"""Create an auditable bilingual summary of the formal multimodal study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


METRICS = (
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece",
    "adaptive_ece",
    "aurc",
)
METHOD_LABELS = {
    "uncalibrated": "Uncalibrated / 未校准",
    "clean_global_ts": "Clean global TS / 干净数据全局温度缩放",
    "pooled_global_ts": "Pooled global TS / 混合退化全局温度缩放",
    "severity_oracle_ts": "Condition-oracle TS / 条件已知温度缩放",
    "quality_aware_ts": "Quality-aware TS / 质量感知温度缩放",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize an empty metric group")
    return {metric: fmean(numeric(row, metric) for row in rows) for metric in METRICS}


def summarize_quality_vs_pooled(rows: list[dict[str, str]]) -> dict:
    by_condition: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition_id"]][row["method"]] = row

    deltas: dict[str, list[float]] = {metric: [] for metric in ("nll", "brier", "ece", "adaptive_ece")}
    wins = {metric: 0 for metric in deltas}
    ties = {metric: 0 for metric in deltas}
    comparisons = 0
    for methods in by_condition.values():
        if "quality_aware_ts" not in methods or "pooled_global_ts" not in methods:
            continue
        comparisons += 1
        quality = methods["quality_aware_ts"]
        pooled = methods["pooled_global_ts"]
        for metric in deltas:
            delta = numeric(quality, metric) - numeric(pooled, metric)
            deltas[metric].append(delta)
            if delta < -1e-12:
                wins[metric] += 1
            elif abs(delta) <= 1e-12:
                ties[metric] += 1
    if comparisons == 0:
        raise ValueError("No paired quality-aware and pooled calibration rows")
    return {
        "condition_comparisons": comparisons,
        "mean_quality_minus_pooled": {
            metric: fmean(values) for metric, values in deltas.items()
        },
        "quality_aware_wins": wins,
        "ties": ties,
    }


def summarize_robustness_auc(path: Path) -> dict:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["method"], row["path"])].append(row)
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, path_name), rows in grouped.items():
        output[method][path_name] = {
            "accuracy_auc_mean": fmean(
                numeric(row, "accuracy_robustness_auc") for row in rows
            ),
            "macro_f1_auc_mean": fmean(
                numeric(row, "macro_f1_robustness_auc") for row in rows
            ),
            "row_count": len(rows),
        }
    return dict(output)


def summarize_output_space(directory: Path) -> dict:
    rows = read_csv(directory / "metrics_by_condition_method.csv")
    if not rows:
        raise ValueError(f"No metrics found in {directory}")
    clean_rows = [
        row
        for row in rows
        if row["geometry"] == "clean" and row["method"] == "uncalibrated"
    ]
    clean_by_mask = {
        row["mask_name"]: {metric: numeric(row, metric) for metric in METRICS}
        for row in clean_rows
    }
    fused_degraded = [
        row
        for row in rows
        if row["mask_name"] == "lidar_mmwave" and row["geometry"] != "clean"
    ]
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fused_degraded:
        by_method[row["method"]].append(row)
    degraded_means = {
        method: mean_metrics(method_rows) for method, method_rows in by_method.items()
    }

    uncalibrated = [row for row in fused_degraded if row["method"] == "uncalibrated"]
    if not uncalibrated:
        raise ValueError("No degraded uncalibrated fusion rows")
    worst = min(uncalibrated, key=lambda row: (numeric(row, "accuracy"), -numeric(row, "nll")))
    worst_methods = {
        row["method"]: {metric: numeric(row, metric) for metric in METRICS}
        for row in fused_degraded
        if row["condition_id"] == worst["condition_id"]
    }

    statistics = json.loads(
        (directory / "calibration_models_and_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "output_space": rows[0]["output_space"],
        "metric_row_count": len(rows),
        "clean_uncalibrated_by_mask": clean_by_mask,
        "degraded_fusion_condition_count": len(
            {row["condition_id"] for row in fused_degraded}
        ),
        "degraded_fusion_mean_by_method": degraded_means,
        "worst_uncalibrated_fusion_condition": {
            "condition_id": worst["condition_id"],
            "geometry": worst["geometry"],
            "corruption_seed": int(worst["corruption_seed"]),
            "lidar_drop_rate": numeric(worst, "lidar_drop_rate"),
            "mmwave_drop_rate": numeric(worst, "mmwave_drop_rate"),
            "methods": worst_methods,
        },
        "quality_aware_vs_pooled": summarize_quality_vs_pooled(fused_degraded),
        "paired_cluster_statistics": statistics["paired_cluster_bootstrap"],
        "geometry_statistics_holm": statistics["geometry_stratified_bootstrap_holm"],
        # New formal runs include subject-cluster sensitivity analyses. Keep
        # older audited fixtures readable so result-summary compatibility does
        # not depend on when this secondary analysis was introduced.
        "paired_subject_cluster_sensitivity": statistics.get(
            "paired_subject_cluster_sensitivity"
        ),
        "geometry_subject_cluster_sensitivity_holm": statistics.get(
            "geometry_stratified_subject_cluster_sensitivity_holm", []
        ),
        "robustness_auc": summarize_robustness_auc(
            directory / "robustness_auc_by_seed.csv"
        ),
    }


def build_summary(result_root: Path) -> dict:
    audit_path = result_root / "analysis" / "formal_multimodal_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing formal audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("Refusing to summarize an experiment that failed audit")

    alignment_path = result_root / "inference" / "data_alignment_audit.json"
    manifest_path = result_root / "inference" / "inference_manifest.json"
    if not alignment_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Formal alignment audit or inference manifest is missing")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_identity = manifest.get("data_audit_identity") or {}

    clean_gate = audit.get("clean_baseline_gate") or {}
    is_formal = (
        int(audit.get("sample_count", -1)) > 0
        and int(audit.get("condition_count", -1)) == 323
        and audit.get("clean_baseline_gate_included") is True
        and clean_gate.get("status") == "PASS"
        and int(clean_gate.get("sample_count", -1)) == 54_433
    )
    return {
        "summary_schema_version": 1,
        "experiment": audit.get("experiment"),
        "evidence_status": "formal_real_data_evidence" if is_formal else "software_smoke_only",
        "claim_permitted": bool(is_formal),
        "audit_status": audit["status"],
        "sample_count": int(audit["sample_count"]),
        "condition_count": int(audit["condition_count"]),
        "target_recording_count": int(alignment["recording_count"]),
        "target_subject_count": int(alignment["subject_count"]),
        "target_action_count": int(alignment["action_count"]),
        "calibration_subject_count": int(audit["calibration_subject_count"]),
        "test_subject_count": int(audit["test_subject_count"]),
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "data_audit_sha256": data_identity.get("sha256"),
        "aligned_pair_content_sha256": data_identity.get(
            "aligned_pair_content_sha256"
        ),
        "archive_provenance_bundle_sha256": data_identity.get(
            "archive_provenance_bundle_sha256"
        )
        or data_identity.get("provenance_bundle_sha256"),
        "clean_baseline_gate": audit.get("clean_baseline_gate"),
        "output_spaces": {
            output_space: summarize_output_space(result_root / "analysis" / output_space)
            for output_space in ("strict27", "conditioned7")
        },
        "claim_boundaries": [
            "Controlled point-loss simulation is not a clinical deployment study.",
            "MM-Fi participants are healthy volunteers; no diagnostic or treatment claim is supported.",
            "The official action-wise split is not a globally unseen-subject recognition protocol.",
            "Temperature scaling changes confidence, not the predicted class or accuracy.",
        ],
    }


def metric_line(metrics: dict[str, float]) -> str:
    return (
        f"Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
        f"NLL={metrics['nll']:.4f}, Brier={metrics['brier']:.4f}, "
        f"ECE={metrics['ece']:.4f}, Adaptive-ECE={metrics['adaptive_ece']:.4f}"
    )


def render_markdown(summary: dict) -> str:
    formal = summary["claim_permitted"]
    clean_gate = summary.get("clean_baseline_gate") or {}
    clean_gate_samples = clean_gate.get("sample_count")
    clean_gate_text = (
        f"{int(clean_gate_samples):,}" if clean_gate_samples is not None else "N/A"
    )
    status_cn = "真实正式实验，可用于受限范围内的研究结论" if formal else "仅软件冒烟测试，不可作为研究证据"
    status_en = "formal real-data evidence with bounded claims" if formal else "software smoke test only; not research evidence"
    lines = [
        "# Multimodal Calibration Key Findings / 多模态校准关键结果",
        "",
        f"**Evidence status / 证据状态:** {status_en} / {status_cn}",
        "",
        f"- Samples / 样本帧: {summary['sample_count']:,}",
        f"- Full clean-gate frames / 完整 clean 门槛帧: {clean_gate_text}",
        f"- Recordings / 受试者-动作记录: {summary['target_recording_count']}",
        f"- Subjects / 目标队列受试者: {summary['target_subject_count']}",
        f"- Target actions / 下肢目标动作: {summary['target_action_count']}",
        f"- Conditions / 实验条件: {summary['condition_count']}",
        f"- Calibration subjects / 校准受试者: {summary['calibration_subject_count']}",
        f"- Test subjects / 测试受试者: {summary['test_subject_count']}",
        f"- Checkpoint SHA-256 / 权重哈希: `{summary['checkpoint_sha256']}`",
        f"- Data-audit SHA-256 / 数据审计哈希: "
        f"`{summary['data_audit_sha256']}`",
        f"- Aligned-pair SHA-256 / 对齐数据对哈希: "
        f"`{summary['aligned_pair_content_sha256']}`",
        f"- Archive-provenance SHA-256 / 原始压缩包来源哈希: "
        f"`{summary['archive_provenance_bundle_sha256']}`",
        "",
    ]
    if not formal:
        lines.extend(
            [
                "> **Warning / 警告:** The numbers below only verify the analysis software. "
                "They must not be reported as empirical findings. / 以下数字仅验证分析代码，严禁作为实证结论。",
                "",
            ]
        )

    for name, payload in summary["output_spaces"].items():
        lines.extend([f"## {name}", "", "### Clean recognition / 干净数据识别", ""])
        for mask, metrics in payload["clean_uncalibrated_by_mask"].items():
            lines.append(f"- `{mask}`: {metric_line(metrics)}")
        worst = payload["worst_uncalibrated_fusion_condition"]
        lines.extend(
            [
                "",
                "### Worst fused degradation / 最差融合退化条件",
                "",
                f"- Condition / 条件: `{worst['condition_id']}`",
                f"- Uncalibrated / 未校准: {metric_line(worst['methods']['uncalibrated'])}",
                "",
                "### Mean calibration performance on degraded fusion / 退化融合上的平均校准表现",
                "",
            ]
        )
        for method, metrics in payload["degraded_fusion_mean_by_method"].items():
            lines.append(f"- {METHOD_LABELS.get(method, method)}: {metric_line(metrics)}")
        comparison = payload["quality_aware_vs_pooled"]
        lines.extend(
            [
                "",
                "### Quality-aware TS vs pooled TS / 质量感知校准与混合全局校准",
                "",
                f"- Paired conditions / 配对条件: {comparison['condition_comparisons']}",
                f"- Mean QA - pooled NLL / 平均NLL差: {comparison['mean_quality_minus_pooled']['nll']:.6f}",
                f"- QA NLL wins / QA在NLL上的胜出条件数: {comparison['quality_aware_wins']['nll']}",
                f"- Cluster sign-flip p / 聚类符号翻转检验p值: "
                f"{payload['paired_cluster_statistics']['sign_flip_p']:.6g}",
                "",
            ]
        )

    lines.extend(["## Claim boundaries / 结论边界", ""])
    translations = (
        "受控点丢失模拟不等同于临床部署研究。",
        "MM-Fi参与者为健康志愿者，不能支持诊断或治疗结论。",
        "官方按动作随机划分不等同于全局未见受试者协议。",
        "温度缩放只改变置信度，不改变类别预测或准确率。",
    )
    for english, chinese in zip(summary["claim_boundaries"], translations):
        lines.append(f"- {english} / {chinese}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or arguments.result_root / "research_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(arguments.result_root)
    (output_dir / "key_findings.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "key_findings_bilingual.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "evidence_status": summary["evidence_status"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
