#!/usr/bin/env python3
"""Build the one-page advisor-facing research brief from audited JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "tables" / "key_findings.json"
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "Yiyang_Zhang_Multimodal_Perception_Reliability_Research_Brief.pdf"
)

INK = HexColor("#17252A")
MUTED = HexColor("#5D6B70")
TEAL = HexColor("#087E8B")
GREEN = HexColor("#2A9D8F")
CORAL = HexColor("#E85D4A")
AMBER = HexColor("#D69E2E")
LIGHT = HexColor("#F3F7F6")
LINE = HexColor("#D7E1DF")
GRAY = HexColor("#9AA7AA")
WHITE = HexColor("#FFFFFF")


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("audit_status") != "PASS":
        raise RuntimeError("Refusing to build brief from results without a PASS audit.")
    return data


def draw_wrapped_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 8.4,
    leading: float = 11.0,
    color=INK,
    max_lines: int | None = None,
) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines is not None:
        lines = lines[:max_lines]

    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def section_title(pdf: canvas.Canvas, text: str, x: float, y: float) -> None:
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 9.2)
    pdf.drawString(x, y, text.upper())


def stat(pdf: canvas.Canvas, value: str, label: str, x: float, y: float) -> None:
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 14.5)
    pdf.drawString(x, y, value)
    draw_wrapped_text(pdf, label, x, y - 13, 115, size=7.2, leading=8.5, color=MUTED)


def bullet(pdf: canvas.Canvas, text: str, x: float, y: float, width: float) -> float:
    pdf.setFillColor(TEAL)
    pdf.circle(x + 2.5, y + 2.2, 1.8, fill=1, stroke=0)
    return draw_wrapped_text(pdf, text, x + 11, y + 6, width - 11, size=8.0, leading=10.2)


def build_brief(results: dict, output: Path, repository_url: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    pdf = canvas.Canvas(str(output), pagesize=A4)
    pdf.setTitle("Multimodal Perception Reliability Research Brief")
    pdf.setAuthor("Yiyang Zhang")
    pdf.setSubject("Independent reproduction and controlled reliability evaluation")

    strict = results["output_spaces"]["strict27"]
    clean = strict["clean_uncalibrated_by_mask"]
    degraded = strict["degraded_fusion_mean_by_method"]
    gate_rows = results["clean_baseline_gate"]["rows"]

    margin = 36
    usable = width - 2 * margin

    # Header
    pdf.setFillColor(TEAL)
    pdf.rect(0, height - 8, width, 8, fill=1, stroke=0)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 7.4)
    pdf.drawString(margin, height - 28, "RESEARCH BRIEF  |  REPRODUCTION AND CONTROLLED EVALUATION  |  13 AUG 2026")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 17.2)
    pdf.drawString(margin, height - 55, "Confidence Calibration for Multimodal Indoor")
    pdf.drawString(margin, height - 76, "Lower-Limb Activity Recognition under Sensing Degradation")
    pdf.setFont("Helvetica", 8.2)
    pdf.setFillColor(MUTED)
    pdf.drawString(margin, height - 94, "Yiyang Zhang  |  MSc Smart Manufacturing, Nanyang Technological University")
    pdf.drawRightString(width - margin, height - 94, "YIYANG021@e.ntu.edu.sg")

    # Research question band
    q_top = height - 111
    pdf.setFillColor(LIGHT)
    pdf.roundRect(margin, q_top - 48, usable, 48, 4, fill=1, stroke=0)
    pdf.setFillColor(TEAL)
    pdf.rect(margin, q_top - 48, 4, 48, fill=1, stroke=0)
    pdf.setFont("Helvetica-Bold", 8.1)
    pdf.drawString(margin + 14, q_top - 15, "RESEARCH QUESTION")
    draw_wrapped_text(
        pdf,
        "When aligned LiDAR and mmWave observations deteriorate, can observable quality cues make a frozen multimodal model's confidence more trustworthy than one pooled temperature?",
        margin + 14,
        q_top - 29,
        usable - 27,
        size=8.7,
        leading=10.6,
    )

    # Protocol and evidence base
    y = q_top - 68
    section_title(pdf, "Study design", margin, y)
    pipeline_y = y - 30
    box_w = 116
    gap = 13
    labels = [
        ("ALIGNED INPUT", "LiDAR + mmWave"),
        ("CONTROLLED LOSS", "uniform / azimuth"),
        ("FROZEN MODEL", "multimodal recognizer"),
        ("POST-HOC TEST", "confidence calibration"),
    ]
    for idx, (heading, detail) in enumerate(labels):
        x = margin + idx * (box_w + gap)
        pdf.setFillColor(LIGHT if idx % 2 == 0 else WHITE)
        pdf.setStrokeColor(LINE)
        pdf.roundRect(x, pipeline_y - 31, box_w, 31, 3, fill=1, stroke=1)
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 6.7)
        pdf.drawCentredString(x + box_w / 2, pipeline_y - 12, heading)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica", 7.2)
        pdf.drawCentredString(x + box_w / 2, pipeline_y - 23, detail)
        if idx < len(labels) - 1:
            arrow_x = x + box_w + 3
            pdf.setStrokeColor(GRAY)
            pdf.line(arrow_x, pipeline_y - 15.5, arrow_x + 7, pipeline_y - 15.5)
            pdf.line(arrow_x + 7, pipeline_y - 15.5, arrow_x + 4, pipeline_y - 12.5)
            pdf.line(arrow_x + 7, pipeline_y - 15.5, arrow_x + 4, pipeline_y - 18.5)

    stats_y = pipeline_y - 58
    stat(pdf, "54,433", "frames in full 27-action reproduction gate", margin, stats_y)
    stat(pdf, "15,315", "aligned target frames from 33 healthy volunteers", margin + 174, stats_y)
    stat(pdf, "323", "clean, degraded, fused and unimodal conditions", margin + 348, stats_y)

    # Two compact evidence panels
    panel_top = stats_y - 49
    left_x = margin
    left_w = 250
    right_x = margin + 270
    right_w = usable - 270
    section_title(pdf, "1. Reproduction gate", left_x, panel_top)
    draw_wrapped_text(
        pdf,
        "The released checkpoint was first checked on the complete clean validation cohort. All three masks were within the frozen 0.03 tolerance.",
        left_x,
        panel_top - 14,
        left_w,
        size=7.3,
        leading=9.0,
        color=MUTED,
    )
    chart_top = panel_top - 51
    row_names = ["Fusion", "LiDAR", "mmWave"]
    for idx, (name, row) in enumerate(zip(row_names, gate_rows)):
        yy = chart_top - idx * 25
        reproduced = row["accuracy"]
        published = row["official_accuracy_reference"]
        pdf.setFont("Helvetica-Bold", 7.4)
        pdf.setFillColor(INK)
        pdf.drawString(left_x, yy + 4, name)
        bar_x = left_x + 50
        bar_w = 126
        pdf.setFillColor(HexColor("#E4EAE9"))
        pdf.roundRect(bar_x, yy, bar_w, 7, 2, fill=1, stroke=0)
        pdf.setFillColor(TEAL)
        pdf.roundRect(bar_x, yy, bar_w * reproduced, 7, 2, fill=1, stroke=0)
        marker_x = bar_x + bar_w * published
        pdf.setStrokeColor(CORAL)
        pdf.setLineWidth(1.4)
        pdf.line(marker_x, yy - 2, marker_x, yy + 9)
        pdf.setFont("Helvetica", 7.1)
        pdf.setFillColor(INK)
        pdf.drawRightString(left_x + left_w, yy + 1, f"{reproduced:.3f} / {published:.3f}")
    pdf.setFont("Helvetica", 6.5)
    pdf.setFillColor(MUTED)
    pdf.drawString(left_x + 50, chart_top - 78, "reproduced bar / published marker")

    section_title(pdf, "2. Confidence reliability", right_x, panel_top)
    draw_wrapped_text(
        pdf,
        "Mean Expected Calibration Error (ECE) across 240 degraded fused test conditions. Lower is better; temperature scaling does not alter class predictions.",
        right_x,
        panel_top - 14,
        right_w,
        size=7.3,
        leading=9.0,
        color=MUTED,
    )
    methods = [
        ("Uncalibrated", degraded["uncalibrated"]["ece"], CORAL),
        ("Pooled TS", degraded["pooled_global_ts"]["ece"], AMBER),
        ("Quality-aware", degraded["quality_aware_ts"]["ece"], GREEN),
    ]
    chart_y = panel_top - 62
    max_ece = 0.30
    for idx, (name, value, color) in enumerate(methods):
        x = right_x + idx * 77
        height_bar = 70 * value / max_ece
        pdf.setFillColor(HexColor("#E8EEED"))
        pdf.roundRect(x, chart_y - 70, 38, 70, 3, fill=1, stroke=0)
        pdf.setFillColor(color)
        pdf.roundRect(x, chart_y - height_bar, 38, height_bar, 3, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 7.4)
        pdf.drawCentredString(x + 19, chart_y + 8, f"{value:.3f}")
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 6.5)
        pdf.drawCentredString(x + 19, chart_y - 81, name)

    # Findings and interpretation
    findings_top = panel_top - 162
    pdf.setStrokeColor(LINE)
    pdf.line(margin, findings_top + 12, width - margin, findings_top + 12)
    section_title(pdf, "What the experiment shows", margin, findings_top)
    y_left = findings_top - 17
    y_left = bullet(
        pdf,
        f"Recognition degraded sharply: fused Accuracy was {clean['lidar_mmwave']['accuracy']:.3f} on clean target data and averaged {degraded['uncalibrated']['accuracy']:.3f} across degraded fused conditions.",
        margin,
        y_left,
        250,
    )
    y_left -= 3
    bullet(
        pdf,
        "Under severe asymmetric degradation, fixed fusion could underperform the stronger available unimodal branch.",
        margin,
        y_left,
        250,
    )

    y_right = findings_top - 17
    y_right = bullet(
        pdf,
        f"Quality-aware temperature scaling reduced mean ECE from {degraded['pooled_global_ts']['ece']:.3f} to {degraded['quality_aware_ts']['ece']:.3f} and NLL from {degraded['pooled_global_ts']['nll']:.3f} to {degraded['quality_aware_ts']['nll']:.3f} versus pooled scaling.",
        right_x,
        y_right,
        right_w,
    )
    y_right -= 3
    bullet(
        pdf,
        "Calibration improved probability reliability, not recognition Accuracy; both methods retained the same predictions.",
        right_x,
        y_right,
        right_w,
    )

    # Implication band
    implication_y = findings_top - 93
    pdf.setFillColor(INK)
    pdf.roundRect(margin, implication_y - 64, usable, 64, 4, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#7AD6D0"))
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.drawString(margin + 14, implication_y - 16, "RESEARCH IMPLICATION")
    draw_wrapped_text(
        pdf,
        "Observable sensing quality can improve how much a frozen multimodal model should be trusted, but cannot repair a wrong prediction. A focused dissertation extension is quality-gated fusion or selective abstention, followed by tests under real sensor faults.",
        margin + 14,
        implication_y - 32,
        usable - 28,
        font="Helvetica-Bold",
        size=8.4,
        leading=10.7,
        color=WHITE,
    )

    # Footer: boundaries and links
    footer_y = implication_y - 83
    section_title(pdf, "Claim boundaries", margin, footer_y)
    boundaries = (
        "Healthy volunteers; controlled software point loss; one released frozen checkpoint; "
        "no rehabilitation-patient, clinical, physical-failure, or real-time deployment validation."
    )
    draw_wrapped_text(pdf, boundaries, margin, footer_y - 14, 355, size=7.1, leading=8.6, color=MUTED)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.4)
    pdf.drawRightString(width - margin, footer_y - 2, "Sources: MM-Fi (NeurIPS 2023); X-Fi (ICLR 2025)")
    repo_text = repository_url or "Repository: local reproducibility package prepared"
    if repository_url:
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 6.7)
        pdf.drawRightString(width - margin, footer_y - 16, repo_text)
        pdf.linkURL(repository_url, (width - margin - 220, footer_y - 20, width - margin, footer_y - 7))
    else:
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 6.7)
        pdf.drawRightString(width - margin, footer_y - 16, repo_text)

    pdf.showPage()
    pdf.save()


def render_preview(pdf_path: Path, preview_path: Path) -> None:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF to render the preview.") from exc
    document = pymupdf.open(pdf_path)
    if len(document) != 1:
        raise RuntimeError(f"Expected one page, found {len(document)}.")
    pixmap = document[0].get_pixmap(
        matrix=pymupdf.Matrix(1.8, 1.8), alpha=False
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(preview_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--repository-url")
    args = parser.parse_args()

    results = load_results(args.results)
    build_brief(results, args.output, args.repository_url)
    if args.preview:
        render_preview(args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
