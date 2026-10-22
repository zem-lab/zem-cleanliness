from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.cleanliness_core.optical_particle_analysis import (
    SIMULATED_ELEMENTS_DISCLAIMER,
)


def _successful_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in manifest.get("particles", [])
        if record.get("status") == "SUCCEEDED"
    ]


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    records = list(manifest.get("particles", []))
    succeeded = _successful_records(manifest)
    errors = [
        float(record["localization"]["centerErrorUm"])
        for record in succeeded
        if record.get("localization", {}).get("centerErrorUm") is not None
    ]
    high = sum(
        record.get("match", {}).get("status") == "MATCHED_HIGH"
        for record in records
    )
    medium = sum(
        record.get("match", {}).get("status") == "MATCHED_MEDIUM"
        for record in records
    )
    return {
        "total": len(records),
        "succeeded": len(succeeded),
        "failed": len(records) - len(succeeded),
        "matchedHigh": high,
        "matchedMedium": medium,
        "successRate": len(succeeded) / max(len(records), 1),
        "centerErrorUm": {
            "mean": sum(errors) / len(errors) if errors else None,
            "minimum": min(errors) if errors else None,
            "maximum": max(errors) if errors else None,
            "rms": (
                math.sqrt(sum(value * value for value in errors) / len(errors))
                if errors
                else None
            ),
        },
    }


def _metric(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _element_text(record: dict[str, Any]) -> str:
    spectrum = record.get("energySpectrum") or {}
    parts = []
    for element in spectrum.get("elements") or []:
        try:
            parts.append(
                f"{element.get('symbol', '')} "
                f"{float(element.get('value', 0.0)):.2f}%"
            )
        except (TypeError, ValueError):
            continue
    return "、".join(parts) or "-"


def _relative_image(record: dict[str, Any], key: str) -> str:
    value = str((record.get("images") or {}).get(key) or "")
    return html.escape(value.replace("\\", "/"))


def write_html_report(
    task_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    task_dir = Path(task_dir)
    cards: list[str] = []
    for record in manifest.get("particles", []):
        match = record.get("match") or {}
        localization = record.get("localization") or {}
        target = record.get("targetStageUm") or {}
        actual = record.get("actualStageUm") or {}
        status = str(record.get("status") or "UNKNOWN")
        confidence = str(match.get("confidence") or "low")
        status_cn = {
            "MATCHED_HIGH": "高置信度找到",
            "MATCHED_MEDIUM": "中置信度找到",
            "FAILED_NOT_FOUND": "未找到",
            "FAILED_LOW_CONFIDENCE": "低置信度失败",
            "FAILED_CAPTURE": "采图失败",
            "FAILED_STAGE_MOVE": "移动失败",
        }.get(str(match.get("status") or status), status)
        error_text = _metric(localization.get("centerErrorUm"))
        cards.append(
            f"""
            <article class="particle-card" data-status="{html.escape(status)}"
                     data-confidence="{html.escape(confidence)}">
              <header>
                <div>
                  <strong>光镜颗粒 #{int(record.get('sequence', 0))}</strong>
                  <span class="pill {html.escape(confidence)}">{html.escape(status_cn)}</span>
                </div>
                <b>{error_text} μm</b>
              </header>
              <div class="images">
                <figure>
                  <img loading="lazy" src="{_relative_image(record, 'opticalImage')}"
                       alt="光镜颗粒 #{int(record.get('sequence', 0))}">
                  <figcaption>光镜厂商原始颗粒图</figcaption>
                </figure>
                <figure>
                  <img loading="lazy" src="{_relative_image(record, 'semParticleImage')}"
                       alt="电镜颗粒 #{int(record.get('sequence', 0))}">
                  <figcaption>电镜实拍目标颗粒图</figcaption>
                </figure>
                <figure>
                  <img loading="lazy" src="{_relative_image(record, 'semAnnotatedImage')}"
                       alt="完整电镜定位证据 #{int(record.get('sequence', 0))}">
                  <figcaption>完整 SEM 视场与自动定位标记</figcaption>
                </figure>
              </div>
              <dl>
                <div><dt>Particle ID</dt><dd>{html.escape(str(record.get('particleId', '')))}</dd></div>
                <div><dt>光镜尺寸</dt><dd>{float(record.get('lengthUm', 0.0)):.1f} × {float(record.get('widthUm', 0.0)):.1f} μm</dd></div>
                <div><dt>预测 Stage</dt><dd>{float(target.get('x', 0.0)):.2f}, {float(target.get('y', 0.0)):.2f} μm</dd></div>
                <div><dt>实际 Stage</dt><dd>{float(actual.get('x', 0.0)):.2f}, {float(actual.get('y', 0.0)):.2f} μm</dd></div>
                <div><dt>候选数 / 分差</dt><dd>{int(match.get('candidateCount') or 0)} / {_metric(match.get('scoreMargin'), 3)}</dd></div>
                <div><dt>模拟元素</dt><dd>{html.escape(_element_text(record))}</dd></div>
              </dl>
              <p class="reason">{html.escape(str(match.get('reason') or record.get('error') or ''))}</p>
            </article>
            """
        )

    error = summary["centerErrorUm"]
    input_info = manifest.get("input") or {}
    calibration = manifest.get("calibration") or {}
    device = manifest.get("device") or {}
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>光镜颗粒 SEM 自动定位验证报告</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
      color:#19242d; background:#f4f7f8;
      --accent:#087f5b; --line:#d8e0e4; --muted:#64727d;
      --danger:#b42318; --warning:#9a6700;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; }}
    .hero {{ color:white; background:linear-gradient(135deg,#113d4b,#12352d);
      padding:40px max(24px,calc((100vw - 1280px)/2)); }}
    .hero h1 {{ margin:4px 0 10px; font-size:clamp(28px,4vw,46px); }}
    .hero p {{ max-width:920px; line-height:1.7; color:#d9ece7; }}
    main {{ width:min(1280px,calc(100% - 30px)); margin:24px auto 60px; }}
    .warning {{ padding:15px 18px; background:#fff4d6; border-left:5px solid #e5a100;
      border-radius:7px; font-weight:650; }}
    .metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr));
      gap:11px; margin:20px 0; }}
    .metric {{ padding:15px; background:white; border:1px solid var(--line);
      border-radius:10px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; margin-top:5px; font-size:24px; }}
    .audit {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; padding:18px;
      background:white; border:1px solid var(--line); border-radius:10px; }}
    .audit dt {{ color:var(--muted); font-size:11px; }}
    .audit dd {{ margin:3px 0 0; font:12px Consolas,monospace; overflow-wrap:anywhere; }}
    h2 {{ margin:30px 0 12px; }}
    .particle-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:14px; }}
    .particle-card {{ overflow:hidden; background:white; border:1px solid var(--line); border-radius:11px; }}
    .particle-card > header {{ display:flex; justify-content:space-between; align-items:center; padding:13px 15px; }}
    .particle-card > header b {{ color:var(--danger); font-size:21px; }}
    .pill {{ display:inline-block; margin-left:8px; padding:3px 8px; border-radius:999px; font-size:11px; }}
    .pill.high {{ color:#086447; background:#dff5ec; }}
    .pill.medium {{ color:#7b5200; background:#fff1c7; }}
    .pill.low {{ color:#9f241b; background:#fee4e2; }}
    .images {{ display:grid; grid-template-columns:.68fr .78fr 1.35fr; gap:1px; background:var(--line); }}
    figure {{ display:grid; grid-template-rows:1fr auto; margin:0; min-width:0; background:#121719; }}
    figure img {{ display:block; width:100%; height:270px; object-fit:contain; }}
    figcaption {{ padding:7px; color:#d4dce0; background:#20272b; font-size:11px; text-align:center; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin:0; padding:14px 15px; }}
    dt {{ color:var(--muted); font-size:11px; }}
    dd {{ margin:3px 0 0; font:12px Consolas,"Microsoft YaHei",sans-serif; overflow-wrap:anywhere; }}
    .reason {{ margin:0; padding:0 15px 14px; color:var(--muted); font-size:12px; }}
    @media(max-width:800px) {{
      .metrics {{ grid-template-columns:1fr 1fr; }}
      .audit,.particle-grid {{ grid-template-columns:1fr; }}
      .images {{ grid-template-columns:1fr; }}
      dl {{ grid-template-columns:1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <small>CORRELATIVE LIGHT–ELECTRON MICROSCOPY</small>
    <h1>光镜颗粒 SEM 自动定位验证报告</h1>
    <p>本报告逐颗并列展示光镜厂商导出的颗粒图、电镜实拍颗粒图以及完整
       SEM 视场定位证据。定位结论来自真实样品台移动和真实 SEM 图像；
       元素含量为模拟数据。</p>
  </header>
  <main>
    <p class="warning">{html.escape(SIMULATED_ELEMENTS_DISCLAIMER)}</p>
    <section class="metrics">
      <div class="metric"><span>任务颗粒</span><strong>{summary['total']}</strong></div>
      <div class="metric"><span>自动找到</span><strong>{summary['succeeded']}</strong></div>
      <div class="metric"><span>高置信度</span><strong>{summary['matchedHigh']}</strong></div>
      <div class="metric"><span>平均中心误差</span><strong>{_metric(error['mean'])} μm</strong></div>
      <div class="metric"><span>最大中心误差</span><strong>{_metric(error['maximum'])} μm</strong></div>
    </section>
    <dl class="audit">
      <div><dt>任务 ID</dt><dd>{html.escape(str(manifest.get('taskId', '')))}</dd></div>
      <div><dt>任务状态</dt><dd>{html.escape(str(manifest.get('status', '')))}</dd></div>
      <div><dt>光镜源文件</dt><dd>{html.escape(str(input_info.get('path', '')))}</dd></div>
      <div><dt>光镜 SHA-256</dt><dd>{html.escape(str(input_info.get('sha256', '')))}</dd></div>
      <div><dt>标定版本</dt><dd>{html.escape(str(calibration.get('calibrationId', '')))}</dd></div>
      <div><dt>电镜设备</dt><dd>{html.escape(str(device.get('host', '')))}:{int(device.get('port') or 0)}</dd></div>
      <div><dt>开始时间</dt><dd>{html.escape(str(manifest.get('startedAt', '')))}</dd></div>
      <div><dt>结束时间</dt><dd>{html.escape(str(manifest.get('completedAt', '')))}</dd></div>
    </dl>
    <h2>逐颗粒双图证据</h2>
    <section class="particle-grid">{''.join(cards)}</section>
  </main>
</body>
</html>
"""
    output = task_dir / "report.html"
    output.write_text(report, encoding="utf-8")
    return output


def _register_pdf_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    font_name = "STSong-Light"
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def _scaled_image(path: Path, max_width: float, max_height: float):
    from reportlab.platypus import Image

    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def write_pdf_report(
    task_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    task_dir = Path(task_dir)
    output = task_dir / "report.pdf"
    font_name = _register_pdf_font()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CjkTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=22,
        leading=30,
        textColor=colors.HexColor("#12352d"),
    )
    heading_style = ParagraphStyle(
        "CjkHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=15,
        leading=21,
        textColor=colors.HexColor("#12352d"),
    )
    body_style = ParagraphStyle(
        "CjkBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=15,
    )
    caption_style = ParagraphStyle(
        "CjkCaption",
        parent=body_style,
        alignment=TA_CENTER,
        fontSize=8,
        textColor=colors.HexColor("#56636d"),
    )
    warning_style = ParagraphStyle(
        "CjkWarning",
        parent=body_style,
        textColor=colors.HexColor("#9a6700"),
        backColor=colors.HexColor("#fff4d6"),
        borderPadding=8,
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#68757f"))
        canvas.drawString(18 * mm, 10 * mm, "ZEM Cleanliness · 光镜颗粒 SEM 定位验证")
        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"第 {document.page} 页",
        )
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="光镜颗粒 SEM 自动定位验证报告",
        author="ZEM Cleanliness",
    )
    story: list[Any] = [
        Paragraph("光镜颗粒 SEM 自动定位验证报告", title_style),
        Spacer(1, 5 * mm),
        Paragraph(
            "逐颗并列展示光镜厂商原始颗粒图、电镜实拍颗粒图和完整 SEM "
            "视场定位证据，用于判断自动流程是否找到同一颗粒。",
            body_style,
        ),
        Spacer(1, 4 * mm),
        Paragraph(SIMULATED_ELEMENTS_DISCLAIMER, warning_style),
        Spacer(1, 5 * mm),
    ]
    error = summary["centerErrorUm"]
    summary_rows = [
        ["任务颗粒", "自动找到", "高置信度", "平均中心误差", "最大中心误差"],
        [
            str(summary["total"]),
            str(summary["succeeded"]),
            str(summary["matchedHigh"]),
            f"{_metric(error['mean'])} μm",
            f"{_metric(error['maximum'])} μm",
        ],
    ]
    summary_table = Table(summary_rows, colWidths=[34 * mm] * 5)
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12352d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccd5da")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 6 * mm)])

    calibration = manifest.get("calibration") or {}
    input_info = manifest.get("input") or {}
    audit_rows = [
        ["任务 ID", str(manifest.get("taskId", ""))],
        ["任务状态", str(manifest.get("status", ""))],
        ["光镜源文件", str(input_info.get("path", ""))],
        ["光镜 SHA-256", str(input_info.get("sha256", ""))],
        ["标定版本", str(calibration.get("calibrationId", ""))],
        ["开始 / 结束", f"{manifest.get('startedAt', '')} / {manifest.get('completedAt', '')}"],
    ]
    audit_table = Table(audit_rows, colWidths=[32 * mm, 138 * mm])
    audit_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f3")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8e0e4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([audit_table])

    for record in manifest.get("particles", []):
        story.append(PageBreak())
        sequence = int(record.get("sequence", 0))
        match = record.get("match") or {}
        localization = record.get("localization") or {}
        story.append(
            Paragraph(
                f"光镜颗粒 #{sequence} · {match.get('status', record.get('status', ''))}",
                heading_style,
            )
        )
        story.append(Spacer(1, 3 * mm))
        optical_path = task_dir / str((record.get("images") or {}).get("opticalImage") or "")
        sem_particle_path = task_dir / str((record.get("images") or {}).get("semParticleImage") or "")
        sem_annotated_path = task_dir / str((record.get("images") or {}).get("semAnnotatedImage") or "")
        image_cells: list[Any] = []
        for path, caption in (
            (optical_path, "光镜厂商原始颗粒图"),
            (sem_particle_path, "电镜实拍目标颗粒图"),
        ):
            if path.is_file():
                image_cells.append(
                    [
                        _scaled_image(path, 78 * mm, 58 * mm),
                        Paragraph(caption, caption_style),
                    ]
                )
            else:
                image_cells.append(
                    [Paragraph("图片缺失", body_style), Paragraph(caption, caption_style)]
                )
        image_table = Table(
            [
                [image_cells[0][0], image_cells[1][0]],
                [image_cells[0][1], image_cells[1][1]],
            ],
            colWidths=[85 * mm, 85 * mm],
        )
        image_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8e0e4")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8e0e4")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([image_table, Spacer(1, 4 * mm)])
        if sem_annotated_path.is_file():
            story.extend(
                [
                    _scaled_image(sem_annotated_path, 170 * mm, 80 * mm),
                    Paragraph("完整 SEM 视场与自动定位标记", caption_style),
                    Spacer(1, 3 * mm),
                ]
            )

        target = record.get("targetStageUm") or {}
        actual = record.get("actualStageUm") or {}
        detail_rows = [
            ["Particle ID", str(record.get("particleId", ""))],
            [
                "光镜尺寸",
                f"{float(record.get('lengthUm', 0.0)):.1f} × "
                f"{float(record.get('widthUm', 0.0)):.1f} μm",
            ],
            [
                "预测 / 实际 Stage",
                f"({float(target.get('x', 0.0)):.2f}, {float(target.get('y', 0.0)):.2f}) / "
                f"({float(actual.get('x', 0.0)):.2f}, {float(actual.get('y', 0.0)):.2f}) μm",
            ],
            [
                "颗粒中心定位误差",
                f"{_metric(localization.get('centerErrorUm'))} μm",
            ],
            ["匹配判定", str(match.get("reason") or record.get("error") or "")],
            ["模拟元素", _element_text(record)],
        ]
        detail_table = Table(detail_rows, colWidths=[42 * mm, 128 * mm])
        detail_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f3")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8e0e4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(detail_table)

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def write_csv_results(
    task_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    rows: list[dict[str, Any]] = []
    for record in manifest.get("particles", []):
        target = record.get("targetStageUm") or {}
        actual = record.get("actualStageUm") or {}
        match = record.get("match") or {}
        localization = record.get("localization") or {}
        rows.append(
            {
                "sequence": record.get("sequence"),
                "particleId": record.get("particleId"),
                "status": record.get("status"),
                "matchStatus": match.get("status"),
                "confidence": match.get("confidence"),
                "targetStageXUm": target.get("x"),
                "targetStageYUm": target.get("y"),
                "actualStageXUm": actual.get("x"),
                "actualStageYUm": actual.get("y"),
                "centerErrorUm": localization.get("centerErrorUm"),
                "elementsMode": (record.get("energySpectrum") or {}).get("mode"),
                "elements": _element_text(record),
            }
        )
    output = Path(task_dir) / "results.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(rows[0]) if rows else ["sequence", "status"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_reports(
    task_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    task_dir = Path(task_dir)
    summary = summarize_manifest(manifest)
    manifest["summary"] = summary
    html_path = write_html_report(task_dir, manifest, summary)
    pdf_path = write_pdf_report(task_dir, manifest, summary)
    csv_path = write_csv_results(task_dir, manifest)
    report_payload = {
        "generatedAt": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "html": html_path.name,
        "pdf": pdf_path.name,
        "csv": csv_path.name,
    }
    report_json = task_dir / "report.json"
    report_json.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "html": str(html_path),
        "pdf": str(pdf_path),
        "csv": str(csv_path),
        "json": str(report_json),
    }
