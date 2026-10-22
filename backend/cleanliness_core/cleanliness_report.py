from __future__ import annotations

import csv
import base64
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.cleanliness_core.html_to_pdf_edge import html_to_pdf_via_edge
from backend.cleanliness_core.zeptools_brand import zeptools_logo_data_uri
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.eds_report_details import detail_sections, csv_values, CSV_FIELDS, STYLE as EDS_STYLE, stop_label

# 报告层默认不展示的元素（原始 task.json 能谱仍完整保留）。
# 配置未写 excludeElements 时沿用此默认；显式 [] 表示不排除。
DEFAULT_EXCLUDE_ELEMENTS: tuple[str, ...] = ("O", "C")


def _successful(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in manifest.get("particles") or []
        if record.get("status") == "SUCCEEDED"
    ]


def _exclude_element_set(report_config: dict[str, Any] | None) -> set[str]:
    """解析报告排除元素集合（统一大写比较）。"""
    if report_config is None or "excludeElements" not in report_config:
        return {symbol.upper() for symbol in DEFAULT_EXCLUDE_ELEMENTS}
    raw = report_config.get("excludeElements")
    if not raw:
        return set()
    return {
        str(symbol).strip().upper()
        for symbol in raw
        if str(symbol).strip()
    }


def _element_column_limit(report_config: dict[str, Any] | None) -> int:
    if report_config is None:
        return 5
    try:
        return max(0, int(report_config.get("elementColumns") or 5))
    except (TypeError, ValueError):
        return 5


def _element_map(record: dict[str, Any]) -> dict[str, float]:
    spectrum = record.get("energySpectrum") or {}
    result: dict[str, float] = {}
    for item in spectrum.get("elements") or []:
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        try:
            result[symbol] = float(item.get("value") or 0.0)
        except (TypeError, ValueError):
            continue
    return result


def _normalized_element_map(
    record: dict[str, Any],
    exclude: set[str] | None = None,
) -> dict[str, float]:
    """报告展示用元素含量：剔除排除项后归一化到 100%（保留 3 位小数）。

    仅用于报告/CSV 展示，不改写原始能谱；排除项为空且原始合计已为 100 时数值不变。
    """
    excluded = {symbol.upper() for symbol in (exclude or set())}
    elements = {
        symbol: value
        for symbol, value in _element_map(record).items()
        if symbol.upper() not in excluded
    }
    total = sum(elements.values())
    if total <= 0:
        return elements
    scale = 100.0 / total
    return {symbol: round(value * scale, 3) for symbol, value in elements.items()}


def _element_columns(
    records: list[dict[str, Any]],
    maximum: int = 5,
    exclude: set[str] | None = None,
) -> list[str]:
    """兼容旧测试：按全任务出现次数取列名。HTML/PDF 已改用每颗颗粒自己的占比。"""
    excluded = {symbol.upper() for symbol in (exclude or set())}
    score: dict[str, float] = {}
    frequency: dict[str, int] = {}
    for record in records:
        for symbol, value in _element_map(record).items():
            if symbol.upper() in excluded:
                continue
            score[symbol] = score.get(symbol, 0.0) + value
            frequency[symbol] = frequency.get(symbol, 0) + 1
    return sorted(
        score,
        key=lambda symbol: (frequency[symbol], score[symbol], symbol),
        reverse=True,
    )[: max(0, int(maximum))]


def _particle_element_columns(
    record: dict[str, Any],
    maximum: int = 5,
    exclude: set[str] | None = None,
) -> list[str]:
    """按本颗粒含量从高到低取前 N 个元素（已排除项不参与）。不足 N 个时不补符号。"""
    excluded = {symbol.upper() for symbol in (exclude or set())}
    ranked = sorted(
        (
            (symbol, value)
            for symbol, value in _element_map(record).items()
            if symbol.upper() not in excluded and value > 0
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    return [symbol for symbol, _ in ranked[: max(0, int(maximum))]]


def _html_particle_element_cells(
    record: dict[str, Any],
    maximum: int,
    exclude: set[str],
) -> tuple[str, str]:
    """生成固定 N 列：按归一化占比填入，不足的表头与单元格留空。

    非排除元素超过 N 个时，最后一列并入「其他」，保证展示合计为 100%。
    """
    elements = _normalized_element_map(record, exclude=exclude)
    ranked = sorted(
        ((symbol, value) for symbol, value in elements.items() if value > 0),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    width = max(0, int(maximum))
    if 0 < width < len(ranked):
        cells = ranked[: width - 1]
        remainder = round(sum(value for _, value in ranked[width - 1:]), 3)
        if remainder > 0:
            cells.append(("其他", remainder))
    else:
        cells = ranked[:width]
    headers: list[str] = []
    values: list[str] = []
    for index in range(width):
        if index < len(cells):
            symbol, value = cells[index]
            headers.append(f"<th>{html.escape(symbol)}</th>")
            values.append(f"<td>{_number(value, 3)}</td>")
        else:
            headers.append("<th></th>")
            values.append("<td>-</td>")
    return "".join(headers), "".join(values)


def _csv_element_columns(
    records: list[dict[str, Any]],
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = {symbol.upper() for symbol in (exclude or set())}
    totals: dict[str, float] = {}
    for record in records:
        for symbol, value in _normalized_element_map(record, exclude=excluded).items():
            totals[symbol] = totals.get(symbol, 0.0) + value
    return sorted(totals, key=lambda symbol: -totals[symbol])


def _relative_image(record: dict[str, Any], key: str) -> str:
    value = str((record.get("images") or {}).get(key) or "")
    return html.escape(value.replace("\\", "/"))


def _image_or_placeholder(source: str, title: str) -> str:
    if source:
        return f'<img src="{source}" alt="{html.escape(title)}">'
    return '<div class="missing-image">未生成该图像</div>'


def _report_optical_context(manifest: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], str]:
    """Read the fingerprint-checked original export without changing task records."""
    info = manifest.get("input") or {}
    if not info.get("path"):
        return {}, "未提供原始光镜文件，掩码及原始坐标可能缺失。"
    try:
        export = load_optical_export(Path(info["path"]))
        if info.get("sha256") and export.sha256 != info["sha256"]:
            return {}, "光镜源文件指纹与任务不一致，未补充掩码及原始坐标。"
    except (OSError, ValueError) as exc:
        return {}, f"无法读取原始光镜文件，未补充掩码及原始坐标：{exc}"
    records = {}
    for particle in export.particles:
        mask_url, mask_note = "", "源数据未提供光镜掩码"
        if particle.mask_data_base64:
            try:
                raw = base64.b64decode(particle.mask_data_base64.split(",", 1)[-1], validate=True)
                # Preserve the customer's exact PNG bytes, including small masks.
                decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                if not raw.startswith(b"\x89PNG\r\n\x1a\n") or decoded is None:
                    raise ValueError("invalid PNG")
                mask_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
                mask_note = ""
            except (ValueError, cv2.error):
                mask_note = "原始光镜掩码无法解码"
        records[particle.sequence] = {
            "particleId": particle.particle_id,
            "sourcePositionUm": {"x": particle.source_x_um, "y": particle.source_y_um},
            "abPositionUm": {"x": particle.ab_x_um, "y": particle.ab_y_um},
            "maskUrl": mask_url, "maskNote": mask_note,
        }
    return records, ""


def _coordinate_pair(value: Any, missing: str = "未记录") -> str:
    if not isinstance(value, dict):
        return missing
    try:
        x, y = float(value["x"]), float(value["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            return missing
    except (KeyError, TypeError, ValueError):
        return missing
    return f"X {x:.3f}　Y {y:.3f} μm"


def _particle_coordinates(record: dict[str, Any], optical: dict[str, Any]) -> str:
    matched = record.get("status") == "SUCCEEDED" or (record.get("match") or {}).get("status") in {"MATCHED_HIGH", "MATCHED_MEDIUM"}
    center_sampling = record.get("acquisitionMode") == "large_particle_center_region"
    positions = [
        ("光镜原始台坐标（sourceStage）", _coordinate_pair(record.get("opticalSourcePositionUm") or optical.get("sourcePositionUm"))),
        ("光镜 AB 坐标", _coordinate_pair(record.get("opticalAbPositionUm") or optical.get("abPositionUm"))),
        (("电镜采样点（中心区域）", _coordinate_pair(record.get("samplingStageUm"))) if center_sampling else
         ("电镜颗粒位置（匹配估计）", _coordinate_pair((record.get("localization") or {}).get("observedStageUm"), "未确认") if matched else "未确认")),
        ("电镜预测台坐标", _coordinate_pair(record.get("targetStageUm"))),
        ("电镜采图台坐标（回读）", _coordinate_pair(record.get("actualStageUm"))),
    ]
    return '<div class="particle-coordinates">' + ''.join(
        f'<div><span>{label}</span><strong>{html.escape(value)}</strong></div>' for label, value in positions
    ) + '</div>'


def _failure_stage(record: dict[str, Any]) -> str:
    if record.get("status") != "FAILED":
        return ""
    if record.get("edsAttempts"):
        return "EDS"
    return "中心采样点检查" if (record.get("centerSampling") or {}).get("status") == "REJECTED" else "SEM 定位/采集"


def _spectrum_image(record: dict[str, Any]) -> str:
    spectrum = record.get("energySpectrum") or {}
    image = spectrum.get("spectrumImage") or {}
    value = image.get("path") if isinstance(image, dict) else image
    return html.escape(str(value or "").replace("\\", "/"))


def _material_classification(record: dict[str, Any]) -> dict[str, Any]:
    spectrum = record.get("energySpectrum") or {}
    value = spectrum.get("materialClassification") or {}
    return value if isinstance(value, dict) else {}


def _sem_evidence_image_key(record: dict[str, Any]) -> str:
    """报告优先使用含识别轮廓与 EDS 打点的 SEM 标注图。"""
    images = record.get("images") or {}
    if images.get("semAnnotatedImage"):
        return "semAnnotatedImage"
    return "semParticleImage"


def _sem_report_image(record: dict[str, Any]) -> tuple[str, str]:
    source = _relative_image(record, _sem_evidence_image_key(record))
    if source:
        if record.get("acquisitionMode") == "large_particle_center_region":
            return source, "SEM · 中心区域采谱，未做轮廓匹配"
        return source, "SEM · 识别轮廓 + EDS 打点" if (record.get("images") or {}).get("semAnnotatedImage") else "SEM · 颗粒图"
    for attempt in reversed(record.get("captureAttempts") or []):
        if attempt.get("displayPath"):
            return html.escape(str(attempt["displayPath"]).replace("\\", "/")), "SEM · 采集视场，未确认匹配"
    return "", "SEM"


def _sem_marker_legend() -> list[dict[str, str]]:
    """电镜标注图图例（整份报告只展示一次）。颜色与 annotate_sem_match 一致。"""
    return [
        {
            "color": "#ffdc00",
            "marker": "＋",
            "title": "正十字（黄）",
            "meaning": "视场中心（SEM 图像中心）",
        },
        {
            "color": "#78e650",
            "marker": "＋",
            "title": "正十字（绿）",
            "meaning": "识别到的颗粒中心（匹配成功）",
        },
        {
            "color": "#ff503c",
            "marker": "＋",
            "title": "正十字（红）",
            "meaning": "识别到的颗粒中心（匹配未通过）",
        },
        {
            "color": "#dc50ff",
            "marker": "×",
            "title": "斜十字（品红）",
            "meaning": "EDS 电子束打点位置",
        },
    ]


def _html_sem_marker_legend() -> str:
    items = "".join(
        f"""<div class="legend-item">
          <div class="legend-mark" style="color:{item['color']}">{item['marker']}</div>
          <div class="legend-text"><strong>{html.escape(item['title'])}</strong>
          <span>{html.escape(item['meaning'])}</span></div>
        </div>"""
        for item in _sem_marker_legend()
    )
    return f"""<p class="legend-title">电镜标注图图例（整份报告通用）</p>
<div class="legend">{items}</div>"""


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _compact_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "-"


def _filter_summary(filter_info: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for label, minimum_key, maximum_key, unit in (
        ("长度", "minLengthUm", "maxLengthUm", "μm"),
        ("宽度", "minWidthUm", "maxWidthUm", "μm"),
        ("面积", "minAreaUm2", "maxAreaUm2", "μm²"),
        ("等效圆直径", "minEcdUm", "maxEcdUm", "μm"),
    ):
        minimum = filter_info.get(minimum_key)
        maximum = filter_info.get(maximum_key)
        if minimum is not None and maximum is not None:
            items.append(
                f"{label} {_compact_number(minimum)}–{_compact_number(maximum)} {unit}"
            )
        elif minimum is not None:
            items.append(f"{label} ≥ {_compact_number(minimum)} {unit}")
        elif maximum is not None:
            items.append(f"{label} ≤ {_compact_number(maximum)} {unit}")
    labels = [str(value).strip() for value in filter_info.get("labels") or [] if str(value).strip()]
    if labels:
        items.append(f"分类：{'、'.join(labels)}")
    maximum_particles = filter_info.get("maxParticles")
    if maximum_particles is not None:
        items.append(f"最多执行 {int(maximum_particles)} 个颗粒")
    return items or ["无额外筛选，执行全部有效颗粒"]


def summarize_cleanliness(manifest: dict[str, Any]) -> dict[str, Any]:
    particles = manifest.get("particles") or []
    succeeded = _successful(manifest)
    errors = [
        float(record.get("localization", {}).get("centerErrorUm"))
        for record in succeeded
        if record.get("localization", {}).get("centerErrorUm") is not None
    ]
    selected = int((manifest.get("selection") or {}).get("count") or 0)
    failed = sum(record.get("status") == "FAILED" for record in particles)
    processed = len(succeeded) + failed
    eds_failed = sum(_failure_stage(record) == "EDS" for record in particles)
    return {
        "selected": selected,
        "processed": processed,
        "succeeded": len(succeeded),
        "failed": failed,
        "localizationFailed": failed - eds_failed,
        "edsFailed": eds_failed,
        "pending": max(0, selected - processed),
        "meanCenterErrorUm": sum(errors) / len(errors) if errors else None,
        "maxCenterErrorUm": max(errors) if errors else None,
    }


def _mock_eds_notice(manifest: dict[str, Any]) -> str | None:
    acceptance = manifest.get("acceptanceTest") or {}
    spectra = [
        record.get("energySpectrum") or {}
        for record in manifest.get("particles") or []
    ]
    if not acceptance.get("fakeEds") and not any(spectrum.get("mock") for spectrum in spectra):
        return None
    return str(
        acceptance.get("notice")
        or next((spectrum.get("notice") for spectrum in spectra if spectrum.get("notice")), "")
        or "模拟能谱，仅用于软件真机验收，不代表真实材料成分"
    )


def _selected_calibration_group(manifest: dict[str, Any]) -> dict[str, Any]:
    calibration = manifest.get("calibration") or {}
    if calibration.get("result"):
        return calibration
    groups = calibration.get("groups") or []
    selected = calibration.get("selectedGroup")
    if selected is None:
        selected = (calibration.get("review") or {}).get("selectedGroup")
    for group in groups:
        if group.get("groupIndex") == selected:
            return group
    if isinstance(selected, int) and 0 <= selected < len(groups):
        return groups[selected]
    return groups[0] if groups else calibration


def write_cleanliness_html(
    task_dir: Path,
    manifest: dict[str, Any],
    report_config: dict[str, Any] | None = None,
    *,
    html_name: str = "cleanliness_report.html",
) -> Path:
    records = manifest.get("particles") or []
    summary = summarize_cleanliness(manifest)
    optical_context, optical_warning = _report_optical_context(manifest)
    show_coordinates = (report_config or {}).get("showCoordinates", True) is not False
    cleanup = manifest.get("cleanup") or {}
    shutdown_note = ("电子枪已关闭（设备反馈确认）" if cleanup.get("gunOffVerified") else
                     "自动关枪未确认：" + str(cleanup.get("gunOffError") or "请人工检查电子枪状态")) if cleanup.get("gunOffRequested") else ""
    column_limit = _element_column_limit(report_config)
    exclude = _exclude_element_set(report_config)
    calibration_group = _selected_calibration_group(manifest)
    calibration = calibration_group.get("result") or {}
    equation = calibration.get("equation") or {}
    validation = calibration.get("validation") or {}
    filter_info = (manifest.get("selection") or {}).get("filter") or {}
    filter_summary = _filter_summary(filter_info)
    selection = manifest.get("selection") or {}
    if selection.get("sequences"):
        filter_summary = [
            f"按指定序号选择 {len(selection['sequences'])} 个颗粒"
        ] + ([item for item in filter_summary if item != "无额外筛选，执行全部有效颗粒"])
    reuse = manifest.get("calibrationReuse") or {}
    calibration_title = "沿用已确认标定" if reuse else "本次标定"
    calibration_source = (
        f'<p class="element-note">来源任务：{html.escape(str(reuse.get("sourceTaskId") or "-"))}；以下为来源标定与验证结果。</p>'
        if reuse else ""
    )
    mock_eds_notice = _mock_eds_notice(manifest)
    excluded_label = "、".join(sorted(exclude))
    element_note = (
        f"元素含量（wt%）已剔除 {html.escape(excluded_label)} 后归一化至 100%；"
        if excluded_label
        else "元素含量（wt%）已归一化至 100%；"
    ) + f"每颗粒最多显示 {column_limit} 列，超出部分并入「其他」，完整数据见 CSV 与 task.json。"
    mock_eds_banner = (
        f'<div class="mock-eds-warning"><strong>验收模式 · 模拟能谱</strong>'
        f'<span>{html.escape(mock_eds_notice)}</span></div>'
        if mock_eds_notice
        else ""
    )

    particle_sections = []
    for rank, record in enumerate(records, start=1):
        optical = optical_context.get(int(record.get("sequence") or 0), {})
        if optical and record.get("particleId") and optical.get("particleId") != record["particleId"]:
            optical = {"maskNote": "颗粒 ID 与光镜源数据不一致"}
        mask_image = (f'<img src="{optical["maskUrl"]}" alt="原始光镜掩码">' if optical.get("maskUrl") else
                      f'<div class="missing-image">{html.escape(optical.get("maskNote") or "原始光镜掩码不可用")}</div>')
        status = str(record.get("status") or "UNKNOWN")
        status_label = {"SUCCEEDED": "成功", "FAILED": "失败", "RUNNING": "处理中", "STOPPED": "已停止"}.get(status, status)
        failure_stage = _failure_stage(record)
        outcome_html = (
            f'<div class="particle-outcome">检测状态：{html.escape(status_label)}'
            + (' · 方式：大颗粒中心区域采谱（跳过轮廓匹配）' if record.get('acquisitionMode')=='large_particle_center_region' else '')
            + (f' · 失败阶段：{html.escape(failure_stage)}' if failure_stage else '')
            + (f' · 原因：{html.escape(str(record["error"]))}' if record.get("error") else '')
            + '</div>'
        )
        element_headers, element_values = _html_particle_element_cells(
            record,
            maximum=column_limit,
            exclude=exclude,
        )
        spectrum_image = _spectrum_image(record)
        sem_source, sem_caption = _sem_report_image(record)
        material = _material_classification(record)
        hardness_label = str(material.get("hardnessLabel") or "").strip()
        material_class = str(material.get("materialClass") or "").strip()
        parent_class = str(material.get("parentClass") or "").strip()
        confidence = material.get("confidence")
        heading_hardness = (
            f" · Oxford 硬度分类 {html.escape(hardness_label)}" if hardness_label else ""
        )
        spectrum_figure = (
            f'<figure><figcaption>能谱谱图 <span>EDS · SPECTRUM</span></figcaption>'
            f'<img src="{spectrum_image}" alt="能谱谱图"></figure>'
            if spectrum_image
            else ""
        )
        classification_items = []
        if material_class:
            classification_items.append(
                f"<div><span>Oxford 材料分类</span><strong>{html.escape(material_class)}</strong></div>"
            )
        if parent_class:
            classification_items.append(
                f"<div><span>母类</span><strong>{html.escape(parent_class)}</strong></div>"
            )
        if hardness_label:
            classification_items.append(
                f'<div class="hardness"><span>HV 硬度分类</span><strong>{html.escape(hardness_label)}</strong></div>'
            )
        if confidence is not None:
            try:
                confidence_text = f"{float(confidence) * 100:.1f}%"
            except (TypeError, ValueError):
                confidence_text = str(confidence)
            classification_items.append(
                f"<div><span>分类置信度</span><strong>{html.escape(confidence_text)}</strong></div>"
            )
        classification_html = (
            f'<div class="eds-classification">{"".join(classification_items)}</div>'
            if classification_items
            else ""
        )
        spectrum = record.get('energySpectrum') or {}
        if spectrum:
            vda = spectrum.get('vdaClassification') or {}
            classification_html += '<div class="particle-meta">VDA 分类：'+html.escape(str(vda.get('materialClass') or '未提供'))+' / '+html.escape(str(vda.get('hardnessLabel') or '未提供'))+'；'+html.escape(stop_label(spectrum))+'。上方元素列为客户端展示值，不代表 VDA 分类成分；完整口径与警告见后续明细。</div>'
        image_count_class = "images-4" if spectrum_image else "images-3"
        particle_sections.append(
            f"""
            <section class="particle">
              <div class="particle-heading">
                <div><span>PARTICLE</span><strong>颗粒 {rank:02d}</strong></div>
                <div class="particle-heading-meta">ID {html.escape(str(record.get('particleId') or record.get('sequence')))} · {html.escape(str(record.get('label') or '未分类'))}{heading_hardness}</div>
              </div>
              {outcome_html}
              <table class="particle-table">
                <thead><tr>
                  <th>Rank</th><th>ID</th><th>Size (μm)</th><th>Width (μm)</th>
                  <th>Class</th><th>Aspect</th><th>Area (μm²)</th>
                  <th>Roundness</th><th>ECD (μm)</th>{element_headers}
                </tr></thead>
                <tbody><tr>
                  <td>{rank}</td><td>{html.escape(str(record.get('particleId') or record.get('sequence')))}</td>
                  <td>{_number(record.get('lengthUm'))}</td><td>{_number(record.get('widthUm'))}</td>
                  <td>{html.escape(str(record.get('label') or '-'))}</td>
                  <td>{_number(record.get('aspect'))}</td><td>{_number(record.get('areaUm2'))}</td>
                  <td>{_number(record.get('roundness'))}</td>
                  <td>{_number(record.get('equivalentDiameterUm'))}</td>{element_values}
                </tr></tbody>
              </table>
              {_particle_coordinates(record, optical) if show_coordinates else ''}
              <div class="images {image_count_class}">
                <figure><figcaption>光镜颗粒 <span>OPTICAL</span></figcaption>{_image_or_placeholder(_relative_image(record, 'opticalImage'), '光镜颗粒')}</figure>
                <figure><figcaption>光镜掩码 <span>原始 MASK</span></figcaption>{mask_image}</figure>
                <figure><figcaption>电镜图像 <span>{sem_caption}</span></figcaption>{_image_or_placeholder(sem_source, '电镜图像')}</figure>
                {spectrum_figure}
              </div>
              {classification_html}
              <div class="particle-meta">sequence #{int(record.get('sequence') or 0)} · 定位误差 {_number((record.get('localization') or {}).get('centerErrorUm'), 1)} μm · EDS {html.escape(str((record.get('energySpectrum') or {}).get('device') or '-'))}</div>
            </section>"""
        )

        for title, body in detail_sections(spectrum):
            particle_sections.append(f'<section class="eds-detail-page"><h2>颗粒 ID {html.escape(str(record.get("particleId") or record.get("sequence")))} · {html.escape(title)}</h2>{body}</section>')

    calibration_pairs = []
    for kind, title in (("anchors", "标定集"), ("validation", "独立验证集")):
        for record in calibration_group.get(kind) or []:
            images = record.get("images") or {}
            calibration_pairs.append(
                f"""
                <div class="cal-pair">
                  <strong>{title} #{int(record.get('sequence') or 0)}</strong>
                  <div><img src="{html.escape(str(images.get('opticalImage') or ''))}"><img src="{html.escape(str(images.get('semParticleImage') or ''))}"></div>
                </div>"""
            )

    generated_at = datetime.now().astimezone()
    generated = generated_at.isoformat(timespec="seconds")
    generated_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    logo_uri = zeptools_logo_data_uri()
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>清洁度检测报告</title>
<style>
:root{{--navy:#12384f;--blue:#1675b8;--red:#e30613;--line:#d7e1e7;--soft:#f4f7f9;--ink:#17242c;--muted:#6a7983}}
*{{box-sizing:border-box}} body{{margin:0;background:#e9eff2;color:var(--ink);font:13px/1.45 Arial,"Microsoft YaHei",sans-serif}}
.page{{max-width:1180px;margin:22px auto;background:white;padding:28px 34px 34px;box-shadow:0 12px 40px #1837461c}}
.report-header{{display:flex;align-items:center;padding-bottom:14px;border-bottom:1px solid var(--line)}} .brand-logo{{width:205px;height:auto}}
.header-copy{{margin-left:auto;text-align:right;color:var(--muted);font-size:11px}} .header-copy strong{{display:block;color:var(--navy);font-size:12px;letter-spacing:.08em}}
.title-block{{padding:25px 0 18px}} .eyebrow{{color:var(--red);font-size:11px;font-weight:700;letter-spacing:.16em}}
h1{{margin:5px 0 3px;color:var(--navy);font-size:30px;line-height:1.2}} h2{{display:flex;align-items:center;gap:10px;margin:25px 0 11px;color:var(--navy);font-size:15px}}
h2::before{{content:"";width:4px;height:16px;background:var(--red);border-radius:2px}} .sub{{color:var(--muted);margin:4px 0 16px}}
.summary{{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);border-top:3px solid var(--navy)}}
.metric{{background:white;padding:12px 14px;border-right:1px solid var(--line)}} .metric:last-child{{border-right:0}} .metric span{{display:block;color:var(--muted);font-size:11px}} .metric strong{{display:block;margin-top:3px;color:var(--navy);font-size:21px}}
.equation{{font-family:Consolas,monospace;background:#f5f7f8;border:1px solid var(--line);border-left:3px solid var(--blue);color:#2a3d49;padding:11px 13px;white-space:pre-wrap}}
.cal-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .cal-pair{{border:1px solid var(--line);padding:8px;background:#fff;break-inside:avoid}}
.cal-pair>div{{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:6px}} .cal-pair img{{width:100%;height:120px;object-fit:contain;background:#101517}}
.particle{{margin:20px 0;border:1px solid var(--line);break-inside:avoid;background:white;box-shadow:0 4px 16px #17394b0b}}
.particle-heading{{display:flex;align-items:center;justify-content:space-between;padding:10px 13px;background:var(--navy);color:white}} .particle-heading span{{display:block;color:#a9cada;font-size:9px;letter-spacing:.16em}} .particle-heading strong{{font-size:15px}} .particle-heading-meta{{font-size:11px;color:#dceaf1}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid var(--line);padding:6px;text-align:center}} th{{background:#edf3f6;color:#324b59;font-weight:600;font-size:10px}} td{{font-size:11px}}
.images{{display:grid;gap:1px;background:var(--line)}} .images-2{{grid-template-columns:1fr 1fr}} .images-3{{grid-template-columns:repeat(3,1fr)}} .images-4{{grid-template-columns:repeat(4,1fr)}}
.particle-coordinates{{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--line);background:#fff}}
.particle-coordinates>div{{padding:5px 8px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}
.particle-coordinates span{{display:block;color:var(--muted);font-size:9px}} .particle-coordinates strong{{display:block;color:var(--navy);font:11px/1.6 Consolas,"Microsoft YaHei",monospace}}
figure{{margin:0;min-width:0;background:#f7f9fa}} figure img{{display:block;width:100%;height:250px;padding:10px;object-fit:contain;background:#f7f9fa}}
figcaption{{background:white;color:var(--navy);padding:8px 11px;font-weight:600;border-bottom:1px solid var(--line)}} figcaption span{{float:right;color:var(--muted);font-size:9px;font-weight:400;letter-spacing:.05em}} .particle-meta{{padding:8px 11px;color:var(--muted);background:#f7f9fa;border-top:1px solid var(--line)}}
.images-3 figure img,.images-4 figure img{{height:220px}} .eds-classification{{display:flex;flex-wrap:wrap;border-top:1px solid var(--line);background:#fff}}
.eds-classification>div{{min-width:150px;flex:1;padding:8px 11px;border-right:1px solid var(--line)}} .eds-classification>div:last-child{{border-right:0}}
.eds-classification span{{display:block;color:var(--muted);font-size:9px}} .eds-classification strong{{display:block;margin-top:2px;color:var(--navy);font-size:12px}}
.eds-classification .hardness{{background:#eef6fb}} .eds-classification .hardness strong{{color:var(--blue);font-size:15px}}
.audit{{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line);border-top:0}} .audit div{{padding:9px 11px}} .audit div+div{{border-left:1px solid var(--line)}}
.filters{{display:flex;flex-wrap:wrap;gap:8px}} .filter-item{{border:1px solid #c9dbe5;background:#f4f8fa;color:#294b5d;padding:7px 11px;border-radius:16px}}
.mock-eds-warning{{display:flex;align-items:center;gap:16px;margin:0 0 16px;padding:12px 14px;border:1px solid #e5ad36;border-left:5px solid #e58b18;background:#fff8e8;color:#6b4700}}
.mock-eds-warning strong{{white-space:nowrap;font-size:14px}} .mock-eds-warning span{{font-size:12px}}
.legend-title{{margin:2px 0 3px;color:var(--muted);font-size:9px}}
.element-note{{margin:4px 0 10px;color:var(--muted);font-size:10px}}
.particle-outcome{{padding:6px 11px;background:#f4f8fa;overflow-wrap:anywhere}}
.missing-image{{height:175px;display:flex;align-items:center;justify-content:center;color:var(--muted);background:#f7f9fa}}
.legend{{display:grid;grid-template-columns:max-content;gap:2px;margin:0 0 10px;padding:5px 7px;border:1px solid var(--line);background:#f7f9fa;width:fit-content}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:9px;line-height:1.2;white-space:nowrap}}
.legend-mark{{width:15px;height:15px;flex:0 0 15px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;line-height:1;background:#fff;border:1px solid var(--blue);border-radius:2px}}
.legend-text strong{{display:inline;color:#243038;font-size:9px;margin-right:3px}} .legend-text span{{color:#65727a;font-size:9px}}
@page{{size:A4 landscape;margin:8mm}}
@media print{{
  html,body{{-webkit-print-color-adjust:exact;print-color-adjust:exact;background:white !important}}
  .page{{box-shadow:none !important;margin:0;max-width:none;padding:0;background:white}}
  .report-header{{padding-bottom:6px}} .brand-logo{{width:150px}}
  .title-block{{padding:8px 0 6px}} h1{{font-size:22px;margin:2px 0}} .sub{{margin:2px 0 8px}}
  h2{{margin:10px 0 6px;font-size:13px}} h2::before{{height:12px}}
  .metric{{padding:6px 8px}} .metric strong{{font-size:16px}}
  .equation{{padding:6px 8px;font-size:11px}} .audit div{{padding:5px 8px}}
  .filters{{gap:4px}} .filter-item{{padding:4px 8px}}
  .cal-grid{{gap:6px;break-inside:avoid;page-break-inside:avoid}}
  .cal-pair{{padding:4px;break-inside:avoid;page-break-inside:avoid;box-shadow:none !important}}
  .cal-pair img{{height:88px !important;background:#101517 !important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  .particle{{break-after:page;break-inside:avoid;box-shadow:none !important;background:white}}
  .images-3 figure img,.images-4 figure img{{height:175px !important}} .eds-classification>div{{padding:5px 8px}}
  .filter-item,.legend,.equation,.metric,figure{{box-shadow:none !important}}
  .summary,.audit,.filters,.cal-grid,.legend{{background:white}}
  .particle-meta,figure,figure img{{background:#f7f9fa}}
}}
{EDS_STYLE}
</style></head><body><main class="page">
<header class="report-header"><img class="brand-logo" src="{logo_uri}" alt="ZEPTOOLS"><div class="header-copy"><strong>CLEANLINESS INSPECTION REPORT</strong>www.zeptools.cn</div></header>
<div class="title-block"><div class="eyebrow">INSPECTION REPORT</div><h1>清洁度检测报告</h1><p class="sub">报告生成：{generated_display}　·　任务编号：{html.escape(str(manifest.get('taskId') or '-'))}　·　状态：{html.escape(str(manifest.get('status') or '-'))}</p></div>
{mock_eds_banner}
{f'<p class="gun-shutdown">{html.escape(shutdown_note)}</p>' if shutdown_note else ''}
<div class="summary">
  <div class="metric"><span>光镜总颗粒</span><strong>{int((manifest.get('input') or {}).get('particleCount') or 0)}</strong></div>
  <div class="metric"><span>筛选任务颗粒</span><strong>{summary['selected']}</strong></div>
  <div class="metric"><span>成功</span><strong>{summary['succeeded']}</strong></div>
  <div class="metric"><span>失败</span><strong>{summary['failed']}</strong></div>
  <div class="metric"><span>平均定位误差</span><strong>{_number(summary['meanCenterErrorUm'], 1)} μm</strong></div>
</div>
<h2>任务筛选</h2><div class="filters">{''.join(f'<div class="filter-item">{html.escape(item)}</div>' for item in filter_summary)}</div>
<p class="element-note">已完成 {summary['processed']} / {summary['selected']}；SEM 定位/采集失败 {summary['localizationFailed']}；EDS 失败 {summary['edsFailed']}；未完成 {summary['pending']}。明细保留全部已有处理记录。</p>
<h2>{calibration_title}</h2>{calibration_source}
<div class="equation">{html.escape(str(equation.get('stageX') or ''))}\n{html.escape(str(equation.get('stageY') or ''))}</div>
<div class="audit"><div>标定 RMS：{_number((calibration.get('fit') or {}).get('rmsErrorUm'), 2)} μm</div><div>验证平均/最大误差：{_number(validation.get('meanErrorUm'), 2)} / {_number(validation.get('maxErrorUm'), 2)} μm</div></div>
<div class="cal-grid">{''.join(calibration_pairs)}</div>
<h2>颗粒明细</h2>
{_html_sem_marker_legend()}
<p class="element-note">{element_note}</p>
<p class="element-note">长度、宽度、面积与 ECD 来自光镜导出；Aspect、Roundness 来自 SEM 匹配轮廓。定位误差为识别位置相对预测位置的偏移量。</p>
<p class="element-note">{'坐标均为 μm：光镜原始台坐标与 AB 坐标系分别列示。电镜预测台坐标、采图回读和匹配估计位置含义不同；未匹配的颗粒位置标为「未确认」。' if show_coordinates else ''}掩码为光镜源文件原始 PNG。</p>
{f'<p class="element-note">{html.escape(optical_warning)}</p>' if optical_warning else ''}
{''.join(particle_sections) if particle_sections else '<p>暂无颗粒处理记录。</p>'}
</main></body></html>"""
    output = Path(task_dir) / html_name
    output.write_text(document, encoding="utf-8")
    return output


def write_cleanliness_pdf(
    task_dir: Path,
    manifest: dict[str, Any],
    report_config: dict[str, Any] | None = None,
    *,
    html_name: str = "cleanliness_report.html",
    pdf_name: str = "cleanliness_report.pdf",
    timeout_s: float = 180.0,
) -> Path:
    """将清洁度报告 HTML 经 Edge headless 打印为 PDF（HTML 为唯一版式来源）。"""
    task_dir = Path(task_dir)
    html_path = write_cleanliness_html(
        task_dir,
        manifest,
        report_config=report_config,
        html_name=html_name,
    )
    pdf_path = task_dir / pdf_name
    return html_to_pdf_via_edge(html_path, pdf_path, timeout_s=timeout_s)


def write_cleanliness_csv(
    task_dir: Path,
    manifest: dict[str, Any],
    report_config: dict[str, Any] | None = None,
) -> Path:
    records = manifest.get("particles") or []
    exclude = _exclude_element_set(report_config)
    element_columns = _csv_element_columns(records, exclude=exclude)
    output = Path(task_dir) / "cleanliness_results.csv"
    fieldnames = [
        "rank",
        "sequence",
        "particleId",
        "status",
        "failureStage",
        "error",
        "class",
        "lengthUm",
        "widthUm",
        "areaUm2",
        "roundness",
        "aspect",
        "equivalentDiameterUm",
        "centerErrorUm",
        "edsDevice",
        "edsMock",
        "edsNotice",
        *element_columns,
        *CSV_FIELDS,
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            row = {
                "rank": index,
                "sequence": record.get("sequence"),
                "particleId": record.get("particleId"),
                "status": record.get("status"),
                "failureStage": _failure_stage(record),
                "error": record.get("error"),
                "class": record.get("label"),
                "lengthUm": record.get("lengthUm"),
                "widthUm": record.get("widthUm"),
                "areaUm2": record.get("areaUm2"),
                "roundness": record.get("roundness"),
                "aspect": record.get("aspect"),
                "equivalentDiameterUm": record.get("equivalentDiameterUm"),
                "centerErrorUm": (record.get("localization") or {}).get(
                    "centerErrorUm"
                ),
                "edsDevice": (record.get("energySpectrum") or {}).get("device"),
                "edsMock": bool((record.get("energySpectrum") or {}).get("mock")),
                "edsNotice": (record.get("energySpectrum") or {}).get("notice"),
            }
            row.update(_normalized_element_map(record, exclude=exclude))
            row.update(csv_values(record.get('energySpectrum') or {}))
            writer.writerow(row)
    return output


def build_cleanliness_reports(
    task_dir: Path,
    manifest: dict[str, Any],
    report_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    task_dir = Path(task_dir)
    html_path = write_cleanliness_html(task_dir, manifest, report_config=report_config)
    pdf_path = write_cleanliness_pdf(task_dir, manifest, report_config=report_config)
    csv_path = write_cleanliness_csv(task_dir, manifest, report_config=report_config)
    report_json = task_dir / "cleanliness_report.json"
    report_json.write_text(
        json.dumps(
            {
                "generatedAt": datetime.now().astimezone().isoformat(),
                "summary": summarize_cleanliness(manifest),
                "acceptanceTest": manifest.get("acceptanceTest"),
                "mockEdsNotice": _mock_eds_notice(manifest),
                "html": html_path.name,
                "pdf": pdf_path.name,
                "csv": csv_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "html": html_path,
        "pdf": pdf_path,
        "csv": csv_path,
        "json": report_json,
    }
