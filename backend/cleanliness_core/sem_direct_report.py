"""Formal direct-SEM report, using the established brand and EDS table rules."""
from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .cleanliness_report import _exclude_element_set, _element_column_limit, _html_particle_element_cells
from .html_to_pdf_edge import html_to_pdf_via_edge
from .optical_particle_analysis import normalize_uint8
from .zeptools_brand import zeptools_logo_data_uri
from .eds_report_details import detail_sections, csv_values, CSV_FIELDS, STYLE as EDS_STYLE, stop_label

STATUS = {'COMPLETED': '已完成', 'COMPLETED_WITH_FAILURES': '部分完成', 'FAILED': '异常中断',
          'STOPPED': '已暂停', 'RUNNING': '分析中', 'DRAFT': '草稿', 'SUCCEEDED': '采谱成功',
          'MEASURED': '形貌完成', 'REJECTED': '已跳过', 'ACQUIRING': '采谱中', 'PENDING': '待分析'}


def esc(value):
    return html.escape(str(value if value is not None else '—'))


def number(value, digits=2):
    return f'{float(value):.{digits}f}' if value is not None and math.isfinite(float(value)) else '—'


def write_image(path, image):
    parameters = [cv2.IMWRITE_JPEG_QUALITY,92] if path.suffix.lower() in {'.jpg','.jpeg'} else []
    if not cv2.imwrite(str(path), image, parameters):
        raise OSError(f'报告图片保存失败：{path}')
    return 'report_assets/'+path.name


FIELD_DETECTED_COLOR = (0, 165, 255)  # BGR, orange
FIELD_SUCCESS_COLOR = (70, 210, 80)  # BGR, green
FIELD_CURRENT_COLOR = (255, 220, 0)  # BGR, cyan


def annotate_field(gray, records, current_sequence):
    """Report-only overlay: current > successful > other detected contours.

    Records must belong to this capture; never reuse another frame's pixels.
    Neither image data nor detection/status records are modified.
    """
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape
    ordered = sorted(records, key=lambda p: p['sequence'] == current_sequence)
    for record in ordered:
        points = np.asarray(record.get('contour', []), np.float32)
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3
                or not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] >= width).any() or (points[:, 1] >= height).any()):
            continue
        current = record['sequence'] == current_sequence
        color = (FIELD_CURRENT_COLOR if current else FIELD_SUCCESS_COLOR
                 if record['status'] in {'SUCCEEDED', 'MEASURED'} else FIELD_DETECTED_COLOR)
        thickness = max(2, round(width/320)) + (1 if current else 0)
        contour = points.astype(np.int32)
        cv2.polylines(image, [contour], True, color, thickness)
        x, y, w, h = cv2.boundingRect(contour)
        if current:
            pad = max(5, round(width/100))
            cv2.rectangle(image, (max(0,x-pad),max(0,y-pad)),
                          (min(width-1,x+w+pad),min(height-1,y+h+pad)), color, max(2,thickness-1))
        label = f'ID {record["sequence"]}' if current else f'#{record["sequence"]}'
        scale = max(.5, width/1100)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        tx = min(max(0,x), max(0,width-tw-6))
        ty = y-max(8,round(width/70))
        if ty < th+5:
            ty = min(height-baseline-5, y+h+th+10)
        cv2.rectangle(image, (tx,ty-th-4), (tx+tw+6,ty+baseline+3), (15,22,28), -1)
        cv2.putText(image,label,(tx+3,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
    return image


def particle_images(directory, record, tile, cache, field_records):
    raw = tile.get('raw')
    if not raw or not (directory/raw).is_file():
        return None
    if raw not in cache:
        image = cv2.imread(str(directory/raw), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'无法读取 SEM 原图：{raw}')
        cache[raw] = normalize_uint8(image)
    gray = cache[raw]
    h, w = gray.shape
    points = np.asarray(record.get('contour', []), np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
        return None
    context = annotate_field(gray, field_records, record['sequence'])
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contour = points.astype(np.int32)
    color = (68, 180, 55)
    cv2.polylines(image, [contour], True, color, max(1, w//512))
    point = record.get('safePoint')
    if point:
        cv2.drawMarker(image, tuple(map(int, point)), (20, 30, 235), cv2.MARKER_CROSS, max(10, w//70), 2)
    x, y, cw, ch = cv2.boundingRect(contour)
    pad = max(25, round(max(cw, ch)*.4))
    left, top, right, bottom = max(0,x-pad), max(0,y-pad), min(w,x+cw+pad), min(h,y+ch+pad)
    detail = image[top:bottom, left:right].copy()
    # Reserve a footer so scale bars never obscure particle pixels.
    scale = float(tile.get('capture', {}).get('pixelSizeUm', 0))
    if scale > 0 and detail.size:
        detail = cv2.copyMakeBorder(detail, 0, 28, 0, 0, cv2.BORDER_CONSTANT, value=(245,247,249))
        length_px = max(1, round((right-left)*.35))
        cy = detail.shape[0]-10
        cv2.line(detail, (8,cy), (8+length_px,cy), (25,45,60), 2)
        cv2.putText(detail, f'{length_px*scale:.1f} um', (8,cy-5), cv2.FONT_HERSHEY_SIMPLEX, .35, (25,45,60), 1, cv2.LINE_AA)
    sequence = record['sequence']
    assets = directory/'report_assets'
    return (write_image(assets/f'p{sequence:05d}_context.jpg', context),
            write_image(assets/f'p{sequence:05d}_detail.png', detail))


def navigation_image(directory, manifest):
    image = cv2.imread(str(directory/(manifest.get('navigation', {}).get('path') or 'navigation.png')))
    if image is None:
        return None
    bounds = manifest['navigation']['boundsUm']
    h, w = image.shape[:2]
    def point(x, y):
        return round((bounds[2]-x)/(bounds[2]-bounds[0])*w), round((y-bounds[1])/(bounds[3]-bounds[1])*h)
    roi = manifest['plan']['roi']
    a, b = point(roi['x1'], roi['y0']), point(roi['x0'], roi['y1'])
    field = manifest['plan']['fieldUm']
    for tile in manifest['plan']['tiles']:
        color = (50,180,60) if tile['status']=='COMPLETED' else (0,165,230)
        cv2.rectangle(image, point(tile['x']+field/2,tile['y']-field/2), point(tile['x']-field/2,tile['y']+field/2), color, 2)
    if roi['shape'] == 'circle':
        cv2.ellipse(image, ((a[0]+b[0])//2,(a[1]+b[1])//2), ((b[0]-a[0])//2,(b[1]-a[1])//2), 0, 0, 360, (20,20,230), 3)
    else:
        cv2.rectangle(image, a, b, (20,20,230), 3)
    cv2.putText(image, 'ROI', (max(0,a[0]-5),max(18,a[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, .8, (20,20,230), 2, cv2.LINE_AA)
    return write_image(directory/'report_assets'/'navigation.jpg', image)


def field_overview(directory, tile, records, header, cache):
    raw = tile.get('raw')
    source = None
    if raw and (directory/raw).is_file():
        if raw not in cache:
            image = cv2.imread(str(directory/raw), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'无法读取 SEM 原图：{raw}')
            cache[raw] = normalize_uint8(image)
        source = write_image(directory/'report_assets'/f'field{tile["id"]:05d}.jpg', annotate_field(cache[raw], records, None))
    successful = sum(p['status'] in {'SUCCEEDED','MEASURED'} for p in records)
    rows = ''.join(f'<tr><td>{p["sequence"]}</td><td>{number(p.get("lengthUm"))}</td><td>{esc(STATUS.get(p["status"],p["status"]))}</td></tr>' for p in records[:12])
    more = '<p class="note">概览列出前 12 个候选，全部已分析颗粒见后续详情，跳过记录见附录。</p>' if len(records)>12 else ''
    picture = f'<img src="{source}" alt="视野 {tile["id"]} 全部识别颗粒">' if source else '<div class="placeholder">该视野尚未保存 SEM 原图</div>'
    return header+f'''<div class="eyebrow">SEM FIELD {tile['id']:02d}</div><h1>视野 {tile['id']:02d} · SEM 原视野 / 目标轮廓</h1>
<p class="sub">识别候选 {len(records)} 个　·　成功分析 {successful} 颗　·　跳过 / 未完成 {len(records)-successful} 个</p>
<div class="field-overview"><figure><figcaption>本视野全部识别颗粒 · 标号为颗粒 ID</figcaption>{picture}</figure><div><h2>本视野颗粒概览</h2><table><tr><th>颗粒 ID</th><th>长度（μm）</th><th>分析状态</th></tr>{rows}</table>{more}<p class="note"><b style="color:#b56a00">■ 橙色：已识别、未完成分析</b><br><b style="color:#258038">■ 绿色：已成功分析</b></p><p class="note">接下来的颗粒详情均属于本视野。详情页中，当前展示颗粒以青色轮廓、外框和 ID 突出显示。</p></div></div>'''


STYLE = '''
:root{--navy:#12384f;--blue:#1675b8;--red:#e30613;--line:#d7e1e7;--soft:#f4f7f9;--ink:#17242c;--muted:#6a7983}
*{box-sizing:border-box}body{margin:0;background:#e9eff2;color:var(--ink);font:13px/1.45 Arial,"Microsoft YaHei",sans-serif}
.sheet{width:281mm;min-height:194mm;margin:22px auto;background:#fff;padding:12mm;position:relative;box-shadow:0 12px 40px #1837461c}
.report-header{display:flex;align-items:center;border-bottom:1px solid var(--line);padding-bottom:10px}.brand-logo{width:180px}.header-copy{margin-left:auto;text-align:right;color:var(--muted);font-size:10px}.header-copy strong{display:block;color:var(--navy);letter-spacing:.08em}
h1{font-size:27px;color:var(--navy);margin:8px 0 5px}h2{font-size:15px;color:var(--navy);margin:16px 0 8px;border-left:4px solid var(--red);padding-left:9px}.eyebrow{color:var(--red);font-size:10px;letter-spacing:.16em;margin-top:15px}.sub,.note{font-size:10px;color:var(--muted);line-height:1.6}.sub{margin:0 0 14px}
.summary{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);border-top:3px solid var(--navy)}.metric{padding:9px 12px;border-right:1px solid var(--line)}.metric span{display:block;font-size:10px;color:var(--muted)}.metric strong{display:block;font-size:23px;color:var(--navy)}
table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{border:1px solid var(--line);padding:7px 6px;text-align:center;overflow-wrap:anywhere}th{background:#edf3f6;color:#324b59;font-size:10px}td{font-size:11px}.setup td{text-align:left}.setup th{width:14%}
.overview{display:grid;grid-template-columns:1fr 1.25fr;gap:18px;margin-top:12px}.nav img{width:100%;height:210px;object-fit:contain}.nav{background:#f5f8fa;border:1px solid var(--line);padding:5px}.reason{color:#875022}
.particle-heading{display:flex;align-items:center;justify-content:space-between;padding:10px 13px;background:var(--navy);color:#fff;margin-top:18px}.particle-heading strong{font-size:18px}.particle-heading small{font-size:11px;color:#dceaf1}.outcome{padding:8px 10px;border:1px solid var(--line);border-top:0;font-size:11px}
.images{display:grid;grid-template-columns:1fr .8fr 1.4fr;gap:1px;background:var(--line);border:1px solid var(--line);margin-top:12px}.images figure{min-width:0;background:#f7f9fa;margin:0}.images img{width:100%;height:245px;object-fit:contain;padding:8px}figcaption{font-size:11px;padding:8px;background:white;color:var(--navy);border-bottom:1px solid var(--line)}.placeholder{height:245px;display:flex;align-items:center;justify-content:center;font-size:11px;color:var(--muted)}
.classification{border:1px solid var(--line);background:#f4f8fa;padding:10px 12px;font-size:11px;margin-top:12px}.coords{display:flex;gap:30px;font-size:11px;padding:9px 0;color:var(--muted)}.footer{border-top:1px solid var(--line);color:var(--muted);font-size:9px;margin-top:16px;padding-top:8px;display:flex;justify-content:space-between}.rejected td{font-size:10px;padding:6px}
.field-overview{display:grid;grid-template-columns:1.3fr 1fr;gap:22px;align-items:start}.field-overview figure{margin:0;border:1px solid var(--line);background:#f7f9fa}.field-overview img{display:block;width:100%;height:355px;object-fit:contain;padding:8px}.field-overview h2{margin-top:0}.field-overview td{padding:5px}.field-overview th{padding:6px}
@page{size:A4 landscape;margin:8mm}@media print{html,body{background:white;-webkit-print-color-adjust:exact;print-color-adjust:exact}.sheet{width:auto;min-height:0;height:193mm;margin:0;padding:0;box-shadow:none;break-after:page;page-break-after:always}.sheet:last-child{break-after:auto;page-break-after:auto}.brand-logo{width:150px}.report-header{padding-bottom:6px}h1{font-size:23px}.eyebrow{margin-top:10px}.images img,.placeholder{height:180px}.images{margin-top:8px}.particle-heading{margin-top:10px;padding:7px 10px}.outcome{padding:5px 8px}.coords{padding:6px 0}.classification{margin-top:8px;padding:7px 10px;font-size:10px}.footer{position:absolute;bottom:0;left:0;right:0}.overview{gap:16px}.nav img{height:205px}}
'''


def build_sem_reports(directory, manifest, *, make_pdf=True):
    directory = Path(directory).resolve()
    (directory/'report_assets').mkdir(exist_ok=True)
    snapshot = directory/'.cleanliness-settings.json'
    config = (json.loads(snapshot.read_text(encoding='utf-8')).get('report') or {}) if snapshot.is_file() else {}
    exclude, columns = _exclude_element_set(config), _element_column_limit(config)
    show_coordinates = manifest['request'].get('showReportCoordinates', True)
    records = manifest.get('particles', [])
    accepted = [p for p in records if p['status'] in {'SUCCEEDED','MEASURED'}]
    other = [p for p in records if p['status'] not in {'SUCCEEDED','MEASURED'}]
    tiles = {t['id']:t for t in manifest['plan']['tiles']}
    records_by_tile,accepted_by_tile,records_by_capture = defaultdict(list),defaultdict(list),defaultdict(list)
    for record in records:
        tile_id=record['tileId']
        records_by_tile[tile_id].append(record)
        records_by_capture[(tile_id,record.get('sourceCapture',tiles[tile_id].get('raw')))].append(record)
        if record['status'] in {'SUCCEEDED','MEASURED'}:
            accepted_by_tile[tile_id].append(record)
    plan = manifest['plan']
    completed = sum(t['status']=='COMPLETED' for t in tiles.values())
    generated = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')
    header = f'<header class="report-header"><img class="brand-logo" src="{zeptools_logo_data_uri()}" alt="ZEPTOOLS"><div class="header-copy"><strong>CLEANLINESS INSPECTION REPORT</strong>直接 SEM 分析 · www.zeptools.cn</div></header>'
    pages = []
    nav = navigation_image(directory, manifest)
    bins = [(0,25),(25,50),(50,100),(100,200),(200,float('inf'))]
    sizes = ''.join(f'<tr><td>{low}–{high if math.isfinite(high) else "以上"}</td><td>{sum(low<=p.get("lengthUm",0)<high for p in accepted)}</td></tr>' for low,high in bins)
    cleanup = manifest.get('cleanup', {})
    gun = '自动关枪已确认' if cleanup.get('gunOffVerified') else ('自动关枪未确认' if cleanup.get('gunOffRequested') else '按迭代设置保留电子枪状态')
    rejected_counts = Counter(p.get('reason') or STATUS.get(p['status'],p['status']) for p in other)
    rejection_summary = '；'.join(f'{reason} {count} 个' for reason,count in rejected_counts.items()) or '无跳过候选'
    partial = completed < len(tiles)
    partial_note = (f'已完成部分报告：规划 {len(tiles)} 个视野，完整完成 {completed} 个；中断视野仅纳入已成功返回的结果。未扫描区域不参与统计，本报告不代表整个 ROI 已完成。' if partial else '')
    pages.append(header+f'''<div class="eyebrow">INSPECTION REPORT</div><h1>清洁度检测报告{' · 已完成部分' if partial else ''}</h1>
<p class="sub">样品：{esc(manifest['input']['workpieceName'])}　·　状态：{esc(STATUS.get(manifest['status'],manifest['status']))}<br>任务编号：{esc(manifest['taskId'])}　·　报告生成：{generated}</p>
{f'<p class="note" style="color:#875022">{esc(partial_note)}</p>' if partial else ''}
<div class="summary"><div class="metric"><span>扫描区域面积</span><strong>{number(plan['areaMm2'],3)} <small>mm²</small></strong></div><div class="metric"><span>完成视野</span><strong>{completed} / {len(tiles)}</strong></div><div class="metric"><span>完成分析颗粒</span><strong>{len(accepted)}</strong></div><div class="metric"><span>EDS 成功</span><strong>{sum(p['status']=='SUCCEEDED' for p in records)}</strong></div><div class="metric"><span>跳过 / 未完成候选</span><strong>{len(other)}</strong></div></div>
<h2>检测参数</h2><table class="setup"><tr><th>区域形状</th><td>{'圆形' if plan['roi']['shape']=='circle' else '正方形'}</td><th>单视野 FOV</th><td>{number(plan['fieldUm'])} μm</td><th>图像分辨率</th><td>{plan['imageSize']} × {plan['imageSize']}</td></tr><tr><th>最小颗粒长度</th><td>{number(plan['minLengthUm'])} μm</td><th>视野重叠</th><td>{number(plan['overlap']*100,1)}%</td><th>采集结束时</th><td>{gun}</td></tr></table>
<p class="note">分割模型：{'SAM3（远程），置信度阈值 '+number(manifest.get('model',{}).get('scoreThreshold'),3) if manifest.get('model',{}).get('backend')=='sam3' else 'YOLO（本地）'}</p>
<div class="overview"><div><h2>导航图与 ROI 扫描区域</h2><div class="nav">{f'<img src="{nav}" alt="导航图与扫描区域">' if nav else '导航图未保存'}</div><p class="note">红色：ROI 边界；绿色：已完成视野；橙色：未完成视野。<br>ROI: {number((plan["roi"]["x1"]-plan["roi"]["x0"])/1000,2)} x {number((plan["roi"]["y1"]-plan["roi"]["y0"])/1000,2)} mm / {number(plan["areaMm2"],3)} mm²</p></div><div><h2>颗粒长度分布</h2><table><tr><th>长度区间（μm）</th><th>完成分析数量</th></tr>{sizes}</table><p class="note">尺寸来自 SEM 原始图的分割轮廓及实际像素尺寸。统计仅包含已完成分析的颗粒；重叠区候选和被拒绝候选另行保留。</p><p class="note">{esc(rejection_summary)}</p><p class="note">本报告记录区域分析结果，不计算标准清洁度等级。</p></div></div>''')
    cache = {}
    element_note = ('按正式报告设置剔除 '+ '、'.join(sorted(exclude))+' 后归一化至 100%。' if exclude else '元素占比归一化至 100%。')+'原始能谱与原始元素含量完整保存在任务数据中。'
    ordered = []
    for tile in tiles.values():
        if not tile.get('raw') and not records_by_tile[tile['id']]:
            continue
        ordered.append((None, tile))
        ordered.extend((p,tile) for p in accepted_by_tile[tile['id']])
    rank = 0
    for p,tile in ordered:
        if p is None:
            cache.clear()  # Retain only the current frame during large surveys.
            pages.append(field_overview(directory,tile,records_by_tile[tile['id']],header,cache))
            continue
        rank += 1
        if p.get('sourceCapture'):
            tile={**tile,'raw':p['sourceCapture']}
        images = particle_images(directory,p,tile,cache,records_by_capture[(p['tileId'],tile.get('raw'))])
        spectrum = p.get('spectrum', {})
        spectrum_path = spectrum.get('spectrumImage', {}).get('path')
        headers, values = _html_particle_element_cells({'energySpectrum':spectrum},columns,exclude)
        def figure(title, path):
            return f'<figure><figcaption>{title}</figcaption>'+ (f'<img src="{esc(path)}" alt="{title}">' if path else '<div class="placeholder">本颗粒未保存此图像</div>')+'</figure>'
        contour = np.asarray(p.get('contour',[]),np.float32)
        perimeter = float(cv2.arcLength(contour,True)) if len(contour)>=3 else 0
        pixel_area = float(cv2.contourArea(contour)) if len(contour)>=3 else 0
        roundness = 4*math.pi*pixel_area/perimeter**2 if perimeter else None
        aspect = p['lengthUm']/p['widthUm'] if p.get('widthUm') else None
        material = spectrum.get('materialClassification', {})
        vda = spectrum.get('vdaClassification', {})
        classification = f"Oxford：{esc(material.get('materialClass') or '未提供')} / {esc(material.get('hardnessLabel') or '未提供')}　·　VDA：{esc(vda.get('materialClass') or '未提供')} / {esc(vda.get('hardnessLabel') or '未提供')}<br>{esc(stop_label(spectrum))}；分类警告、原始成分与采集参数见后续 EDS 明细。"
        coords = ''
        if show_coordinates:
            coords = f'<div class="coords"><span>颗粒估计台坐标（μm）：{esc([round(v,3) for v in p.get("stageUm",[])])}</span><span>实际采图台坐标（μm）：{esc(tile.get("actualStageUm"))}</span><span>采谱点（px）：{esc(p.get("safePoint"))}</span></div>'
        pages.append(header+f'''<div class="particle-heading"><strong>视野 {p['tileId']:02d} / 颗粒 {rank:02d}</strong><small>ID {p['sequence']} · {STATUS[p['status']]}</small></div>
<div class="outcome">分析方式：SEM 分割后直接自动采谱　·　实际像素尺寸：{number(tile.get('capture',{}).get('pixelSizeUm'),4)} μm/px　·　EDS 尝试次数：{len(p.get('edsAttempts',[])) or (1 if spectrum else 0)}</div>
<h2>形貌测量</h2><table><tr><th>长度（μm）</th><th>宽度（μm）</th><th>面积（μm²）</th><th>等效圆直径（μm）</th><th>长宽比</th><th>圆度</th></tr><tr><td>{number(p.get('lengthUm'))}</td><td>{number(p.get('widthUm'))}</td><td>{number(p.get('areaUm2'))}</td><td>{number(p.get('equivalentDiameterUm'))}</td><td>{number(aspect,3)}</td><td>{number(roundness,3)}</td></tr></table>
{coords}<div class="images">{figure('SEM 原视野 / 目标轮廓',images[0] if images else None)}{figure('颗粒局部 / 采谱位置',images[1] if images else None)}{figure('EDS 能谱谱图',spectrum_path)}</div>
<p class="note">原视野：<b style="color:#b56a00">■ 橙色：已识别、未完成分析</b>　<b style="color:#258038">■ 绿色：已成功分析</b>　<b style="color:#007e91">■ 青色加框：当前展示颗粒</b>；编号对应颗粒 ID。<br>局部图：绿色为本颗粒轮廓，红色十字为实际下发的采谱点；下方显示比例尺。</p>
<h2>元素含量（wt%，客户端展示值）</h2><table><tr>{headers}</tr><tr>{values}</tr></table><p class="note">{esc(element_note)}此表不代表 VDA 分类成分。</p>
<div class="classification">{classification if spectrum else '本颗粒仅进行了形貌分析，未采集能谱。'}</div>''')
        for title, body in detail_sections(spectrum):
            pages.append(header+f'<h2>视野 {p["tileId"]} / 颗粒 ID {p["sequence"]} · {esc(title)}</h2>'+body)
    for start in range(0,len(other),14):
        rows = ''.join(f'<tr><td>{p["sequence"]}</td><td>{p["tileId"]}</td><td>{esc(STATUS.get(p["status"],p["status"]))}</td><td>{number(p.get("lengthUm"))}</td><td style="text-align:left">{esc(p.get("reason") or "—")}</td></tr>' for p in other[start:start+14])
        pages.append(header+f'<h1>跳过与未完成候选</h1><p class="sub">以下记录不计入完成分析颗粒数，未获准的候选不进入 EDS。</p><table class="rejected"><colgroup><col style="width:7%"><col style="width:7%"><col style="width:14%"><col style="width:13%"><col></colgroup><tr><th>ID</th><th>视野</th><th>状态</th><th>长度（μm）</th><th>原因</th></tr>{rows}</table>')
    total = len(pages)
    path = directory/'sem_report.html'
    with path.open('w',encoding='utf-8') as output:
        output.write(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>清洁度检测报告</title><style>{STYLE}{EDS_STYLE}</style></head><body>')
        for i,body in enumerate(pages):
            output.write(f'<section class="sheet">{body}<footer class="footer"><span>ZEPTOOLS · 清洁度检测报告 · {esc(manifest["taskId"])}</span><span>{i+1} / {total}</span></footer></section>')
        output.write('</body></html>')
    with (directory/'sem_results.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['颗粒ID','视野','状态','长度 μm','宽度 μm','面积 μm²','等效圆直径 μm','原始元素含量 wt%','原因',*CSV_FIELDS])
        for p in records:
            writer.writerow([p['sequence'],p['tileId'],STATUS.get(p['status'],p['status']),p.get('lengthUm'),p.get('widthUm'),p.get('areaUm2'),p.get('equivalentDiameterUm'),'; '.join(f"{e['symbol']}:{e['value']}" for e in p.get('spectrum',{}).get('elements',[])),p.get('reason',''),*csv_values(p.get('spectrum') or {}).values()])
    manifest['reports']={'html':'sem_report.html','csv':'sem_results.csv'}
    if make_pdf:
        pdf=directory/'sem_report.pdf'
        if len(pages)<=60:
            html_to_pdf_via_edge(path,pdf)
        else:
            # Inline images only one bounded chunk at a time. A 30 mm survey
            # can contain thousands of pages; one gigantic browser document
            # otherwise retains all decoded image copies at once.
            from pypdf import PdfReader, PdfWriter
            part_dir=directory/'report_parts'
            part_dir.mkdir(exist_ok=True)
            writer=PdfWriter()
            try:
                for offset in range(0,total,40):
                    part_sheets=''.join(f'<section class="sheet">{body}<footer class="footer"><span>ZEPTOOLS · {esc(manifest["taskId"])}</span><span>{offset+j+1} / {total}</span></footer></section>' for j,body in enumerate(pages[offset:offset+40]))
                    part_html=directory/f'.sem-report-part-{offset//40+1:04d}.html'
                    part_pdf=part_dir/f'{offset//40+1:04d}.pdf'
                    part_html.write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{STYLE}{EDS_STYLE}</style></head><body>{part_sheets}</body></html>',encoding='utf-8')
                    try:
                        html_to_pdf_via_edge(part_html,part_pdf)
                        reader=PdfReader(part_pdf)
                        expected=min(40,total-offset)
                        if len(reader.pages)!=expected:
                            raise ValueError(f'PDF 分卷页数异常：{len(reader.pages)} != {expected}')
                        writer.append(reader)
                    finally:
                        part_html.unlink(missing_ok=True)
                temporary=pdf.with_suffix('.assembling.pdf')
                with temporary.open('wb') as output:
                    writer.write(output)
                temporary.replace(pdf)
            finally:
                writer.close()
        manifest['reports']['pdf']=pdf.name
    return manifest['reports']
