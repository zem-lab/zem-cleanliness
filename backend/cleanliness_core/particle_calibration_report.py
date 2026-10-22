"""Portable HTML audit for particle calibration (success AND failures). No browser/service needed."""
from __future__ import annotations

import html
import json
from pathlib import Path


def stage_coverage_svg(audit: dict) -> str:
    """Recorded acquisition centers only, not a claim of fully usable FOV coverage."""
    points = [(s['captureStageUm'], s.get('phase', 'coarse'), s['index'])
              for s in audit.get('captures', []) if s.get('status') == 'CAPTURED' and s.get('captureStageUm')]
    if not points:
        return '<p>尚无完成采图的台坐标。</p>'
    xvals = [p[0][0]/1000 for p in points]
    yvals = [p[0][1]/1000 for p in points]
    xmin, xmax, ymin, ymax = min(xvals)-1, max(xvals)+1, min(yvals)-1, max(yvals)+1
    span = max(xmax-xmin, ymax-ymin, 2)
    xmid, ymid = (xmin+xmax)/2, (ymin+ymax)/2
    xmin, ymin = xmid-span/2, ymid-span/2
    def mapped(point):
        return 65+(point[0]/1000-xmin)/span*460, 505-(point[1]/1000-ymin)/span*460
    elements = ['<rect x="65" y="45" width="460" height="460" fill="#f6fafb" stroke="#c8d7dd"/>']
    for i in range(5):
        pos = 65+i*115
        y = 505-i*115
        elements.append(f'<path d="M {pos} 45 V 505 M 65 {y} H 525" stroke="#dce5e8"/>')
        elements.append(f'<text x="{pos}" y="526" text-anchor="middle">{xmin+span*i/4:.1f}</text>')
        elements.append(f'<text x="56" y="{y+4}" text-anchor="end">{ymin+span*i/4:.1f}</text>')
    for point, phase, index in points:
        x, y = mapped(point)
        color = '#087f78' if phase == 'fine' else '#377eb8'
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" fill-opacity=".7"><title>采图 #{html.escape(str(index))}：{point[0]:.2f}, {point[1]:.2f} µm</title></circle>')
    for record in audit.get('observations', []):
        target = record.get('targetStageUm')
        if target and str(record.get('role', '')).startswith('seed'):
            x, y = mapped([target['x'], target['y']])
            if 65 <= x <= 525 and 45 <= y <= 505:
                elements.append(f'<path d="M {x-6:.2f} {y:.2f} h 12 M {x:.2f} {y-6:.2f} v 12" stroke="#b96b15" stroke-width="2"/>')
                elements.append(f'<text x="{x+9:.2f}" y="{y-7:.2f}">#{html.escape(str(record["sequence"]))}</text>')
    return ('<svg viewBox="0 0 590 570" role="img" aria-label="实际采图台坐标分布" style="width:100%;max-width:590px;font:12px system-ui">'
            + ''.join(elements) + '<text x="295" y="553" text-anchor="middle">SEM 台 X / mm</text><text x="10" y="24">SEM 台 Y / mm</text></svg>'
            '<p>蓝点：粗倍率；绿点：精倍率；橙色十字：搜索初始目标。仅展示完成采图时记录的台面中心，不表示视场全幅有效或已经匹配成功。</p>')


def build_particle_calibration_report(manifest: dict, task_dir: Path) -> Path:
    audit = manifest['calibration']['abPointResult']['semRefinement']
    gate = audit.get('acceptance') or {}
    def esc(value):
        return html.escape(str(value if value is not None else '—'))
    def metric(value):
        return f'{float(value):.2f}' if isinstance(value, (float, int)) else '—'
    def evidence(records, validation=False, search=False):
        rows = []
        shots = {s['index']: s for s in audit.get('captures', [])}
        for r in records:
            images = r.get('images') or {}
            pictures = ''.join(f'<a href="../../{esc(images[k])}"><img loading="lazy" src="../../{esc(images[k])}" alt="{label} #{esc(r["sequence"])}"></a>'
                               for k, label in (('opticalImage', '光镜'), ('semAnnotatedImage', 'SEM')) if images.get(k))
            error = r.get('predictionErrorUm') if validation or search else r.get('fitResidualUm')
            if validation and r.get('status') != 'MATCH':
                error = None
            latest_attempt = next((a for a in reversed(r.get('attempts') or []) if 'captureIndex' in a), None)
            context = ((latest_attempt if search and latest_attempt else r).get('contextVerification') or {})
            context_text = (f'<p>同图邻域复核：{esc(context.get("status"))} · '
                            f'{esc(context.get("confirmedCount"))} / {esc(context.get("requiredCount"))} · '
                            f'{esc(context.get("reason"))}</p><details><summary>邻域形状与相对坐标证据</summary>'
                            f'<pre>{esc(json.dumps(context, ensure_ascii=False, indent=2))}</pre></details>' if context else '')
            attempt_text = ''
            if search and r.get('attempts'):
                attempts = []
                for attempt in r['attempts']:
                    index = attempt.get('captureIndex')
                    shot = shots.get(index, {})
                    check = attempt.get('contextVerification') or {}
                    image_link = ''
                    raw = (shot.get('capture') or {}).get('rawPath')
                    if raw:
                        try:
                            relative = Path(raw).resolve().relative_to(Path(task_dir).resolve()).as_posix()
                            image_link = f'<a href="../../{esc(relative)}">#{esc(index)}</a>'
                        except ValueError:
                            image_link = esc(index)
                    attempts.append(f'<tr><td>{image_link or esc(index)}</td><td>{esc(attempt.get("status"))}</td>'
                                    f'<td>{metric(attempt.get("score"))}</td><td>{esc(check.get("confirmedCount"))} / {esc(check.get("requiredCount"))}</td>'
                                    f'<td>{esc(shot.get("targetUm") or attempt.get("targetUm"))}</td></tr>')
                attempt_text = ('<details><summary>逐视场搜索历史（含被否决候选）</summary><div class="table-scroll"><table>'
                                '<tr><th>原始图</th><th>状态</th><th>形状分数</th><th>邻域确认</th><th>计划台坐标 µm</th></tr>'
                                + ''.join(attempts) + '</table></div><p>邻域不通过的单候选不更新坐标；继续预定网格，不向候选居中。主颗粒存在多候选歧义时仍保守阻断。</p></details>')
            rows.append(f'<article><h3>#{esc(r["sequence"])} · {esc(r.get("status"))}</h3><div class="pair">{pictures}</div>'
                        f'<p>{"相对粗预测的候选偏移" if search else "独立预测误差" if validation else "训练拟合残差"} {metric(error)} µm · '
                        f'匹配相关系数 {metric(r.get("score"))} · 候选间隔 {metric(r.get("margin"))}'
                        f'{"" if validation else " · 稳健权重 " + metric(r.get("robustWeight"))}</p>{context_text}{attempt_text}</article>')
        return ''.join(rows) or '<p>本阶段尚无证据。</p>'
    transform = audit.get('frozenTransform') or {}
    passed = audit.get('status') == 'SUCCEEDED' and gate.get('withinLimits') is True
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AB 导航 + 颗粒精校准报告</title><style>
body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#1b2b40;margin:0}}main{{max-width:1120px;margin:auto;padding:32px}}
h1{{font-size:28px}}section,article{{background:white;border:1px solid #d9e1ec;border-radius:10px;padding:20px;margin:20px 0}}
.state{{font-size:20px;color:{'#08764a' if passed else '#a63d12'}}}pre{{overflow:auto;padding:16px;background:#edf2f7;white-space:pre-wrap;overflow-wrap:anywhere}}
.pair{{display:flex;gap:16px}}.pair a{{flex:1;min-width:0}}img{{width:100%;height:360px;object-fit:contain;background:#f3f6fa}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.table-scroll{{overflow:auto}}
@media(max-width:650px){{main{{padding:16px}}.pair{{flex-direction:column}}}}@media print{{article{{break-inside:avoid}}}}
</style><main><h1>AB 导航 + 颗粒精校准</h1><p>任务 {esc(manifest.get('taskId'))} · {esc(audit.get('completedAt'))}</p>
<section><p class="state">{'数值验收通过，仍需人工确认' if passed else '未通过 / 未完成，不可用于自动检测'} · {esc(audit.get('status'))}</p>
<p>{esc(audit.get('error') or '；'.join(gate.get('reasons') or []))}</p>
<table><tr><td>原始光镜 SHA256</td><td>{esc(manifest['input']['sha256'])}</td></tr>
<tr><td>自动选点 / 搜索 / 拟合</td><td>本地掩膜匹配 → 多点几何检查 → 稳健仿射；无 GPT、无云端分割</td></tr>
<tr><td>独立验证（有效 / 计划）</td><td>{esc(gate.get('count'))} / {esc(gate.get('requiredCount'))}</td></tr>
<tr><td>独立预测误差（平均 / 最大）</td><td>{metric(gate.get('meanErrorUm'))} / {metric(gate.get('maxErrorUm'))} µm</td></tr>
<tr><td>总拍摄张数 / 上限</td><td>{len(audit.get('captures') or [])} / {esc(audit.get('settings', {}).get('max_captures'))}</td></tr>
<tr><td>冻结时间</td><td>{esc(audit.get('frozenAt'))}</td></tr></table>
<p>独立验证颗粒在采集前预留，冻结后按预测位置直接实拍，不参与拟合，不通过居中后照片掩盖预测误差。相关系数不是概率；本次结果不代表其他样品精度。</p></section>
<section><h2>冻结矩阵（光镜 AB µm → SEM 台坐标 µm）</h2><pre>{esc(json.dumps(transform, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>实际搜索位置</h2>{stage_coverage_svg(audit)}</section>
<section><h2>独立验证证据</h2>{evidence(audit.get('validation') or [], True)}</section>
<section><h2>训练锚点与异常记录</h2>{evidence(audit.get('anchors') or [])}</section>
<details><summary>低倍率搜索候选（单图匹配不等于坐标对应已确认）</summary>{evidence([r for r in audit.get('observations') or [] if str(r.get('role', '')).startswith('seed')], search=True)}</details>
<section><h2>设备恢复</h2><pre>{esc(json.dumps(manifest.get('cleanup'), ensure_ascii=False, indent=2))}</pre></section>
<details><summary>参数、选点计划与像素轴测量</summary><pre>{esc(json.dumps({k: audit.get(k) for k in ('algorithmVersion', 'settings', 'settingsSha256', 'navigationInitial', 'plan', 'seedPlans', 'pixelAxes', 'axisFailures', 'axisFailureDetails', 'bootstrapRejections', 'fitRejections')}, ensure_ascii=False, indent=2))}</pre></details>
<p>完整任务审计：<a href="../../task.json">task.json</a>。请保持本报告与任务目录相对位置。</p></main></html>'''
    path = Path(task_dir) / audit['runDirectory'] / 'report.html'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path
