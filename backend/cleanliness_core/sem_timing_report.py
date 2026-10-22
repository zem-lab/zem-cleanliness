"""Non-overlapping measured worker timings, plus separately labelled EDS time."""
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

LABELS = {'preflight':'开工检查','connect':'设备连接与参数确认','move':'移台与稳定确认',
          'capture':'SEM 拍图、传输及原图保存','segmentation':'颗粒分割',
          'candidateReview':'轮廓筛选','eds':'逐颗 EDS 完整调用','cleanup':'设备恢复',
          'report':'正式报告及图片生成','persistence':'任务清单落盘','other':'其他软件处理'}


def write_timing_report(directory, manifest):
    directory = Path(directory)
    runs = manifest.get('timingRuns', [])
    totals = defaultdict(float)
    worker = sum(r.get('wallSeconds',0) for r in runs)
    events = []
    for run_index, run in enumerate(runs,1):
        for stage, seconds in run.get('stages',{}).items():
            totals[stage] += seconds
        totals['persistence'] += run.get('persistenceSeconds',0)
        events.extend({'run':run_index,**event} for event in run.get('events',[]))
    totals['other'] = max(0,worker-sum(totals.values()))
    totals['preflight'] = sum(p['seconds'] for p in manifest.get('preflightTimings',[]))
    total = worker+totals['preflight']
    tiles = sorted([t for t in manifest['plan']['tiles'] if t.get('wallSeconds') is not None],key=lambda t:t['wallSeconds'],reverse=True)
    particles = sorted([p for p in manifest.get('particles',[]) if p.get('timing',{}).get('eds') is not None],key=lambda p:p['timing']['eds'],reverse=True)
    device_eds = sum(p.get('spectrum',{}).get('elapsedSeconds',0) for p in manifest.get('particles',[]))
    spectra = [p.get('spectrum',{}) for p in manifest.get('particles',[]) if p.get('spectrum')]
    acquisition_times = {key:{'reportedCount':sum(key in s for s in spectra),
                             'seconds':sum(s.get(key,0) for s in spectra)}
                         for key in ('realTimeSeconds','liveTimeSeconds','elapsedSeconds','clientCallSeconds')}
    below_target = sum(bool(s.get('targetCounts')) and s.get('accumulatedCounts') is not None
                       and s['accumulatedCounts']<s['targetCounts'] for s in spectra)
    retries=sum(max(0,len(p.get('edsAttempts',[]))-1) for p in manifest.get('particles',[]))
    summary = {'taskId':manifest['taskId'],'status':manifest['status'],'measuredRunCount':len(runs),
               'totalMeasuredSeconds':total,'stages':dict(totals),'edsDeviceSeconds':device_eds,'edsReportedSeconds':device_eds,'edsRetryCount':retries,
               'configuration':{'fieldUm':manifest['plan']['fieldUm'],'imageSize':manifest['plan']['imageSize'],
                   'areaMm2':manifest['plan']['areaMm2'],'captureLevel':manifest['request'].get('captureLevel'),
                   'minLengthUm':manifest['plan']['minLengthUm'],'edsRetryLimit':manifest['request'].get('edsRetryCount',2)},
               'acquisitionTimes':acquisition_times,'belowTargetCount':below_target,
               'completedTiles':sum(t['status']=='COMPLETED' for t in manifest['plan']['tiles']),
               'plannedTiles':manifest['plan']['tileCount'],'slowestTiles':[{'tileId':t['id'],'seconds':t['wallSeconds']} for t in tiles[:20]],
               'slowestParticles':[{'sequence':p['sequence'],'tileId':p['tileId'],'edsSeconds':p['timing']['eds']} for p in particles[:20]],
               'priorAttempts':manifest.get('priorTimingEstimates',[]),
               'measurementNotes':['perf_counter 实测；阶段合计不重复计算。',
                   '其他软件处理为总用时减已计时阶段的差值，包含像素校验、跨视野检查、叠加图保存和状态读取。',
                   'EDS 服务上报耗时只对成功结果求和；并非独立测得的纯硬件时间。仅作对照，不再次累加。',
                   '末次清单保存和本时间报告生成不计入总用时；暂停等待与没有计时的旧执行不计入。']}
    (directory/'timing_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (directory/'timing_events.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','stage','tileId','sequence','startedAt','seconds','outcome'],extrasaction='ignore')
        writer.writeheader();writer.writerows(events)
    esc=lambda v:html.escape(str(v))
    rows=''.join(f'<tr><td>{LABELS.get(k,k)}</td><td>{v:.3f}</td><td>{v/total*100 if total else 0:.1f}%</td><td><div class="bar" style="width:{v/total*100 if total else 0:.1f}%"></div></td></tr>' for k,v in sorted(totals.items(),key=lambda item:item[1],reverse=True))
    tile_rows=''.join(f'<tr><td>{t["id"]}</td><td>{t["wallSeconds"]:.3f}</td></tr>' for t in tiles[:20])
    particle_rows=''.join(f'<tr><td>{p["sequence"]}</td><td>{p["tileId"]}</td><td>{p["timing"]["eds"]:.3f}</td></tr>' for p in particles[:20])
    nav = 'report_assets/navigation.jpg'
    acquisition_rows = ''.join(f'<tr><td>{label}</td><td>{acquisition_times[key]["reportedCount"]}</td><td>{acquisition_times[key]["seconds"]:.3f}</td></tr>'
        for key,label in [('realTimeSeconds','仪器实际时间'),('liveTimeSeconds','仪器活时间'),('elapsedSeconds','服务端耗时'),('clientCallSeconds','客户端完整调用')])
    prior = ''.join(f'<li>{esc(a.get("startedAt"))}：约 {a.get("seconds",0):.2f} 秒；{esc(a.get("error"))}（根据作业日志时间计算，未分环节计时，不纳入本表实测占比）</li>' for a in manifest.get('priorTimingEstimates',[]))
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SEM 时间开销分析</title><style>
body{{font:14px/1.7 "Microsoft YaHei",sans-serif;color:#233e53;max-width:1100px;margin:36px auto}}h1,h2{{color:#12384f}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d7e1e7;padding:8px 12px;text-align:left}}th{{background:#edf3f6}}.bar{{height:12px;background:#1675b8}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.note{{font-size:12px;color:#647887}}img{{width:380px;max-height:240px;object-fit:contain}}@page{{size:A4 landscape;margin:8mm}}@media print{{body{{margin:0;max-width:none;font-size:11px;line-height:1.35}}td,th{{padding:3px 6px;font-size:10px;line-height:1.3}}img{{width:260px;max-height:150px}}.grid{{display:grid;gap:16px;break-before:page;break-inside:avoid}}.grid>div{{break-inside:avoid}}.note{{font-size:10px}}tr{{break-inside:avoid}}}}</style>
<h1>SEM 时间开销分析</h1><p>任务 {esc(manifest['taskId'])} · 状态 {esc(manifest['status'])}</p><p>视野完成 {summary['completedTiles']} / {summary['plannedTiles']} · 已计时执行 {len(runs)} 次 · 实测总用时 {total/60:.2f} 分钟</p>
<p>测量条件：面积 {manifest['plan']['areaMm2']:.3f} mm²，FOV {manifest['plan']['fieldUm']:.2f} μm，图像 {manifest['plan']['imageSize']}×{manifest['plan']['imageSize']}，拍图质量档位 {manifest['request'].get('captureLevel','—')}，最小长度 {manifest['plan']['minLengthUm']:g} μm，EDS 重试上限 {manifest['request'].get('edsRetryCount',2)} 次。</p>
<img src="{nav}" alt="导航图与 ROI"><h2>时间花在哪里</h2><table><tr><th>环节</th><th>秒</th><th>占比</th><th style="width:35%">占比图</th></tr>{rows}</table>
<p>EDS 服务上报耗时（成功结果合计）：{device_eds:.3f} 秒，仅作对照，不再次累加。累计重试 {retries} 次。平均视野时间：{sum(t['wallSeconds'] for t in tiles)/len(tiles) if tiles else 0:.3f} 秒；平均颗粒 EDS 完整调用：{sum(p['timing']['eds'] for p in particles)/len(particles) if particles else 0:.3f} 秒。</p>
<div class="grid"><div><h2>最慢视野（前 20）</h2><table><tr><th>视野</th><th>秒</th></tr>{tile_rows}</table></div><div><h2>最慢采谱颗粒（前 20）</h2><table><tr><th>颗粒</th><th>视野</th><th>秒</th></tr>{particle_rows}</table></div></div>{'<h2>早期未分环节计时的尝试</h2><ul>'+prior+'</ul>' if prior else ''}<h2>计时口径</h2><ul class="note">{''.join('<li>'+esc(n)+'</li>' for n in summary['measurementNotes'])}</ul><p><a href="timing_events.csv">下载逐阶段计时 CSV</a> · <a href="sem_report.pdf">检测 PDF</a></p></html>'''
    acquisition_section = '<h2>EDS 时间口径对照（不累加）</h2><table><tr><th>时间口径</th><th>有数据的结果数</th><th>合计秒数</th></tr>'+acquisition_rows+f'</table><p>已知计数未达标结果：{below_target}。各时间口径有包含关系，不相加；未上报的旧结果不补零参与平均。</p>'
    document = document.replace('<h2>计时口径</h2>',acquisition_section+'<h2>计时口径</h2>')
    (directory/'timing_report.html').write_text(document,encoding='utf-8')
    manifest.setdefault('reports',{}).update(timingHtml='timing_report.html',timingJson='timing_report.json',timingCsv='timing_events.csv')
    return summary
