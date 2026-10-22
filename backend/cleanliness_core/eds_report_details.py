"""Shared report presentation of final EDS data; never recompute classification."""
import html
import json
import math
import re
import textwrap

CSV_FIELDS = ['edsFinalResultOnly','edsRawQuantification','edsOxfordClassification','edsVdaClassification',
              'edsOxfordComposition','edsVdaComposition','edsStartedAt','edsEndedAt','edsRealTimeSeconds',
              'edsLiveTimeSeconds','edsServerSeconds','edsClientSeconds','edsTargetCounts','edsActualCounts',
              'edsStopReason','edsTargetReached','edsQuantMethod','edsClassificationReport']


def _hidden_oxford_warning(text):
    return bool(re.fullmatch(
        r'Wt% Sigma is unavailable for Z=\d+; UseError uses the nominal weight percent\.?',
        ' '.join(str(text).split())))


def esc(value):
    return html.escape(str(value)) if value is not None and value != '' else '未提供'


def number(value):
    if value is None: return '—'
    try: return f'{float(value):.3f}'.rstrip('0').rstrip('.') if math.isfinite(float(value)) else '—'
    except (TypeError,ValueError): return '—'


def stop_label(s):
    code=s.get('stopReasonCode')
    reason=s.get('stopReason','')
    if code==1 or reason=='SPECTRUM_STOP_REASON_TARGET_REACHED': label='达到目标'
    elif code==2 or reason=='SPECTRUM_STOP_REASON_MAXIMUM_DURATION': label='达到最长采集时间'
    elif not code and (not reason or reason=='SPECTRUM_STOP_REASON_UNSPECIFIED'): label='未提供停止原因'
    else: label='未知停止原因：'+str(reason or code)
    if s.get('targetCounts'):
        actual=s.get('accumulatedCounts')
        if actual is None: label+='；计数达标情况未知'
        elif actual<s['targetCounts']: label+='；计数未达标'
        else: label+='；计数已达标'
    return label


def table(headers, rows):
    return '<table class="eds-data"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def detail_sections(s):
    """Bounded sections, one printable page each; old results explicitly missing."""
    if not s: return []
    acquisition=table(['项目','结果','项目','结果'],[
        ['开始时间',s.get('startedAt'),'结束时间',s.get('endedAt')],
        ['实际时间（秒）',number(s.get('realTimeSeconds')),'活时间（秒）',number(s.get('liveTimeSeconds'))],
        ['服务端耗时（秒）',number(s.get('elapsedSeconds')),'客户端完整调用（秒）',number(s.get('clientCallSeconds'))],
        ['目标计数',s.get('targetCounts'),'实际计数',s.get('accumulatedCounts')],
        ['停止原因',stop_label(s),'定量方法',s.get('quantMethod')]])
    note='<p class="eds-note">实际时间、活时间、服务端耗时及客户端耗时分别记录，不相加。服务端起止时间为其本地时钟。缺失字段不推算。</p>'
    quants=s.get('elementQuantifications') or s.get('elements') or []
    raw={q['symbol']:q for q in quants}
    ox=s.get('oxfordWeightPercent') or {}
    vda=s.get('vdaAnalysisComposition') or {}
    symbols=sorted(set(raw)|set(ox)|set(vda))
    rows=[[e,number(raw.get(e,{}).get('value')),number(raw.get(e,{}).get('sigma')),number(ox.get(e)),number(vda.get(e))] for e in symbols]
    composition_note='<p class="eds-note">原始 wt% 与 σ（标准差）来自仪器；Oxford / VDA 成分为服务端分类实际使用值，不再剔除或归一化。“—”表示未提供或该口径未包含此元素；明确置零的元素显示 0。分类硬度为规则类别，不是直接硬度测量。</p>'
    sections=[]
    for start in range(0,max(1,len(rows)),12):
        sections.append(('EDS 采集与成分'+('（续）' if start else ''),
            (acquisition+note if start==0 else '')+table(['元素','仪器原始 wt%','σ（wt%）','Oxford wt%','VDA wt%'],rows[start:start+12] or [['未提供','—','—','—','—']])+composition_note))
    classifications=[('Oxford',s.get('materialClassification') or {}),('VDA',s.get('vdaClassification') or {})]
    summary=table(['来源','材料分类','上级分类','硬度类别 / 区间','置信度','分类状态'],[
        [name,m.get('materialClass'),m.get('parentClass'),
         (str(m.get('hardnessLabel') or '未提供')+' / '+str(m.get('hardnessBand') or '未提供')),
         number(m.get('confidence')),('未提供' if not m else '未分类' if m.get('isUnclassified') else '服务端已分类')]
        for name,m in classifications])
    notes=[]
    for name,m in classifications:
        for key,label in [('matchedCriteria','命中规则'),('boundaryNotes','临界元素'),('warnings','警告')]:
            for text in m.get(key,[]):
                if name == 'Oxford' and key == 'warnings' and _hidden_oxford_warning(text):
                    continue
                # Bound line lengths before pagination, including long error text.
                notes.extend(textwrap.wrap(f'{name} {label}：{text}',width=86,replace_whitespace=True) or [''])
    if not notes: notes=['无其他需展示的命中规则、临界元素或警告。']
    for start in range(0,len(notes),16):
        sections.append(('Oxford / VDA 分类'+('（续）' if start else ''),
            (summary if start==0 else '')+'<div class="eds-lines">'+''.join('<p>'+esc(line)+'</p>' for line in notes[start:start+16])+'</div>'+
            '<p class="eds-note">兜底分类和其他警告按服务原文保留，不等同于材料已被人工确认。服务端完整分类原文保存在任务数据及 CSV 的 edsClassificationReport 列。</p>'))
    return sections


def csv_values(s):
    def encoded(value): return json.dumps(value,ensure_ascii=False,separators=(',',':')) if value is not None else ''
    values=[s.get('finalResultOnly'),encoded(s.get('elementQuantifications') or s.get('elements')),
        encoded(s.get('materialClassification')),encoded(s.get('vdaClassification')),
        encoded(s.get('oxfordWeightPercent')),encoded(s.get('vdaAnalysisComposition')),
        s.get('startedAt'),s.get('endedAt'),s.get('realTimeSeconds'),s.get('liveTimeSeconds'),
        s.get('elapsedSeconds'),s.get('clientCallSeconds'),s.get('targetCounts'),s.get('accumulatedCounts'),
        s.get('stopReason'),s.get('targetReached'),s.get('quantMethod'),s.get('classificationReport')]
    return dict(zip(CSV_FIELDS,values,strict=True))


STYLE = '''
.eds-data{width:100%;border-collapse:collapse;font-size:11px;table-layout:fixed}
.eds-data th,.eds-data td{border:1px solid #ccd8e1;padding:5px 7px;text-align:left;overflow-wrap:anywhere}
.eds-data th{background:#edf3f6}.eds-note{font-size:10px;color:#526777;line-height:1.5}
.eds-lines p{font-size:11px;line-height:1.5;margin:4px 0;overflow-wrap:anywhere}
.eds-detail-page{break-before:page;break-after:page;padding:18px 0}
@media print{.eds-detail-page{padding:0}.eds-data th,.eds-data td{padding:4px 6px}}
'''
