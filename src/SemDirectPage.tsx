import { useEffect, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import { ArrowLeft, CirclePause, FileText, FolderOpen, Microscope, Play, RefreshCw, ScanLine } from 'lucide-react'
import { openPath, request } from './api'
import type { JobState, TaskManifest } from './api'
import './SemDirectPage.css'

type Roi = { shape: 'square' | 'circle'; x0: number; y0: number; x1: number; y1: number }
type Tile = { id: number; x: number; y: number; status: string; preview?: string; error?: string }
type Plan = { roi: Roi; fieldUm: number; overlap: number; imageSize: number; minLengthUm: number; areaMm2: number; tileCount: number; tiles: Tile[] }
type Particle = { sequence: number; tileId: number; status: string; lengthUm?: number; areaUm2?: number; reason?: string; spectrum?: { elements: Array<{ symbol: string; value: number }> } }
type SemTask = {
  stats?: { completedTiles: number; measured: number; rejected: number; particleCount: number }
  reportState?: 'DEFERRED' | 'READY' | 'FAILED'
  reportError?: string
  model?: { backend?: 'yolo' | 'sam3'; url?: string; texts?: string[]; scoreThreshold?: number }
  taskDirectory?: string; reports?: Record<string, string>
  taskId: string; status: string; previewDataUrl?: string; plan?: Plan
  navigation: { boundsUm: number[]; imageShape: number[]; path: string; fieldUm: number; magnification: number }
  request: { edsEnabled: boolean; gunOffAfterTask: boolean; [key: string]: unknown }
  particles: Particle[]
  cleanup?: { gunOffRequested?: boolean; gunOffVerified?: boolean; gunOffError?: string }
  postCompletionGunShutdown?: { verified?: boolean; at?: string }
}
type SemProgress = { cursor: string; unchanged?: boolean; reset?: boolean; task?: Omit<SemTask, 'plan'> & { plan: Omit<Plan, 'tiles'> }; tiles?: Tile[] }
function mergeProgress(previous: SemTask | null, progress: SemProgress): SemTask | null {
  if (progress.unchanged || !progress.task) return previous
  const changes = new Map((progress.tiles || []).map(t => [t.id, t]))
  const tiles = progress.reset ? (progress.tiles || []) : (previous?.plan?.tiles || []).map(t => changes.get(t.id) || t)
  return { ...previous, ...progress.task, plan: { ...progress.task.plan, tiles } }
}
const labels: Record<string, string> = { DRAFT: '待开始', RUNNING: '分析中', STOPPED: '已暂停', FAILED: '异常中断', COMPLETED: '已完成', COMPLETED_WITH_FAILURES: '部分完成', PENDING: '待分析', ACQUIRING: '采谱中', SUCCEEDED: '采谱完成', MEASURED: '形貌完成', REJECTED: '已跳过', INCOMPLETE: '待复查' }

export default function SemDirectPage({ initialTask, apiBase, onBack }: { initialTask: TaskManifest | null; apiBase: string; onBack: () => void }) {
  const [task, setTask] = useState<SemTask | null>(initialTask?.mode === 'sem_direct' ? initialTask as SemTask : null)
  const [roi, setRoi] = useState<Roi | null>((initialTask?.plan as Plan | undefined)?.roi || null)
  const [shape, setShape] = useState<'square' | 'circle'>(roi?.shape || 'square')
  const [name, setName] = useState(String(initialTask?.input?.workpieceName || ''))
  const [field, setField] = useState(Number(((initialTask?.plan as Plan | undefined)?.fieldUm || 1000).toFixed(3)))
  const [overlap, setOverlap] = useState((initialTask?.plan as Plan | undefined)?.overlap ?? .1)
  const [size, setSize] = useState((initialTask?.plan as Plan | undefined)?.imageSize || 1024)
  const [minimum, setMinimum] = useState((initialTask?.plan as Plan | undefined)?.minLengthUm ?? 5)
  const [eds, setEds] = useState((initialTask?.request as SemTask['request'] | undefined)?.edsEnabled ?? true)
  const savedModel = (initialTask as SemTask | null)?.model
  const [segmentation, setSegmentation] = useState<'yolo' | 'sam3'>(savedModel?.backend || 'yolo')
  const [samUrl, setSamUrl] = useState(savedModel?.url || 'http://192.168.0.42:8765')
  const [samTexts, setSamTexts] = useState((savedModel?.texts || ['white particle', 'bright particle', 'bright white blob', 'flake', 'fiber', 'debris']).join(', '))
  const [samScore, setSamScore] = useState(savedModel?.scoreThreshold ?? .1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [job, setJob] = useState<JobState | null>(null)
  const [pauseRequested, setPauseRequested] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [selectedTile, setSelectedTile] = useState<number | null>(null)
  const anchor = useRef<[number, number] | null>(null)
  const mounted = useRef(true)
  const running = job?.status === 'running' || job?.status === 'queued'
  const reporting = running && job?.operation === 'sem_report'
  const editable = !busy && !running && (!task || task.status === 'DRAFT')
  const plan = dirty ? undefined : task?.plan
  const bounds = task?.navigation.boundsUm
  const finished = task?.stats?.completedTiles ?? task?.plan?.tiles.filter(t => t.status === 'COMPLETED').length ?? 0
  const measured = task?.stats?.measured ?? task?.particles.filter(p => ['SUCCEEDED', 'MEASURED'].includes(p.status)).length ?? 0
  const activeTile = task?.plan?.tiles.find(t => t.id === selectedTile) || [...(task?.plan?.tiles || [])].reverse().find(t => t.preview)
  const fileUrl = (path: string) => `${apiBase}/tasks/${task?.taskId}/files/${path}`
  const update = () => { setDirty(true); setError('') }

  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => {
    if (!running || !job || !task) return
    let cancelled = false
    let cursor = ''
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const [nextJob, progressReply] = await Promise.all([
          request<JobState>('GET', `/jobs/${job.id}`),
          request<SemProgress>('GET', `/tasks/${task.taskId}/sem/progress?cursor=${encodeURIComponent(cursor)}`),
        ])
        if (cancelled) return
        let progress = progressReply
        if (!['queued', 'running'].includes(nextJob.status)) {
          progress = await request<SemProgress>('GET', `/tasks/${task.taskId}/sem/progress`)
          if (cancelled) return
        }
        if (!progress.unchanged && progress.task) {
          setTask(previous => mergeProgress(previous, progress))
        }
        cursor = progress.cursor
        setJob(nextJob)
        if (nextJob.status === 'failed') setError(nextJob.error || nextJob.message)
        if (['queued', 'running'].includes(nextJob.status)) timer = setTimeout(() => void poll(), 1000)
      } catch (e) {
        if (!cancelled && String(e).includes('作业不存在')) {
          try {
            const active = await request<{ job: JobState | null }>('GET', `/sem/active/${task.taskId}`)
            const latest = await request<SemProgress>('GET', `/tasks/${task.taskId}/sem/progress`)
            if (cancelled) return
            setTask(previous => mergeProgress(previous, latest))
            setJob(active.job)
            setError(active.job ? '' : '原作业记录已失效，已恢复最新任务数据；若任务中断，可检查后继续扫描。')
            return
          } catch { /* Retry if the backend itself is still unavailable. */ }
        }
        if (!cancelled) { setError(String(e)); timer = setTimeout(() => void poll(), 2000) }
      }
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [job?.id, running, task?.taskId])

  // Reattach to an existing job when the user returns from task history.
  useEffect(() => {
    if (!task || job) return
    void request<{ job: JobState | null }>('GET', `/sem/active/${task.taskId}`).then(result => {
      if (mounted.current && result.job) setJob(result.job)
    }).catch(e => { if (mounted.current) setError(String(e)) })
  }, [task?.taskId, job])

  async function loadNavigation() {
    setBusy(true); setError('')
    try {
      const result = await request<SemTask>('POST', '/sem/navigation')
      if (!mounted.current) return
      setTask(result); setRoi(null); setField(Number(result.navigation.fieldUm.toFixed(3))); setDirty(false); setSelectedTile(null)
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  async function savePlan() {
    if (!task || !roi) return null
    const result = await request<SemTask>('POST', `/tasks/${task.taskId}/sem/plan`, { name, roi, fieldUm: field, overlap, imageSize: size, minLengthUm: minimum, edsEnabled: eds,
      segmentation: { backend: segmentation, url: samUrl, texts: samTexts.split(/[,，、\n]/).map(t => t.trim()).filter(Boolean), scoreThreshold: samScore } })
    setTask(result); setField(Number(result.plan!.fieldUm.toFixed(3))); setDirty(false)
    return result
  }
  async function previewPlan() {
    setBusy(true); setError('')
    try { await savePlan() } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  async function start() {
    if (!task) return
    setBusy(true); setError(''); setPauseRequested(false)
    try {
      if (task.status === 'DRAFT') await savePlan()
      setJob(await request<JobState>('POST', `/tasks/${task.taskId}/sem/start`))
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  async function pause() {
    if (!job) return
    try { await request('POST', `/jobs/${job.id}/cancel`); setPauseRequested(true) } catch (e) { setError(String(e)) }
  }
  async function generateReport() {
    if (!task) return
    setBusy(true); setError(''); setPauseRequested(false)
    try { setJob(await request<JobState>('POST', `/tasks/${task.taskId}/sem/report`)) }
    catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  function stagePoint(event: PointerEvent<SVGSVGElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect()
    const u = Math.max(0, Math.min(1, (event.clientX-rect.left)/rect.width))
    const v = Math.max(0, Math.min(1, (event.clientY-rect.top)/rect.height))
    return [bounds![2]-u*(bounds![2]-bounds![0]), bounds![1]+v*(bounds![3]-bounds![1])]
  }
  function draw(event: PointerEvent<SVGSVGElement>) {
    if (!anchor.current || !bounds) return
    const [ax, ay] = anchor.current, [bx, by] = stagePoint(event)
    const sx = bx >= ax ? 1 : -1, sy = by >= ay ? 1 : -1
    const side = Math.min(Math.max(Math.abs(bx-ax), Math.abs(by-ay)), sx > 0 ? bounds[2]-ax : ax-bounds[0], sy > 0 ? bounds[3]-ay : ay-bounds[1])
    setRoi({ shape, x0: Math.min(ax, ax+sx*side), y0: Math.min(ay, ay+sy*side), x1: Math.max(ax, ax+sx*side), y1: Math.max(ay, ay+sy*side) })
    update()
  }
  const vx = (x: number) => bounds ? (bounds[2]-x)/(bounds[2]-bounds[0])*1000 : 0
  const vy = (y: number) => bounds ? (y-bounds[1])/(bounds[3]-bounds[1])*1000 : 0
  const roiWidth = roi && bounds ? (roi.x1-roi.x0)/(bounds[2]-bounds[0])*1000 : 0
  const roiHeight = roi && bounds ? (roi.y1-roi.y0)/(bounds[3]-bounds[1])*1000 : 0
  async function openResult(kind: string) {
    try {
      const relative = task?.reports?.[kind]
      if (relative) await openPath(`${task?.taskDirectory}/${relative}`)
    } catch (e) { setError(String(e)) }
  }
  return <div className="page semDirect" data-task-id={task?.taskId || ''}>
    <div className="pageTitle"><div><button className="semBack" onClick={onBack}><ArrowLeft size={14} /> 返回</button><h1>直接 SEM 分析 <span className="semTag">无需光镜数据</span></h1><p>圈定扫描区域，自动拍图、识别颗粒并逐颗采集能谱。</p></div><span className={`semState ${running ? 'active' : ''}`}>{reporting ? '报告生成中' : running ? '自动分析中' : labels[task?.status || 'DRAFT']}</span></div>
    <div className="semSteps"><span className={!task ? 'active' : ''}>01 获取导航图</span><span className={task && !running && task.status === 'DRAFT' ? 'active' : ''}>02 圈选区域与参数</span><span className={running ? 'active' : ''}>03 自动分割与采谱</span><span className={task?.status.startsWith('COMPLETED') ? 'active' : ''}>04 查看结果</span></div>
    {error && <div className="semError" role="alert">{error}</div>}
    <div className="semLayout">
      <div className="semMain">
        <section className="semCard">
          <div className="semCardHead"><h2><ScanLine size={17} />扫描区域</h2><button onClick={() => void loadNavigation()} disabled={!editable}>{busy ? '正在读取…' : <><RefreshCw size={14} />{task ? '重新获取导航图' : '获取导航图'}</>}</button></div>
          <div className="semMap">
            {!task ? <div className="semEmpty"><Microscope size={42} /><strong>从导航图开始扫描</strong><p>连接电镜并获取当前导航图，然后拖动画出扫描区域。</p><button onClick={() => void loadNavigation()} disabled={busy}>获取导航图</button></div> : <>
              <div className="semMapCanvas" style={{ aspectRatio: `${task.navigation.imageShape[1]} / ${task.navigation.imageShape[0]}`, maxWidth: `${520*task.navigation.imageShape[1]/task.navigation.imageShape[0]}px` }}>
                <img src={task.previewDataUrl || fileUrl(task.navigation.path)} alt="电镜导航图" draggable={false} />
                <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="拖动绘制扫描区域" style={{ cursor: editable ? 'crosshair' : 'default' }}
                  onPointerDown={e => { if (!editable || !bounds || e.button !== 0) return; anchor.current = stagePoint(e); e.currentTarget.setPointerCapture(e.pointerId) }}
                  onPointerMove={draw} onPointerUp={e => { draw(e); anchor.current = null }} onPointerCancel={() => { anchor.current = null }}>
                  {plan?.tiles.map(t => <rect key={t.id} x={vx(t.x+plan.fieldUm/2)} y={vy(t.y-plan.fieldUm/2)} width={plan.fieldUm/(bounds![2]-bounds![0])*1000} height={plan.fieldUm/(bounds![3]-bounds![1])*1000} className={`semGrid ${t.status.toLowerCase()}`} onClick={() => setSelectedTile(t.id)} />)}
                  {roi && (roi.shape === 'circle' ? <ellipse className="semRoi" cx={vx((roi.x0+roi.x1)/2)} cy={vy((roi.y0+roi.y1)/2)} rx={roiWidth/2} ry={roiHeight/2} /> : <rect className="semRoi" x={vx(roi.x1)} y={vy(roi.y0)} width={roiWidth} height={roiHeight} />)}
                </svg>
              </div>
              <div className="semMapCaption"><span>{roi ? `${roi.shape === 'circle' ? '圆形直径' : '正方形边长'} ${(roi.x1-roi.x0).toFixed(0)} μm` : '在图中按住鼠标拖动，画出扫描区域'}</span><span>绿色：已完成 · 蓝色：待扫描</span></div>
            </>}
          </div>
        </section>
        <section className="semCard semResults"><div className="semCardHead"><h2>分析进度</h2><span>{job?.message || '每个视野完成拍图、分割和采谱后进入下一个视野'}</span></div>
          <div className="semStats"><div><strong>{finished}<small> / {task?.plan?.tileCount || '—'}</small></strong><span>已完成视野</span></div><div><strong>{measured}</strong><span>已分析颗粒</span></div><div><strong>{task?.stats?.rejected ?? task?.particles.filter(p => p.status === 'REJECTED').length ?? 0}</strong><span>已跳过候选</span></div><div><strong>{plan?.areaMm2.toFixed(3) || '—'}<small> mm²</small></strong><span>选区面积</span></div></div>
          <progress max={task?.plan?.tileCount || 1} value={finished} />
          {pauseRequested && running && <p className="semHint">已请求暂停，当前视野的采谱完成后停止。</p>}
          {task?.cleanup?.gunOffRequested && <p className="semHint">{task.cleanup.gunOffVerified ? '电子枪已关闭；继续扫描前请在电镜端重新开枪。' : `未确认关枪：${task.cleanup.gunOffError || '请检查设备'}`}</p>}
          {task?.postCompletionGunShutdown?.verified && <p className="semHint">完整扫描与报告已核验，电子枪已关闭并回读确认。</p>}
          {activeTile?.preview && <div className="semFrame"><img src={fileUrl(activeTile.preview)} alt={`视野 ${activeTile.id} 分割结果`} /><div><strong>视野 {activeTile.id}</strong><p>{labels[activeTile.status] || activeTile.status}</p><p>绿色轮廓：进入分析<br />橙色轮廓：已跳过</p>{activeTile.error && <p>{activeTile.error}</p>}</div></div>}
          <div className="semTableWrap"><table><thead><tr><th>颗粒 / 视野</th><th>长度 μm</th><th>主要元素</th><th>状态 / 原因</th></tr></thead><tbody>{task?.particles.slice(-100).map(p => <tr key={p.sequence}><td>#{p.sequence} <small> / {p.tileId}</small></td><td>{p.lengthUm?.toFixed(2) || '—'}</td><td>{p.spectrum?.elements.slice(0, 3).map(e => `${e.symbol} ${e.value.toFixed(1)}%`).join(' · ') || '—'}</td><td>{labels[p.status] || p.status}{p.reason && <small>{p.reason}</small>}</td></tr>)}</tbody></table>{!task?.particles.length && <div className="semNoResults">分析开始后，颗粒尺寸与能谱结果会显示在这里。</div>}</div>
          {(task?.stats?.particleCount ?? task?.particles.length ?? 0) > 100 && <p className="semHint">展示最近 100 条，完整结果保存在报告中。</p>}
        </section>
      </div>
      <aside className="semAside">
        {task?.plan && task.status !== 'DRAFT' && <section className="semCard"><div className="semCardHead"><h2>正式报告</h2></div><div className="semExports"><p>全部扫描结束后自动生成；暂停或异常时先保存数据，可按需导出当前结果。</p>{task.reportError && <p role="alert">{task.reportError}</p>}<button disabled={running || busy || task.status === 'RUNNING'} onClick={() => void generateReport()}><FileText size={15} />{reporting ? '正在生成…' : '生成当前结果报告'}</button></div></section>}
        <section className="semCard"><div className="semCardHead"><h2>扫描设置</h2></div><fieldset disabled={!editable}>
          <label>样品名称<input placeholder="输入样品名称" value={name} onChange={e => { setName(e.target.value); update() }} /></label>
          <label>区域形状<div className="semShape"><button className={shape === 'square' ? 'selected' : ''} onClick={() => { setShape('square'); if (roi) setRoi({ ...roi, shape: 'square' }); update() }}>正方形</button><button className={shape === 'circle' ? 'selected' : ''} onClick={() => { setShape('circle'); if (roi) setRoi({ ...roi, shape: 'circle' }); update() }}>圆形</button></div></label>
          <label>单视野 FOV（μm）<input type="number" min="1" value={field} onChange={e => { setField(Number(e.target.value)); update() }} /><small>以实际视野回读为准；越小越适合小颗粒。</small></label>
          <div className="semFieldPair"><label>图像分辨率<select value={size} onChange={e => { setSize(Number(e.target.value)); update() }}>{[512, 1024, 2048].map(s => <option key={s} value={s}>{s} × {s}</option>)}</select></label><label>视野重叠（%）<input type="number" min="0" max="40" value={Math.round(overlap*100)} onChange={e => { setOverlap(Number(e.target.value)/100); update() }} /></label></div>
          <label>最小颗粒长度（μm）<input type="number" min="0" value={minimum} onChange={e => { setMinimum(Number(e.target.value)); update() }} /></label>
          <label>颗粒分割模型<select value={segmentation} onChange={e => { setSegmentation(e.target.value as 'yolo' | 'sam3'); update() }}><option value="yolo">YOLO（本地）</option><option value="sam3">SAM3（远程）</option></select></label>
          {segmentation === 'sam3' && <>
            <label>SAM3 服务地址<input value={samUrl} onChange={e => { setSamUrl(e.target.value); update() }} /></label>
            <label>分割提示词<input value={samTexts} onChange={e => { setSamTexts(e.target.value); update() }} /><small>多个提示词用逗号分隔。</small></label>
            <label>SAM3 置信度阈值<input type="number" min="0.001" max="1" step="0.01" value={samScore} onChange={e => { setSamScore(Number(e.target.value)); update() }} /><small>默认 0.1；降低阈值会增加候选，也可能增加误检。服务异常时停止分析。</small></label>
          </>}
          <label className="semCheck"><input type="checkbox" checked={eds} onChange={e => { setEds(e.target.checked); update() }} /><span>分割后自动采集 EDS<small>使用能谱服务的自动采集参数</small></span></label>
        </fieldset>
          <div className="semSettingsFoot"><p>{task?.request.gunOffAfterTask === false ? '任务结束后保持电子枪状态' : '结束或暂停后自动关枪'}</p><p>边缘截断与归属不明确的颗粒会跳过并保留记录。</p>
            {editable && <button className="semPreviewButton" disabled={!roi || busy || roi.x1 <= roi.x0} onClick={() => void previewPlan()}>预览扫描网格</button>}
            {reporting ? <button className="semStart" disabled>正在生成完整报告…</button> : running ? <button className="semStart" disabled={pauseRequested} onClick={() => void pause()}><CirclePause size={16} />{pauseRequested ? '等待当前视野完成' : '暂停扫描'}</button> : <button className="semStart" disabled={busy || !task || !roi || roi.x1 <= roi.x0 || task.status.startsWith('COMPLETED')} onClick={() => void start()}><Play size={16} />{task && ['STOPPED', 'FAILED', 'RUNNING'].includes(task.status) ? '继续扫描' : '开始自动分析'}</button>}
          </div>
        </section>
        {task?.reports && <section className="semCard"><div className="semCardHead"><h2>结果文件</h2></div><div className="semExports">{task.reports.pdf && <button onClick={() => void openResult('pdf')}><FileText size={15} />打开正式 PDF 报告</button>}{task.reports.timingHtml && <button onClick={() => void openResult('timingHtml')}><FileText size={15} />打开时间开销报告</button>}<button onClick={() => void openResult('html')}><FileText size={15} />打开网页版报告</button><button onClick={() => void openResult('csv')}>导出数据 CSV</button><button onClick={() => void openPath(task.taskDirectory!).catch(e => setError(String(e)))}><FolderOpen size={15} />打开任务目录</button></div></section>}
      </aside>
    </div>
  </div>
}
