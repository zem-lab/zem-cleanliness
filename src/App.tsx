import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import type { LucideIcon } from 'lucide-react'
import {
  Activity,
  Bot,
  Check,
  ChevronRight,
  CircleStop,
  Clock3,
  Copy,
  Database,
  FileJson,
  FileText,
  Filter,
  FolderOpen,
  Home,
  Info,
  KeyRound,
  LockKeyhole,
  Microscope,
  MoreHorizontal,
  Play,
  Plus,
  RefreshCw,
  ScanLine,
  Search,
  Settings,
  Save,
  RotateCcw,
  TriangleAlert,
  Trash2,
  X,
} from 'lucide-react'
import zeptoolsLogo from './assets/zeptools-logo.png'
import { chooseOpticalJson, fileName, formatBytes, getRuntime, logClientEvent, openPath, request, waitForJob } from './api'
import type { DeviceResource, JobState, ModelResource, OpticalPreview, TaskManifest, TaskSummary, PreflightReport } from './api'
import { completedTaskSummary, gunShutdownSummary } from './taskOutcome'
import { createErrorRecord, diagnoseError, errorNotice, loadErrorRecords, mergeErrorRecord, saveErrorRecords } from './errors'
import type { ErrorRecord, NoticeInput, NoticeKind } from './errors'
import './App.css'
import SemDirectPage from './SemDirectPage'
import { ChatPanel, ChatToggleButton } from './ChatPanel'

type PageId = 'home' | 'choose-mode' | 'sem-direct' | 'task' | 'calibration' | 'running' | 'reports' | 'settings'
type WorkflowStep = 1 | 2 | 3 | 4
type CalibrationMode = 'multi_particle' | 'ab_points' | 'ab_particle' | 'ab_manual'
const calibrationModeLabels: Record<CalibrationMode, string> = {
  multi_particle: '经验矩阵', ab_points: 'AB 标定', ab_particle: 'AB 导航 + 颗粒精校准', ab_manual: '手动 AB 标定',
}

type NavItem = {
  id: PageId
  label: string
  icon: LucideIcon
}

type Notify = (notice: NoticeInput) => void

type AppNotice = {
  kind: NoticeKind
  title?: string
  message: string
  details?: string
}

const navigation: NavItem[] = [
  { id: 'home', label: '首页', icon: Home },
  { id: 'reports', label: '任务记录', icon: FileText },
  { id: 'settings', label: '系统设置', icon: Settings },
]

const stepPages: Record<WorkflowStep, PageId> = {
  1: 'task',
  2: 'calibration',
  3: 'running',
  4: 'reports',
}

const acceptedCalibrationStatuses = new Set(['CALIBRATION_ACCEPTED', 'RUNNING', 'COMPLETED', 'COMPLETED_WITH_FAILURES', 'STOPPED', 'FAILED'])
const completedTaskStatuses = new Set(['COMPLETED', 'COMPLETED_WITH_FAILURES'])
const inputEditableStatuses = new Set(['DRAFT', 'CALIBRATION_PENDING_REVIEW', 'CALIBRATION_ACCEPTED', 'AB_POINTS_PENDING_SEM_REFINEMENT'])

const emptyPreview: OpticalPreview | null = null

function App() {
  const [page, setPage] = useState<PageId>('home')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [notice, setNotice] = useState<AppNotice | null>(null)
  const [errors, setErrors] = useState<ErrorRecord[]>(loadErrorRecords)
  const [errorCenterOpen, setErrorCenterOpen] = useState(false)
  const [preview, setPreview] = useState<OpticalPreview | null>(emptyPreview)
  const [task, setTask] = useState<TaskManifest | null>(null)
  const [tasks, setTasks] = useState<TaskSummary[]>([])
  const [devices, setDevices] = useState<DeviceResource[]>([])
  const [models, setModels] = useState<ModelResource[]>([])
  const [backendReady, setBackendReady] = useState(false)
  const [apiBase, setApiBase] = useState('http://127.0.0.1:8765')
  const [logDirectory, setLogDirectory] = useState('')
  const [appVersion, setAppVersion] = useState('')
  const [agentControlled, setAgentControlled] = useState(false)
  const [selectedMode, setSelectedMode] = useState<CalibrationMode>('multi_particle')
  const [job, setJob] = useState<JobState | null>(null)
  const [engineerOpen, setEngineerOpen] = useState(false)

  const notify = useCallback<Notify>((input) => {
    const normalized = typeof input === 'string'
      ? { kind: 'success' as const, message: input }
      : { kind: input.kind || 'info', message: input.message, context: input.context, details: input.details }
    if (normalized.kind === 'error') {
      logClientEvent('error', 'user_visible_error', { context: normalized.context || '', message: normalized.details || normalized.message })
      const record = createErrorRecord(normalized.details || normalized.message, normalized.context || '')
      setErrors((current) => mergeErrorRecord(current, record))
      setNotice({ kind: 'error', title: record.title, message: record.summary, details: record.technicalDetails })
      return
    }
    setNotice({ kind: normalized.kind, message: normalized.message })
  }, [])

  useEffect(() => {
    if (!notice || notice.kind === 'error') return
    const timer = window.setTimeout(() => setNotice(null), 3000)
    return () => window.clearTimeout(timer)
  }, [notice])

  useEffect(() => { saveErrorRecords(errors) }, [errors])

  useEffect(() => {
    let cancelled = false
    void Promise.all([
      request<{ tasks: TaskSummary[] }>('GET', '/tasks'),
      request<{ devices: DeviceResource[] }>('GET', '/devices'),
      request<{ models: ModelResource[] }>('GET', '/models'),
    ]).then(([taskResult, deviceResult, modelResult]) => {
      if (cancelled) return
      setTasks(taskResult.tasks)
      setDevices(deviceResult.devices)
      setModels(modelResult.models)
      setBackendReady(true)
    }).catch((error: unknown) => {
      if (!cancelled) notify(errorNotice(error, '应用初始化'))
    })
    return () => { cancelled = true }
  }, [notify])

  useEffect(() => {
    void getRuntime()
      .then((runtime) => {
        setApiBase(runtime.backendBaseUrl)
        setAppVersion(runtime.version)
        setLogDirectory(runtime.logDirectory)
        setAgentControlled(Boolean(runtime.agentControlled))
        logClientEvent('info', 'runtime_ready', { version: runtime.version, packaged: runtime.packaged, logDirectory: runtime.logDirectory })
      })
      .catch((error: unknown) => notify(errorNotice(error, '读取运行环境')))
  }, [notify])

  useEffect(() => {
    const onAgentControl = (event: Event) => {
      setAgentControlled(Boolean((event as CustomEvent<boolean>).detail))
    }
    window.addEventListener('zem-agent-control', onAgentControl)
    return () => window.removeEventListener('zem-agent-control', onAgentControl)
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'l') {
        event.preventDefault()
        setEngineerOpen(true)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  useEffect(() => {
    const onWindowError = (event: ErrorEvent) => {
      logClientEvent('error', 'window_error', { message: event.message, filename: event.filename, line: event.lineno, column: event.colno })
      notify(errorNotice(event.error || event.message, '前端运行'))
    }
    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      logClientEvent('error', 'unhandled_promise_rejection', { reason: event.reason instanceof Error ? event.reason.message : String(event.reason) })
      notify(errorNotice(event.reason, '前端异步操作'))
    }
    window.addEventListener('error', onWindowError)
    window.addEventListener('unhandledrejection', onUnhandledRejection)
    return () => {
      window.removeEventListener('error', onWindowError)
      window.removeEventListener('unhandledrejection', onUnhandledRejection)
    }
  }, [notify])

  const refreshDevices = async () => {
    try {
      const result = await request<{ devices: DeviceResource[] }>('GET', '/devices')
      setDevices(result.devices)
      return result.devices
    } catch (error) {
      notify(errorNotice(error, '刷新设备状态'))
      return []
    }
  }

  useEffect(() => {
    if (page !== 'home') return
    void request<{ tasks: TaskSummary[] }>('GET', '/tasks').then(result => setTasks(result.tasks)).catch(error => notify(errorNotice(error, '刷新任务记录')))
  }, [page, notify])

  const go = (next: PageId) => {
    setPage(next)
    window.scrollTo({ top: 0 })
  }

  const startNewTask = () => {
    setTask(null)
    setPreview(null)
    setJob(null)
    setSelectedMode('multi_particle')
    go('choose-mode')
  }

  const canNavigateStep = (step: WorkflowStep): boolean => {
    if (step === 1) return !task || inputEditableStatuses.has(String(task.status || ''))
    if (step === 2) return Boolean(task)
    if (step === 3) return Boolean(task && acceptedCalibrationStatuses.has(String(task.status || '')))
    return Boolean(task && completedTaskStatuses.has(String(task.status || '')))
  }

  const isStepComplete = (step: WorkflowStep): boolean => {
    if (step === 1) return Boolean(task)
    if (step === 2) return Boolean(task && acceptedCalibrationStatuses.has(String(task.status || '')))
    return Boolean(task && completedTaskStatuses.has(String(task.status || '')))
  }

  const goStep = (step: WorkflowStep) => {
    if (!canNavigateStep(step)) {
      notify({ kind: 'info', message: step === 1 ? '当前任务已开始检测，不能再修改导入和筛选条件' : step === 2 ? '请先导入数据并创建任务' : step === 3 ? '请先确认坐标标定结果' : '检测完成后才能查看报告' })
      return
    }
    go(stepPages[step])
  }

  const openTask = async (summary: TaskSummary) => {
    try {
      const manifest = await request<TaskManifest>('GET', `/tasks/${summary.id}`)
      setTask(manifest)
      setPreview(null)
      if (manifest.mode === 'sem_direct') {
        go('sem-direct')
        return
      }
      const calibration = manifest.calibration as { strategy?: string } | undefined
      const requestSnapshot = manifest.request as { mode?: string } | undefined
      const strategy = requestSnapshot?.mode === 'ab_manual' ? 'ab_manual' : calibration?.strategy || requestSnapshot?.mode
      setSelectedMode(strategy === 'ab_manual' ? 'ab_manual' : strategy === 'ab_particle' ? 'ab_particle' : strategy === 'ab_points' ? 'ab_points' : 'multi_particle')
      const status = String(manifest.status || '')
      if (manifest.error) notify(errorNotice(manifest.error, `任务 ${summary.id}`))
      go(['DRAFT', 'AB_POINTS_PENDING_SEM_REFINEMENT', 'CALIBRATION_RUNNING', 'CALIBRATION_PENDING_REVIEW'].includes(status) ? 'calibration' : 'running')
    } catch (error) {
      notify(errorNotice(error, '打开任务'))
    }
  }

  return (
    <div className="app">
      {agentControlled && <div className="agentControlFrame" aria-hidden="true" />}
      <header className="appHeader">
        {agentControlled && <div className="agentControlBadge" role="status"><Bot size={15} /><i className="agentControlDot" />当前软件正在被 Agent 控制</div>}
        <div className="brandArea">
          <img src={zeptoolsLogo} alt="ZEPTOOLS" />
          <span className="brandDivider" />
          <div className="productName">
            <strong>清洁度检测</strong>
            <span>Cleanliness Inspection</span>
          </div>
        </div>
        <div className="headerActions">
          <ChatToggleButton onClick={() => setChatOpen((open) => !open)} />
          <button className="connectionButton" type="button" onClick={() => setDrawerOpen(true)}>
            <StatusDot active={devices.some((device) => device.id === 'microscope' && device.status === 'connected')} /> 电镜{deviceLabel(devices.find((device) => device.id === 'microscope'))}
          </button>
          <button className="connectionButton" type="button" onClick={() => setDrawerOpen(true)}>
            <StatusDot active={devices.some((device) => device.id === 'eds' && device.status === 'connected')} /> 能谱{deviceLabel(devices.find((device) => device.id === 'eds'))}
          </button>
          <button className={`headerIconButton errorHistoryButton ${errors.length ? 'hasErrors' : ''}`} type="button" aria-label={`错误记录，共 ${errors.length} 条`} title="错误记录" onClick={() => setErrorCenterOpen(true)}>
            <TriangleAlert size={17} />{errors.length > 0 && <span>{Math.min(errors.length, 99)}</span>}
          </button>
          <button className="headerIconButton" type="button" aria-label="更多"><MoreHorizontal size={18} /></button>
        </div>
      </header>

      <div className={chatOpen ? 'appBody withChat' : 'appBody'}>
        <aside className="sidebar">
          <button className="createButton" type="button" onClick={startNewTask}>
            <Plus size={16} /> 新建检测任务
          </button>
          <nav aria-label="主导航">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <button
                  key={item.id}
                  type="button"
                  className={page === item.id ? 'navItem active' : 'navItem'}
                  onClick={() => go(item.id)}
                >
                  <Icon size={17} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </nav>
          <div className="sidebarBottom">
            <div className="licenseInfo">
              <Check size={15} />
              <div><strong>授权有效</strong><span>版本 {appVersion || '—'}</span></div>
            </div>
          </div>
        </aside>

        <main className="content">
          {page === 'home' && <HomePage go={go} startNewTask={startNewTask} openTask={(summary) => void openTask(summary)} tasks={tasks} backendReady={backendReady} />}
          {page === 'choose-mode' && <div className="page"><PageTitle title="新建检测任务" description="选择颗粒来源，开始清洁度分析" /><div className="taskModeChoice"><button onClick={() => go('task')}><FileJson size={30} /><strong>光镜关联分析</strong><span>导入光镜颗粒数据，通过坐标标定找到目标颗粒，完成 SEM 精定位与能谱分析。</span><span>导入数据 → 坐标标定 → 分析</span></button><button onClick={() => go('sem-direct')}><ScanLine size={30} /><strong>直接 SEM 分析</strong><span>在导航图上画出区域，自动扫描、分割颗粒并采集能谱，无需光镜数据。</span><span>圈选区域 → 自动拍图 → 分割与采谱</span></button></div></div>}
          {page === 'sem-direct' && <SemDirectPage key={task?.mode === 'sem_direct' ? task.taskId : 'new'} initialTask={task} apiBase={apiBase} onBack={() => go('home')} />}
          {page === 'task' && <TaskPage task={task} preview={preview} setPreview={setPreview} setTask={setTask} next={() => go('calibration')} notify={notify} mode={selectedMode} onStep={goStep} canNavigate={canNavigateStep} isComplete={isStepComplete} />}
          {page === 'calibration' && <CalibrationPage task={task} setTask={setTask} mode={selectedMode} setMode={setSelectedMode} next={() => go('running')} notify={notify} apiBase={apiBase} models={models} onStep={goStep} canNavigate={canNavigateStep} isComplete={isStepComplete} />}
          {page === 'running' && <RunningPage task={task} setTask={setTask} job={job} setJob={setJob} notify={notify} apiBase={apiBase} onStep={goStep} canNavigate={canNavigateStep} isComplete={isStepComplete} />}
          {page === 'reports' && <ReportsPage tasks={tasks} notify={notify} openTask={(summary) => void openTask(summary)} />}
          {page === 'settings' && <SettingsPage devices={devices} models={models} refreshDevices={refreshDevices} notify={notify} logDirectory={logDirectory} />}
        </main>
        {chatOpen && <ChatPanel close={() => setChatOpen(false)} />}
      </div>

      <footer className="statusBar">
        <div><StatusDot active={backendReady} /><span>{backendReady ? '后端已连接' : '后端连接中'}</span><i /> <StatusDot /><span>License</span><i /> <StatusDot active={devices.some((device) => device.id === 'microscope' && device.status === 'connected')} /><span>电镜</span><i /> <StatusDot active={devices.some((device) => device.id === 'eds' && device.status === 'connected')} /><span>能谱</span></div>
        <div><span>任务目录由软件自动管理</span><i /><span>v{appVersion || '—'}</span></div>
      </footer>
      {drawerOpen && <DeviceDrawer devices={devices} refresh={() => void refreshDevices()} close={() => setDrawerOpen(false)} />}
      <EngineerPanel open={engineerOpen} close={() => setEngineerOpen(false)} notify={notify} />
      {errorCenterOpen && <ErrorCenter records={errors} close={() => setErrorCenterOpen(false)} clear={() => setErrors([])} />}
      {notice && <NoticeToast notice={notice} close={() => setNotice(null)} openErrors={() => setErrorCenterOpen(true)} />}
    </div>
  )
}

function StatusDot({ active = true }: { active?: boolean }) {
  return <span className={active ? 'statusDot' : 'statusDot inactive'} />
}

function PageTitle({ title, description, action }: { title: string; description: string; action?: React.ReactNode }) {
  return (
    <div className="pageTitle">
      <div><h1>{title}</h1><p>{description}</p></div>
      {action && <div>{action}</div>}
    </div>
  )
}

function PrimaryButton({ children, onClick, disabled }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return <button className="primaryButton" type="button" onClick={onClick} disabled={disabled}>{children}</button>
}

function SecondaryButton({ children, onClick, disabled }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return <button className="secondaryButton" type="button" onClick={onClick} disabled={disabled}>{children}</button>
}

function Section({ title, extra, children, className = '' }: { title: string; extra?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={`section ${className}`}>
      <div className="sectionHeader"><h2>{title}</h2>{extra}</div>
      <div className="sectionBody">{children}</div>
    </section>
  )
}

function NoticeToast({ notice, close, openErrors }: { notice: AppNotice; close: () => void; openErrors: () => void }) {
  const Icon = notice.kind === 'error' ? TriangleAlert : notice.kind === 'success' ? Check : Info
  return (
    <div className={`toast ${notice.kind}`} role={notice.kind === 'error' ? 'alert' : 'status'} aria-live={notice.kind === 'error' ? 'assertive' : 'polite'}>
      <Icon size={17} />
      <div className="toastContent">
        {notice.title && <strong>{notice.title}</strong>}
        <span>{notice.message}</span>
        {notice.kind === 'error' && <button type="button" onClick={openErrors}>查看错误记录和技术详情</button>}
      </div>
      <button className="toastClose" type="button" aria-label="关闭提示" onClick={close}><X size={15} /></button>
    </div>
  )
}

function ErrorCenter({ records, close, clear }: { records: ErrorRecord[]; close: () => void; clear: () => void }) {
  return (
    <div className="errorCenterMask" role="presentation" onMouseDown={close}>
      <aside className="errorCenter" role="dialog" aria-modal="true" aria-label="错误记录" onMouseDown={(event) => event.stopPropagation()}>
        <div className="errorCenterHeader">
          <div><span>运行诊断</span><h2>错误记录</h2><p>保留最近 {records.length} 条错误，刷新应用后仍可查看。</p></div>
          <button type="button" aria-label="关闭错误记录" onClick={close}><X size={18} /></button>
        </div>
        <div className="errorCenterToolbar">
          <span><TriangleAlert size={15} />{records.length ? `${records.length} 条记录` : '暂无错误'}</span>
          {records.length > 0 && <button type="button" onClick={clear}><Trash2 size={14} />清空记录</button>}
        </div>
        <div className="errorRecordList">
          {records.length ? records.map((record) => (
            <article className="errorRecord" key={record.id}>
              <div className="errorRecordTitle">
                <span><TriangleAlert size={16} /></span>
                <div><strong>{record.title}</strong><small>{record.context || '未标记操作'} · {new Date(record.createdAt).toLocaleString('zh-CN')}{record.count > 1 ? ` · 重复 ${record.count} 次` : ''}</small></div>
              </div>
              <p>{record.summary}</p>
              <div className="errorGuidance">{record.guidance.map((item) => <span key={item}>{item}</span>)}</div>
              <details>
                <summary>技术详情</summary>
                <pre>{record.technicalDetails}</pre>
                <button type="button" onClick={() => copyText(record.technicalDetails)}><Copy size={13} />复制错误信息</button>
              </details>
            </article>
          )) : <div className="errorCenterEmpty"><Check size={24} /><strong>当前没有错误记录</strong><span>发生错误后会在这里保留诊断摘要和原始信息。</span></div>}
        </div>
      </aside>
    </div>
  )
}

function PersistentErrorPanel({ error, context }: { error: unknown; context: string }) {
  const diagnosis = diagnoseError(error, context)
  return (
    <section className="persistentError" role="alert">
      <span className="persistentErrorIcon"><TriangleAlert size={20} /></span>
      <div className="persistentErrorBody">
        <strong>{diagnosis.title}</strong>
        <p>{diagnosis.summary}</p>
        <div className="persistentErrorGuidance">{diagnosis.guidance.map((item) => <span key={item}>{item}</span>)}</div>
        <details><summary>查看技术详情</summary><pre>{diagnosis.technicalDetails}</pre><button type="button" onClick={() => copyText(diagnosis.technicalDetails)}><Copy size={13} />复制错误信息</button></details>
      </div>
    </section>
  )
}

function copyText(value: string): void {
  if (!navigator.clipboard) return
  void navigator.clipboard.writeText(value).catch(() => undefined)
}

function deviceLabel(device?: DeviceResource): string {
  if (!device) return '检测中'
  if (device.status === 'connected') return '已连接'
  if (device.status === 'busy') return '忙碌'
  if (device.status === 'offline') return '离线'
  return '未知'
}

function displayStatus(status: string): string {
  const values: Record<string, string> = {
    DRAFT: '待标定',
    CALIBRATING: '标定中',
    CALIBRATION_PENDING_REVIEW: '待确认',
    CALIBRATION_ACCEPTED: '待检测',
    AB_POINTS_PENDING_SEM_REFINEMENT: '待精定位',
    RUNNING: '检测中',
    COMPLETED: '已完成',
    COMPLETED_WITH_FAILURES: '已完成（部分颗粒未通过）',
    FAILED: '失败',
    STOPPED: '已停止',
  }
  return values[status] || status || '未知'
}

function HomePage({ go, startNewTask, openTask, tasks, backendReady }: { go: (page: PageId) => void; startNewTask: () => void; openTask: (task: TaskSummary) => void; tasks: TaskSummary[]; backendReady: boolean }) {
  const activeTask = tasks.find((item) => !['COMPLETED', 'COMPLETED_WITH_FAILURES', 'FAILED', 'STOPPED'].includes(item.status))
  return (
    <div className="page">
      <PageTitle
        title="首页"
        description="清洁度检测任务概览"
        action={<PrimaryButton onClick={startNewTask}><Plus size={15} />新建检测</PrimaryButton>}
      />

      <div className="homeGrid">
        <div className="homeMain">
          <Section title="当前任务" extra={activeTask ? <button className="linkButton" type="button" onClick={() => openTask(activeTask)}>查看详情 <ChevronRight size={14} /></button> : undefined}>
            {activeTask ? <div className="currentTask">
              <div className="currentTaskTop"><div><TaskStatus status={displayStatus(activeTask.status)} /><h3>{activeTask.input?.workpieceName || fileName(activeTask.input?.path || '未命名样品')}</h3><p>{activeTask.id} · {displayStatus(activeTask.status)}</p></div></div>
              <div className="currentTaskMeta"><span>光镜颗粒 <b>{activeTask.input?.particleCount || 0}</b></span><span>进入任务 <b>{activeTask.input?.selectedCount || activeTask.input?.particleCount || 0}</b></span></div>
            </div> : <div className="emptyState"><Microscope size={24} /><strong>{backendReady ? '暂无进行中的任务' : '正在连接本地检测服务'}</strong><span>{backendReady ? '导入光镜颗粒结果后即可开始检测' : '请稍候'}</span></div>}
          </Section>

          <Section title="最近任务" extra={<button className="linkButton" type="button" onClick={() => go('reports')}>全部记录 <ChevronRight size={14} /></button>}>
            <div className="simpleTable">
              <div className="tableHeader"><span>任务 / 样品</span><span>颗粒数</span><span>时间</span><span>状态</span><span /></div>
              {tasks.length ? tasks.slice(0, 8).map((task) => <button className="tableRow" type="button" key={task.id} onClick={() => ['COMPLETED', 'COMPLETED_WITH_FAILURES'].includes(task.status) ? go('reports') : openTask(task)}>
                <span><strong>{task.input?.workpieceName || fileName(task.input?.path || '未命名样品')}</strong><small>{task.id}</small></span><span>{task.input?.particleCount || 0}</span><span>{task.updatedAt ? new Date(task.updatedAt).toLocaleString('zh-CN') : '—'}</span><span><TaskStatus status={displayStatus(task.status)} /></span><ChevronRight size={15} />
              </button>) : <div className="emptyTable">暂无检测记录</div>}
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}

function DeviceSummary({ icon: Icon, name, detail, status = 'connected' }: { icon: LucideIcon; name: string; detail: string; status?: DeviceResource['status'] }) {
  return (
    <div className="deviceSummary"><span className="deviceIcon"><Icon size={19} /></span><div><strong>{name}</strong><span className={status === 'connected' ? '' : 'offlineText'}><StatusDot active={status === 'connected'} />{deviceLabel({ id: 'microscope', name, host: '', port: 0, status, detail })}</span><small>{detail}</small></div></div>
  )
}

function TaskStatus({ status }: { status: string }) {
  const type = status === '已完成' ? 'success' : ['检测中', '标定中', '待检测', '待确认', '待精定位'].includes(status) ? 'running' : 'warning'
  return <span className={`plainStatus ${type}`}><i />{status}</span>
}

function StepBar({ active, onStep, canNavigate, isComplete }: { active: WorkflowStep; onStep: (step: WorkflowStep) => void; canNavigate: (step: WorkflowStep) => boolean; isComplete: (step: WorkflowStep) => boolean }) {
  const steps = ['导入与筛选', '坐标标定', '检测运行', '查看报告']
  return (
    <nav className="stepBar" aria-label="检测任务步骤">
      {steps.map((step, index) => {
        const number = (index + 1) as WorkflowStep
        const current = number === active
        const complete = isComplete(number) && !current
        const enabled = canNavigate(number)
        return <Fragment key={step}>
          <button className={current ? 'step active' : complete ? 'step done' : enabled ? 'step available' : 'step'} type="button" onClick={() => onStep(number)} disabled={!enabled || current} aria-current={current ? 'step' : undefined}>
            <span>{complete ? <Check size={13} /> : number}</span>
            <strong>{step}</strong>
          </button>
          {index < steps.length - 1 && <i aria-hidden="true" />}
        </Fragment>
      })}
    </nav>
  )
}

function filterSignature(filters: Record<string, unknown>): string {
  return JSON.stringify(Object.entries(filters).filter(([, value]) => value !== null && value !== '').map(([key, value]) => [key, String(value)]).sort(([left], [right]) => left.localeCompare(right)))
}

function TaskPage({ task, preview, setPreview, setTask, next, notify, mode, onStep, canNavigate, isComplete }: { task: TaskManifest | null; preview: OpticalPreview | null; setPreview: (value: OpticalPreview | null) => void; setTask: (value: TaskManifest | null) => void; next: () => void; notify: Notify; mode: CalibrationMode; onStep: (step: WorkflowStep) => void; canNavigate: (step: WorkflowStep) => boolean; isComplete: (step: WorkflowStep) => boolean }) {
  const existingTaskId = String(task?.folderId || task?.taskId || '')
  const existingInputPath = String(task?.input?.path || '')
  const existingRequest = useMemo(() => (task?.request || {}) as Record<string, unknown>, [task])
  const existingSelection = useMemo(() => (task?.selection || {}) as Record<string, unknown>, [task])
  const existingFilters = useMemo(() => (existingSelection.filter || existingRequest.filters || {}) as Record<string, unknown>, [existingRequest, existingSelection])
  const initialFilterKey = Object.keys(existingFilters).find((key) => (key.startsWith('min') || key.startsWith('max')) && existingFilters[key] !== null && existingFilters[key] !== '') || ''
  const initialMetric = initialFilterKey === 'minEcdUm' ? 'ecd' : initialFilterKey === 'minAreaUm2' ? 'area' : 'length'
  const initialMinKey = initialMetric === 'length' ? 'minLengthUm' : initialMetric === 'ecd' ? 'minEcdUm' : 'minAreaUm2'
  const initialMaxKey = initialMetric === 'length' ? 'maxLengthUm' : initialMetric === 'ecd' ? 'maxEcdUm' : 'maxAreaUm2'
  const [inputPath, setInputPath] = useState(preview?.path || existingInputPath)
  const [minimum, setMinimum] = useState(existingTaskId ? (existingFilters[initialMinKey] == null ? '' : String(existingFilters[initialMinKey])) : '5.00')
  const [maximum, setMaximum] = useState(existingTaskId ? (existingFilters[initialMaxKey] == null ? '' : String(existingFilters[initialMaxKey])) : '120.00')
  const [metric, setMetric] = useState<'length' | 'ecd' | 'area'>(initialMetric)
  const [loading, setLoading] = useState(false)
  const filters = useMemo(() => {
    const suffix = metric === 'length' ? 'LengthUm' : metric === 'ecd' ? 'EcdUm' : 'AreaUm2'
    return { [`min${suffix}`]: minimum || null, [`max${suffix}`]: maximum || null }
  }, [maximum, metric, minimum])
  const [appliedFilterSignature, setAppliedFilterSignature] = useState(preview && existingTaskId ? filterSignature(existingFilters) : '')
  const filtersApplied = appliedFilterSignature === filterSignature(filters)

  const loadPreview = async (path = inputPath, filterOverride: Record<string, unknown> = filters) => {
    if (!path) return
    setLoading(true)
    try {
      const result = await request<OpticalPreview>('POST', '/optical/preview', { inputPath: path, filters: filterOverride })
      setInputPath(path)
      setPreview(result)
      setAppliedFilterSignature(filterSignature(filterOverride))
      notify(`已载入 ${fileName(path)}，筛选后 ${result.selectedCount} 个颗粒`)
    } catch (error) {
      notify(errorNotice(error, '导入光镜结果'))
    } finally {
      setLoading(false)
    }
  }

  const existingSequences = Array.isArray(existingSelection.sequences) ? existingSelection.sequences.map(Number) : []
  const currentSequences = preview?.particles.map((particle) => particle.sequence) || []
  const revisionChanged = Boolean(existingTaskId && (
    existingInputPath !== inputPath
    || filterSignature(existingFilters) !== filterSignature(filters)
    || JSON.stringify(existingSequences) !== JSON.stringify(currentSequences)
    || String(existingRequest.mode || mode) !== mode
  ))
  const requiresResetConfirmation = Boolean(task && revisionChanged && task.status !== 'DRAFT')

  useEffect(() => {
    if (preview || !existingInputPath) return
    const storedFilters = existingFilters
    const filterKey = Object.keys(storedFilters).find((key) => (key.startsWith('min') || key.startsWith('max')) && storedFilters[key] !== null && storedFilters[key] !== '') || ''
    const storedMetric = filterKey === 'minEcdUm' ? 'ecd' : filterKey === 'minAreaUm2' ? 'area' : 'length'
    const minKey = storedMetric === 'length' ? 'minLengthUm' : storedMetric === 'ecd' ? 'minEcdUm' : 'minAreaUm2'
    const maxKey = storedMetric === 'length' ? 'maxLengthUm' : storedMetric === 'ecd' ? 'maxEcdUm' : 'maxAreaUm2'
    setMetric(storedMetric)
    setMinimum(storedFilters[minKey] == null ? '' : String(storedFilters[minKey]))
    setMaximum(storedFilters[maxKey] == null ? '' : String(storedFilters[maxKey]))
    let cancelled = false
    setLoading(true)
    void request<OpticalPreview>('POST', '/optical/preview', { inputPath: existingInputPath, filters: storedFilters })
      .then((result) => {
        if (cancelled) return
        setInputPath(existingInputPath)
        setPreview(result)
        setAppliedFilterSignature(filterSignature(storedFilters))
      })
      .catch((error: unknown) => {
        if (!cancelled) notify(errorNotice(error, '恢复任务导入数据'))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [existingFilters, existingInputPath, notify, preview, setPreview])

  const selectFile = async () => {
    try {
      const path = await chooseOpticalJson()
      if (path) await loadPreview(path)
    } catch (error) {
      notify(errorNotice(error, '选择光镜文件'))
    }
  }

  const createTask = async () => {
    if (!preview || !inputPath) return
    if (requiresResetConfirmation && !window.confirm('修改导入或筛选条件会使当前坐标标定结果失效，需要重新标定。确定继续吗？')) return
    setLoading(true)
    try {
      const payload = {
        inputPath,
        filters,
        selectedSequences: preview.particles.map((particle) => particle.sequence),
        mode,
        edsEnabled: true,
        restoreStartPosition: true,
        confirmReset: requiresResetConfirmation,
      }
      const result = await request<TaskManifest>('POST', task ? `/tasks/${existingTaskId}/input` : '/tasks', payload)
      setTask(result)
      next()
    } catch (error) {
      notify(errorNotice(error, task ? '更新检测任务' : '创建检测任务'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <PageTitle title="导入与筛选" description={task ? '调整当前任务的光镜数据和检测范围' : '导入光镜颗粒结果并设置检测范围'} />
      <StepBar active={1} onStep={onStep} canNavigate={canNavigate} isComplete={isComplete} />
      <div className="taskLayout">
        <div className="taskMain">
          <Section title="光镜结果">
            <button className="fileRow" type="button" onClick={selectFile} disabled={loading}>
              <span className="fileTypeIcon"><FileJson size={21} /></span>
              <span><strong>{inputPath ? fileName(inputPath) : '请选择光镜颗粒 JSON'}</strong><small>{preview ? `${preview.particleCount} 个有效颗粒 · ${preview.coordinateSpace || '未知坐标系'}` : '软件会校验颗粒、坐标和图片数据'}</small></span>
              <span className="fileAction">{inputPath ? '更换文件' : '选择文件'}</span>
            </button>
            <div className="formGrid threeColumns">
              <Field label="样品名称"><input value={preview?.workpieceName || ''} readOnly placeholder="从光镜文件读取" /></Field>
              <Field label="项目编号"><input value={preview?.projectId || ''} readOnly placeholder="从光镜文件读取" /></Field>
              <Field label="工件编号"><input value={preview?.workpieceId || ''} readOnly placeholder="从光镜文件读取" /></Field>
            </div>
          </Section>

          <Section title="颗粒筛选" extra={<span className="sectionNote">已选 {preview?.selectedCount || 0} / {preview?.particleCount || 0}</span>}>
            <div className="filterToolbar">
              <div className="tabs"><button className={metric === 'length' ? 'active' : ''} type="button" onClick={() => setMetric('length')}>长度</button><button className={metric === 'ecd' ? 'active' : ''} type="button" onClick={() => setMetric('ecd')}>等效圆直径</button><button className={metric === 'area' ? 'active' : ''} type="button" onClick={() => setMetric('area')}>面积</button></div>
              <button className="filterPreset" type="button" onClick={() => void loadPreview()} disabled={!inputPath || loading}><Filter size={14} />应用筛选</button>
            </div>
            <div className="formGrid filterFields">
              <Field label="最小值"><div className="unitInput"><input value={minimum} onChange={(event) => setMinimum(event.target.value)} inputMode="decimal" /><span>{metric === 'area' ? 'μm²' : 'μm'}</span></div></Field>
              <Field label="最大值"><div className="unitInput"><input value={maximum} onChange={(event) => setMaximum(event.target.value)} inputMode="decimal" /><span>{metric === 'area' ? 'μm²' : 'μm'}</span></div></Field>
            </div>
            <div className="estimate"><Clock3 size={14} />检测耗时取决于颗粒数量、采图次数和 EDS 采集时间</div>
          </Section>

          <Section title="颗粒空间分布" extra={<span className="sectionNote">AB 坐标 · μm</span>}>
            <ParticleMap particles={preview?.particles || []} />
          </Section>
        </div>
        <aside className="taskAside">
          <Section title="任务摘要">
            <div className="summaryRows">
              <SummaryRow label="光镜颗粒" value={String(preview?.particleCount || 0)} />
              <SummaryRow label="进入任务" value={String(preview?.selectedCount || 0)} />
              <SummaryRow label="尺寸范围" value={`${minimum || '不限'}–${maximum || '不限'} ${metric === 'area' ? 'μm²' : 'μm'}`} />
              <SummaryRow label="能谱分析" value="启用" />
              <SummaryRow label="标定方式" value={calibrationModeLabels[mode]} />
            </div>
          </Section>
          <div className="asideActions"><PrimaryButton onClick={() => void createTask()} disabled={!preview || preview.selectedCount === 0 || loading || !filtersApplied}>下一步：坐标标定 <ChevronRight size={15} /></PrimaryButton></div>
        </aside>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="field"><span>{label}</span>{children}</label>
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>
}

function ParticleMap({ particles }: { particles: OpticalPreview['particles'] }) {
  const points = useMemo(() => {
    if (!particles.length) return []
    const xs = particles.map((particle) => particle.abPositionUm.x)
    const ys = particles.map((particle) => particle.abPositionUm.y)
    const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys)
    const rangeX = Math.max(maxX - minX, 1)
    const rangeY = Math.max(maxY - minY, 1)
    const span = Math.max(rangeX, rangeY)
    const offsetX = (span - rangeX) / 2
    const offsetY = (span - rangeY) / 2
    return particles.map((particle) => [8 + 84 * (offsetX + particle.abPositionUm.x - minX) / span, 8 + 84 * (offsetY + particle.abPositionUm.y - minY) / span, particle.sequence] as const)
  }, [particles])
  return (
    <div className="particleMap">
      <div className="particleMapCanvas">
        <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" aria-label={`筛选后颗粒空间分布，共 ${particles.length} 个颗粒`}>
          <defs><pattern id="particle-grid" width="12.5" height="12.5" patternUnits="userSpaceOnUse"><path d="M 12.5 0 L 0 0 0 12.5" /></pattern></defs>
          <rect width="100" height="100" className="particleMapGrid" />
          {points.map(([x, y, sequence]) => <circle key={sequence} cx={x} cy={y} r=".72" />)}
        </svg>
      </div>
      <div className="mapLegend"><span><i className="blue" />筛选后颗粒</span><span>{particles.length} 个 · 横纵坐标等比例</span></div>
    </div>
  )
}

type CalibrationGroup = {
  groupIndex?: number
  anchorSequences?: number[]
  validationSequences?: number[]
  result?: {
    validation?: { meanErrorUm?: number; maxErrorUm?: number; rmsErrorUm?: number }
    acceptance?: { withinLimits?: boolean }
    equation?: { stageX?: string; stageY?: string }
  }
}

function CalibrationPage({ task, setTask, mode, setMode, next, notify, apiBase, models, onStep, canNavigate, isComplete }: { task: TaskManifest | null; setTask: (value: TaskManifest | null) => void; mode: CalibrationMode; setMode: (value: CalibrationMode) => void; next: () => void; notify: Notify; apiBase: string; models: ModelResource[]; onStep: (step: WorkflowStep) => void; canNavigate: (step: WorkflowStep) => boolean; isComplete: (step: WorkflowStep) => boolean }) {
  const [selected, setSelected] = useState(0)
  const [job, setJob] = useState<JobState | null>(null)
  const [busy, setBusy] = useState(false)
  const taskId = String(task?.folderId || task?.taskId || '')
  const calibration = task?.calibration as { groups?: CalibrationGroup[] } | undefined
  const groups = calibration?.groups || []
  const result = groups[selected]?.result
  const taskStatus = String(task?.status || '')
  const canStartCalibration = taskStatus === 'DRAFT'
  const canAcceptCalibration = taskStatus === 'CALIBRATION_PENDING_REVIEW'
  const startCalibration = async () => {
    if (!taskId) return notify('请先创建检测任务')
    if (!canStartCalibration) return notify({ kind: 'info', message: '当前任务状态不能重新开始标定' })
    setBusy(true)
    try {
      const started = await request<JobState>('POST', `/tasks/${taskId}/calibration`, {})
      setJob(started)
      const completed = await waitForJob(started.id, setJob)
      const manifest = completed.result || await request<TaskManifest>('GET', `/tasks/${taskId}`)
      setTask(manifest)
      if (manifest.error) notify(errorNotice(manifest.error, '多颗粒标定'))
      else if (manifest.status === 'CALIBRATION_PENDING_REVIEW') notify('标定完成，请审核方案')
      else if (String(manifest.status || '').includes('FAILED')) notify(errorNotice(`标定任务状态：${displayStatus(String(manifest.status))}`, '多颗粒标定'))
      else notify({ kind: 'info', message: String(manifest.status || '标定结束') })
    } catch (error) {
      notify(errorNotice(error, '多颗粒标定'))
    } finally {
      setBusy(false)
    }
  }
  const accept = async () => {
    if (!taskId) return
    setBusy(true)
    try {
      const started = await request<JobState>('POST', `/tasks/${taskId}/review`, { accepted: true, selectedGroup: selected })
      const completed = await waitForJob(started.id, setJob)
      setTask(completed.result || null)
      next()
    } catch (error) {
      notify(errorNotice(error, '确认标定结果'))
    } finally { setBusy(false) }
  }
  return (
    <div className="page">
      <PageTitle title="坐标标定" description="确认光镜坐标到 SEM 样品台坐标的映射" />
      <StepBar active={2} onStep={onStep} canNavigate={canNavigate} isComplete={isComplete} />
      {task?.error && <PersistentErrorPanel error={task.error} context="坐标标定" />}
      <div className="modeTabs">{(Object.keys(calibrationModeLabels) as CalibrationMode[]).map((value) => <button key={value} className={mode === value ? 'active' : ''} type="button" onClick={() => setMode(value)} disabled={busy || Boolean(task?.status && task.status !== 'DRAFT')}>{calibrationModeLabels[value]}</button>)}</div>
      <div className="verificationGuide"><Info size={16} /><span>{mode === 'ab_manual' ? '直接输入电镜样品台 A/B 坐标（mm），复用 AB 坐标变换，跳过导航图识别和自动找孔，再实拍验证颗粒。' : mode === 'ab_particle' ? '适用于电镜下看不到 AB 孔的样品。AB 导航只作粗定位，再用本地颗粒匹配、实测像素轴与稳健拟合精校准；独立实拍验证通过后才能确认。不依赖 GPT 或在线服务。' : mode === 'ab_points' ? '导航图确定 A/B 后，在电镜中寻找孔中心完成标定。适用于 SEM 下孔位可见的样品。' : '以已保存的经验矩阵预定位，保留多颗粒匹配与分组验证。更换样品后仍需本次标定验收。'}</span></div>
      {mode === 'multi_particle' ? (
        <div className="calibrationLayout">
          <div className="calibrationMain">
            <Section title="标定进度" extra={<span className="sectionNote">{job?.message || displayStatus(String(task?.status || 'DRAFT'))}</span>}>
              {job && ['queued', 'running'].includes(job.status) ? <div className="jobProgress"><div className="progress"><span style={{ width: `${job.total ? Math.round(job.progress / job.total * 100) : 8}%` }} /></div><span>{job.message}</span></div> : <div className="emptyState"><ScanLine size={24} /><strong>{groups.length ? '标定证据已生成' : '准备执行多颗粒标定'}</strong><span>{groups.length ? '请选择误差最小且验收通过的方案' : '软件将自动选择空间分散的锚点和验证颗粒'}</span></div>}
            </Section>
            <Section title="匹配证据">
              {groups.length ? <div className="evidenceList">{(groups[selected]?.anchorSequences || []).map((sequence) => <EvidenceRow key={sequence} sequence={sequence} taskId={taskId} group={selected} apiBase={apiBase} />)}</div> : <div className="emptyState compact"><Info size={20} /><span>执行标定后显示光镜与 SEM 匹配证据</span></div>}
            </Section>
          </div>
          <aside className="calibrationAside">
            <Section title="标定方案">
              <div className="groupList">
                {groups.length ? groups.map((group, index) => (
                  <button className={selected === index ? 'group active' : 'group'} type="button" key={index} onClick={() => setSelected(index)}>
                    <span className="radio"><i /></span>
                    <span><strong>第 {index + 1} 组 {index === 0 && <em>候选</em>}</strong><small>平均误差 {formatMetric(group.result?.validation?.meanErrorUm)} μm</small></span>
                  </button>
                )) : <div className="emptyTable">尚无标定方案</div>}
              </div>
            </Section>
            <Section title="计算结果">
              <div className="resultRows"><SummaryRow label="拟合 RMS" value={`${formatMetric(result?.validation?.rmsErrorUm)} μm`} /><SummaryRow label="平均误差" value={`${formatMetric(result?.validation?.meanErrorUm)} μm`} /><SummaryRow label="最大误差" value={`${formatMetric(result?.validation?.maxErrorUm)} μm`} /><SummaryRow label="验收结果" value={result ? (result.acceptance?.withinLimits ? '通过' : '未通过') : '—'} /></div>
              {result?.equation && <div className="formula">{result.equation.stageX}<br />{result.equation.stageY}</div>}
            </Section>
            <div className="asideActions">{groups.length ? <PrimaryButton onClick={() => void accept()} disabled={busy || !canAcceptCalibration || !result?.acceptance?.withinLimits}><Check size={15} />{canAcceptCalibration ? '确认此方案' : '标定已确认'}</PrimaryButton> : <PrimaryButton onClick={() => void startCalibration()} disabled={busy || !taskId || !canStartCalibration}><Play size={15} />开始标定</PrimaryButton>}</div>
          </aside>
        </div>
      ) : <ABPage key={`${taskId}-${mode}`} mode={mode} task={task} setTask={setTask} next={next} notify={notify} models={models} apiBase={apiBase} />}
    </div>
  )
}

function formatMetric(value?: number): string { return Number.isFinite(value) ? Number(value).toFixed(1) : '—' }

function EvidenceRow({ sequence, taskId, group, apiBase }: { sequence: number; taskId: string; group: number; apiBase: string }) {
  return (
    <div className="evidenceRow"><div className="evidenceImage"><span>光镜</span><img src={`${apiBase}/tasks/${taskId}/files/optical/${String(sequence).padStart(4, '0')}.png`} alt="" /></div><div className="evidenceImage sem"><span>SEM</span><img src={`${apiBase}/tasks/${taskId}/files/g${group}_cal_annotated/${String(sequence).padStart(4, '0')}.png`} alt="" /></div><div><strong>颗粒 #{sequence}</strong><span className="matchOk"><Check size={13} />已生成证据</span></div></div>
  )
}

type ABHole = { confidence: number; centerPixel: number[]; boxXyxy?: number[] }
type ABNavigation = {
  previewDataUrl: string
  rawPreviewDataUrl?: string
  imageShape: number[]
  detection: { fixture?: { boxXyxy?: number[] }; holes: ABHole[] }
}
type ABValidationRecord = {
  sequence: number
  lengthUm?: number
  targetStageUm?: { x?: number; y?: number }
  moveErrorUm?: number
  error?: string | null
  status?: string
  predictionErrorUm?: number
  score?: number
  margin?: number
  role?: string
  contextVerification?: { status?: string; confirmedCount?: number; requiredCount?: number; reason?: string }
  inlier?: boolean
  robustWeight?: number
  fitResidualUm?: number
  images?: { opticalImage?: string; semRawImage?: string; semDisplayImage?: string; semAnnotatedImage?: string }
}
type ABSemRefinement = {
  status?: string
  error?: string
  warnings?: Array<{ hole?: string; captureIndex?: number; message?: string; positionErrorUm?: number }>
  validation?: ABValidationRecord[]
  anchors?: ABValidationRecord[]
  observations?: ABValidationRecord[]
  axisFailureDetails?: Array<{ phase?: string; error?: string; retryable?: boolean; diagnostics?: { repeatImageShiftPx?: number; stageReturnDeltaUm?: number; measuredPixelSizeUm?: number; pixelSizeUm?: number } }>
  plan?: { anchors?: number[]; validation?: number[] }
  frozenAt?: string
  runDirectory?: string
  htmlReport?: string
  acceptance?: { withinLimits?: boolean; reasons?: string[]; count?: number; requiredCount?: number; meanErrorUm?: number; maxErrorUm?: number; coverage?: number; limits?: { meanErrorUm?: number; maxErrorUm?: number } }
  pixelAxes?: { coarse?: { returnErrorUm?: number }; fine?: { returnErrorUm?: number } }
  refinedTransform?: { fit?: { rmsErrorUm?: number; maxErrorUm?: number }; scale?: number; rotationDeg?: number }
}
type ABPointResult = {
  manualStagePoints?: { a: { x: number; y: number }; b: { x: number; y: number } }
  navigation?: { image?: string; annotatedImage?: string; imageShape?: number[] }
  detection?: ABNavigation['detection']
  assignment?: { a?: { candidateIndex?: number }; b?: { candidateIndex?: number } }
  semRefinement?: ABSemRefinement
}

function PreflightSummary({ task, job }: { task: TaskManifest | null; job: JobState | null }) {
  const report = job?.preflight || ((!job || job.status === 'completed') ? task?.preflight as PreflightReport | undefined : undefined)
  if (!report) return null
  const labels = { PASS: '通过', WARN: '提醒', FAIL: '阻止启动', SKIP: '未检查' }
  return <Section title="开工检查" extra={<span className="sectionNote">{!report.enabled ? '已关闭' : report.passed ? '检查通过' : '未通过，未执行设备动作'}</span>}>
    <p>{report.items.filter(item => item.status === 'FAIL').length} 项阻止启动 · {report.items.filter(item => item.status === 'WARN').length} 项提醒</p>
    <details open={!report.passed}><summary>查看检查明细</summary><div className="resultRows">{report.items.map(item => <SummaryRow key={item.id} label={`${item.label} · ${labels[item.status]}`} value={item.detail} />)}</div></details>
  </Section>
}

function taskFileUrl(apiBase: string, taskId: string, relative?: string): string {
  if (!relative) return ''
  return `${apiBase}/tasks/${encodeURIComponent(taskId)}/files/${relative.split(/[\\/]/).map(encodeURIComponent).join('/')}`
}

function ABPage({ task, setTask, next, notify, models, apiBase, mode }: { task: TaskManifest | null; setTask: (value: TaskManifest | null) => void; next: () => void; notify: Notify; models: ModelResource[]; apiBase: string; mode: CalibrationMode }) {
  const particleMode = mode === 'ab_particle'
  const manualMode = mode === 'ab_manual'
  const [manualValues, setManualValues] = useState(['', '', '', ''])
  const manualValid = manualValues.every(value => value.trim() !== '' && Number.isFinite(Number(value)) && Math.abs(Number(value)) <= 50)
    && Math.hypot(Number(manualValues[2]) - Number(manualValues[0]), Number(manualValues[3]) - Number(manualValues[1])) >= 1
  const [navigation, setNavigation] = useState<ABNavigation | null>(null)
  const [aIndex, setAIndex] = useState(0)
  const [bIndex, setBIndex] = useState(1)
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState<JobState | null>(null)
  const taskId = String(task?.folderId || task?.taskId || '')
  const calibration = task?.calibration as { strategy?: string; result?: ABSemRefinement['refinedTransform']; abPointResult?: ABPointResult } | undefined
  const abResult = calibration?.abPointResult
  useEffect(() => {
    const points = abResult?.manualStagePoints
    if (points) setManualValues([points.a.x, points.a.y, points.b.x, points.b.y].map(value => String(value / 1000)))
  }, [abResult?.manualStagePoints])
  const refinement = abResult?.semRefinement
  const validation = refinement?.validation || []
  const refinementWarnings = refinement?.warnings || []
  const savedNavigation = useMemo<ABNavigation | null>(() => {
    const saved = abResult?.navigation
    if (!saved?.imageShape || !abResult?.detection || !taskId) return null
    return {
      previewDataUrl: taskFileUrl(apiBase, taskId, saved.annotatedImage || saved.image),
      rawPreviewDataUrl: taskFileUrl(apiBase, taskId, saved.image || saved.annotatedImage),
      imageShape: saved.imageShape,
      detection: abResult.detection,
    }
  }, [abResult, apiBase, taskId])
  const effectiveNavigation = navigation || savedNavigation

  useEffect(() => {
    const assignedA = abResult?.assignment?.a?.candidateIndex
    const assignedB = abResult?.assignment?.b?.candidateIndex
    if (Number.isInteger(assignedA)) setAIndex(Number(assignedA))
    if (Number.isInteger(assignedB)) setBIndex(Number(assignedB))
  }, [abResult?.assignment?.a?.candidateIndex, abResult?.assignment?.b?.candidateIndex])

  const fetchNavigation = async () => {
    if (!taskId) return notify('请先创建检测任务')
    setBusy(true)
    try {
      const result = await request<ABNavigation>('POST', `/tasks/${taskId}/ab/navigation`, {})
      setNavigation(result)
      setAIndex(0); setBIndex(Math.min(1, result.detection.holes.length - 1))
      notify(`检测到 ${result.detection.holes.length} 个孔候选`)
    } catch (error) { notify(errorNotice(error, '获取 AB 导航图')) } finally { setBusy(false) }
  }
  const refine = async () => {
    if (manualMode ? !manualValid : (!effectiveNavigation || aIndex === bIndex)) return notify(manualMode ? '请完整填写有效的 AB 坐标（mm），两点间距至少 1 mm' : '请为 A、B 选择不同的孔位')
    if (!window.confirm(manualMode ? `使用 A (${manualValues[0]}, ${manualValues[1]}) mm、B (${manualValues[2]}, ${manualValues[3]}) mm 建立标定，并移动样品台实拍验证颗粒，确认开始？` : particleMode ? '将连接电镜、移动样品台并拍摄颗粒，期间会切换倍率和测量像素轴。请确认电镜空闲、样品装夹安全，是否开始？' : '将连接电镜并移动样品台执行 AB 孔精定位，确认开始？')) return
    setBusy(true)
    try {
      if (particleMode) {
        const health = await request<{ capabilities?: { abParticleCalibration?: boolean } }>('GET', '/health')
        if (!health.capabilities?.abParticleCalibration) throw new Error('当前后端不支持颗粒精校准，请重启或升级后端；未发送任何标定启动命令')
      }
      if (task?.status === 'DRAFT') {
        const assigned = manualMode
          ? await request<TaskManifest>('POST', `/tasks/${taskId}/ab/manual`, { coordinateUnit: 'mm', a: { x: Number(manualValues[0]), y: Number(manualValues[1]) }, b: { x: Number(manualValues[2]), y: Number(manualValues[3]) } })
          : await request<TaskManifest>('POST', `/tasks/${taskId}/ab/assign`, { aIndex, bIndex, handedness: 'right', mode })
        setTask(assigned)
      }
      const started = await request<JobState>('POST', `/tasks/${taskId}/ab/refine`, {})
      setJob(started)
      notify({ kind: 'info', message: '已开始本次 SEM 验证，上一次错误保留在错误记录中' })
      const completed = await waitForJob(started.id, setJob)
      const manifest = completed.result || null
      setTask(manifest)
      const completedCalibration = manifest?.calibration as { abPointResult?: ABPointResult } | undefined
      const completedRefinement = completedCalibration?.abPointResult?.semRefinement
      if (completedRefinement?.status === 'VALIDATION_FAILED') notify(errorNotice(completedRefinement.acceptance?.reasons?.join('；') || '独立实拍误差超限', '颗粒精校准验收'))
      else if (completedRefinement?.status === 'FAILED') notify(errorNotice(completedRefinement.error || '请检查设备与参数后重试', 'SEM 精定位'))
      else if (manifest?.status === 'CALIBRATION_PENDING_REVIEW' && completedRefinement?.warnings?.length) notify({ kind: 'info', message: `SEM 精定位已完成，存在 ${completedRefinement.warnings.length} 条样品台回读警告；已按实际坐标继续，请结合验证图片确认` })
      else if (manifest?.status === 'CALIBRATION_PENDING_REVIEW') notify(manualMode ? '手动 AB 标定与验证拍摄完成，请逐张对比后确认' : 'SEM 精定位与验证拍摄完成，请逐张对比后确认')
      else if (manifest?.error) notify(errorNotice(manifest.error, 'SEM 精定位'))
      else if (String(manifest?.status || '').includes('FAILED')) notify(errorNotice(`精定位任务状态：${displayStatus(String(manifest?.status))}`, 'SEM 精定位'))
      else notify({ kind: 'info', message: String(manifest?.status || '精定位结束') })
    } catch (error) { notify(errorNotice(error, 'SEM 精定位')) } finally { setBusy(false) }
  }
  const retry = async () => {
    setBusy(true)
    try {
      const manifest = await request<TaskManifest>('POST', `/tasks/${taskId}/ab/retry`, {})
      setTask(manifest); setJob(null)
      notify('已返回待精定位；设备不会自动移动，请确认后重新点击开始')
    } catch (error) { notify(errorNotice(error, '重新执行 SEM 精定位')) } finally { setBusy(false) }
  }
  const accept = async () => {
    setBusy(true)
    try {
      const started = await request<JobState>('POST', `/tasks/${taskId}/review`, { accepted: true, selectedGroup: 0 })
      const completed = await waitForJob(started.id, setJob)
      setTask(completed.result || null); next()
    } catch (error) { notify(errorNotice(error, '确认 AB 标定')) } finally { setBusy(false) }
  }
  const readyToAccept = task?.status === 'CALIBRATION_PENDING_REVIEW'
  const refining = job?.status === 'queued' || job?.status === 'running'
  const refineStopped = !busy && !refining && refinement?.status === 'STOPPED'
  const refineFailed = !busy && !refining && (refinement?.status === 'FAILED' || refinement?.status === 'VALIDATION_FAILED' || job?.status === 'failed')
  const completedEvidence = validation.filter((record) => !record.error && record.images?.opticalImage && (record.images?.semAnnotatedImage || record.images?.semDisplayImage)).length
  const evidenceComplete = validation.length > 0 && completedEvidence === validation.length
  const numericPassed = !particleMode || (refinement?.status === 'SUCCEEDED' && refinement.acceptance?.withinLimits === true)
  const progressPercent = readyToAccept ? 100 : job?.total ? Math.min(99, Math.round(job.progress / job.total * 100)) : refining ? 6 : 0
  const canChooseHoles = task?.status === 'DRAFT'
  const canRunRefinement = task?.status === 'DRAFT' || task?.status === 'AB_POINTS_PENDING_SEM_REFINEMENT'
  return (
    <div className="abLayout">
      <div className="abMain">
        <PreflightSummary task={task} job={job} />
        {particleMode && refinement && <ParticleRefinementSummary refinement={refinement} taskId={taskId} apiBase={apiBase} />}
        {manualMode && <Section title="手动输入 A/B 台坐标" extra={<span className="sectionNote">单位：mm（毫米）</span>}>
          <div className="formGrid">{['A 点 X（mm）', 'A 点 Y（mm）', 'B 点 X（mm）', 'B 点 Y（mm）'].map((label, index) => <Field key={label} label={label}><input aria-label={label} type="number" step="0.001" min="-50" max="50" placeholder="请输入坐标" value={manualValues[index]} disabled={busy || task?.status !== 'DRAFT'} onChange={event => setManualValues(values => values.map((value, i) => i === index ? event.target.value : value))} /></Field>)}</div>
          <p className="holeHelp">填写电镜样品台坐标，注意正负号及 A/B 顺序。每个坐标须在 ±50 mm 内，AB 间距至少 1 mm。提交后保留原始坐标，修改坐标请新建任务。</p>
          {manualValid && <p>AB 间距：{Math.hypot(Number(manualValues[2]) - Number(manualValues[0]), Number(manualValues[3]) - Number(manualValues[1])).toFixed(3)} mm</p>}
        </Section>}
        {!manualMode && <Section title="SEM 导航图" extra={canChooseHoles ? <SecondaryButton onClick={() => void fetchNavigation()}><RefreshCw size={14} />{effectiveNavigation ? '重新获取' : '获取导航图'}</SecondaryButton> : <span className="sectionNote">完整图像 · 等比例显示</span>}>
          {effectiveNavigation ? <NavigationPreview navigation={effectiveNavigation} aIndex={aIndex} bIndex={bIndex} /> : <div className="navigationImage emptyNavigation"><Microscope size={30} /><strong>尚未获取电镜导航图</strong><span>该操作只读取导航图，不会移动样品台</span></div>}
        </Section>}

        {(job || refinement) && <Section title="SEM 精定位进度" extra={<span className={`refineState ${refineFailed ? 'failed' : readyToAccept ? 'success' : ''}`}>{refineStopped ? '已停止' : refineFailed ? '执行失败' : readyToAccept ? '等待人工验收' : refining ? '正在执行' : displayStatus(String(task?.status || ''))}</span>}>
          <div className="refineProgressHeader"><div><strong>{refineStopped ? '精定位已停止，不会自动继续' : refineFailed ? '精定位未完成' : readyToAccept ? '精定位和验证拍摄已完成' : refining ? '电镜正在执行闭环精定位' : '等待操作员启动精定位'}</strong><span>{refineStopped ? '保留本次证据；再次操作设备前需重新确认设备空闲' : refineFailed ? refinement?.error || job?.error || job?.message : readyToAccept ? `已取得 ${completedEvidence} / ${validation.length} 组光镜与 SEM 对比证据` : job?.message || '尚未启动设备操作'}</span></div><b>{progressPercent}%</b></div>
          <div className="progress refineProgress"><span style={{ width: `${progressPercent}%` }} /></div>
          {job && <div className="progressMeta"><span>已完成 {Math.min(job.progress, job.total || job.progress)} / {job.total || '—'} 步</span><span>{job.message}</span></div>}
          {refinementWarnings.length > 0 && <div className="refinementWarnings"><TriangleAlert size={16} /><div><strong>精定位已继续，但有 {refinementWarnings.length} 次样品台回读超差</strong>{refinementWarnings.map((warning, index) => <span key={`${warning.hole || '孔'}-${warning.captureIndex || index}`}>{warning.hole || '孔'} · 第 {warning.captureIndex || '—'} 张：{warning.message || `${formatMetric(warning.positionErrorUm)} μm`}</span>)}</div></div>}
        </Section>}

        {(readyToAccept || (particleMode && validation.length > 0)) && <Section title="光镜 / SEM 对比验收" extra={<span className="sectionNote">请由人工逐张确认</span>}>
          <div className="verificationGuide"><Info size={16} /><span>{particleMode ? '这些颗粒未参与拟合。SEM 图是在冻结矩阵的预测位置直接拍摄，绿色十字是视场中心，红圈是匹配位置；同时满足数值门槛与人工形貌确认后才能进入检测。' : '左侧是光镜颗粒，右侧是按标定坐标实拍的 SEM 图。请确认 SEM 图中心标记附近能看到形貌相符的颗粒；软件不会替你自动判定通过。'}</span></div>
          {validation.length ? <div className="abEvidenceList">{validation.map((record) => <ABEvidenceCard key={record.sequence} record={record} taskId={taskId} apiBase={apiBase} />)}</div> : <div className="emptyState compact"><TriangleAlert size={20} /><span>没有生成验证图片，不能确认标定结果，请重新精定位</span></div>}
        </Section>}
      </div>
      <aside className="abAside">
        {!manualMode && <Section title="定位步骤">
          <div className="simpleSteps"><StepLine done={Boolean(effectiveNavigation)} active={!effectiveNavigation} title="获取导航图" detail={effectiveNavigation ? `${effectiveNavigation.imageShape[1]} × ${effectiveNavigation.imageShape[0]} px` : '等待设备'} /><StepLine done={Boolean(effectiveNavigation)} active={Boolean(effectiveNavigation) && canChooseHoles && !job} title="选择 A/B 孔位" detail={effectiveNavigation ? `${effectiveNavigation.detection.holes.length} 个候选孔` : '使用本地 ONNX'} /><StepLine done={refinement?.status === 'SUCCEEDED'} active={refining} title="SEM 闭环精定位" detail={job?.message || (refineFailed ? refinement?.error || '执行失败' : '需确认后移动样品台')} /><StepLine done={readyToAccept && evidenceComplete} active={readyToAccept} title="人工对比验收" detail={readyToAccept ? `${completedEvidence} / ${validation.length} 组证据待确认` : '完成后显示光镜 / SEM 对比'} /></div>
        </Section>}
        {effectiveNavigation && <Section title="A/B 孔位"><div className="formGrid abSelectors"><Field label="A 点"><select value={aIndex} disabled={!canChooseHoles} onChange={(event) => setAIndex(Number(event.target.value))}>{effectiveNavigation.detection.holes.map((hole, index) => <option value={index} key={index}>候选孔 {index + 1} · {(hole.confidence * 100).toFixed(1)}%</option>)}</select></Field><Field label="B 点"><select value={bIndex} disabled={!canChooseHoles} onChange={(event) => setBIndex(Number(event.target.value))}>{effectiveNavigation.detection.holes.map((hole, index) => <option value={index} key={index}>候选孔 {index + 1} · {(hole.confidence * 100).toFixed(1)}%</option>)}</select></Field></div><p className="holeHelp">图上的“候选孔 1/2”是模型检测到的孔位编号；选定后分别标为 A 点和 B 点。</p></Section>}
        {!manualMode && <Section title="模型资源"><div className="modelList">{models.filter((model) => model.id === 'fixture' || model.id === 'hole').map((model) => <span key={model.id}>{model.file} <b className={model.ready ? '' : 'modelMissing'}>{model.ready ? '就绪' : '缺失'}</b></span>)}</div></Section>}
        {readyToAccept ? <div className="asideActions"><PrimaryButton onClick={() => void accept()} disabled={busy || !evidenceComplete || !numericPassed}><Check size={15} />{particleMode && !numericPassed ? '数值验收未通过，不能确认' : '对比一致，确认标定'}</PrimaryButton><SecondaryButton onClick={() => void retry()} disabled={busy}><RotateCcw size={14} />{particleMode ? '重新精定位（保留本次记录）' : '对比不一致，重新精定位'}</SecondaryButton></div> : <PrimaryButton onClick={() => void refine()} disabled={(manualMode ? !manualValid : (!effectiveNavigation || aIndex === bIndex)) || busy || refining || !canRunRefinement || !taskId}><Play size={15} />{canRunRefinement ? (refining ? 'SEM 验证进行中' : manualMode ? '建立标定并实拍验证' : particleMode ? '开始颗粒精校准' : refineFailed ? '重新开始 SEM 精定位' : '开始 SEM 精定位') : '标定已确认'}</PrimaryButton>}
        {refining && job && <SecondaryButton onClick={async () => { try { await request('POST', `/jobs/${job.id}/cancel`, {}); notify('已请求停止，将在当前设备调用结束后恢复设备') } catch (error) { notify(errorNotice(error, '停止精校准')) } }}><CircleStop size={14} />停止并恢复设备</SecondaryButton>}
      </aside>
    </div>
  )
}

function ParticleRefinementSummary({ refinement, taskId, apiBase }: { refinement?: ABSemRefinement; taskId: string; apiBase: string }) {
  const gate = refinement?.acceptance
  const contextRecords = refinement?.observations?.filter((record) => record.contextVerification) || []
  return <Section title="颗粒精校准 · 数值验收" extra={<span className="sectionNote">纯本地算法 · 不在电镜中找孔</span>}>
    <div className="resultRows">
      <SummaryRow label="锚点 / 预留验证点" value={refinement?.plan ? `${refinement.anchors?.length || 0} / ${refinement.plan.validation?.length || 0}` : '自动选择显著、分散的颗粒'} />
      <SummaryRow label="冻结时间" value={refinement?.frozenAt || '拟合完成后冻结，验证数据不回流'} />
      <SummaryRow label="像素轴回原点误差" value={`${formatMetric(refinement?.pixelAxes?.fine?.returnErrorUm)} μm`} />
      {contextRecords.length > 0 && <SummaryRow label="种子邻域复核（通过 / 检查）" value={`${contextRecords.filter((record) => record.contextVerification?.status === 'VERIFIED').length} / ${contextRecords.length}`} />}
      <SummaryRow label="有效独立验证" value={gate ? `${gate.count} / ${gate.requiredCount}` : '尚未验证'} />
      <SummaryRow label="平均 / 最大预测误差" value={`${formatMetric(gate?.meanErrorUm)} / ${formatMetric(gate?.maxErrorUm)} μm`} />
      <SummaryRow label="验收阈值（平均 / 最大）" value={gate?.limits ? `${formatMetric(gate.limits.meanErrorUm)} / ${formatMetric(gate.limits.maxErrorUm)} μm` : '使用任务快照中的 abParticle 配置'} />
      <SummaryRow label="验证覆盖率" value={gate?.coverage != null ? `${(gate.coverage * 100).toFixed(1)}%` : '—'} />
      <SummaryRow label="数值验收" value={gate ? (gate.withinLimits && refinement?.status === 'SUCCEEDED' ? '通过，仍需人工确认' : '未通过，禁止进入检测') : '待执行'} />
    </div>
    {gate?.reasons?.map((reason) => <p className="holeHelp" key={reason}>{reason}</p>)}
    {refinement?.error && <p className="holeHelp">{refinement.error}</p>}
    {refinement?.axisFailureDetails?.filter((failure) => failure.retryable === false).map((failure, index) => <p className="holeHelp" key={index}>{failure.diagnostics?.measuredPixelSizeUm != null ? `图像尺度异常：元数据 ${failure.diagnostics.pixelSizeUm?.toFixed(3)} μm/px，实测 ${failure.diagnostics.measuredPixelSizeUm.toFixed(3)} μm/px。${failure.error || '需检查倍率与采图链路。'}` : `图像 / 台坐标一致性异常：回原点图像位移 ${formatMetric(failure.diagnostics?.repeatImageShiftPx)} px，台坐标差 ${formatMetric(failure.diagnostics?.stageReturnDeltaUm)} μm。已停止自动换区域重试，需检查采图与定位链路。`}</p>)}
    {!!refinement?.anchors?.length && <details><summary>锚点残差与降权记录（{refinement.anchors.length}）</summary><div className="resultRows">{refinement.anchors.map((anchor) => <SummaryRow key={anchor.sequence} label={`#${anchor.sequence}${anchor.inlier === false ? ' · 异常点' : ''}`} value={`残差 ${formatMetric(anchor.fitResidualUm)} μm · 权重 ${anchor.robustWeight?.toFixed(3) ?? '—'}`} />)}</div></details>}
    {refinement?.htmlReport && <a href={taskFileUrl(apiBase, taskId, refinement.htmlReport)} target="_blank" rel="noreferrer">打开标定报告（HTML）</a>}
    {refinement?.runDirectory && refinement.frozenAt && <p><a href={taskFileUrl(apiBase, taskId, `${refinement.runDirectory}/frozen_transform.json`)} target="_blank" rel="noreferrer">查看冻结矩阵（JSON）</a></p>}
    <p className="holeHelp">匹配分数是相关系数，不是概率。种子还需通过同图邻近颗粒复核；预留验证颗粒不参与复核。未找到、歧义、覆盖不足或误差超限都会停止或拒绝通过；新样品需重新校准。</p>
  </Section>
}

function NavigationPreview({ navigation, aIndex, bIndex }: { navigation: ABNavigation; aIndex: number; bIndex: number }) {
  const height = Number(navigation.imageShape[0] || 1)
  const width = Number(navigation.imageShape[1] || 1)
  const fixture = navigation.detection.fixture?.boxXyxy
  const fontSize = Math.max(18, width / 80)
  return <div className="navigationPreview"><div className="navigationViewport" style={{ aspectRatio: `${width} / ${height}` }}>
    <img src={navigation.rawPreviewDataUrl || navigation.previewDataUrl} alt="完整 SEM 导航图" />
    <svg className="navigationOverlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-label="导航图孔位候选标注">
      {fixture && <rect className="fixtureBox" x={fixture[0]} y={fixture[1]} width={fixture[2] - fixture[0]} height={fixture[3] - fixture[1]} />}
      {navigation.detection.holes.map((hole, index) => {
        const [x, y] = hole.centerPixel
        const box = hole.boxXyxy || [x - 24, y - 24, x + 24, y + 24]
        const radius = Math.max(18, Math.max(box[2] - box[0], box[3] - box[1]) / 2)
        const selected = index === aIndex ? 'a' : index === bIndex ? 'b' : 'candidate'
        const label = index === aIndex ? `A 点（候选孔 ${index + 1}）` : index === bIndex ? `B 点（候选孔 ${index + 1}）` : `候选孔 ${index + 1}`
        const labelX = Math.max(8, Math.min(width - fontSize * 8.5, x + radius + 10))
        const labelY = Math.max(fontSize + 6, Math.min(height - 8, y - radius - 8))
        return <g className={`navigationHole ${selected}`} key={index}><title>{label}，置信度 {(hole.confidence * 100).toFixed(1)}%</title><circle cx={x} cy={y} r={radius} /><path d={`M ${x - radius * .55} ${y} H ${x + radius * .55} M ${x} ${y - radius * .55} V ${y + radius * .55}`} /><text x={labelX} y={labelY} fontSize={fontSize}>{label} · {(hole.confidence * 100).toFixed(1)}%</text></g>
      })}
    </svg>
  </div><div className="navigationLegend"><span><i className="legendFixture" />夹具范围</span><span><i className="legendA" />A 点</span><span><i className="legendB" />B 点</span><b>{width} × {height} px · 完整图像</b></div></div>
}

function ABEvidenceCard({ record, taskId, apiBase }: { record: ABValidationRecord; taskId: string; apiBase: string }) {
  const optical = taskFileUrl(apiBase, taskId, record.images?.opticalImage)
  const sem = taskFileUrl(apiBase, taskId, record.images?.semAnnotatedImage || record.images?.semDisplayImage || record.images?.semRawImage)
  return <article className={`abEvidenceCard ${record.error ? 'failed' : ''}`}>
    <div className="comparisonImages"><figure><div className="comparisonImage optical">{optical ? <img src={optical} alt={`光镜颗粒 ${record.sequence}`} /> : <span>缺少光镜图</span>}<b>光镜颗粒</b></div></figure><figure><div className="comparisonImage sem">{sem ? <img src={sem} alt={`SEM 验证颗粒 ${record.sequence}`} /> : <span>缺少 SEM 图</span>}<b>SEM 实拍 · 中心为预测位置</b></div></figure></div>
    <div className="evidenceMeta"><div><strong>验证颗粒 #{record.sequence}</strong><span>{record.error ? `采集失败：${record.error}` : record.status && record.status !== 'MATCH' ? `未通过：${record.status === 'AMBIGUOUS' ? '候选存在歧义' : '未取得可靠匹配'}` : '请人工比较形貌、方向和尺寸是否一致'}</span></div><dl><div><dt>光镜长度</dt><dd>{formatMetric(record.lengthUm)} μm</dd></div><div><dt>移台到位误差</dt><dd>{formatMetric(record.moveErrorUm)} μm</dd></div>{record.status && <div><dt>独立预测误差</dt><dd>{record.status === 'MATCH' ? `${formatMetric(record.predictionErrorUm)} μm` : '未可靠匹配，不计误差'}</dd></div>}<div><dt>目标台坐标</dt><dd>{Number.isFinite(record.targetStageUm?.x) && Number.isFinite(record.targetStageUm?.y) ? `${Number(record.targetStageUm?.x).toFixed(1)}, ${Number(record.targetStageUm?.y).toFixed(1)} μm` : '—'}</dd></div></dl></div>
  </article>
}

function StepLine({ title, detail, done, active }: { title: string; detail: string; done?: boolean; active?: boolean }) {
  return <div className={active ? 'stepLine active' : done ? 'stepLine done' : 'stepLine'}><span>{done ? <Check size={13} /> : ''}</span><div><strong>{title}</strong><small>{detail}</small></div></div>
}

function RunningPage({ task, setTask, job, setJob, notify, apiBase, onStep, canNavigate, isComplete }: { task: TaskManifest | null; setTask: (value: TaskManifest | null) => void; job: JobState | null; setJob: (value: JobState | null) => void; notify: Notify; apiBase: string; onStep: (step: WorkflowStep) => void; canNavigate: (step: WorkflowStep) => boolean; isComplete: (step: WorkflowStep) => boolean }) {
  const taskId = String(task?.folderId || task?.taskId || '')
  const records = (task?.particles || []) as Array<{ sequence?: number; status?: string; error?: string; acquisitionMode?: string; lengthUm?: number; equivalentDiameterUm?: number; localization?: { centerErrorUm?: number }; match?: { score?: number }; images?: { semAnnotatedImage?: string }; energySpectrum?: { elements?: Array<{ symbol: string }> } }>
  const total = Number((task?.selection as { count?: number } | undefined)?.count || (task?.input as { selectedCount?: number } | undefined)?.selectedCount || 0)
  const completed = records.filter((record) => record.status && record.status !== 'RUNNING').length
  const percent = total ? Math.min(100, Math.round(completed / total * 100)) : 0
  const running = job?.status === 'queued' || job?.status === 'running'
  const interrupted = ['RUNNING', 'STOPPED', 'FAILED'].includes(String(task?.status))
  const resumable = interrupted && completed > 0 && completed < total && !running
  const restartable = ['RUNNING', 'STOPPED', 'FAILED', 'COMPLETED_WITH_FAILURES'].includes(String(task?.status)) && !running
  const latest = [...records].reverse().find((record) => record.images?.semAnnotatedImage) || records.at(-1)
  const visibleError = job?.status === 'failed' ? job.error || job.message : task?.error
  const hasFailure = Boolean(visibleError) || task?.status === 'FAILED'
  const partialCompletion = task?.status === 'COMPLETED_WITH_FAILURES' && !hasFailure
  const failedRecords = records.filter(record => record.status === 'FAILED')

  const launchAnalysis = async (mode: 'start' | 'resume' | 'restart') => {
    if (!taskId) return notify('请先创建并完成标定任务')
    if (mode === 'start' && task?.status !== 'CALIBRATION_ACCEPTED') return notify('必须先确认标定结果')
    if (mode === 'resume' && !resumable) return notify('当前任务没有可以继续的未处理颗粒')
    if (mode === 'restart' && !restartable) return notify('当前任务不能从头重新检测')
    if (mode === 'restart' && !window.confirm(`将清空当前 ${completed} 条检测记录并从头检测，确定继续吗？`)) return
    try {
      if (mode === 'restart') {
        const reset = await request<TaskManifest>('POST', `/tasks/${taskId}/analysis/reset`, {})
        setTask(reset)
        notify('已保留上次检测摘要，正在从头重新检测')
      } else if (mode === 'resume') {
        notify({ kind: 'info', message: `将保留已有 ${completed} 条结果，继续检测剩余 ${total - completed} 个颗粒` })
      }
      const route = mode === 'resume' ? `/tasks/${taskId}/analysis/resume` : `/tasks/${taskId}/analysis`
      const started = await request<JobState>('POST', route, {})
      setJob(started)
      const pollManifest = window.setInterval(() => {
        void request<TaskManifest>('GET', `/tasks/${taskId}`).then(setTask).catch((error: unknown) => notify(errorNotice(error, '刷新检测进度')))
      }, 1000)
      try {
        const finished = await waitForJob(started.id, setJob)
        const result = finished.result || await request<TaskManifest>('GET', `/tasks/${taskId}`)
        setTask(result)
        if (result.error) notify(errorNotice(result.error, '检测运行'))
        else if (result.status === 'COMPLETED') notify(`检测完成，报告已生成。${gunShutdownSummary(result)}`)
        else if (result.status === 'COMPLETED_WITH_FAILURES') notify({ kind: 'info', message: completedTaskSummary(result) })
        else if (result.status === 'STOPPED') notify({ kind: 'info', message: '检测已停止' })
        else notify(errorNotice(`检测任务状态：${displayStatus(String(result.status || '未知'))}`, '检测运行'))
      } finally { window.clearInterval(pollManifest) }
    } catch (error) { notify(errorNotice(error, '检测运行')) }
  }

  const stop = async () => {
    if (!job) return
    try { await request('POST', `/jobs/${job.id}/cancel`, {}); notify('正在安全停止任务') } catch (error) { notify(errorNotice(error, '停止检测')) }
  }
  return (
    <div className="page">
      <PageTitle title="检测运行" description="SEM 定位与 EDS 分析执行状态" action={<span className={`runningLabel ${hasFailure ? 'failed' : partialCompletion ? 'partial' : ''}`}><StatusDot active={running || task?.status === 'COMPLETED'} />{running ? '正在运行' : displayStatus(String(task?.status || '未创建任务'))}</span>} />
      <StepBar active={3} onStep={onStep} canNavigate={canNavigate} isComplete={isComplete} />
      {visibleError && <PersistentErrorPanel error={visibleError} context="检测运行" />}
      <PreflightSummary task={task} job={job} />
      {!running && task && gunShutdownSummary(task) && <div className="verificationGuide">{gunShutdownSummary(task)}</div>}
      {!running && partialCompletion && task && <Section title="检测完成统计">
        <p className="completionSummary">{completedTaskSummary(task)}</p>
        <p>普通颗粒匹配未通过时不采谱；大颗粒按中心区域检查结果采谱。原因和采谱方式保留在报告中。</p>
        <details><summary>查看 {failedRecords.length} 个失败颗粒及原因</summary><div className="resultRows">
          {failedRecords.map(record => <SummaryRow key={record.sequence} label={`颗粒 #${record.sequence}`} value={record.error || '未记录详细原因'} />)}
        </div></details>
      </Section>}
      <div className="runHeader">
        <div><strong>{String((task?.input as { workpieceName?: string } | undefined)?.workpieceName || '未选择任务')}</strong><span>{taskId || '—'} · {completed} / {total}</span></div>
        <div className="runProgress"><span>总体进度</span><strong>{percent}%</strong><div className="progress"><span style={{ width: `${percent}%` }} /></div></div>
        <div className="runButtons">{running
          ? <PrimaryButton onClick={() => void stop()}><CircleStop size={15} />停止</PrimaryButton>
          : resumable
            ? <><PrimaryButton onClick={() => void launchAnalysis('resume')}><Play size={15} />继续剩余 {total - completed} 个</PrimaryButton><SecondaryButton onClick={() => void launchAnalysis('restart')}><RotateCcw size={14} />从头重新检测</SecondaryButton></>
            : <PrimaryButton onClick={() => void launchAnalysis(restartable ? 'restart' : 'start')} disabled={!taskId || (!restartable && task?.status !== 'CALIBRATION_ACCEPTED')}><Play size={15} />{restartable ? '从头重新检测' : '开始检测'}</PrimaryButton>}</div>
      </div>
      <div className="runningLayout">
        <Section title={`SEM 图像${latest?.sequence ? ` · 颗粒 #${String(latest.sequence).padStart(3, '0')}` : ''}`} extra={<span className="sectionNote">{job?.status === 'failed' ? '执行失败，请查看错误详情' : job?.message || '等待执行'}</span>} className="semSection">
          {latest?.images?.semAnnotatedImage ? <div className="semImage actual"><img src={`${apiBase}/tasks/${taskId}/files/${latest.images.semAnnotatedImage}`} alt={`颗粒 ${latest.sequence} SEM 标注图`} /></div> : <div className="semImage emptySem"><Microscope size={32} /><strong>暂无 SEM 图像</strong><span>开始检测后显示实时标注结果</span></div>}
          <div className="imageMetrics"><SummaryRow label={latest?.acquisitionMode === 'large_particle_center_region' ? '采谱方式' : '定位偏差'} value={latest?.acquisitionMode === 'large_particle_center_region' ? '中心区域（未做轮廓匹配）' : `${formatMetric(latest?.localization?.centerErrorUm)} μm`} /><SummaryRow label="颗粒长度" value={`${formatMetric(latest?.lengthUm)} μm`} /><SummaryRow label="等效圆直径" value={`${formatMetric(latest?.equivalentDiameterUm)} μm`} /><SummaryRow label="任务状态" value={displayStatus(String(latest?.status || task?.status || '—'))} /></div>
        </Section>
        <aside className="queueSection">
          <Section title="执行队列" extra={<span className="sectionNote">{completed} / {total}</span>}>
            <div className="queueList">
              {records.length ? records.slice(-12).map((record) => {
                const state = record.status === 'SUCCEEDED' ? 'done' : record.status === 'FAILED' ? 'failed' : 'active'
                const elements = record.energySpectrum?.elements?.map((item) => item.symbol).slice(0, 4).join(' · ')
                return <div className={`queueRow ${state}`} key={record.sequence}><span>#{String(record.sequence || 0).padStart(3, '0')}</span><div><strong>{record.status === 'SUCCEEDED' ? '检测完成' : record.status === 'FAILED' ? '检测失败' : '执行中'}</strong><small>{record.acquisitionMode === 'large_particle_center_region' ? '中心采谱 · ' : ''}{elements || record.error || '—'}</small></div><i>{state === 'done' ? <Check size={13} /> : state === 'failed' ? <X size={13} /> : <Activity size={13} />}</i></div>
              }) : <div className="emptyTable">尚未开始检测</div>}
            </div>
          </Section>
        </aside>
      </div>
    </div>
  )
}

function ReportsPage({ tasks, notify, openTask }: { tasks: TaskSummary[]; notify: Notify; openTask: (task: TaskSummary) => void }) {
  const [searchText, setSearchText] = useState('')
  const [rows, setRows] = useState(tasks)
  useEffect(() => {
    void request<{ tasks: TaskSummary[] }>('GET', '/tasks').then((result) => setRows(result.tasks)).catch((error: unknown) => notify(errorNotice(error, '读取报告记录')))
  }, [notify])
  const filtered = rows.filter((task) => `${task.id} ${task.input?.workpieceName || ''} ${task.input?.path || ''}`.toLowerCase().includes(searchText.toLowerCase()))
  const openReport = async (taskId: string, kind: 'pdf' | 'csv' | 'html') => {
    try {
      const manifest = await request<TaskManifest>('GET', `/tasks/${taskId}`)
      const relative = manifest.reports?.[kind]
      if (!relative || !manifest.taskDirectory) throw new Error(`该任务尚未生成 ${kind.toUpperCase()} 报告`)
      await openPath(`${manifest.taskDirectory}\\${relative.replaceAll('/', '\\')}`)
    } catch (error) { notify(errorNotice(error, '打开检测报告')) }
  }
  return (
    <div className="page">
      <PageTitle title="任务记录" description="查看历史检测任务与导出结果" />
      <Section title="检测记录">
        <div className="reportToolbar"><div className="search"><Search size={15} /><input placeholder="搜索任务编号或样品名称" value={searchText} onChange={(event) => setSearchText(event.target.value)} /></div></div>
        <div className="simpleTable reportsTable">
          <div className="tableHeader"><span>任务 / 样品</span><span>颗粒数</span><span>完成时间</span><span>状态</span><span>报告</span></div>
          {filtered.length ? filtered.map((task) => <div className="tableRow static" key={task.id}><span><strong>{task.input?.workpieceName || fileName(task.input?.path || '未命名样品')}</strong><small>{task.id}</small></span><span>{task.input?.selectedCount || task.input?.particleCount || 0}</span><span>{task.updatedAt ? new Date(task.updatedAt).toLocaleString('zh-CN') : '—'}</span><span><TaskStatus status={displayStatus(task.status)} /></span><span className="reportActions"><button type="button" onClick={() => openTask(task)}>打开</button><button type="button" onClick={() => void openReport(task.id, task.mode === 'sem_direct' ? 'html' : 'pdf')}><FileText size={15} />{task.mode === 'sem_direct' ? '报告' : 'PDF'}</button><button type="button" onClick={() => void openReport(task.id, 'csv')}><Database size={15} />CSV</button></span></div>) : <div className="emptyTable">暂无符合条件的检测记录</div>}
        </div>
      </Section>
    </div>
  )
}

type AppSettings = { microscopeHost: string; microscopePort: number; edsHost: string; edsPort: number; edsEnabled: boolean; edsTargetCounts: number | null; restoreStartPosition: boolean; gunOffAfterTask: boolean; showReportCoordinates: boolean; preflightEnabled: boolean; captureLevel: number; captureImageSize: number; candidateScoreMargin: number; templateMatchMinScore: number; contourOverlapMin: number; maxCaptureAttempts: number; taskRoot: string }
type EngineerDefinition = { path: string; group: string; groupLabel: string; label: string; unit: string; kind: 'number' | 'integer'; minimum: number; maximum: number; step: number; defaultValue: number; summary: string; lowerEffect: string; higherEffect: string; example: string; risk: 'medium' | 'high' }
type EngineerSettings = { definitions: EngineerDefinition[]; values: Record<string, number>; defaults: Record<string, number>; updatedAt: string | null }

function SettingsPage({ devices, models, refreshDevices, notify, logDirectory }: { devices: DeviceResource[]; models: ModelResource[]; refreshDevices: () => Promise<DeviceResource[]>; notify: Notify; logDirectory: string }) {
  const [settings, setSettings] = useState<AppSettings>({ microscopeHost: '', microscopePort: 61557, edsHost: '', edsPort: 6200, edsEnabled: true, edsTargetCounts: null, restoreStartPosition: true, gunOffAfterTask: true, showReportCoordinates: true, preflightEnabled: true, captureLevel: 7, captureImageSize: 512, candidateScoreMargin: 0.07, templateMatchMinScore: 0.65, contourOverlapMin: 0.5, maxCaptureAttempts: 2, taskRoot: '' })
  const [saving, setSaving] = useState(false)
  useEffect(() => { void request<AppSettings>('GET', '/settings').then(setSettings).catch((error: unknown) => notify(errorNotice(error, '读取系统设置'))) }, [notify])
  const save = async () => {
    setSaving(true)
    try { setSettings(await request<AppSettings>('POST', '/settings', settings)); notify('设置已保存') } catch (error) { notify(errorNotice(error, '保存系统设置')) } finally { setSaving(false) }
  }
  const patchSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => setSettings((current) => ({ ...current, [key]: value }))
  return (
    <div className="page">
      <PageTitle title="系统设置" description="设备连接、模型资源与任务默认选项" action={<PrimaryButton onClick={() => void save()} disabled={saving}><Check size={15} />保存设置</PrimaryButton>} />
      <div className="settingsGrid">
        <Section title="设备连接">
          <SettingDevice icon={Microscope} name="ZEM 扫描电镜" device={devices.find((item) => item.id === 'microscope')} host={settings.microscopeHost} port={settings.microscopePort} setHost={(value) => patchSetting('microscopeHost', value)} setPort={(value) => patchSetting('microscopePort', value)} test={() => void refreshDevices()} />
          <SettingDevice icon={Activity} name="EDS 能谱服务" device={devices.find((item) => item.id === 'eds')} host={settings.edsHost} port={settings.edsPort} setHost={(value) => patchSetting('edsHost', value)} setPort={(value) => patchSetting('edsPort', value)} test={() => void refreshDevices()} />
        </Section>
        <Section title="模型资源">
          <div className="settingsModels">{models.map((model) => <ModelItem key={model.id} name={model.id === 'particle' ? '颗粒分割模型' : model.id === 'fixture' ? '夹具检测模型' : 'AB 孔检测模型'} model={model} />)}</div>
        </Section>
        <Section title="默认任务选项">
          <div className="optionRows"><OptionRow title="执行真实 EDS 分析" detail="逐颗粒进行能谱采集" enabled={settings.edsEnabled} setEnabled={(value) => patchSetting('edsEnabled', value)} /><OptionRow title="任务结束后恢复样品台位置" detail="返回任务开始前的安全位置" enabled={settings.restoreStartPosition} setEnabled={(value) => patchSetting('restoreStartPosition', value)} /></div>
          <div className="captureLevelSetting"><Field label="光镜关联分析 · EDS 停止条件"><select aria-label="EDS 停止条件" value={settings.edsTargetCounts == null ? 'automatic' : 'target_counts'} onChange={(event) => patchSetting('edsTargetCounts', event.target.value === 'automatic' ? null : 100000)}><option value="automatic">自动（10 万计数，5–60 秒）</option><option value="target_counts">指定目标计数</option></select></Field><span>自动模式保留原有采集策略。指定目标计数用于比较不同计数下的采谱耗时，最长采集时间仍为 60 秒。</span></div>
          {settings.edsTargetCounts != null && <div className="captureLevelSetting"><Field label="EDS 目标计数（万次）"><input aria-label="EDS 目标计数（万次）" type="number" min="0.0001" step="0.1" value={settings.edsTargetCounts ? settings.edsTargetCounts / 10000 : ''} onChange={(event) => patchSetting('edsTargetCounts', Number(event.target.value) * 10000)} /></Field><span>例如 2 表示 20,000 次，5 表示 50,000 次，10 表示 100,000 次。新建任务会冻结此设置；报告记录设备返回的目标计数、实际计数和停止原因。</span></div>}
          <OptionRow title="检测结束后自动关枪" detail="完成、停止或异常清理后关枪并确认状态；标定中途不关枪" enabled={settings.gunOffAfterTask} setEnabled={(value) => patchSetting('gunOffAfterTask', value)} />
          <div className="captureLevelSetting"><Field label="拍照档位（Tab2 定位与 Tab3 分析通用）"><select aria-label="拍照档位" value={settings.captureLevel || 7} onChange={(event) => patchSetting('captureLevel', Number(event.target.value))}>{Array.from({ length: 7 }, (_, index) => index + 1).map((level) => <option key={level} value={level}>{level}{level === 1 ? '（最快）' : level === 7 ? '（最清晰）' : ''}</option>)}</select></Field><span>档位越高，像素平均次数越多、图像信噪比越好，但单张拍摄时间也越长。新建任务会冻结所选档位。</span></div>
          <div className="captureLevelSetting"><Field label="光镜关联分析 · 拍照图幅"><select aria-label="拍照图幅" value={settings.captureImageSize ?? 512} onChange={(event) => patchSetting('captureImageSize', Number(event.target.value))}><option value={512}>512 × 512</option><option value={1024}>1024 × 1024</option></select></Field><span>同一倍率下图幅只改变采样像素数量，视场范围不变。1024 图幅保留更多细节，也增加拍照及处理耗时。新建任务会冻结图幅，定位验证与正式分析使用同一图幅。</span></div>
          <div className="captureLevelSetting"><Field label="拍照张数"><select aria-label="拍照张数" value={settings.maxCaptureAttempts || 2} onChange={(event) => patchSetting('maxCaptureAttempts', Number(event.target.value))}>{Array.from({ length: 5 }, (_, index) => index + 1).map((count) => <option key={count} value={count}>{count}</option>)}</select></Field><span>每个颗粒最多拍这么多张。第一张已经匹配通过就不再拍，没对上才拍下一张。标定采图失败时也按这个上限重试。默认 2，范围 1 到 5。新建任务会冻结该值。</span></div>
          <div className="captureLevelSetting"><Field label="候选颗粒分差阈值"><input aria-label="候选颗粒分差阈值" type="number" min="0.01" max="0.99" step="0.01" value={settings.candidateScoreMargin ?? 0.07} onChange={(event) => patchSetting('candidateScoreMargin', Number(event.target.value))} /></Field><span>前两名不同位置候选的分差低于该值时拒绝匹配。数值越大越保守、越容易判为歧义；默认 0.07。新建任务会冻结该阈值。</span></div>
          <div className="captureLevelSetting"><Field label="模板相关分数阈值"><input aria-label="模板相关分数阈值" type="number" min="0.01" max="0.99" step="0.01" value={settings.templateMatchMinScore ?? 0.65} onChange={(event) => patchSetting('templateMatchMinScore', Number(event.target.value))} /></Field><span>光镜形状盖到电镜图上的最低相似分数。低于该值拒绝匹配，也不进能谱。数值越大越严格；默认 0.65。新建任务会冻结该值。</span></div>
          <div className="captureLevelSetting"><Field label="轮廓重叠"><input aria-label="轮廓重叠" type="number" min="0.01" max="0.99" step="0.01" value={settings.contourOverlapMin ?? 0.5} onChange={(event) => patchSetting('contourOverlapMin', Number(event.target.value))} /></Field><span>光镜形状和电镜重新描出的边，重合面积至少要达到这个比例。低于该值拒绝匹配，也不进能谱。数值越大越严格；默认 0.5。新建任务会冻结该值。</span></div>
          <p className="sectionNote">默认选项用于新任务；已创建的任务保留原设置。</p>
        </Section>
        <Section title="开工检查">
          <OptionRow title="开工前状态检查" detail="每次标定、检测或继续前检查；开关对下一次启动生效，运行中的任务不受影响" enabled={settings.preflightEnabled} setEnabled={(value) => patchSetting('preflightEnabled', value)} />
          <p className="sectionNote">检查设备空闲、开枪状态、样品台反馈、换样、电压、成像参数、所需能谱和模型、输入及存储条件。不试移台、不拍图、不自动开枪；无法读取的状态会明确提示。</p>
        </Section>
        <Section title="PDF 报告">
          <p>默认展示光镜原图、原始光镜掩码和 SEM 图。</p>
          <OptionRow title="显示光镜/电镜坐标" detail="在颗粒明细中显示坐标系、单位和位置来源；关闭后仍保留光镜掩码" enabled={settings.showReportCoordinates} setEnabled={(value) => patchSetting('showReportCoordinates', value)} />
        </Section>
        <Section title="存储位置">
          <div className="storageRow"><FolderOpen size={18} /><div><strong>任务数据目录</strong><span>{settings.taskRoot || '由软件自动管理'}</span></div></div>
          <div className="storageRow"><FileText size={18} /><div><strong>运行日志目录</strong><span>{logDirectory || '浏览器开发模式不写入桌面日志'}</span></div>{logDirectory && <SecondaryButton onClick={() => void openPath(logDirectory).catch((error: unknown) => notify(errorNotice(error, '打开日志目录')))}>打开目录</SecondaryButton>}</div>
        </Section>
      </div>
    </div>
  )
}

function SettingDevice({ icon: Icon, name, device, host, port, setHost, setPort, test }: { icon: LucideIcon; name: string; device?: DeviceResource; host: string; port: number; setHost: (value: string) => void; setPort: (value: number) => void; test: () => void }) {
  return <div className="settingDevice"><span className="deviceIcon"><Icon size={19} /></span><div className="settingName"><strong>{name}</strong><span className={device?.status === 'connected' ? '' : 'offlineText'}><StatusDot active={device?.status === 'connected'} />{deviceLabel(device)}</span></div><Field label="主机地址"><input value={host} onChange={(event) => setHost(event.target.value)} /></Field><Field label="端口"><input value={port} onChange={(event) => setPort(Number(event.target.value) || 0)} inputMode="numeric" /></Field><SecondaryButton onClick={test}>测试连接</SecondaryButton></div>
}

function ModelItem({ name, model }: { name: string; model: ModelResource }) {
  return <div className="modelItem"><span className="modelFileIcon"><Database size={17} /></span><div><strong>{name}</strong><span>{model.file} · {model.sha256?.slice(0, 10) || '未校验'}</span></div><small>{formatBytes(model.size)}</small><b className={model.ready ? '' : 'modelMissing'}>{model.ready ? <><Check size={13} />就绪</> : <><X size={13} />缺失</>}</b></div>
}

function OptionRow({ title, detail, enabled, setEnabled }: { title: string; detail: string; enabled: boolean; setEnabled: (value: boolean) => void }) {
  return <div className="optionRow"><div><strong>{title}</strong><span>{detail}</span></div><button className={enabled ? 'switch on' : 'switch'} type="button" role="switch" aria-label={title} aria-checked={enabled} onClick={() => setEnabled(!enabled)}><span /></button></div>
}

function EngineerPanel({ open, close, notify }: { open: boolean; close: () => void; notify: Notify }) {
  const [password, setPassword] = useState('')
  const [token, setToken] = useState<string | null>(null)
  const [settings, setSettings] = useState<EngineerSettings | null>(null)
  const [values, setValues] = useState<Record<string, number>>({})
  const [saving, setSaving] = useState(false)
  const [unlocking, setUnlocking] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [reviewing, setReviewing] = useState(false)

  useEffect(() => {
    if (!open) { setPassword(''); setToken(null); setSettings(null); setValues({}); setDirty(false); setReviewing(false) }
  }, [open])

  const unlock = async () => {
    setUnlocking(true)
    try {
      const result = await request<{ token: string }>('POST', '/engineer/unlock', { password })
      const loaded = await request<EngineerSettings>('POST', '/engineer/settings', { token: result.token })
      setToken(result.token); setSettings(loaded); setValues(loaded.values); setPassword(''); notify('工程师面板已解锁')
    } catch (error) { notify(errorNotice(error, '解锁工程师面板')) } finally { setUnlocking(false) }
  }
  const lock = async () => { if (token) await request('POST', '/engineer/lock', { token }).catch((error: unknown) => notify(errorNotice(error, '锁定工程师面板'))); close() }
  const save = async () => {
    if (!token) return
    setSaving(true)
    try { const loaded = await request<EngineerSettings>('POST', '/engineer/settings/save', { token, values }); setSettings(loaded); setValues(loaded.values); setDirty(false); notify('工程师配置已保存，新建任务时生效') } catch (error) { notify(errorNotice(error, '保存工程师配置')) } finally { setSaving(false) }
  }
  const resetGroup = (group: string) => {
    if (!settings) return
    const next = { ...values }; settings.definitions.filter((field) => field.group === group).forEach((field) => { next[field.path] = field.defaultValue }); setValues(next); setDirty(true)
  }
  if (!open) return null
  return <div className="engineerMask" onMouseDown={() => void lock()}><aside className="engineerPanel" onMouseDown={(event) => event.stopPropagation()}>
    <div className="engineerHeader"><div><div className="eyebrow"><LockKeyhole size={13} />受保护区域</div><h2>工程师面板</h2><p>调整算法阈值前请确认设备与样品状态。</p></div><button type="button" className="closeButton" onClick={() => void lock()} aria-label="关闭"><X size={18} /></button></div>
    {!settings ? <div className="engineerLogin"><div className="engineerLoginIcon"><KeyRound size={25} /></div><h3>输入工程师密码</h3><p>面板用于调整检测与标定参数，普通任务无需使用。</p><Field label="密码"><input autoFocus type="password" value={password} onChange={(event) => setPassword(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void unlock() }} /></Field><PrimaryButton onClick={() => void unlock()} disabled={unlocking || !password}><KeyRound size={14} />{unlocking ? '验证中…' : '解锁面板'}</PrimaryButton><small className="engineerShortcut">快捷键：Ctrl + L</small></div> : <>
      <div className="engineerToolbar"><span><LockKeyhole size={14} />已解锁，修改后点击保存</span><div><SecondaryButton onClick={() => void lock()}><LockKeyhole size={14} />立即锁定</SecondaryButton><PrimaryButton onClick={() => setReviewing(true)} disabled={saving || !dirty}><Save size={14} />保存配置</PrimaryButton></div></div>
      <div className="engineerNotice"><TriangleAlert size={15} /><span>高风险参数可能影响移台和标定结果。配置只影响新建任务，正在运行的任务保持原有快照。</span></div>
      <div className="engineerGroups">{['navigation', 'sem', 'calibration'].map((group) => <section className="engineerGroup" key={group}><div className="engineerGroupHeader"><div><h3>{settings.definitions.find((field) => field.group === group)?.groupLabel}</h3><span>{group === 'navigation' ? '导航图与 AB 孔粗定位' : group === 'sem' ? 'AB 孔闭环圆心精定位' : '光镜到电镜坐标标定验收'}</span></div><button type="button" className="resetButton" onClick={() => resetGroup(group)}><RotateCcw size={13} />恢复本组默认</button></div>{settings.definitions.filter((field) => field.group === group).map((field) => <EngineerField key={field.path} field={field} value={values[field.path]} setValue={(value) => { setValues((current) => ({ ...current, [field.path]: value })); setDirty(true) }} />)}</section>)}</div>
      {reviewing && <EngineerReview definitions={settings.definitions} before={settings.values} after={values} saving={saving} cancel={() => setReviewing(false)} confirm={() => { setReviewing(false); void save() }} />}
    </>}
  </aside></div>
}

function EngineerReview({ definitions, before, after, saving, cancel, confirm }: { definitions: EngineerDefinition[]; before: Record<string, number>; after: Record<string, number>; saving: boolean; cancel: () => void; confirm: () => void }) {
  const changes = definitions.filter((field) => before[field.path] !== after[field.path])
  const hasHighRisk = changes.some((field) => field.risk === 'high')
  return <div className="reviewMask"><div className="reviewDialog"><div className="reviewHeader"><div><h3>确认配置变更</h3><p>以下参数将在新建任务时生效。</p></div><button type="button" className="closeButton" onClick={cancel} aria-label="关闭确认"><X size={16} /></button></div>{hasHighRisk && <div className="engineerNotice"><TriangleAlert size={15} /><span>本次包含高风险参数，可能改变移台范围、采图行为或标定验收结果。</span></div>}<div className="changeList">{changes.map((field) => <div key={field.path}><span><strong>{field.label}</strong><small>{field.groupLabel}</small></span><code>{before[field.path]}{field.unit ? ` ${field.unit}` : ''}</code><ChevronRight size={14} /><code className="newValue">{after[field.path]}{field.unit ? ` ${field.unit}` : ''}</code></div>)}</div><div className="reviewActions"><SecondaryButton onClick={cancel}>返回修改</SecondaryButton><PrimaryButton onClick={confirm} disabled={saving}><Check size={14} />确认并保存</PrimaryButton></div></div></div>
}

function EngineerField({ field, value, setValue }: { field: EngineerDefinition; value: number; setValue: (value: number) => void }) {
  const displayValue = Number.isFinite(value) ? value : field.defaultValue
  return <div className={`engineerField ${field.risk === 'high' ? 'highRisk' : ''}`}><div className="engineerFieldLabel"><strong>{field.label}</strong><span>{field.unit || '无单位'} · 默认 {field.defaultValue}</span></div><div className="engineerFieldControl"><input type="number" min={field.minimum} max={field.maximum} step={field.step} value={displayValue} onChange={(event) => setValue(Number(event.target.value))} /><span>{field.unit}</span><button type="button" className="helpButton" aria-label={`${field.label}说明`}><Info size={14} /><div className="engineerTooltip"><strong>{field.label}</strong><p>{field.summary}</p><p><b>调低：</b>{field.lowerEffect}</p><p><b>调高：</b>{field.higherEffect}</p><p><b>例子：</b>{field.example}</p><small>范围：{field.minimum} - {field.maximum}{field.unit ? ` ${field.unit}` : ''} · {field.risk === 'high' ? '高风险' : '需谨慎'}</small></div></button></div></div>
}

function DeviceDrawer({ devices, refresh, close }: { devices: DeviceResource[]; refresh: () => void; close: () => void }) {
  return (
    <div className="drawerMask" onMouseDown={close}>
      <aside className="drawer" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawerHeader"><h2>设备状态</h2><button type="button" onClick={close}><X size={18} /></button></div>
        {devices.map((device) => <div key={device.id}><DeviceSummary icon={device.id === 'microscope' ? Microscope : Activity} name={device.name} detail={`${device.host}:${device.port}`} status={device.status} /><div className="drawerDetails"><SummaryRow label="连接状态" value={deviceLabel(device)} /><SummaryRow label="设备响应" value={device.detail || '—'} /></div></div>)}
        <PrimaryButton onClick={refresh}><RefreshCw size={14} />刷新状态</PrimaryButton>
      </aside>
    </div>
  )
}

export default App
