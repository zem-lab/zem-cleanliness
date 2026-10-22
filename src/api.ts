export type ApiMethod = 'GET' | 'POST'

export type RuntimeInfo = {
  backendBaseUrl: string
  packaged: boolean
  version: string
  logDirectory: string
  agentControlled?: boolean
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export type AcceptanceTestOptions = {
  opticalJson: string
  fakeEds: boolean
  expectedParticles: number
}

export type ModelResource = {
  id: 'particle' | 'fixture' | 'hole'
  file: string
  path: string
  ready: boolean
  size: number
  sha256: string | null
}

export type DeviceResource = {
  id: 'microscope' | 'eds'
  name: string
  host: string
  port: number
  status: 'connected' | 'busy' | 'offline' | 'unknown'
  detail: string
}

export type ParticlePreview = {
  sequence: number
  particleId: string
  label: string
  abPositionUm: { x: number; y: number }
  lengthUm: number
  widthUm: number
  areaUm2: number
  equivalentDiameterUm: number
}

export type OpticalPreview = {
  path: string
  sha256: string
  projectId: string
  workpieceId: string
  workpieceName: string
  coordinateUnit: string
  coordinateSpace: string
  registrationMode: string
  particleCount: number
  selectedCount: number
  particles: ParticlePreview[]
}

export type TaskSummary = {
  id: string
  mode?: string
  status: string
  input?: {
    path?: string
    particleCount?: number
    selectedCount?: number
    workpieceName?: string
  }
  updatedAt?: string
}

export type PreflightReport = {
  enabled: boolean
  passed: boolean
  checkedAt?: string
  items: Array<{ id: string; label: string; status: 'PASS' | 'WARN' | 'FAIL' | 'SKIP'; detail: string }>
}

export type JobState = {
  id: string
  taskId: string
  operation: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  total: number
  message: string
  error?: string
  result?: TaskManifest
  preflight?: PreflightReport
}

export type TaskManifest = Record<string, unknown> & {
  taskId?: string
  folderId?: string
  taskDirectory?: string
  status?: string
  input?: Record<string, unknown>
  selection?: Record<string, unknown>
  calibration?: Record<string, unknown>
  particles?: Array<Record<string, unknown>>
  reports?: Record<string, string>
  error?: string
}

type DesktopBridge = {
  request<T>(method: ApiMethod, route: string, body?: unknown, requestId?: string): Promise<T>
  chooseOpticalJson(): Promise<string | null>
  openPath(target: string): Promise<void>
  getRuntime(): Promise<RuntimeInfo>
  getAcceptanceTest?(): Promise<AcceptanceTestOptions | null>
  log(level: LogLevel, event: string, details?: Record<string, unknown>): Promise<void>
}

declare global {
  interface Window {
    zemCleanliness?: DesktopBridge
  }
}

const browserBackend = 'http://127.0.0.1:8765'

export const isDesktop = () => Boolean(window.zemCleanliness)

export function logClientEvent(level: LogLevel, event: string, details: Record<string, unknown> = {}): void {
  if (window.zemCleanliness) {
    void window.zemCleanliness.log(level, event, details).catch(() => undefined)
    return
  }
  const method = level === 'error' ? console.error : level === 'warn' ? console.warn : console.info
  method(`[zem-cleanliness:${event}]`, details)
}

export async function request<T>(method: ApiMethod, route: string, body?: unknown): Promise<T> {
  const requestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
  const startedAt = performance.now()
  logClientEvent('debug', 'api_request_started', { requestId, method, route })
  try {
    if (window.zemCleanliness) {
      const result = await window.zemCleanliness.request<T>(method, route, body, requestId)
      logClientEvent('debug', 'api_request_finished', { requestId, method, route, durationMs: Math.round(performance.now() - startedAt) })
      return result
    }
    const response = await fetch(`${browserBackend}${route}`, {
      method,
      headers: {
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        'X-Request-Id': requestId
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
    const payload = await response.json() as T & { error?: string }
    if (!response.ok) throw new Error(payload.error || `请求失败 (${response.status})`)
    logClientEvent('debug', 'api_request_finished', { requestId, method, route, status: response.status, durationMs: Math.round(performance.now() - startedAt) })
    return payload
  } catch (error) {
    logClientEvent('error', 'api_request_failed', { requestId, method, route, durationMs: Math.round(performance.now() - startedAt), error: error instanceof Error ? error.message : String(error) })
    throw error
  }
}

export async function chooseOpticalJson(): Promise<string | null> {
  if (!window.zemCleanliness) throw new Error('请在 ZEM Cleanliness 桌面软件中选择文件')
  return window.zemCleanliness.chooseOpticalJson()
}

export async function openPath(target: string): Promise<void> {
  if (!window.zemCleanliness) throw new Error('请在桌面软件中打开报告')
  await window.zemCleanliness.openPath(target)
}

export async function getRuntime(): Promise<RuntimeInfo> {
  if (window.zemCleanliness) return window.zemCleanliness.getRuntime()
  return { backendBaseUrl: browserBackend, packaged: false, version: '0.1.7', logDirectory: '' }
}

export async function backendFileUrl(path: string): Promise<string> {
  const base = window.zemCleanliness ? (await getRuntime()).backendBaseUrl : browserBackend
  return `${base}/chat/file?path=${encodeURIComponent(path)}`
}

export async function getAcceptanceTest(): Promise<AcceptanceTestOptions | null> {
  if (window.zemCleanliness?.getAcceptanceTest) return window.zemCleanliness.getAcceptanceTest()
  return null
}

export async function waitForJob(jobId: string, onUpdate: (job: JobState) => void): Promise<JobState> {
  logClientEvent('info', 'job_polling_started', { jobId })
  for (;;) {
    const job = await request<JobState>('GET', `/jobs/${jobId}`)
    onUpdate(job)
    if (job.status === 'completed') {
      logClientEvent('info', 'job_polling_completed', { jobId, taskId: job.taskId, operation: job.operation })
      return job
    }
    if (job.status === 'failed') {
      logClientEvent('error', 'job_polling_failed', { jobId, taskId: job.taskId, operation: job.operation, error: job.error || job.message })
      throw new Error(job.error || job.message)
    }
    await new Promise((resolve) => window.setTimeout(resolve, 750))
  }
}

export function formatBytes(size: number): string {
  if (!size) return '0 MB'
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() || path
}
