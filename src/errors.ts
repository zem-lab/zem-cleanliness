export type NoticeKind = 'success' | 'info' | 'error'

export type NoticeInput = string | {
  kind?: NoticeKind
  message: string
  context?: string
  details?: string
}

export type ErrorDiagnosis = {
  title: string
  summary: string
  guidance: string[]
  technicalDetails: string
}

export type ErrorRecord = ErrorDiagnosis & {
  id: string
  context: string
  createdAt: string
  fingerprint: string
  count: number
}

const ERROR_STORAGE_KEY = 'zem-cleanliness-error-records-v1'
const MAX_ERROR_RECORDS = 40

export function errorNotice(error: unknown, context: string): NoticeInput {
  const details = error instanceof Error ? error.message : String(error)
  return { kind: 'error', message: details, details, context }
}

export function diagnoseError(error: unknown, context = ''): ErrorDiagnosis {
  const technicalDetails = error instanceof Error ? error.message : String(error || '未知错误')
  const normalized = technicalDetails.toLowerCase()
  const address = extractAddress(technicalDetails)
  const service = inferService(technicalDetails, address?.port)

  if (/connection refused|actively refused|积极拒绝|10061/.test(normalized)) {
    return {
      title: `${service}连接失败`,
      summary: address
        ? `无法连接 ${address.host}:${address.port}，目标端口拒绝连接，通常表示设备端服务未启动或已经退出。`
        : '目标服务拒绝连接，通常表示服务未启动、已经退出或端口配置不正确。',
      guidance: [
        `确认${service}已启动并处于可连接状态。`,
        address ? `核对设备地址 ${address.host}:${address.port} 与系统设置是否一致。` : '核对系统设置中的设备 IP 和端口。',
        '恢复服务后重新执行；若仍失败，再检查防火墙和设备网络。',
      ],
      technicalDetails,
    }
  }

  if (/deadline exceeded|timed out|timeout|连接超时|请求超时/.test(normalized)) {
    return {
      title: `${service}连接超时`,
      summary: address
        ? `连接 ${address.host}:${address.port} 超时，设备可能离线、网络不通或服务响应过慢。`
        : '请求等待超时，目标服务可能离线、网络不通或响应过慢。',
      guidance: ['确认设备在线且网络可达。', '检查设备服务是否卡住，必要时重启设备端服务。', '确认 IP、端口和防火墙配置。'],
      technicalDetails,
    }
  }

  if (/failed to fetch|networkerror|network error|backend|本地检测服务|fetch failed/.test(normalized)) {
    return {
      title: '本地检测服务不可用',
      summary: '前端无法访问本地检测后端，可能是后端尚未启动、已经退出或端口发生变化。',
      guidance: ['确认桌面程序或 pnpm dev 仍在运行。', '查看启动终端中 Python 后端是否退出。', '重新启动应用后再试。'],
      technicalDetails,
    }
  }

  if (/model/.test(normalized) && /not found|missing|不存在|缺失/.test(normalized)) {
    return {
      title: '检测模型缺失',
      summary: '检测所需的模型文件不存在或无法读取，当前步骤不能继续。',
      guidance: ['在系统设置中检查模型状态。', '确认 resources/models 下的模型文件完整。', '重新安装或补齐模型资源后再试。'],
      technicalDetails,
    }
  }

  if (/no such file|file not found|不存在|找不到/.test(normalized)) {
    return {
      title: '文件或资源不存在',
      summary: firstUsefulLine(technicalDetails),
      guidance: ['确认文件没有被移动、删除或重命名。', '重新选择输入文件或恢复缺失资源后再试。'],
      technicalDetails,
    }
  }

  if (/permission denied|access denied|拒绝访问|没有权限/.test(normalized)) {
    return {
      title: '文件访问被拒绝',
      summary: '程序没有权限读取或写入目标文件。',
      guidance: ['确认文件未被其他程序占用。', '检查任务目录和目标文件的读写权限。'],
      technicalDetails,
    }
  }

  return {
    title: context ? `${context}失败` : '操作执行失败',
    summary: firstUsefulLine(technicalDetails),
    guidance: ['根据下方技术详情检查失败步骤和相关参数。', '问题处理后重新执行；重复出现时可复制技术详情用于排查。'],
    technicalDetails,
  }
}

export function createErrorRecord(error: unknown, context = ''): ErrorRecord {
  const diagnosis = diagnoseError(error, context)
  const fingerprint = `${context}|${diagnosis.title}|${diagnosis.summary}`
  return {
    ...diagnosis,
    id: createId(),
    context,
    createdAt: new Date().toISOString(),
    fingerprint,
    count: 1,
  }
}

export function mergeErrorRecord(records: ErrorRecord[], incoming: ErrorRecord): ErrorRecord[] {
  const existing = records.find((record) => record.fingerprint === incoming.fingerprint)
  if (!existing) return [incoming, ...records].slice(0, MAX_ERROR_RECORDS)
  const updated = {
    ...existing,
    ...incoming,
    id: existing.id,
    count: existing.count + 1,
  }
  return [updated, ...records.filter((record) => record.id !== existing.id)].slice(0, MAX_ERROR_RECORDS)
}

export function loadErrorRecords(): ErrorRecord[] {
  try {
    const stored = window.localStorage.getItem(ERROR_STORAGE_KEY)
    if (!stored) return []
    const parsed = JSON.parse(stored)
    return Array.isArray(parsed) ? parsed.slice(0, MAX_ERROR_RECORDS) : []
  } catch {
    return []
  }
}

export function saveErrorRecords(records: ErrorRecord[]): void {
  try {
    window.localStorage.setItem(ERROR_STORAGE_KEY, JSON.stringify(records.slice(0, MAX_ERROR_RECORDS)))
  } catch {
    // Error history is supplementary and must never break the inspection flow.
  }
}

function extractAddress(message: string): { host: string; port: string } | null {
  const match = message.match(/(?:ipv4:)?((?:\d{1,3}\.){3}\d{1,3}):(\d{2,5})/i)
  return match ? { host: match[1], port: match[2] } : null
}

function inferService(message: string, port?: string): string {
  if (/opensession|eds|能谱/i.test(message) || port === '6200') return 'EDS 能谱服务'
  if (/microscope|sem|电镜/i.test(message) || port === '61557') return '电镜服务'
  return '设备服务'
}

function firstUsefulLine(message: string): string {
  const line = message
    .split(/\r?\n/)
    .map((item) => item.trim())
    .find(Boolean) || '发生未知错误，请展开技术详情查看原始信息。'
  return line.length > 220 ? `${line.slice(0, 217)}...` : line
}

function createId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}
