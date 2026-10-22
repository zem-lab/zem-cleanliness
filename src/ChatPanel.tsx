import { useEffect, useRef, useState } from 'react'
import { MessageCircle, Plus, Send, Square, X } from 'lucide-react'
import { backendFileUrl, request } from './api'

type ChatEventKind = 'user' | 'assistant' | 'action' | 'system' | 'error' | 'image' | 'debug'

type ChatEvent = {
  seq: number
  ts: string
  kind: ChatEventKind
  text: string
}

type ChatSession = {
  id: string
  title: string
  createdAt: string
  status: 'idle' | 'running' | 'failed'
  codexSessionId: string | null
  lastEventAt?: string
  events?: ChatEvent[]
  cursor?: number
}

type SessionSummary = {
  id: string
  title: string
  status: string
  createdAt: string
  lastEventAt?: string
}

type ChatRuntime = {
  engine?: string
  provider?: string
  launcherConfigured?: boolean
  model: string
  credentialConfigured: boolean | null
  mcp: { available: boolean; mode: string; captureEnabled: boolean; moveEnabled: boolean; uiMcp?: boolean; servers?: string[] }
}

const POLL_INTERVAL_MS = 750

/** 与后端 IMAGE_MARKER_RE 对齐：从气泡文本里抽出要展示的图片路径。 */
const IMAGE_MARKER_RE = /\[图片\][ \t]*(?:\()?[ \t]*(?:<)?[ \t]*("?)([A-Za-z]:[\\/][^\r\n`|]+?\.(?:png|jpe?g|bmp|gif))\1(?:>)?(?:\))?/gi

function extractImagePaths(text: string): string[] {
  const paths: string[] = []
  const seen = new Set<string>()
  for (const match of text.matchAll(IMAGE_MARKER_RE)) {
    const path = (match[2] || '').trim()
    if (!path || seen.has(path)) continue
    seen.add(path)
    paths.push(path)
  }
  return paths
}

function stripImageMarkers(text: string): string {
  return text.replace(IMAGE_MARKER_RE, '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
}

function isImageMarkerOnly(text: string): boolean {
  return extractImagePaths(text).length > 0 && !stripImageMarkers(text)
}

function normalizePathKey(path: string): string {
  return path.replace(/\//g, '\\').replace(/\\+/g, '\\').trim().toLowerCase()
}

function buildVisibleEvents(events: ChatEvent[]): ChatEvent[] {
  const shownImages = new Set<string>()
  const result: ChatEvent[] = []
  for (const event of events) {
    if (event.kind === 'debug') continue
    // 纯 [图片] 路径的 action 交给后续 image / 内联出图，避免只剩路径字串。
    if (event.kind === 'action' && isImageMarkerOnly(event.text)) continue
    if (event.kind === 'assistant' || event.kind === 'action') {
      for (const path of extractImagePaths(event.text)) shownImages.add(normalizePathKey(path))
      result.push(event)
      continue
    }
    if (event.kind === 'image') {
      const key = normalizePathKey(event.text)
      if (shownImages.has(key)) continue
      shownImages.add(key)
      result.push(event)
      continue
    }
    result.push(event)
  }
  return result
}

export function ChatToggleButton({ onClick }: { onClick: () => void }) {
  return (
    <button className="headerIconButton" type="button" aria-label="vibe 助手" title="vibe 助手" onClick={onClick}>
      <MessageCircle size={18} />
    </button>
  )
}

export function ChatPanel({ close }: { close: () => void }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [active, setActive] = useState<ChatSession | null>(null)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [runtime, setRuntime] = useState<ChatRuntime | null>(null)
  const [error, setError] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    request<{ sessions: SessionSummary[] }>('GET', '/chat/sessions')
      .then((payload) => { if (!cancelled) setSessions(payload.sessions) })
      .catch((reason: unknown) => { if (!cancelled) setError(`读取会话失败：${String(reason)}`) })
    request<ChatRuntime>('GET', '/chat/runtime')
      .then((payload) => { if (!cancelled) setRuntime(payload) })
      .catch((reason: unknown) => { if (!cancelled) setError(`读取助手连接信息失败：${String(reason)}`) })
    return () => { cancelled = true }
  }, [])

  // 轮询当前会话的事件增量。
  useEffect(() => {
    if (!active) return
    let stopped = false
    const tick = async () => {
      try {
        const session = await request<ChatSession>('GET', `/chat/sessions/${active.id}?after=${active.cursor ?? 0}`)
        if (stopped) return
        setActive((current) => {
          if (!current || current.id !== session.id) return current
          const incoming = session.events || []
          if (!incoming.length && session.status === current.status) return current
          return {
            ...current,
            status: session.status,
            codexSessionId: session.codexSessionId,
            events: [...(current.events || []), ...incoming],
            cursor: session.cursor ?? current.cursor,
          }
        })
      } catch {
        // 会话被删除等场景：静默，下一轮重新同步。
      }
    }
    const timer = window.setInterval(() => void tick(), POLL_INTERVAL_MS)
    void tick()
    return () => { stopped = true; window.clearInterval(timer) }
  }, [active?.id, active?.cursor])

  // 新事件自动滚到底部。
  useEffect(() => {
    const node = listRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [active?.events?.length])

  const createSession = async () => {
    try {
      setSending(true)
      setError('')
      const session = await request<ChatSession>('POST', '/chat/sessions', {})
      setActive(session)
      setSessions((current) => [{
        id: session.id, title: session.title, status: session.status, createdAt: session.createdAt,
      }, ...current])
    } catch (reason: unknown) {
      setError(`创建会话失败：${String(reason)}`)
    } finally {
      setSending(false)
    }
  }

  const switchSession = (id: string) => {
    if (active?.id === id) return
    request<ChatSession>('GET', `/chat/sessions/${id}`)
      .then((session) => setActive(session))
      .catch((reason: unknown) => setError(`读取会话失败：${String(reason)}`))
  }

  const send = async () => {
    const text = input.trim()
    if (!text || !active || sending || active.status === 'running') return
    setSending(true)
    setError('')
    try {
      await request('POST', `/chat/sessions/${active.id}/messages`, { text })
      setInput('')
    } catch (reason: unknown) {
      setError(`发送失败：${String(reason)}`)
    } finally {
      setSending(false)
    }
  }

  const cancelRun = async () => {
    if (!active) return
    setError('')
    try { await request('POST', `/chat/sessions/${active.id}/cancel`, {}) }
    catch (reason: unknown) { setError(`取消请求失败：${String(reason)}`) }
  }

  const busy = active?.status === 'running'
  const visibleEvents = buildVisibleEvents(active?.events || [])

  return (
    <aside className="chatPanel" aria-label="vibe 助手">
      <header className="chatHeader">
        <strong>vibe 助手</strong>
        <div className="chatHeaderActions">
          <select
            className="chatSessionSelect"
            value={active?.id || ''}
            onChange={(event) => { if (event.target.value) switchSession(event.target.value) }}
            aria-label="选择会话"
          >
            {!active && <option value="">选择会话</option>}
            {sessions.map((session) => (
              <option key={session.id} value={session.id}>{session.title}</option>
            ))}
            {active && !sessions.some((session) => session.id === active.id) && (
              <option value={active.id}>{active.title}</option>
            )}
          </select>
          <button className="chatIconButton" type="button" aria-label="新会话" title="新会话" onClick={() => void createSession()} disabled={sending}>
            <Plus size={15} />
          </button>
          <button className="chatIconButton" type="button" aria-label="收起" title="收起" onClick={close}>
            <X size={15} />
          </button>
        </div>
      </header>

      {runtime && (
        <div className="chatLine chatSystem" role="status">
          {runtime.mcp.available
            ? `${runtime.mcp.mode === 'simulated' ? '模拟设备' : '真实电镜接口'} · ${runtime.mcp.uiMcp ? '软件界面 MCP 已接入' : '软件界面 MCP 未接入'} · ${runtime.mcp.moveEnabled ? '移动已开放' : '移动未开放'} · ${runtime.mcp.captureEnabled ? '采图已开放' : '采图未开放'}`
            : '设备 MCP 尚未就绪'}
          {runtime.engine === 'codex' ? (
            <div>{runtime.launcherConfigured
              ? `Codex · CPA（${runtime.provider}）· 启动入口就绪，实际连接需通过对话验证。`
              : 'Codex 启动入口不存在，请检查本机 ZEM_VIBE_CODEX_CMD 配置。'}</div>
          ) : <>
            {runtime.credentialConfigured === false && <div>请先在本机 .env.local 填写 DeepSeek 密钥，再重启软件。</div>}
            {runtime.credentialConfigured === true && <div>{runtime.model} 配置已填写，实际连接需通过对话验证。</div>}
          </>}
        </div>
      )}
      {error && <div className="chatLine chatError" role="alert">{error}</div>}

      {busy && (
        <div className="chatBusyBanner">
          <span className="chatBusyDot" /> AI 正在操作软件…
          <button className="chatCancelButton" type="button" onClick={() => void cancelRun()}>
            <Square size={11} /> 取消
          </button>
        </div>
      )}

      <div className="chatMessages" ref={listRef}>
        {!active && <div className="chatEmpty">选择一个会话，或点击 + 开始新会话。<br />试试：“查询电镜状态”。</div>}
        {active && visibleEvents.length === 0 && <div className="chatEmpty">描述你想做的事情，例如：<br />“查询电镜支持哪些操作，再获取已有导航图”。</div>}
        {visibleEvents.map((event) => <ChatBubble key={event.seq} event={event} />)}
      </div>

      <footer className="chatInputRow">
        <textarea
          className="chatInput"
          value={input}
          placeholder={busy ? 'AI 正在执行，请等待…' : '输入指令，Enter 发送，Shift+Enter 换行'}
          disabled={!active || busy}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              void send()
            }
          }}
        />
        <div className="chatInputActions">
          <span className="chatInputHint">Enter 发送 · Shift+Enter 换行</span>
          <button className="chatSendButton" type="button" onClick={() => void send()} disabled={!active || busy || sending || !input.trim()}>
            <Send size={14} />
          </button>
        </div>
      </footer>
    </aside>
  )
}

function ChatBubble({ event }: { event: ChatEvent }) {
  if (event.kind === 'user') {
    return <div className="chatBubble chatUser"><span>{event.text}</span></div>
  }
  if (event.kind === 'assistant') {
    const paths = extractImagePaths(event.text)
    const body = stripImageMarkers(event.text)
    // 助手正文去掉 [图片] 行；有路径时直接出图（不依赖单独 image 事件）。
    return (
      <div className="chatAssistantGroup">
        {body ? <div className="chatBubble chatAssistant"><span>{body}</span></div> : null}
        {paths.map((path) => <ChatImage key={`inline-${path}`} path={path} />)}
      </div>
    )
  }
  if (event.kind === 'image') {
    return <ChatImage path={event.text} />
  }
  if (event.kind === 'action') {
    return <ChatAction event={event} />
  }
  if (event.kind === 'error') {
    return <div className="chatLine chatError">{event.text}</div>
  }
  return <div className="chatLine chatSystem">{event.text}</div>
}

function ChatAction({ event }: { event: ChatEvent }) {
  const [expanded, setExpanded] = useState(false)
  const paths = extractImagePaths(event.text)
  const body = stripImageMarkers(event.text)
  const isCommand = event.text.startsWith('$ ') || event.text.startsWith('⚙ ')
  const display = body || event.text
  const firstLine = display.split('\n')[0] || ''
  const summary = firstLine.length > 120 ? `${firstLine.slice(0, 120)}…` : firstLine
  if (isCommand) {
    return <pre className="chatLine chatAction chatActionExpanded">{event.text}</pre>
  }
  return (
    <>
      {body ? (
        <pre
          className={expanded ? 'chatLine chatAction chatActionExpanded' : 'chatLine chatAction chatActionCollapsed'}
          onClick={() => setExpanded((value) => !value)}
          title={expanded ? '点击收起' : '点击展开全部输出'}
        >
          {expanded ? display : `${summary}\n▸ 点击展开工具输出`}
        </pre>
      ) : null}
      {paths.map((path) => <ChatImage key={`action-${path}`} path={path} />)}
    </>
  )
}

function ChatImage({ path }: { path: string }) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    let cancelled = false
    let objectUrl = ''
    const load = async () => {
      try {
        const href = await backendFileUrl(path)
        // 先拉成 blob，避免 file:// 页面对 http 图片偶发空白；失败再退回直链。
        try {
          const response = await fetch(href)
          if (!response.ok) throw new Error(`HTTP ${response.status}`)
          const blob = await response.blob()
          if (cancelled) return
          objectUrl = URL.createObjectURL(blob)
          setUrl(objectUrl)
          setError('')
          return
        } catch {
          if (cancelled) return
          setUrl(href)
          setError('')
        }
      } catch (reason: unknown) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason))
      }
    }
    void load()
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [path])
  if (error) {
    return <div className="chatLine chatError">图片加载失败：{path.split(/[\\/]/).pop() || path}（{error}）</div>
  }
  if (!url) return <div className="chatLine chatSystem">图片加载中…</div>
  return (
    <div className="chatImageWrap">
      <img
        src={url}
        alt={path.split(/[\\/]/).pop() || '图片'}
        title={path}
        onClick={() => window.open(url, '_blank')}
        onError={() => setError('无法解码或地址不可达')}
      />
    </div>
  )
}
