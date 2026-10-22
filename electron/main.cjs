const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const { spawn, spawnSync } = require('node:child_process')
const crypto = require('node:crypto')
const fs = require('node:fs')
const net = require('node:net')
const path = require('node:path')

let backend = null
let backendBaseUrl = ''
let acceptanceTestOptions = null
let backendStopRequested = false
let logDirectory = ''
let electronLogPath = ''
const sessionId = crypto.randomUUID()

function allocateListenPort() {
  const server = net.createServer()
  server.listen(0, '127.0.0.1')
  const address = server.address()
  const port = typeof address === 'object' && address ? address.port : 9333
  server.close()
  return port
}

const cdpPort = Number(process.env.ZEM_CDP_PORT) || allocateListenPort()
app.commandLine.appendSwitch('remote-debugging-port', String(cdpPort))
app.commandLine.appendSwitch('remote-debugging-address', '127.0.0.1')

const LOG_MAX_BYTES = 10 * 1024 * 1024
const SENSITIVE_FIELD = /password|token|secret|authorization|api[-_]?key/i

function ensureLogDirectory() {
  if (logDirectory) return logDirectory
  const base = app.isReady() ? app.getPath('userData') : path.join(__dirname, '..', '.tmp')
  logDirectory = path.join(base, 'logs')
  fs.mkdirSync(logDirectory, { recursive: true })
  electronLogPath = path.join(logDirectory, 'electron.jsonl')
  rotateLog(electronLogPath)
  return logDirectory
}

function rotateLog(file) {
  try {
    if (!fs.existsSync(file) || fs.statSync(file).size < LOG_MAX_BYTES) return
    const archived = file.replace(/\.jsonl$/, '.previous.jsonl')
    if (fs.existsSync(archived)) fs.rmSync(archived)
    fs.renameSync(file, archived)
  } catch (error) {
    console.error('[logging] rotation failed', error)
  }
}

function sanitizeLogValue(value, depth = 0) {
  if (value instanceof Error) return { name: value.name, message: value.message, stack: value.stack }
  if (value === null || value === undefined || typeof value === 'number' || typeof value === 'boolean') return value
  if (typeof value === 'string') return value.length > 8000 ? `${value.slice(0, 8000)}...[truncated]` : value
  if (depth >= 4) return '[max-depth]'
  if (Array.isArray(value)) return value.slice(0, 50).map((item) => sanitizeLogValue(item, depth + 1))
  if (typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).slice(0, 80).map(([key, item]) => [
      key,
      SENSITIVE_FIELD.test(key) ? '[redacted]' : sanitizeLogValue(item, depth + 1)
    ]))
  }
  return String(value)
}

function writeLog(level, event, details = {}) {
  try {
    ensureLogDirectory()
    const record = {
      timestamp: new Date().toISOString(),
      level,
      process: 'electron',
      pid: process.pid,
      sessionId,
      event,
      ...sanitizeLogValue(details)
    }
    fs.appendFileSync(electronLogPath, `${JSON.stringify(record)}\n`, 'utf8')
  } catch (error) {
    console.error('[logging] write failed', error)
  }
}

function summarizeRequestBody(body) {
  if (!body || typeof body !== 'object') return undefined
  const summary = { keys: Object.keys(body).filter((key) => !SENSITIVE_FIELD.test(key)) }
  for (const key of ['inputPath', 'mode', 'selectedGroup', 'accepted', 'aIndex', 'bIndex', 'handedness']) {
    if (Object.hasOwn(body, key)) summary[key] = sanitizeLogValue(body[key])
  }
  if (Array.isArray(body.selectedSequences)) summary.selectedCount = body.selectedSequences.length
  return summary
}

function logBackendChunk(stream, chunk) {
  const content = chunk.toString('utf8').trim()
  if (!content) return
  for (const line of content.split(/\r?\n/)) writeLog(stream === 'stderr' ? 'error' : 'info', `backend_${stream}`, { message: line })
}

function readAcceptanceTestOptions() {
  const marker = process.env.ZEM_ACCEPTANCE_TEST_FILE
  if (!marker || !fs.existsSync(marker)) return null
  try {
    const parsed = JSON.parse(fs.readFileSync(marker, 'utf8'))
    return {
      opticalJson: String(parsed.opticalJson || ''),
      fakeEds: Boolean(parsed.fakeEds),
      expectedParticles: Number(parsed.expectedParticles || 0)
    }
  } catch (error) {
    console.error(`[acceptance] invalid marker: ${error}`)
    return null
  }
}

function stopBackend() {
  if (!backend) return
  const pid = backend.pid
  backendStopRequested = true
  writeLog('info', 'backend_stop_requested', { backendPid: pid })
  if (pid) {
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
    } else if (!backend.killed) {
      backend.kill()
    }
  }
  backend = null
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 8765
      server.close(() => resolve(port))
    })
  })
}

async function waitForBackend(url) {
  let lastError
  const startedAt = Date.now()
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`${url}/health`)
      if (response.ok) {
        writeLog('info', 'backend_ready', { url, attempt: attempt + 1, durationMs: Date.now() - startedAt })
        return
      }
    } catch (error) {
      lastError = error
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`本地检测服务启动超时：${lastError ? String(lastError) : '无响应'}`)
}

async function startBackend() {
  backendStopRequested = false
  acceptanceTestOptions = readAcceptanceTestOptions()
  const port = await getFreePort()
  backendBaseUrl = `http://127.0.0.1:${port}`
  const taskRoot = process.env.ZEM_TASK_ROOT || path.join(app.getPath('documents'), 'ZEM Cleanliness', 'tasks')
  const environment = {
    ...process.env,
    ZEM_TASK_ROOT: taskRoot,
    ZEM_LOG_DIR: ensureLogDirectory(),
    ZEM_SESSION_ID: sessionId,
    ZEM_RESOURCE_ROOT: app.isPackaged
      ? path.join(process.resourcesPath, 'resources')
      : path.join(__dirname, '..', 'resources'),
    // Keep Windows' locale for subprocess output decoded by the legacy ZEM
    // discovery helper, while making this backend's own stdout UTF-8 for Electron.
    PYTHONIOENCODING: 'utf-8',
    PYTHONUNBUFFERED: '1',
    ZEM_MCP_CDP_URL: `http://127.0.0.1:${cdpPort}`,
    ZEM_MCP_ATTACH: '1',
    ZEM_MCP_ALLOW_HARDWARE: '1'
  }
  if (acceptanceTestOptions?.fakeEds) environment.ZEM_FAKE_EDS = '1'
  let executable
  let args
  let cwd
  if (app.isPackaged) {
    executable = path.join(process.resourcesPath, 'backend', 'zem-cleanliness-backend.exe')
    args = ['--host', '127.0.0.1', '--port', String(port)]
    cwd = path.dirname(executable)
  } else {
    const projectRoot = path.join(__dirname, '..')
    const virtualEnvPython = process.platform === 'win32'
      ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
      : path.join(projectRoot, '.venv', 'bin', 'python')
    executable = process.env.ZEM_PYTHON || (fs.existsSync(virtualEnvPython) ? virtualEnvPython : 'python')
    args = [path.join(__dirname, '..', 'backend', 'server.py'), '--host', '127.0.0.1', '--port', String(port)]
    cwd = projectRoot
  }
  writeLog('info', 'backend_start_requested', {
    executable,
    args,
    cwd,
    taskRoot,
    resourceRoot: environment.ZEM_RESOURCE_ROOT,
    packaged: app.isPackaged,
    acceptanceTest: Boolean(acceptanceTestOptions)
  })
  backend = spawn(executable, args, {
    cwd,
    env: environment,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  writeLog('info', 'backend_started', { backendPid: backend.pid, baseUrl: backendBaseUrl })
  backend.stdout.on('data', (chunk) => {
    console.log(`[backend] ${chunk.toString().trim()}`)
    logBackendChunk('stdout', chunk)
  })
  backend.stderr.on('data', (chunk) => {
    console.error(`[backend] ${chunk.toString().trim()}`)
    logBackendChunk('stderr', chunk)
  })
  backend.on('error', (error) => writeLog('error', 'backend_process_error', { error }))
  backend.on('exit', (code, signal) => {
    writeLog(backendStopRequested || code === 0 || code === null ? 'info' : 'error', 'backend_exited', { code, signal, expected: backendStopRequested })
    console.log(`[backend] exited code=${code} signal=${signal || ''}`)
  })
  await waitForBackend(backendBaseUrl)
}

async function apiRequest(method, route, body, suppliedRequestId) {
  if (!backendBaseUrl) throw new Error('本地检测服务尚未启动')
  const requestId = typeof suppliedRequestId === 'string' && suppliedRequestId ? suppliedRequestId : crypto.randomUUID()
  const startedAt = Date.now()
  writeLog('info', 'api_request_started', { requestId, method, route, body: summarizeRequestBody(body) })
  try {
    const response = await fetch(`${backendBaseUrl}${route}`, {
      method,
      headers: {
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        'X-Request-Id': requestId
      },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
    const payload = await response.json()
    writeLog(response.ok ? 'info' : 'warn', 'api_request_finished', {
      requestId,
      method,
      route,
      status: response.status,
      durationMs: Date.now() - startedAt,
      error: response.ok ? undefined : payload.error
    })
    if (!response.ok) throw new Error(payload.error || `请求失败 (${response.status})`)
    return payload
  } catch (error) {
    writeLog('error', 'api_request_failed', { requestId, method, route, durationMs: Date.now() - startedAt, error })
    throw error
  }
}

function registerIpc() {
  ipcMain.handle('cleanliness:request', (_event, method, route, body, requestId) => apiRequest(method, route, body, requestId))
  ipcMain.handle('cleanliness:chooseOpticalJson', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: '光镜颗粒结果', extensions: ['json'] }]
    })
    return result.canceled ? null : result.filePaths[0] || null
  })
  ipcMain.handle('cleanliness:openPath', async (_event, target) => {
    writeLog('info', 'path_open_requested', { target: String(target) })
    const error = await shell.openPath(String(target))
    if (error) throw new Error(error)
  })
  ipcMain.handle('cleanliness:getRuntime', () => ({ backendBaseUrl, packaged: app.isPackaged, version: app.getVersion(), logDirectory: ensureLogDirectory(), agentControlled: process.env.ZEM_AGENT_CONTROL === '1' }))
  ipcMain.handle('cleanliness:getAcceptanceTest', () => acceptanceTestOptions)
  ipcMain.handle('cleanliness:log', (_event, level, event, details) => {
    const normalizedLevel = ['debug', 'info', 'warn', 'error'].includes(level) ? level : 'info'
    writeLog(normalizedLevel, `renderer_${String(event || 'event')}`, details && typeof details === 'object' ? details : { message: String(details || '') })
  })
}

function createWindow() {
  const windowIcon = app.isPackaged
    ? path.join(process.resourcesPath, 'icon.ico')
    : path.join(__dirname, '..', 'build', 'icon.ico')
  const window = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#eef2f6',
    autoHideMenuBar: true,
    show: false,
    icon: windowIcon,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })
  writeLog('info', 'window_created', { windowId: window.id, size: window.getSize() })
  window.once('ready-to-show', () => {
    writeLog('info', 'window_ready_to_show', { windowId: window.id })
    window.show()
  })
  window.webContents.on('did-finish-load', () => writeLog('info', 'renderer_loaded', { windowId: window.id, url: window.webContents.getURL() }))
  window.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL) => writeLog('error', 'renderer_load_failed', { windowId: window.id, errorCode, errorDescription, url: validatedURL }))
  window.webContents.on('render-process-gone', (_event, details) => writeLog('error', 'renderer_process_gone', { windowId: window.id, details }))
  window.on('unresponsive', () => writeLog('warn', 'window_unresponsive', { windowId: window.id }))
  window.on('responsive', () => writeLog('info', 'window_responsive', { windowId: window.id }))
  window.on('closed', () => writeLog('info', 'window_closed', { windowId: window.id }))
  if (!app.isPackaged && process.env.VITE_DEV_SERVER_URL) {
    window.loadURL(process.env.VITE_DEV_SERVER_URL)
  } else {
    window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  }
}

app.whenReady().then(async () => {
  ensureLogDirectory()
  writeLog('info', 'application_start', {
    version: app.getVersion(),
    packaged: app.isPackaged,
    platform: process.platform,
    arch: process.arch,
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
    userData: app.getPath('userData'),
    cdpUrl: `http://127.0.0.1:${cdpPort}`
  })
  registerIpc()
  try {
    await startBackend()
  } catch (error) {
    writeLog('error', 'application_start_failed', { error })
    dialog.showErrorBox('启动失败', error instanceof Error ? error.message : String(error))
    app.quit()
    return
  }
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  writeLog('info', 'all_windows_closed')
  if (process.platform !== 'darwin') app.quit()
})

app.on('will-quit', () => {
  writeLog('info', 'application_will_quit')
  stopBackend()
})

process.on('warning', (warning) => writeLog('warn', 'process_warning', { warning }))
process.on('unhandledRejection', (reason) => writeLog('error', 'unhandled_rejection', { reason }))
process.on('uncaughtException', (error) => writeLog('error', 'uncaught_exception', { error }))
