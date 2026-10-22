// MCP server that lets agents drive the real ZEPTOOLS 清洁度 desktop app.
// The app is launched via Playwright's Electron support (same stack as e2e/*.cjs)
// with ZEM_AGENT_CONTROL=1, so the user sees the UI operated like a human and the
// window shows the agent-control badge and blue frame.
//
// 范围约定：光电联用（光镜关联分析）功能不进 MCP；ROI 圈选只能由人操作；
// 工程师面板（密码保护的高风险参数区）不进 MCP。观测类工具通过应用自身的
// window.zemCleanliness.request 读取数据（界面无变化），操作类一律真实 UI 点击。
const fs = require('node:fs')
const path = require('node:path')
const { createRequire } = require('node:module')
const { z } = require('zod')
const { _electron: electron } = require('playwright-core')
const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js')
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js')

const REPO_ROOT = path.resolve(__dirname, '..', '..')
const SESSIONS_ROOT = path.join(REPO_ROOT, '.tmp', 'mcp-sessions')

const state = {
  app: null,
  browser: null,
  page: null,
  sessionDir: '',
  screenshotsDir: SESSIONS_ROOT,
  consoleErrors: [],
  allowHardware: false,
  attached: false,
}

const log = (...args) => process.stderr.write(`[zem-mcp] ${args.join(' ')}\n`)

function requirePage() {
  if (!state.page) throw new Error('应用尚未启动，请先调用 app_launch')
  return state.page
}

function requireHardware(operation) {
  if (!state.allowHardware) {
    throw new Error(`「${operation}」会操作真实电镜设备，已拒绝执行。确认设备与样品安全后，用 app_launch({ allowHardware: true }) 或设置 ZEM_MCP_ALLOW_HARDWARE=1 再试。`)
  }
}

async function markAgentControl(page, active) {
  try {
    await page.evaluate((value) => {
      window.dispatchEvent(new CustomEvent('zem-agent-control', { detail: value }))
    }, active)
  } catch (error) {
    log('mark agent control failed:', error)
  }
}

async function attachExistingApp() {
  const cdpUrl = process.env.ZEM_MCP_CDP_URL
  if (!cdpUrl) return false
  log('attaching via CDP', cdpUrl)
  const { chromium } = require('playwright-core')
  const browser = await chromium.connectOverCDP(cdpUrl)
  const pages = browser.contexts().flatMap((context) => context.pages())
  const page = pages.find((item) => {
    const url = item.url()
    return url.includes('index.html') || url.includes('127.0.0.1') || url.includes('localhost')
  }) || pages[0]
  if (!page) {
    await browser.close().catch(() => undefined)
    throw new Error(`已连接调试端口但没有窗口：${cdpUrl}`)
  }
  if (state.app && !state.attached) await closeApp()
  state.browser = browser
  state.attached = true
  state.app = { close: async () => { try { await browser.close() } catch (error) { log('cdp close:', error) } } }
  state.page = page
  state.allowHardware = process.env.ZEM_MCP_ALLOW_HARDWARE !== '0'
  state.sessionDir = path.join(SESSIONS_ROOT, `attached-${process.pid}`)
  state.screenshotsDir = path.join(state.sessionDir, 'screenshots')
  fs.mkdirSync(state.screenshotsDir, { recursive: true })
  attachConsoleWatch(page)
  await markAgentControl(page, true)
  return {
    running: true,
    attached: true,
    sessionDir: state.sessionDir,
    allowHardware: state.allowHardware,
    cdpUrl,
    url: page.url(),
  }
}

// 观测透传：借用应用自己的 IPC 通道读后端数据，不改变界面
async function backendRequest(method, route, body) {
  const page = requirePage()
  return page.evaluate(async ({ method: m, route: r, body: b }) => {
    const payload = await window.zemCleanliness.request(m, r, b)
    return JSON.parse(JSON.stringify(payload))
  }, { method, route, body })
}

async function currentSemTaskId() {
  const page = requirePage()
  return page.evaluate(() => document.querySelector('.semDirect')?.getAttribute('data-task-id') || '')
}

async function attachConsoleWatch(page) {
  state.consoleErrors = []
  page.on('console', (message) => {
    if (message.type() === 'error') state.consoleErrors.push(message.text())
  })
  page.on('pageerror', (error) => state.consoleErrors.push(error.message))
}

async function launchApp({ pythonPath, taskRoot, allowHardware }) {
  if (process.env.ZEM_MCP_CDP_URL) {
    const attached = await attachExistingApp()
    if (allowHardware !== undefined) state.allowHardware = Boolean(allowHardware)
    return { ...attached, allowHardware: state.allowHardware }
  }
  if (state.app) await closeApp()
  fs.mkdirSync(SESSIONS_ROOT, { recursive: true })
  const sessionDir = path.join(SESSIONS_ROOT, `${new Date().toISOString().replace(/[:.]/g, '-')}-${process.pid}`)
  state.sessionDir = sessionDir
  state.screenshotsDir = path.join(sessionDir, 'screenshots')
  fs.mkdirSync(state.screenshotsDir, { recursive: true })
  state.allowHardware = Boolean(allowHardware)
  const effectiveTaskRoot = taskRoot || path.join(sessionDir, 'tasks')
  const rootRequire = createRequire(path.join(REPO_ROOT, 'package.json'))
  const electronPath = rootRequire('electron')
  log('launching electron:', electronPath, 'taskRoot:', effectiveTaskRoot, 'allowHardware:', state.allowHardware)
  const app = await electron.launch({
    executablePath: electronPath,
    args: ['.'],
    cwd: REPO_ROOT,
    env: {
      ...process.env,
      ZEM_AGENT_CONTROL: '1',
      ZEM_TASK_ROOT: effectiveTaskRoot,
      ...(pythonPath ? { ZEM_PYTHON: pythonPath } : {}),
    },
  })
  // 后端 Python 冷启动可能拖慢首窗出现，给足容忍
  const page = await app.firstWindow({ timeout: 120000 })
  state.app = app
  state.page = page
  attachConsoleWatch(page)
  await page.getByRole('heading', { name: '首页' }).waitFor({ timeout: 120000 })
  await page.getByText('后端已连接').waitFor({ timeout: 120000 })
  const runtime = await page.evaluate(() => window.zemCleanliness.getRuntime())
  await page.locator('.agentControlBadge').waitFor({ timeout: 10000 })
  return {
    running: true,
    sessionDir,
    taskRoot: effectiveTaskRoot,
    allowHardware: state.allowHardware,
    version: runtime.version,
    backendBaseUrl: runtime.backendBaseUrl,
    logDirectory: runtime.logDirectory,
  }
}

async function closeApp() {
  if (!state.app) return { running: false, note: '应用未在运行' }
  const app = state.app
  const attached = state.attached
  if (state.page) await markAgentControl(state.page, false).catch(() => undefined)
  state.app = null
  state.browser = null
  state.page = null
  state.attached = false
  try {
    await app.close()
  } catch (error) {
    log('app close error:', error)
  }
  return { running: false, attached, sessionDir: state.sessionDir, note: attached ? '已断开当前窗口，软件仍在运行' : undefined }
}

const NAV_TARGETS = {
  home: { name: '首页' },
  reports: { name: '任务记录' },
  settings: { name: '系统设置' },
}

async function navigateTo(target) {
  const page = requirePage()
  if (target === 'new-task' || target === 'sem-direct') {
    await page.getByRole('button', { name: '新建检测任务', exact: true }).click()
    if (target === 'sem-direct') {
      await page.getByRole('button', { name: '直接 SEM 分析' }).click()
      await page.getByRole('heading', { name: '直接 SEM 分析' }).waitFor({ timeout: 15000 })
      return { page: 'sem-direct' }
    }
    return { page: 'choose-mode' }
  }
  const targetInfo = NAV_TARGETS[target]
  if (!targetInfo) throw new Error(`未知页面: ${target}（可选 home/reports/settings/new-task/sem-direct）`)
  await page.getByRole('button', { name: targetInfo.name }).click()
  await page.getByRole('heading', { name: targetInfo.name }).waitFor({ timeout: 15000 })
  return { page: target }
}

async function ensureSemDirectPage() {
  const page = requirePage()
  const heading = page.getByRole('heading', { name: '直接 SEM 分析' })
  if ((await heading.count()) === 0) await navigateTo('sem-direct')
  return page
}

async function readUi() {
  const page = requirePage()
  return page.evaluate(() => {
    const clean = (value) => (value || '').replace(/\s+/g, ' ').trim()
    const badge = document.querySelector('.agentControlBadge')
    const fields = [...document.querySelectorAll('label')].slice(0, 80).map((label) => {
      const control = label.querySelector('input, select, textarea')
      const name = clean(label.textContent).split('\n')[0]
      if (!control) return null
      const value = control.type === 'checkbox' ? String(control.checked) : String(control.value ?? '')
      return `${name} = ${value}`
    }).filter(Boolean)
    return {
      agentBadge: badge ? clean(badge.textContent) : null,
      headings: [...document.querySelectorAll('h1, h2')].slice(0, 30).map((node) => clean(node.textContent)),
      statusLabels: [...document.querySelectorAll('.semState, .plainStatus, .statusBar span')].slice(0, 20).map((node) => clean(node.textContent)),
      buttons: [...document.querySelectorAll('button')].slice(0, 80).map((button) => clean(button.getAttribute('aria-label') || button.textContent)).filter(Boolean),
      fields,
      alerts: [...document.querySelectorAll('[role="alert"], .semError, .toast')].map((node) => clean(node.textContent)).filter(Boolean),
    }
  })
}

// ---------- 任务管理 ----------

async function getTasks() {
  const result = await backendRequest('GET', '/tasks')
  return {
    total: result.tasks.length,
    tasks: result.tasks.map((task) => ({
      id: task.id,
      mode: task.mode,
      status: task.status,
      workpieceName: task.input?.workpieceName || null,
      particleCount: task.input?.particleCount ?? task.input?.selectedCount ?? null,
      updatedAt: task.updatedAt || null,
    })),
  }
}

async function openTask(taskId) {
  const list = await getTasks()
  const target = list.tasks.find((task) => task.id === taskId)
  if (!target) throw new Error(`任务不存在: ${taskId}。可用 get_tasks 查看任务列表。`)
  if (target.mode !== 'sem_direct') {
    throw new Error(`任务 ${taskId} 是光镜关联分析任务（mode=${target.mode}），该模块不在 MCP 支持范围内，请人工在软件中打开。`)
  }
  const page = requirePage()
  await navigateTo('reports')
  const row = page.locator('.reportsTable .tableRow').filter({ hasText: taskId })
  await row.waitFor({ timeout: 10000 })
  await row.getByRole('button', { name: '打开', exact: true }).click()
  await page.getByRole('heading', { name: '直接 SEM 分析' }).waitFor({ timeout: 15000 })
  return { opened: taskId, page: 'sem-direct' }
}

async function getReportData(taskId) {
  if (!taskId) taskId = await currentSemTaskId()
  if (!taskId) throw new Error('未指定 taskId，且当前「直接 SEM 分析」页没有打开的任务')
  return backendRequest('GET', `/tasks/${taskId}/report-data`)
}

// ---------- 设备 ----------

async function getDevices() {
  const result = await backendRequest('GET', '/devices')
  return { devices: result.devices.map(({ id, name, host, port, status, detail }) => ({ id, name, host, port, status, detail: detail || null })) }
}

async function refreshDevices() {
  const page = requirePage()
  const before = await getDevices()
  await page.locator('.headerActions .connectionButton').first().click()
  await page.getByRole('button', { name: '刷新状态' }).waitFor({ timeout: 10000 })
  await page.getByRole('button', { name: '刷新状态' }).click()
  await page.waitForTimeout(1500)
  const after = await getDevices()
  await page.locator('.drawerMask', { includeInvisible: false }).first().click({ position: { x: 8, y: 8 } }).catch(() => undefined)
  return {
    before: before.devices.map((device) => ({ id: device.id, status: device.status, detail: device.detail })),
    after: after.devices.map((device) => ({ id: device.id, status: device.status, detail: device.detail })),
  }
}

// ---------- 系统设置 ----------

const SETTING_FIELDS = {
  microscopeHost: { kind: 'text', pick: 0 },
  microscopePort: { kind: 'text', pick: 0 },
  edsHost: { kind: 'text', pick: 1 },
  edsPort: { kind: 'text', pick: 1 },
}

const SETTING_SWITCHES = {
  edsEnabled: '执行真实 EDS 分析',
  restoreStartPosition: '任务结束后恢复样品台位置',
  gunOffAfterTask: '检测结束后自动关枪',
  preflightEnabled: '开工前状态检查',
}

async function getSettings() {
  return backendRequest('GET', '/settings')
}

async function setSettings(patch) {
  const page = requirePage()
  const wasOnSemDirect = (await page.getByRole('heading', { name: '直接 SEM 分析' }).count()) > 0
  await navigateTo('settings')
  for (const [key, label] of Object.entries(SETTING_SWITCHES)) {
    if (patch[key] === undefined) continue
    const control = page.getByRole('switch', { name: label })
    await control.waitFor({ timeout: 10000 })
    const checked = (await control.getAttribute('aria-checked')) === 'true'
    if (checked !== Boolean(patch[key])) await control.click()
  }
  if (patch.captureLevel !== undefined) {
    await page.getByLabel('拍照档位').selectOption(String(patch.captureLevel))
  }
  for (const key of Object.keys(SETTING_FIELDS)) {
    if (patch[key] === undefined) continue
    const info = SETTING_FIELDS[key]
    const box = key.endsWith('Port')
      ? page.locator('label').filter({ hasText: '端口' }).nth(info.pick).locator('input')
      : page.locator('label').filter({ hasText: '主机地址' }).nth(info.pick).locator('input')
    await box.fill(String(patch[key]))
  }
  await page.getByRole('button', { name: '保存设置' }).click()
  await page.getByText('设置已保存').waitFor({ timeout: 15000 })
  const saved = await getSettings()
  if (wasOnSemDirect) await navigateTo('sem-direct')
  return { saved: true, settings: saved }
}

// ---------- 错误中心 ----------

async function readErrors() {
  const page = requirePage()
  const badge = page.locator('.errorHistoryButton')
  const countText = await badge.locator('span').first().textContent().catch(() => null)
  const headerCount = countText ? Number(countText) : 0
  if (!headerCount) return { count: 0, records: [] }
  await badge.click()
  await page.locator('.errorCenter').waitFor({ timeout: 10000 })
  const records = await page.evaluate(() => {
    const clean = (value) => (value || '').replace(/\s+/g, ' ').trim()
    return [...document.querySelectorAll('.errorRecord')].map((node) => ({
      title: clean(node.querySelector('.errorRecordTitle strong')?.textContent),
      meta: clean(node.querySelector('.errorRecordTitle small')?.textContent),
      summary: clean(node.querySelector('p')?.textContent),
    }))
  })
  await page.getByRole('button', { name: '关闭错误记录' }).click()
  return { count: records.length, records }
}

async function clearErrors() {
  const page = requirePage()
  const before = await readErrors()
  if (!before.count) return { cleared: true, hadErrors: 0 }
  await page.locator('.errorHistoryButton').click()
  await page.locator('.errorCenter').waitFor({ timeout: 10000 })
  await page.getByRole('button', { name: '清空记录' }).click()
  await page.getByRole('button', { name: '关闭错误记录' }).click()
  return { cleared: true, hadErrors: before.count }
}

// ---------- SEM 直接分析 ----------

async function semGetNavigation() {
  requireHardware('获取导航图')
  const page = await ensureSemDirectPage()
  const button = page.getByRole('button', { name: '获取导航图', exact: true }).first()
  await button.waitFor({ timeout: 15000 })
  await button.click()
  await page.locator('img[alt="电镜导航图"]').waitFor({ timeout: 120000 })
  const taskId = await currentSemTaskId()
  const task = taskId ? await backendRequest('GET', `/tasks/${taskId}`) : null
  return {
    taskId,
    navigation: task?.navigation || null,
    fieldUm: task?.navigation?.fieldUm || null,
    magnification: task?.navigation?.magnification || null,
  }
}

async function setScanSettings(patch) {
  const page = await ensureSemDirectPage()
  if (patch.name !== undefined) await page.getByLabel('样品名称').fill(patch.name)
  if (patch.shape !== undefined) {
    await page.locator('.semShape').getByRole('button', { name: patch.shape === 'circle' ? '圆形' : '正方形', exact: true }).click()
  }
  if (patch.fieldUm !== undefined) await page.getByLabel('单视野 FOV（μm）').fill(String(patch.fieldUm))
  if (patch.imageSize !== undefined) await page.locator('label').filter({ hasText: '图像分辨率' }).locator('select').selectOption(String(patch.imageSize))
  if (patch.overlapPercent !== undefined) await page.getByLabel('视野重叠（%）').fill(String(patch.overlapPercent))
  if (patch.minLengthUm !== undefined) await page.getByLabel('最小颗粒长度（μm）').fill(String(patch.minLengthUm))
  if (patch.edsEnabled !== undefined) {
    const box = page.locator('.semCheck input[type="checkbox"]')
    const checked = await box.isChecked()
    if (checked !== Boolean(patch.edsEnabled)) await box.click()
  }
  await page.waitForTimeout(200)
  return readScanSettings()
}

async function readScanSettings() {
  const page = requirePage()
  return {
    name: await page.getByLabel('样品名称').inputValue().catch(() => ''),
    shape: await page.locator('.semShape button.selected').textContent().catch(() => null),
    fieldUm: await page.getByLabel('单视野 FOV（μm）').inputValue().catch(() => null),
    imageSize: await page.locator('label').filter({ hasText: '图像分辨率' }).locator('select').inputValue().catch(() => null),
    overlapPercent: await page.getByLabel('视野重叠（%）').inputValue().catch(() => null),
    minLengthUm: await page.getByLabel('最小颗粒长度（μm）').inputValue().catch(() => null),
    edsEnabled: await page.locator('.semCheck input[type="checkbox"]').isChecked().catch(() => null),
    note: '设置在「预览扫描网格」或「开始自动分析」时才保存到任务；扫描区域（ROI）需由人在导航图上圈选',
  }
}

async function setSegmentationModel({ backend, url, texts, score }) {
  const page = await ensureSemDirectPage()
  const modelSelect = page.locator('label').filter({ hasText: '颗粒分割模型' }).locator('select')
  await modelSelect.waitFor({ timeout: 15000 })
  await modelSelect.selectOption(backend)
  // SAM3 字段的 label 内含 <small> 说明文字，必须用子串匹配（同 e2e 脚本）
  if (backend === 'sam3') {
    if (url !== undefined) await page.getByLabel('SAM3 服务地址').fill(url)
    if (texts !== undefined) await page.getByLabel('分割提示词').fill(texts)
    if (score !== undefined) await page.getByLabel('SAM3 置信度阈值').fill(String(score))
  }
  await page.waitForTimeout(200)
  const result = { backend, currentBackend: await modelSelect.inputValue() }
  if (result.currentBackend === 'sam3') {
    result.currentUrl = await page.getByLabel('SAM3 服务地址').inputValue()
    result.currentTexts = await page.getByLabel('分割提示词').inputValue()
    result.currentScore = await page.getByLabel('SAM3 置信度阈值').inputValue()
  }
  return result
}

async function semPreviewGrid() {
  const page = await ensureSemDirectPage()
  const button = page.getByRole('button', { name: '预览扫描网格' })
  await button.waitFor({ timeout: 15000 })
  if (!(await button.isEnabled())) {
    throw new Error('「预览扫描网格」不可用：需要先获取导航图并由人在导航图上圈选扫描区域（ROI 圈选只能人工完成）')
  }
  await button.click()
  await page.locator('.semGrid').first().waitFor({ timeout: 30000 })
  await page.waitForTimeout(500)
  return semGetStatus()
}

async function semStartAnalysis() {
  requireHardware('开始自动分析')
  const page = await ensureSemDirectPage()
  const button = page.getByRole('button', { name: /开始自动分析|继续扫描/ })
  await button.waitFor({ timeout: 15000 })
  if (!(await button.isEnabled())) {
    throw new Error('开始按钮不可用：需要先获取导航图、由人圈选扫描区域（ROI），并完成扫描设置。当前状态可用 sem_get_status 查看。')
  }
  await button.click()
  await page.getByRole('button', { name: '暂停扫描' }).waitFor({ timeout: 30000 })
  return semGetStatus()
}

async function semPauseAnalysis() {
  const page = await ensureSemDirectPage()
  const button = page.getByRole('button', { name: '暂停扫描' })
  await button.waitFor({ timeout: 15000 })
  await button.click()
  return { paused: true, note: '已请求暂停，当前视野采谱完成后停止' }
}

async function semGenerateReport() {
  const page = await ensureSemDirectPage()
  const button = page.getByRole('button', { name: /生成当前结果报告|正在生成…/ })
  await button.waitFor({ timeout: 15000 })
  if (!(await button.isEnabled())) {
    throw new Error('生成报告不可用：需要已有扫描计划且任务不在运行中。可用 sem_get_status 查看当前状态。')
  }
  await button.click()
  const taskId = await currentSemTaskId()
  const deadline = Date.now() + 180000
  while (Date.now() < deadline) {
    await page.waitForTimeout(2000)
    const task = taskId ? await backendRequest('GET', `/tasks/${taskId}`) : null
    if (task?.reportState === 'READY') return { reportState: 'READY', reports: task.reports || {}, taskDirectory: task.taskDirectory || null }
    if (task?.reportState === 'FAILED') throw new Error(`报告生成失败: ${task.reportError || '未知原因'}`)
  }
  throw new Error('报告生成超时（180 秒），可用 sem_get_status 查看当前状态后重试')
}

async function semGetStatus() {
  const page = requirePage()
  const dom = await page.evaluate(() => {
    const clean = (value) => (value || '').replace(/\s+/g, ' ').trim()
    const one = (selector) => document.querySelector(selector)
    return {
      onSemDirectPage: Boolean(one('.semDirect')),
      stateLabel: one('.semState') ? clean(one('.semState').textContent) : null,
      steps: [...document.querySelectorAll('.semSteps span')].map((node) => clean(node.textContent)),
      stats: [...document.querySelectorAll('.semStats > div')].map((node) => ({
        label: clean(node.querySelector('span')?.textContent),
        value: clean(node.querySelector('strong')?.textContent),
      })),
      errorBanner: one('.semError') ? clean(one('.semError').textContent) : null,
      jobMessage: one('.semResults .semCardHead span') ? clean(one('.semResults .semCardHead span').textContent) : null,
      particleRowsShown: document.querySelectorAll('.semTableWrap tbody tr').length,
      recentParticles: [...document.querySelectorAll('.semTableWrap tbody tr')].slice(-8).map((row) => clean(row.textContent)),
    }
  })
  const taskId = dom.onSemDirectPage ? await currentSemTaskId() : ''
  let task = null
  if (taskId) {
    task = await backendRequest('GET', `/tasks/${taskId}`)
  }
  return {
    ...dom,
    taskId: taskId || null,
    taskStatus: task?.status || null,
    reportState: task?.reportState || null,
    reportError: task?.reportError || null,
    reports: task?.reports || null,
    taskDirectory: task?.taskDirectory || null,
    model: task?.model || null,
    totalParticles: task?.particles?.length ?? null,
    gunOff: task?.cleanup || null,
  }
}

async function semGetParticles({ limit }) {
  const taskId = await currentSemTaskId()
  if (!taskId) throw new Error('当前「直接 SEM 分析」页没有打开的任务。先用 sem_get_navigation 获取导航图（需硬件），或用 open_task 打开历史任务。')
  const task = await backendRequest('GET', `/tasks/${taskId}`)
  const particles = task.particles || []
  const effectiveLimit = Math.max(1, limit || 100)
  return {
    taskId,
    total: particles.length,
    returned: Math.min(effectiveLimit, particles.length),
    particles: particles.slice(0, effectiveLimit).map((particle) => ({
      sequence: particle.sequence,
      tileId: particle.tileId,
      status: particle.status,
      lengthUm: particle.lengthUm ?? null,
      areaUm2: particle.areaUm2 ?? null,
      reason: particle.reason || null,
      elements: particle.spectrum?.elements?.slice(0, 5).map((element) => `${element.symbol} ${Number(element.value).toFixed(1)}%`) || null,
    })),
  }
}

async function semOpen(target) {
  const page = await ensureSemDirectPage()
  const taskId = await currentSemTaskId()
  if (!taskId) throw new Error('当前没有打开的 SEM 任务')
  const task = await backendRequest('GET', `/tasks/${taskId}`)
  const kindMap = { report_pdf: 'pdf', report_html: 'html', report_csv: 'csv', report_timing: 'timingHtml' }
  if (target === 'task_directory') {
    if (!task.taskDirectory) throw new Error('任务目录未知')
    await page.evaluate((dir) => window.zemCleanliness.openPath(dir), task.taskDirectory)
    return { opened: 'task_directory', path: task.taskDirectory }
  }
  const kind = kindMap[target]
  const relative = task.reports?.[kind]
  if (!relative) throw new Error(`该任务没有 ${target}（reports: ${Object.keys(task.reports || {}).join(', ') || '无'}）`)
  const fullPath = `${task.taskDirectory}/${relative}`
  await page.evaluate((file) => window.zemCleanliness.openPath(file), fullPath)
  return { opened: target, path: fullPath }
}

// ---------- 通用 ----------

async function takeScreenshot({ fullPage }) {
  const page = requirePage()
  const file = path.join(state.screenshotsDir, `${Date.now()}-${fullPage ? 'full' : 'view'}.png`)
  const buffer = await page.screenshot({ path: file, fullPage: Boolean(fullPage) })
  return { file, base64: buffer.toString('base64') }
}

async function failureDetail(error) {
  const lines = [`错误: ${error && error.message ? error.message : String(error)}`]
  try {
    if (state.page) {
      const file = path.join(state.screenshotsDir, `${Date.now()}-failure.png`)
      await state.page.screenshot({ path: file })
      lines.push(`现场截图: ${file}`)
    }
  } catch (screenshotError) {
    log('failure screenshot error:', screenshotError)
  }
  return lines.join('\n')
}

const server = new McpServer({ name: 'zem-cleanliness', version: '0.2.0' })

function registerTool(name, { description, schema, annotations }, handler) {
  const wrapped = async (args) => {
    try {
      const data = await handler(args || {})
      if (data && data.__image) {
        log(`tool ${name}: ok(image)`)
        return {
          content: [
            { type: 'image', data: data.__image.base64, mimeType: 'image/png' },
            { type: 'text', text: JSON.stringify(data.__meta) },
          ],
        }
      }
      log(`tool ${name}: ok`)
      return { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }] }
    } catch (error) {
      log(`tool ${name}: error -> ${error && error.message ? error.message : String(error)}`)
      return { content: [{ type: 'text', text: await failureDetail(error) }], isError: true }
    }
  }
  if (typeof server.registerTool === 'function') {
    server.registerTool(name, { description, inputSchema: schema, annotations }, wrapped)
  } else {
    server.tool(name, description, schema, wrapped)
  }
}

// 生命周期
registerTool('app_launch', {
  description: '启动或连接清洁度软件。内置聊天已注入当前窗口时会连接到现有软件，不再新开窗口。外部 Agent 才会新开 Electron。默认隔离任务目录。默认 allowHardware=false（当前窗口连接时可由环境放行）：获取导航图、开始自动分析等触碰电镜的操作需确认后放行。',
  schema: {
    pythonPath: z.string().optional().describe('后端 Python 解释器路径；缺省时使用仓库 .venv 或系统 python'),
    taskRoot: z.string().optional().describe('自定义任务数据目录；缺省时使用本次会话的隔离目录'),
    allowHardware: z.boolean().optional().describe('是否允许触碰真实电镜设备的操作（获取导航图、开始自动分析），默认 false'),
  },
}, (args) => launchApp(args))

registerTool('app_close', {
  description: '关闭由 app_launch 启动的软件。',
  schema: {},
}, () => closeApp())

registerTool('app_status', {
  description: '查询软件是否在运行、当前窗口地址、会话目录与硬件操作许可。',
  schema: {},
}, async () => {
  if (!state.page && process.env.ZEM_MCP_CDP_URL) {
    try { await attachExistingApp() } catch (error) { return { running: false, error: String(error) } }
  }
  if (!state.app) return { running: false }
  return {
    running: true,
    attached: state.attached,
    url: state.page ? state.page.url() : null,
    sessionDir: state.sessionDir,
    allowHardware: state.allowHardware,
    consoleErrorCount: state.consoleErrors.length,
  }
})

// 导航与原子操作
registerTool('navigate', {
  description: '切换软件页面：home 首页 / reports 任务记录 / settings 系统设置 / new-task 新建检测任务（模式选择页）/ sem-direct 直接 SEM 分析。',
  schema: {
    target: z.enum(['home', 'reports', 'settings', 'new-task', 'sem-direct']).describe('目标页面'),
  },
}, ({ target }) => navigateTo(target))

registerTool('ui_read', {
  description: '读取当前界面结构：标题、状态、按钮、表单字段值、错误提示，以及 Agent 控制标识是否显示。',
  schema: {},
}, () => readUi())

registerTool('ui_click', {
  description: '点击界面按钮（按可见文字匹配，不区分大小写、支持子串）。',
  schema: {
    text: z.string().describe('按钮文字，例如「获取导航图」「保存设置」'),
    exact: z.boolean().optional().describe('是否精确匹配，默认否'),
  },
}, ({ text, exact }) => requirePage().getByRole('button', { name: text, exact: Boolean(exact) }).first().click({ timeout: 15000 }).then(() => ({ clicked: text })))

registerTool('ui_fill', {
  description: '按标签填写输入框；同名标签出现多次时用 index 指定（从 0 开始）。',
  schema: {
    label: z.string().describe('字段标签文字'),
    value: z.string().describe('要填入的值'),
    index: z.number().optional().describe('同名第几个字段，默认 0'),
  },
}, ({ label, value, index }) => requirePage().getByLabel(label).nth(index || 0).fill(value, { timeout: 15000 }).then(() => ({ filled: label, value })))

registerTool('ui_select', {
  description: '按标签选择下拉框选项；同名标签出现多次时用 index 指定。',
  schema: {
    label: z.string().describe('字段标签文字'),
    value: z.string().describe('选项 value'),
    index: z.number().optional().describe('同名第几个下拉框，默认 0'),
  },
}, async ({ label, value, index }) => {
  const page = requirePage()
  const select = page.locator('label').filter({ hasText: label }).locator('select').nth(index || 0)
  await select.waitFor({ timeout: 15000 })
  await select.selectOption(value)
  return { selected: label, value: await select.inputValue() }
})

// 观测
registerTool('screenshot', {
  description: '截取软件当前界面并返回图片，同时保存到会话目录。',
  schema: {
    fullPage: z.boolean().optional().describe('是否截取整页（含滚动区域），默认仅视口'),
  },
}, async ({ fullPage }) => {
  const shot = await takeScreenshot({ fullPage })
  return { __image: { base64: shot.base64 }, __meta: { file: shot.file } }
})

registerTool('console_errors', {
  description: '返回启动以来渲染进程的 console 错误与未捕获异常列表。',
  schema: {},
}, () => state.consoleErrors)

registerTool('get_errors', {
  description: '读取软件错误中心的记录（用户可见错误的摘要，含重复次数）。',
  schema: {},
}, () => readErrors())

registerTool('clear_errors', {
  description: '清空软件错误中心的记录。',
  schema: {},
}, () => clearErrors())

// 任务管理
registerTool('get_tasks', {
  description: '列出全部检测任务（含直接 SEM 分析与光镜关联任务；光镜任务只能查看，不能通过 MCP 打开）。',
  schema: {},
}, () => getTasks())

registerTool('open_task', {
  description: '在任务记录页打开指定的直接 SEM 分析任务并进入其页面。光镜关联分析任务不属于 MCP 支持范围。',
  schema: {
    taskId: z.string().describe('任务 ID，可从 get_tasks 获取'),
  },
}, ({ taskId }) => openTask(taskId))

registerTool('get_report_data', {
  description: '获取任务的完整报告数据（manifest + 颗粒明细等，数据量可能较大）。不指定 taskId 时取「直接 SEM 分析」页当前打开的任务。',
  schema: {
    taskId: z.string().optional().describe('任务 ID；缺省取当前打开的 SEM 任务'),
  },
}, ({ taskId }) => getReportData(taskId))

// 设备
registerTool('get_devices', {
  description: '读取电镜与 EDS 能谱服务的连接状态。',
  schema: {},
}, () => getDevices())

registerTool('refresh_devices', {
  description: '触发软件重新检测设备连接状态（等价于界面上的测试连接/刷新状态）。',
  schema: {},
}, () => refreshDevices())

// 系统设置
registerTool('get_settings', {
  description: '读取系统设置（设备地址、EDS、自动关枪、开工检查、拍照档位等）。',
  schema: {},
}, () => getSettings())

registerTool('set_settings', {
  description: '修改系统设置并保存（真实填写设置页并点击保存）。仅支持通用字段；光镜关联分析专属参数与工程师面板不在 MCP 范围内。',
  schema: {
    microscopeHost: z.string().optional().describe('电镜主机地址'),
    microscopePort: z.number().optional().describe('电镜端口'),
    edsHost: z.string().optional().describe('EDS 服务主机地址'),
    edsPort: z.number().optional().describe('EDS 服务端口'),
    edsEnabled: z.boolean().optional().describe('执行真实 EDS 分析'),
    restoreStartPosition: z.boolean().optional().describe('任务结束后恢复样品台位置'),
    gunOffAfterTask: z.boolean().optional().describe('检测结束后自动关枪'),
    preflightEnabled: z.boolean().optional().describe('开工前状态检查'),
    captureLevel: z.number().min(1).max(7).optional().describe('拍照档位 1（最快）–7（最清晰）'),
  },
}, (patch) => setSettings(patch))

// SEM 直接分析
registerTool('sem_get_navigation', {
  description: '「直接 SEM 分析」页点击获取导航图（连接电镜读取导航图，不移动样品台）。需要 app_launch 时 allowHardware=true。获取后由人圈选扫描区域（ROI）。',
  schema: {},
}, () => semGetNavigation())

registerTool('set_scan_settings', {
  description: '填写「直接 SEM 分析」的扫描设置：样品名称、区域形状、单视野 FOV、图像分辨率、视野重叠、最小颗粒长度、是否自动采谱。返回界面回读的当前值。设置在预览网格或开始分析时才保存到任务。',
  schema: {
    name: z.string().optional().describe('样品名称'),
    shape: z.enum(['square', 'circle']).optional().describe('区域形状'),
    fieldUm: z.number().optional().describe('单视野 FOV（μm）'),
    imageSize: z.union([z.literal(512), z.literal(1024), z.literal(2048)]).optional().describe('图像分辨率'),
    overlapPercent: z.number().min(0).max(40).optional().describe('视野重叠（%）'),
    minLengthUm: z.number().optional().describe('最小颗粒长度（μm）'),
    edsEnabled: z.boolean().optional().describe('分割后自动采集 EDS'),
  },
}, (patch) => setScanSettings(patch))

registerTool('set_segmentation_model', {
  description: '选择颗粒分割模型：yolo（本地）或 sam3（远程）；选 sam3 时可同时设置服务地址、提示词与置信度阈值。返回界面回读的当前配置。',
  schema: {
    backend: z.enum(['yolo', 'sam3']).describe('分割模型'),
    url: z.string().optional().describe('SAM3 服务地址（仅 sam3）'),
    texts: z.string().optional().describe('分割提示词，逗号分隔（仅 sam3）'),
    score: z.number().optional().describe('SAM3 置信度阈值 0.001–1（仅 sam3）'),
  },
}, (args) => setSegmentationModel(args))

registerTool('sem_preview_grid', {
  description: '点击「预览扫描网格」：按当前设置与已圈选区域计算扫描网格（不碰设备）。需要人先在导航图上圈选 ROI。返回视野数与选区面积。',
  schema: {},
}, () => semPreviewGrid())

registerTool('sem_start_analysis', {
  description: '点击开始自动分析/继续扫描（会移动样品台、拍照、按设置采谱，任务结束可能自动关枪）。需要 app_launch 时 allowHardware=true，且人已圈选扫描区域。',
  schema: {},
}, () => semStartAnalysis())

registerTool('sem_pause_analysis', {
  description: '请求暂停扫描：当前视野采谱完成后停止，进度保留。',
  schema: {},
}, () => semPauseAnalysis())

registerTool('sem_generate_report', {
  description: '生成当前结果报告（等价于界面「生成当前结果报告」按钮），等待完成后返回报告文件列表与任务目录。',
  schema: {},
}, () => semGenerateReport())

registerTool('sem_get_status', {
  description: '读取「直接 SEM 分析」任务状态：分析状态、视野/颗粒统计、分割模型、报告状态与文件、任务目录、关枪记录、最近颗粒。',
  schema: {},
}, () => semGetStatus())

registerTool('sem_get_particles', {
  description: '获取当前 SEM 任务的颗粒明细（尺寸、状态、能谱元素），界面仅展示最近 100 条，本工具返回全量（默认前 100 条，可用 limit 调整）。',
  schema: {
    limit: z.number().min(1).max(1000).optional().describe('返回条数上限，默认 100'),
  },
}, (args) => semGetParticles(args))

registerTool('sem_open', {
  description: '打开当前 SEM 任务的结果文件或任务目录（会在系统中真实打开，供用户查看）：report_pdf 正式 PDF / report_html 网页报告 / report_csv 数据 CSV / report_timing 时间开销报告 / task_directory 任务目录。同时返回路径。',
  schema: {
    target: z.enum(['report_pdf', 'report_html', 'report_csv', 'report_timing', 'task_directory']).describe('要打开的目标'),
  },
}, ({ target }) => semOpen(target))

const transport = new StdioServerTransport()
async function startServer() {
  if (process.env.ZEM_MCP_CDP_URL) {
    try {
      await attachExistingApp()
    } catch (error) {
      log('startup attach failed:', error)
    }
  }
  await server.connect(transport)
  log('ready (repo root:', REPO_ROOT + ')')
}
startServer().catch((error) => {
  log('startup failed:', error)
  process.exit(1)
})

process.on('SIGINT', () => {
  log('SIGINT, closing app')
  closeApp().finally(() => process.exit(0))
})
process.on('SIGTERM', () => {
  log('SIGTERM, closing app')
  closeApp().finally(() => process.exit(0))
})
