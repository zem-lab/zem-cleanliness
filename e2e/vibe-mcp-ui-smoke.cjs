// Isolated browser check: all desktop requests are mocked; no backend or hardware.
const assert = require('node:assert/strict')
const fs = require('node:fs/promises')
const http = require('node:http')
const path = require('node:path')
const { chromium } = require('playwright')

const root = path.resolve(__dirname, '..')
const dist = path.join(root, 'dist')
const output = path.join(root, '.tmp', 'vibe-mcp-ui')

async function main() {
  await fs.access(path.join(dist, 'index.html'))
  await fs.mkdir(output, { recursive: true })
  const server = http.createServer(async (req, res) => {
    try {
      const pathname = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname)
      const file = path.resolve(dist, '.' + (pathname === '/' ? '/index.html' : pathname))
      if (!file.startsWith(dist + path.sep)) { res.writeHead(403).end(); return }
      const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' }
      res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' })
      res.end(await fs.readFile(file))
    } catch { res.writeHead(404).end() }
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const origin = `http://127.0.0.1:${server.address().port}`
  let browser
  const results = []
  try {
    browser = await chromium.launch({ channel: 'msedge', headless: true })
    for (const scenario of ['simulated', 'real', 'codex', 'unavailable']) {
      const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } })
      const pageErrors = []
      const externalRequests = []
      page.on('pageerror', (error) => pageErrors.push(error.message))
      await page.route('**/*', (route) => {
        if (new URL(route.request().url()).origin === origin) return route.continue()
        externalRequests.push(route.request().url())
        return route.abort()
      })
      await page.addInitScript(({ scenario, origin }) => {
        const state = window.__vibeMock = {
          failCreate: true, failSend: true, failCancel: true,
          session: { id: 'ui-mcp', title: '模拟界面检查', status: 'idle', createdAt: '2026-10-16T00:00:00Z',
            codexSessionId: null, events: [], cursor: 0 },
          calls: [],
        }
        window.zemCleanliness = {
          log: async () => {},
          getRuntime: async () => ({ backendBaseUrl: origin, packaged: false, version: 'ui-test', logDirectory: '' }),
          getAcceptanceTest: async () => null,
          request: async (method, route, body) => {
            state.calls.push({ method, route })
            if (route === '/tasks') return { tasks: [] }
            if (route === '/devices') return { devices: [] }
            if (route === '/models') return { models: [] }
            if (route === '/chat/runtime') {
              if (scenario === 'unavailable') throw new Error('模拟：助手服务不可用')
              return { engine: scenario === 'codex' ? 'codex' : 'claude', provider: 'cpa', launcherConfigured: true,
                model: 'deepseek-chat', credentialConfigured: scenario === 'real', mcp: {
                available: true, mode: scenario === 'codex' ? 'real' : scenario, captureEnabled: scenario === 'simulated', moveEnabled: scenario === 'simulated',
              } }
            }
            if (route === '/chat/sessions' && method === 'GET') return { sessions: [] }
            if (route === '/chat/sessions' && method === 'POST') {
              if (state.failCreate) throw new Error('模拟：无法创建会话')
              return structuredClone(state.session)
            }
            if (route === '/chat/sessions/ui-mcp/messages') {
              if (state.failSend) throw new Error('模拟：模型请求失败')
              state.session.status = 'running'
              state.session.events.push({ seq: 1, kind: 'user', text: body.text })
              state.session.cursor = 1
              return { ok: true }
            }
            if (route === '/chat/sessions/ui-mcp/cancel') {
              if (state.failCancel) throw new Error('模拟：取消请求失败')
              state.session.status = 'idle'
              state.session.events.push({ seq: 2, kind: 'system', text: '对话已取消；设备作业仍需查询状态。' })
              state.session.cursor = 2
              return { ok: true }
            }
            if (route.startsWith('/chat/sessions/ui-mcp')) {
              const after = Number(new URL(route, origin).searchParams.get('after') || 0)
              return { ...structuredClone(state.session), events: state.session.events.filter((event) => event.seq > after) }
            }
            throw new Error(`Unexpected isolated UI request: ${method} ${route}`)
          },
        }
      }, { scenario, origin })
      try {
        await page.goto(origin)
        await page.getByRole('button', { name: 'vibe 助手', exact: true }).click()
        const panel = page.getByRole('complementary', { name: 'vibe 助手' })
        const panelBounds = await panel.boundingBox()
        const contentBounds = await page.getByRole('main').boundingBox()
        assert.ok(panelBounds.width >= 280, 'Assistant must have enough width to read and type')
        assert.ok(panelBounds.x >= contentBounds.x + contentBounds.width - 1, 'Assistant must be beside the main content')
        assert.ok(Math.abs(panelBounds.y - contentBounds.y) <= 1, 'Assistant must stay in the main content row')
        if (scenario === 'unavailable') {
          await panel.getByRole('alert').filter({ hasText: '读取助手连接信息失败' }).waitFor()
        } else if (scenario === 'codex') {
          await panel.getByText('Codex · CPA（cpa）· 启动入口就绪，实际连接需通过对话验证。').waitFor()
          assert.equal(await panel.getByText('请先在本机 .env.local 填写 DeepSeek 密钥，再重启软件。').count(), 0)
        } else if (scenario === 'real') {
          await panel.getByRole('status').filter({ hasText: '真实电镜接口 · 软件界面 MCP 未接入 · 移动未开放 · 采图未开放' }).waitFor()
          await panel.getByText('deepseek-chat 配置已填写，实际连接需通过对话验证。').waitFor()
          assert.equal(await panel.getByText('请先在本机 .env.local 填写 DeepSeek 密钥，再重启软件。').count(), 0)
        } else {
          await panel.getByRole('status').filter({ hasText: '模拟设备 · 软件界面 MCP 未接入 · 移动已开放 · 采图已开放' }).waitFor()
          await panel.getByText('请先在本机 .env.local 填写 DeepSeek 密钥，再重启软件。').waitFor()
          assert.equal(await panel.getByRole('textbox').isDisabled(), true)
          await panel.getByRole('button', { name: '新会话' }).click()
          await panel.getByRole('alert').filter({ hasText: '创建会话失败' }).waitFor()
          await page.evaluate(() => { window.__vibeMock.failCreate = false })
          await panel.getByRole('button', { name: '新会话' }).click()
          await panel.getByRole('textbox').fill('查询电镜状态')
          await panel.getByRole('textbox').press('Enter')
          await panel.getByRole('alert').filter({ hasText: '发送失败' }).waitFor()
          assert.equal(await panel.getByRole('textbox').inputValue(), '查询电镜状态')
          await page.evaluate(() => { window.__vibeMock.failSend = false })
          await panel.getByRole('textbox').press('Enter')
          await panel.getByText('AI 正在操作软件…').waitFor()
          await panel.getByRole('button', { name: '取消', exact: true }).click()
          await panel.getByRole('alert').filter({ hasText: '取消请求失败' }).waitFor()
          await page.evaluate(() => { window.__vibeMock.failCancel = false })
          await panel.getByRole('button', { name: '取消', exact: true }).click()
          await panel.getByText('对话已取消；设备作业仍需查询状态。').waitFor()
          assert.equal(await panel.getByRole('alert').count(), 0)
          const calls = await page.evaluate(() => window.__vibeMock.calls)
          assert.equal(calls.some(({ route }) => route.startsWith('/mcp/call') || route.startsWith('/sem/')), false)

          // 助手回复带 [图片] 标记时，应渲染真实 <img>，而不是只显示路径字串。
          const pngBytes = Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
            'base64',
          )
          await page.route('**/chat/file**', async (route) => {
            await route.fulfill({ status: 200, contentType: 'image/png', body: pngBytes })
          })
          await page.evaluate(() => {
            const state = window.__vibeMock
            state.session.status = 'idle'
            state.session.events.push(
              { seq: 3, ts: '2026-10-16T00:00:01Z', kind: 'assistant', text: '已取到导航图。\n[图片] %USERPROFILE%/Documents/ZEM Cleanliness/mcp/images/demo/preview.png' },
              { seq: 4, ts: '2026-10-16T00:00:01Z', kind: 'image', text: 'D:\\Documents\\ZEM Cleanliness\\mcp\\images\\demo\\preview.png' },
            )
            state.session.cursor = 4
          })
          await page.waitForTimeout(900)
          await panel.getByText('已取到导航图。').waitFor()
          assert.equal(await panel.getByText(/\[图片\]/).count(), 0, 'Raw [图片] marker must be stripped from assistant bubble')
          const image = panel.locator('.chatImageWrap img')
          await image.waitFor()
          assert.equal(await image.count(), 1, 'Inline image + image event must dedupe to one thumbnail')
          assert.ok(await image.evaluate((node) => node.complete && node.naturalWidth > 0), 'Chat image must finish decoding')

          await page.evaluate(() => {
            const state = window.__vibeMock
            const lines = Array.from({ length: 24 }, (_, index) => `工具输出第 ${index + 1} 行：${'详情'.repeat(8)}`)
            state.session.events.push({
              seq: 5, ts: '2026-10-16T00:00:02Z', kind: 'action', text: lines.join('\n'),
            })
            state.session.cursor = 5
          })
          const collapsed = panel.locator('.chatActionCollapsed').filter({ hasText: '点击展开工具输出' })
          await collapsed.waitFor()
          const collapsedBox = await collapsed.boundingBox()
          await collapsed.click()
          const expanded = panel.locator('.chatActionExpanded').filter({ hasText: '工具输出第 24 行' })
          await expanded.waitFor()
          const expandedBox = await expanded.boundingBox()
          assert.ok(expandedBox && expandedBox.height >= 80, `Expanded tool output too short: ${expandedBox && expandedBox.height}`)
          assert.ok(expandedBox.height > (collapsedBox?.height || 0) + 16, 'Clicking expand must grow the action block instead of collapsing it')
        }
        await page.screenshot({ path: path.join(output, `${scenario}.png`) })
        assert.deepEqual(pageErrors, [])
        assert.deepEqual(externalRequests, [])
        await panel.getByRole('button', { name: '收起', exact: true }).click()
        assert.equal(await panel.count(), 0)
        assert.ok((await page.getByRole('main').boundingBox()).width > contentBounds.width, 'Closing assistant must restore content width')
        results.push({ scenario, passed: true, hardwareConnections: 0, externalRequests: 0 })
      } finally { await page.close() }
    }
    await fs.writeFile(path.join(output, 'results.json'), JSON.stringify(results, null, 2))
    console.log(JSON.stringify({ passed: results.length, results, output }, null, 2))
  } finally {
    if (browser) await browser.close()
    await new Promise((resolve) => server.close(resolve))
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1 })
