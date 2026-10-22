// 端到端自测：以 MCP 客户端身份通过 stdio 启动 server.cjs，
// 覆盖全量工具中无需硬件的部分：启动→导航→扫描设置→模型切换→设置读写→
// 任务/设备/错误读取→硬件门拒绝→截图回读。真机操作（获取导航图/开始分析）
// 仅验证默认被硬件门拦截。
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { Client } = require('@modelcontextprotocol/sdk/client/index.js')
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js')

const serverPath = path.join(__dirname, 'server.cjs')
const REPO_ROOT = path.resolve(__dirname, '..', '..')

function textOf(result) {
  return result.content.filter((block) => block.type === 'text').map((block) => block.text).join('\n')
}

function jsonOf(result) {
  return JSON.parse(textOf(result))
}

async function main() {
  const client = new Client({ name: 'zem-mcp-smoke', version: '0.0.0' })
  const transport = new StdioClientTransport({ command: process.execPath, args: [serverPath], cwd: __dirname })
  await client.connect(transport)
  try {
    await runScenario(client)
  } finally {
    await client.callTool({ name: 'app_close', arguments: {} }).catch(() => undefined)
    await client.close().catch(() => undefined)
  }
}

async function runScenario(client) {

  const pythonCandidates = [
    process.env.ZEM_MCP_SMOKE_PYTHON,
    path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(path.dirname(REPO_ROOT), 'zem_cleanliness', '.venv', 'Scripts', 'python.exe'),
  ].filter(Boolean)
  const pythonPath = pythonCandidates.find((candidate) => fs.existsSync(candidate))

  // 启动（默认 allowHardware=false）
  const launch = await client.callTool({ name: 'app_launch', arguments: pythonPath ? { pythonPath } : {} })
  assert.ok(!launch.isError, `app_launch failed: ${textOf(launch)}`)
  const launchInfo = jsonOf(launch)
  assert.equal(launchInfo.running, true)
  assert.equal(launchInfo.allowHardware, false)

  // 导航 + 扫描设置
  await client.callTool({ name: 'navigate', arguments: { target: 'sem-direct' } })
  const scan = await client.callTool({
    name: 'set_scan_settings',
    arguments: { name: 'MCP 全量自测样品', shape: 'circle', fieldUm: 800, imageSize: 1024, overlapPercent: 15, minLengthUm: 3, edsEnabled: true },
  })
  assert.ok(!scan.isError, `set_scan_settings failed: ${textOf(scan)}`)
  const scanInfo = jsonOf(scan)
  assert.equal(scanInfo.name, 'MCP 全量自测样品')
  assert.equal(scanInfo.shape, '圆形')
  assert.equal(scanInfo.fieldUm, '800')
  assert.equal(scanInfo.imageSize, '1024')
  assert.equal(scanInfo.overlapPercent, '15')
  assert.equal(scanInfo.minLengthUm, '3')
  assert.equal(scanInfo.edsEnabled, true)

  // 分割模型切换 + 回读
  const model = await client.callTool({
    name: 'set_segmentation_model',
    arguments: { backend: 'sam3', url: 'http://192.168.0.207:8767', score: 0.05 },
  })
  assert.ok(!model.isError, `set_segmentation_model failed: ${textOf(model)}`)
  const modelInfo = jsonOf(model)
  assert.equal(modelInfo.currentBackend, 'sam3')
  assert.equal(modelInfo.currentUrl, 'http://192.168.0.207:8767')
  assert.equal(modelInfo.currentScore, '0.05')
  const backToYolo = await client.callTool({ name: 'set_segmentation_model', arguments: { backend: 'yolo' } })
  assert.equal(jsonOf(backToYolo).currentBackend, 'yolo')

  // 硬件门：默认拒绝触碰设备
  const navigation = await client.callTool({ name: 'sem_get_navigation', arguments: {} })
  assert.ok(navigation.isError, 'sem_get_navigation should be blocked without allowHardware')
  assert.match(textOf(navigation), /allowHardware/)
  const start = await client.callTool({ name: 'sem_start_analysis', arguments: {} })
  assert.ok(start.isError, 'sem_start_analysis should be blocked without allowHardware')
  assert.match(textOf(start), /allowHardware/)

  // 系统设置读写
  const settingsBefore = jsonOf(await client.callTool({ name: 'get_settings', arguments: {} }))
  assert.ok(typeof settingsBefore.captureLevel === 'number')
  const saved = await client.callTool({ name: 'set_settings', arguments: { captureLevel: 5, edsEnabled: true } })
  assert.ok(!saved.isError, `set_settings failed: ${textOf(saved)}`)
  const settingsAfter = jsonOf(saved).settings
  assert.equal(settingsAfter.captureLevel, 5)
  assert.equal(settingsAfter.edsEnabled, true)

  // 任务 / 设备 / 错误 / 状态
  const tasks = jsonOf(await client.callTool({ name: 'get_tasks', arguments: {} }))
  assert.equal(tasks.total, 0)
  const devices = jsonOf(await client.callTool({ name: 'get_devices', arguments: {} }))
  assert.ok(Array.isArray(devices.devices) && devices.devices.length >= 1, 'should list microscope device')
  const errors = jsonOf(await client.callTool({ name: 'get_errors', arguments: {} }))
  assert.equal(errors.count, 0)
  const status = await client.callTool({ name: 'sem_get_status', arguments: {} })
  assert.ok(!status.isError, textOf(status))
  const statusInfo = jsonOf(status)
  assert.equal(statusInfo.onSemDirectPage, true)
  assert.equal(statusInfo.taskId, null, 'no task without navigation image')

  // 打开不存在任务 / 无任务颗粒明细 → 明确报错
  const openMissing = await client.callTool({ name: 'open_task', arguments: { taskId: 'does-not-exist' } })
  assert.ok(openMissing.isError && /任务不存在/.test(textOf(openMissing)))
  const particles = await client.callTool({ name: 'sem_get_particles', arguments: {} })
  assert.ok(particles.isError, 'sem_get_particles should fail without an open task')

  // 截图 + 界面回读 + console 干净
  const shot = await client.callTool({ name: 'screenshot', arguments: {} })
  assert.ok(shot.content.some((block) => block.type === 'image'), 'screenshot should return image content')
  const shotPath = JSON.parse(textOf(shot)).file
  assert.ok(fs.existsSync(shotPath), `screenshot file missing: ${shotPath}`)
  const view = jsonOf(await client.callTool({ name: 'ui_read', arguments: {} }))
  assert.match(view.agentBadge, /Agent 控制/, 'agent-control badge should be visible')
  assert.ok(view.headings.some((heading) => heading.includes('直接 SEM 分析')))
  const consoleErrors = jsonOf(await client.callTool({ name: 'console_errors', arguments: {} }))
  assert.deepEqual(consoleErrors, [], 'renderer console should be clean')

  console.log(`MCP full-scope smoke passed; screenshot: ${shotPath}`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
