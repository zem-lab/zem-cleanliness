// Run via playwright-browser-check skill runner. All API traffic is mocked;
// this test cannot connect to a microscope or the production backend.
const fs = require('node:fs')
const path = require('node:path')
module.exports = async ({ page, assert, log, args }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  let passed = false
  let strategy = 'ab_particle'
  let status = 'CALIBRATION_PENDING_REVIEW'
  let refinementStatus = null
  let axisFailureDetails = []
  const posts = []
  const image = fs.readFileSync(path.resolve('artifacts/ab_particle_live_20260907/navigation/navigation_raw.png'))
  const records = [101, 102, 103, 104].map(sequence => ({ sequence, status: 'MATCH', lengthUm: 100,
    predictionErrorUm: 18, moveErrorUm: .5, score: .85, margin: .17, targetStageUm: { x: 100, y: 200 },
    images: { opticalImage: 'test.png', semAnnotatedImage: 'test.png' } }))
  const manifest = () => ({ taskId: 'offline-review', status,
    input: { path: 'offline-optical.json', sha256: 'offline', workpieceName: '离线界面测试样品', particleCount: 100 },
    selection: { count: 100, sequences: [101, 102, 103, 104] },
    request: { mode: strategy },
    calibration: { strategy, groups: [], abPointResult: {
      navigation: { image: 'test.png', imageShape: [1080, 1920] },
      assignment: { a: { candidateIndex: 0 }, b: { candidateIndex: 1 } },
      detection: { holes: [{ centerPixel: [1320, 158], confidence: .8 }, { centerPixel: [445, 953], confidence: .75 }] },
      semRefinement: { status: refinementStatus || (passed ? 'SUCCEEDED' : 'VALIDATION_FAILED'), validation: records,
        axisFailureDetails,
        observations: [{ sequence: 77, role: 'seed_h0', status: 'CONTEXT_UNVERIFIED', contextVerification: { status: 'UNVERIFIED', confirmedCount: 0, requiredCount: 2 } }],
        plan: { anchors: [1, 2, 3, 4, 5, 6, 7, 8], validation: [101, 102, 103, 104] },
        anchors: [{ sequence: 1, inlier: true, fitResidualUm: 12, robustWeight: 1 }, { sequence: 5, inlier: false, fitResidualUm: 264, robustWeight: .12 }],
        frozenAt: '2026-09-07T14:00:00+08:00', pixelAxes: { fine: { returnErrorUm: .4 } },
        acceptance: { withinLimits: passed, count: 4, requiredCount: 4, meanErrorUm: passed ? 18 : 45,
          maxErrorUm: passed ? 38 : 100, coverage: .55, limits: { meanErrorUm: 30, maxErrorUm: 60 },
          reasons: passed ? [] : ['独立验证平均或最大误差超限'] } }
    } } })
  await page.route('http://127.0.0.1:8765/**', async route => {
    const req = route.request()
    const pathname = new URL(req.url()).pathname
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': '*' } })
    if (req.method() === 'POST') posts.push(pathname)
    let body
    if (pathname.endsWith('/files/test.png')) return route.fulfill({ status: 200, contentType: 'image/png', body: image, headers: { 'Access-Control-Allow-Origin': '*' } })
    if (pathname === '/tasks') body = { tasks: [{ id: 'offline-review', status, input: manifest().input }] }
    else if (pathname === '/tasks/offline-review') body = manifest()
    else if (pathname === '/devices') body = { devices: [] }
    else if (pathname === '/models') body = { models: [{ id: 'fixture', file: 'fixture.onnx', ready: true }, { id: 'hole', file: 'hole.onnx', ready: true }] }
    else if (pathname === '/health') body = { status: 'ok' }
    else if (pathname === '/settings') body = {}
    else throw new Error(`Unexpected API call in offline UI test: ${req.method()} ${pathname}`)
    return route.fulfill({ json: body, headers: { 'Access-Control-Allow-Origin': '*' } })
  })
  await page.goto(args['app-url'] || 'http://127.0.0.1:5174', { waitUntil: 'networkidle' })
  const openTask = async () => {
    await page.locator('.tableRow').filter({ hasText: 'offline-review' }).click()
    await page.getByRole('heading', { name: '坐标标定', exact: true }).waitFor()
  }
  await openTask()
  for (const name of ['经验矩阵', 'AB 标定', 'AB 导航 + 颗粒精校准']) {
    assert(await page.getByRole('button', { name, exact: true }).count() === 1, `Missing mode: ${name}`)
  }
  assert(await page.getByRole('button', { name: '数值验收未通过，不能确认' }).isDisabled(), 'Failed numeric gate must disable confirmation')
  await page.getByText('独立验证平均或最大误差超限', { exact: true }).waitFor()
  assert(await page.locator('.abEvidenceCard').count() === 4, 'Validation evidence cards missing')
  await page.getByText('种子邻域复核（通过 / 检查）', { exact: true }).waitFor()
  assert(posts.length === 0, 'Opening a saved task must not initiate hardware')
  const out = path.resolve(args.out || 'artifacts/ab_particle_implementation/ui')
  fs.mkdirSync(out, { recursive: true })
  await page.screenshot({ path: path.join(out, 'failed-gate.png'), fullPage: true })
  passed = true
  await page.getByRole('button', { name: '首页', exact: true }).click()
  await openTask()
  assert(await page.getByRole('button', { name: '对比一致，确认标定' }).isEnabled(), 'Passed numeric + evidence gate should allow manual review')
  await page.screenshot({ path: path.join(out, 'passed-gate.png'), fullPage: true })
  status = 'AB_POINTS_PENDING_SEM_REFINEMENT'
  refinementStatus = 'STOPPED'
  await page.getByRole('button', { name: '首页', exact: true }).click()
  await openTask()
  await page.getByText('精定位已停止，不会自动继续', { exact: true }).waitFor()
  assert(await page.getByText('电镜正在执行闭环精定位', { exact: true }).count() === 0, 'Stopped task must not claim live acquisition')
  refinementStatus = 'FAILED'
  axisFailureDetails = [{ phase: 'fine', retryable: false, diagnostics: { repeatImageShiftPx: 351.69, stageReturnDeltaUm: 0 } }]
  await page.getByRole('button', { name: '首页', exact: true }).click()
  await openTask()
  await page.getByText(/图像 \/ 台坐标一致性异常：回原点图像位移 351.7 px/).waitFor()
  assert(posts.length === 0, 'Viewing stopped/failure diagnostics must not trigger hardware')
  refinementStatus = null
  axisFailureDetails = []
  status = 'CALIBRATION_PENDING_REVIEW'
  strategy = 'ab_points'
  await page.getByRole('button', { name: '首页', exact: true }).click()
  await openTask()
  assert(await page.getByRole('button', { name: 'AB 标定', exact: true }).getAttribute('class') === 'active', 'Legacy AB mode must reopen correctly')
  assert(await page.getByRole('button', { name: '对比一致，确认标定' }).isEnabled(), 'Legacy AB evidence review regressed')
  status = 'DRAFT'
  strategy = 'multi_particle'
  await page.getByRole('button', { name: '首页', exact: true }).click()
  await openTask()
  await page.getByRole('button', { name: 'AB 导航 + 颗粒精校准', exact: true }).click()
  await page.getByRole('button', { name: '开始颗粒精校准', exact: true }).waitFor()
  page.once('dialog', dialog => dialog.dismiss())
  await page.getByRole('button', { name: '开始颗粒精校准', exact: true }).click()
  assert(posts.length === 0, 'Dismissing hardware confirmation must issue no start/assignment request')
  assert(errors.length === 0, errors.join('\n'))
  log('three_modes=ok; legacy_ab=ok; numeric_review_gate=ok; stopped_status=ok; axis_contradiction=ok; cancelled_start=ok; hardware_requests=0')
}
