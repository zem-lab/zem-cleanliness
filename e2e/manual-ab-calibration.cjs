// UI contract test: all API traffic is intercepted; no hardware operations.
module.exports = async ({ page, assert, log }) => {
  const fs = require('node:fs');
  const out = 'artifacts/manual_ab_live_20260909/ui_contract';
  fs.mkdirSync(out, { recursive: true });
  let status = 'DRAFT', saved = false, failed = false, holdJob = false;
  const posts = [], errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const points = { a: { x: -20100, y: -18890 }, b: { x: 20380, y: 20060 } };
  const manifest = () => ({ taskId: 'manual-ui', status,
    input: { path: 'sample2.json', workpieceName: '样品 2 · 手动 AB', particleCount: 100 },
    selection: { count: 100 }, request: { mode: saved ? 'ab_manual' : 'multi_particle' },
    calibration: saved ? { strategy: 'ab_points', abPointResult: { manualStagePoints: points,
      ...(failed ? { semRefinement: { status: 'FAILED', error: 'previous dependency error' } } : {}) } } : {} });
  await page.route('http://127.0.0.1:8765/**', async route => {
    const req = route.request(), pathname = new URL(req.url()).pathname;
    const headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': '*' };
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
    let body = {};
    if (req.method() === 'POST') posts.push({ path: pathname, body: req.postDataJSON() });
    if (pathname === '/tasks') body = { tasks: [{ id: 'manual-ui', status, input: manifest().input }] };
    else if (pathname === '/tasks/manual-ui') body = manifest();
    else if (pathname === '/tasks/manual-ui/ab/manual') { saved = true; status = 'AB_POINTS_PENDING_SEM_REFINEMENT'; body = manifest(); }
    else if (pathname === '/tasks/manual-ui/ab/refine') body = { id: 'test-job', status: 'queued', progress: 0, total: 1 };
    else if (pathname === '/jobs/test-job') body = holdJob
      ? { id: 'test-job', status: 'running', progress: 1, total: 4, message: 'current retry running' }
      : { id: 'test-job', status: 'succeeded', result: manifest() };
    else if (pathname === '/devices') body = { devices: [] };
    else if (pathname === '/models') body = { models: [] };
    else if (!['/settings', '/health'].includes(pathname)) throw Error(`Unexpected call ${pathname}`);
    await route.fulfill({ json: body, headers });
  });
  await page.goto('http://127.0.0.1:5174', { waitUntil: 'networkidle' });
  await page.locator('.tableRow').filter({ hasText: 'manual-ui' }).click();
  await page.getByRole('button', { name: '手动 AB 标定', exact: true }).click();
  const start = page.getByRole('button', { name: '建立标定并实拍验证' });
  assert(await start.isDisabled(), 'Blank inputs must block start');
  const labels = ['A 点 X（mm）', 'A 点 Y（mm）', 'B 点 X（mm）', 'B 点 Y（mm）'];
  for (let i = 0; i < 4; i++) await page.getByRole('spinbutton', { name: labels[i] }).fill(['-20.1', '-18.89', '20.38', '20.06'][i]);
  assert(await start.isEnabled(), 'Valid manual coordinates should enable start without navigation/models');
  await page.getByRole('spinbutton', { name: labels[0] }).fill('500');
  assert(await start.isDisabled(), 'Out of range coordinate must block start');
  await page.getByRole('spinbutton', { name: labels[0] }).fill('-20.1');
  await page.screenshot({ path: `${out}/tab2-manual-input.png`, fullPage: true });
  page.once('dialog', dialog => dialog.dismiss()); await start.click();
  assert(posts.length === 0, 'Cancelled confirmation must not submit');
  page.once('dialog', dialog => dialog.accept()); await start.click();
  await page.waitForFunction(() => document.querySelector('input[aria-label="A 点 X（mm）"]')?.disabled);
  await page.waitForTimeout(1500);
  assert(posts.length === 2 && posts[0].path.endsWith('/ab/manual') && posts[1].path.endsWith('/ab/refine'), 'Manual save must precede refinement');
  assert(posts[0].body.coordinateUnit === 'mm' && posts[0].body.a.y === -18.89 && posts[0].body.b.x === 20.38, 'UI must transmit exact mm inputs');
  await page.getByRole('button', { name: '首页', exact: true }).click();
  await page.locator('.tableRow').filter({ hasText: 'manual-ui' }).click();
  assert(await page.getByRole('button', { name: '手动 AB 标定', exact: true }).getAttribute('class') === 'active', 'Saved mode must reopen');
  assert(await page.getByRole('spinbutton', { name: labels[1] }).inputValue() === '-18.89', 'Saved micrometers must display as mm');
  failed = true;
  await page.getByRole('button', { name: '首页', exact: true }).click();
  await page.locator('.tableRow').filter({ hasText: 'manual-ui' }).click();
  await page.getByText('previous dependency error', { exact: true }).waitFor();
  holdJob = true;
  page.once('dialog', dialog => dialog.accept()); await start.click();
  await page.getByText('current retry running', { exact: true }).first().waitFor();
  assert((await page.locator('.refineState').innerText()) === '正在执行', 'Retry must not show the previous failure as current status');
  assert(!(await page.locator('.refineProgressHeader').innerText()).includes('previous dependency error'), 'Hide stale error during retry');
  failed = false; holdJob = false;
  assert(errors.length === 0, errors.join(';'));
  fs.writeFileSync(`${out}/ui-test.json`, JSON.stringify({ passed: true, posts, errors }, null, 2));
  log('manual_ab_ui=passed; validation=passed; cancel=passed; units=passed; reopen=passed');
};
