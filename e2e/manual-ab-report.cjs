module.exports = async ({ page, assert, log }) => {
  await page.waitForLoadState('load');
  assert(await page.locator('.particle').count() === 8, 'Eight validation evidence cards required');
  assert(await page.locator('.download').count() === 4, 'Four embedded JSON downloads required');
  assert((await page.locator('#ab-code').textContent()).includes('0.719591669217'), 'Exact sample 2 matrix required');
  assert(await page.locator('img').evaluateAll(async imgs => {
    await Promise.all(imgs.filter(i => i.src).map(i => i.decode()));
    return imgs.filter(i => i.src).every(i => i.complete && i.naturalWidth > 0);
  }), 'All evidence images must decode');
  await page.locator('.frame').first().click();
  assert(await page.locator('#lightbox').evaluate(el => el.open), 'Full image dialog must open');
  await page.locator('#full-image').evaluate(img => img.decode());
  await page.getByRole('button', { name: '关闭大图' }).click();
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('.download').first().click()]);
  assert(download.suggestedFilename() === 'manual_ab_predictions.json', 'Embedded predictions download required');
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Mobile page must not overflow');
  await page.screenshot({ path: 'artifacts/manual_ab_sample2_20260908/report-mobile.png', fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'artifacts/manual_ab_sample2_20260908/report-desktop.png', fullPage: true });
  log('manual_ab_report=passed; images=decoded; lightbox=passed; downloads=passed; responsive=passed');
};
