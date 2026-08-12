/**
 * 시연 영상용 화면 녹화 — 실제 서버를 그대로 조작해 찍는다.
 *
 * 장면마다 '얼마나 멈춰 있을지' 를 밖에서 받는다. 나레이션 길이에 맞춰
 * 화면을 잡아두면 나중에 음성과 붙일 때 어긋나지 않는다.
 *
 * 사전 조건: 서버 실행(npm start), playwright 경로를 NODE_PATH 로 지정.
 *
 *   node tools/record_demo.mjs --scenes outputs/demo/scenes.json --out outputs/demo
 *
 * scenes.json: [{ "id": "photo", "holdMs": 4200 }, ...]  (id 는 아래 ACTIONS 의 키)
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';

const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const argOf = (n, d) => {
  const i = args.indexOf(n);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};

const BASE = argOf('--base', 'http://localhost:4173');
const OUT = resolve(argOf('--out', 'outputs/demo'));
const SCENES = JSON.parse(readFileSync(argOf('--scenes', join(OUT, 'scenes.json')), 'utf8'));
const PHOTO = argOf('--photo', 'data/onfarm_cv/valid/배/특/pear_chuhwang_L_26-10.jpg');

let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  console.error('playwright 를 찾지 못했습니다. NODE_PATH 로 설치 경로를 지정하세요.');
  process.exit(2);
}

mkdirSync(OUT, { recursive: true });

/**
 * 장면별 동작. 각 함수는 '화면을 그 상태로 만드는' 것까지만 하고,
 * 멈춰 있는 시간은 바깥에서 holdMs 로 준다.
 */
const ACTIONS = {
  async home(page) {
    await page.goto(`${BASE}/farmer/`, { waitUntil: 'networkidle' });
  },
  async photo(page) {
    await page.goto(`${BASE}/farmer/sell.html`, { waitUntil: 'networkidle' });
    await page.waitForSelector('#stepPhoto', { state: 'visible' });
  },
  async analyze(page) {
    await page.setInputFiles('#photoInput', PHOTO);
  },
  async candidates(page) {
    await page.waitForSelector('#stepResult', { state: 'visible', timeout: 20000 });
  },
  async alternative(page) {
    // '여기 없어요' 를 강조만 하고 누르지는 않는다 — 있다는 사실을 보여주는 장면
    await page.locator('#resultNo').hover();
  },
  async pick(page) {
    await page.locator('#candidateGrid button').first().click();
    await page.waitForSelector('#stepSku', { state: 'visible' });
  },
  async quantity(page) {
    await page.click('#qtyPlus');
    await page.waitForTimeout(500);
    await page.click('#qtyPlus');
  },
  async confirm(page) {
    await page.click('#skuNext');
    await page.waitForSelector('#stepConfirm', { state: 'visible' });
  },
  async done(page) {
    await page.click('#submitBtn');
    await page.waitForSelector('#stepDone', { state: 'visible', timeout: 20000 });
  },
  async store(page) {
    // 소비자 화면은 소비자 권한이 필요하다(서버가 실제로 막는다).
    // 계정 고르는 화면은 제품의 일부가 아니므로 API 로 조용히 바꾸고 넘어간다.
    await page.evaluate(async () => {
      const { accounts } = await (await fetch('/api/accounts')).json();
      const acc = accounts.find((a) => a.role === 'consumer');
      await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ userId: acc.id }),
      });
    });
    await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
  },
};

const browser = await chromium.launch();
// 녹화 해상도는 뷰포트의 CSS 픽셀과 같다(deviceScaleFactor 로 올라가지 않는다).
// 그래서 뷰포트 자체를 폰 레이아웃이 유지되는 최대치(컨테이너 max-width 560px)로 잡는다.
// 390x844 와 같은 비율이라 화면 모양은 그대로이면서 픽셀만 1.4배 늘어난다.
// recordVideo.size 를 뷰포트보다 크게 주면 확대되지 않고 회색 여백만 생긴다 — 같게 맞춘다.
const VIEW = { width: 560, height: 1212 };
const ctx = await browser.newContext({
  viewport: VIEW,
  locale: 'ko-KR',
  recordVideo: { dir: OUT, size: VIEW },
});
const page = await ctx.newPage();
// 녹화는 페이지가 생기는 순간 시작된다. 모든 구간을 이 시각 기준으로 잰다.
const t0 = Date.now();

// 로그인은 녹화 전에 끝낸다 — 계정 고르는 화면은 제품의 일부가 아니다
await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
const who = await page.evaluate(async () => {
  const { accounts } = await (await fetch('/api/accounts')).json();
  const acc = accounts.find((a) => a.name === '김복순') ?? accounts.find((a) => a.role === 'farmer');
  await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ userId: acc.id }),
  });
  return `${acc.name} · ${acc.farmName} · ${acc.region}`;
});
console.log(`로그인: ${who}`);

const timeline = [];
for (const scene of SCENES) {
  const act = ACTIONS[scene.id];
  if (!act) throw new Error(`알 수 없는 장면: ${scene.id}`);
  await act(page);
  // 나레이션은 '동작이 끝나 화면이 그 상태가 된 뒤' 시작한다.
  // 동작에 걸린 시간(이동·판정 대기)을 세지 않으면 자막과 음성이 뒤로 밀린다.
  const start = Date.now() - t0;
  await page.waitForTimeout(scene.holdMs);
  const end = Date.now() - t0;
  timeline.push({ id: scene.id, start: start / 1000, end: end / 1000 });
  console.log(`  ${scene.id.padEnd(12)} ${(start / 1000).toFixed(1)}초부터 ${(scene.holdMs / 1000).toFixed(1)}초`);
}

const video = page.video();
await ctx.close();
await browser.close();

const raw = await video.path();
writeFileSync(join(OUT, 'timeline.json'), JSON.stringify(timeline, null, 2), 'utf8');
writeFileSync(join(OUT, 'video_path.txt'), raw, 'utf8');
console.log(`\n녹화: ${raw}`);
console.log(`구간표: ${join(OUT, 'timeline.json')}`);
