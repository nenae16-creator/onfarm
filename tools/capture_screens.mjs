/**
 * 발표 자료용 실제 화면 캡처 — 목업이 아니라 돌아가는 서버를 찍는다.
 *
 * 심사 항목 「활용 가능성 20점」은 "내일부터 쓸 수 있는가"를 본다.
 * 손으로 그린 화면을 넣으면 그 자리에서 신뢰를 잃으므로,
 * 실제 등록 흐름(사진 → 후보 선택 → 수량 → 등록 완료)을 그대로 걸어가며 찍는다.
 *
 * 사전 조건:
 *   1) 서버 실행:  npm start        (기본 http://localhost:4173)
 *   2) playwright 는 이 저장소의 의존성이 아니다(런타임 의존성 0 원칙 유지).
 *      npx 캐시나 전역 설치본을 NODE_PATH 로 빌려 쓴다.
 *
 * 실행:
 *   NODE_PATH=<playwright 설치 경로> node tools/capture_screens.mjs
 *   node tools/capture_screens.mjs --base http://localhost:4173
 */
import { mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const BASE = argOf('--base', 'http://localhost:4173');
const OUT = argOf('--out', join(process.cwd(), 'docs', 'screens'));

let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  console.error(
    'playwright 를 찾지 못했습니다. 설치 경로를 NODE_PATH 로 지정하세요.\n' +
      '예) NODE_PATH="$HOME/AppData/Local/npm-cache/_npx/<해시>/node_modules" node tools/capture_screens.mjs',
  );
  process.exit(2);
}

mkdirSync(OUT, { recursive: true });

/** 화면이 실제로 그 단계에 도달했는지 확인하고 찍는다. 안 보이면 실패시킨다. */
async function shot(page, name, selector) {
  if (selector) {
    await page.waitForSelector(selector, { state: 'visible', timeout: 15000 });
  }
  await page.waitForTimeout(400); // 폰트·이미지 안착
  const file = join(OUT, `${name}.png`);
  await page.screenshot({ path: file });
  console.log('✔', name);
  return file;
}

/**
 * 데모 계정으로 로그인한다. 비밀번호가 없는 시연용 계정이지만
 * 역할별 권한 검사는 서버에서 실제로 동작하므로, 화면마다 맞는 역할로 들어가야 한다.
 */
async function loginAs(page, { role, name }) {
  const who = await page.evaluate(
    async ([role, name]) => {
      const res = await fetch('/api/accounts');
      const { accounts } = await res.json();
      const acc =
        accounts.find((a) => a.role === role && (!name || a.name === name)) ??
        accounts.find((a) => a.role === role);
      if (!acc) throw new Error(`계정 없음: ${role}`);
      await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ userId: acc.id }),
      });
      return `${acc.name} (${acc.farmName ?? acc.role}${acc.region ? ' · ' + acc.region : ''})`;
    },
    [role, name],
  );
  console.log(`  로그인: ${who}`);
  return who;
}

const captured = [];

const browser = await chromium.launch();

// ── 농민 화면: 실제 휴대폰 크기로 찍는다 ───────────────────────────────
const phone = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2, // 인쇄·슬라이드에서 뭉개지지 않게
  locale: 'ko-KR',
});
const page = await phone.newPage();

// 천안 농가 계정으로 들어간다 — 발표 대상 지역과 맞춘다
await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
const farmer = await loginAs(page, { role: 'farmer', name: '김복순' });

await page.goto(`${BASE}/farmer/`, { waitUntil: 'networkidle' });
captured.push(await shot(page, 'farmer-01-home'));

await page.goto(`${BASE}/farmer/sell.html`, { waitUntil: 'networkidle' });
captured.push(await shot(page, 'farmer-02-photo', '#stepPhoto'));

// 사진을 태운다. 기본은 학습에 쓰지 않은 valid 분할의 실제 배 사진이다.
// (합성 샘플 버튼도 있지만, 발표 자료에는 합성 이미지를 쓰지 않는다)
const PHOTO = argOf('--photo', 'data/onfarm_cv/valid/배/특/pear_chuhwang_S_1-1.jpg');
if (PHOTO === 'sample') {
  await page.click('#sampleBtn');
} else {
  await page.setInputFiles('#photoInput', PHOTO);
}
captured.push(await shot(page, 'farmer-03-candidates', '#stepResult'));

// ★ 이 화면이 제안의 핵심이다 — 1·2·3번 중 고르기
const firstChoice = page.locator('#candidateGrid button').first();
await firstChoice.waitFor({ state: 'visible', timeout: 15000 });
const choiceText = (await firstChoice.innerText()).replace(/\s+/g, ' ').trim();
await firstChoice.click();
captured.push(await shot(page, 'farmer-04-quantity', '#stepSku'));

await page.click('#qtyPlus');
await page.click('#skuNext');
captured.push(await shot(page, 'farmer-05-confirm', '#stepConfirm'));

await page.click('#submitBtn');
captured.push(await shot(page, 'farmer-06-done', '#stepDone'));

// ── 소비자·거점 화면: 데스크톱 ────────────────────────────────────────
const desk = await browser.newContext({
  viewport: { width: 1280, height: 900 },
  deviceScaleFactor: 2,
  locale: 'ko-KR',
});
const wide = await desk.newPage();

await wide.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
await loginAs(wide, { role: 'consumer' });
await wide.goto(`${BASE}/`, { waitUntil: 'networkidle' });
captured.push(await shot(wide, 'store-01-list'));

await loginAs(wide, { role: 'hub_operator' });
await wide.goto(`${BASE}/hub/`, { waitUntil: 'networkidle' });
captured.push(await shot(wide, 'hub-01-inspect'));

await browser.close();

console.log(`\n${captured.length}장 저장: ${OUT}`);
console.log(`농가 계정: ${farmer}`);
console.log(`AI 첫 후보: ${choiceText}`);
