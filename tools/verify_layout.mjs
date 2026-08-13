/**
 * 시연에 걸린 화면 약속이 실제 픽셀에서 지켜지는지 검사한다.
 *
 * 이 제품의 핵심 문장은 "화면은 항상 1·2·3번 중에 고르게 되어 있습니다" 다.
 * 그런데 후보 2·3번이 접힘선 아래로 내려가면 그 문장은 화면에서 거짓이 된다.
 * CSS 한 줄이 바뀌어도 조용히 깨지는 종류라, 자동 테스트로는 잡히지 않는다.
 * 그래서 실제 브라우저에서 좌표를 재서 확인한다.
 *
 *   node tools/verify_layout.mjs                    # 서버가 떠 있어야 한다
 *   node tools/verify_layout.mjs --base http://localhost:4173
 *
 * playwright 는 이 저장소의 의존성이 아니다(런타임 의존성 0 원칙).
 * NODE_PATH 로 설치본을 빌려 쓴다.
 */
import { readdirSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const argOf = (n, d) => {
  const i = args.indexOf(n);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};

const BASE = argOf('--base', 'http://localhost:4173');
const PHOTO_DIR = argOf('--photo-dir', 'data/onfarm_cv/valid/배/특/');

/** 기준 기기 — 흔한 폰 중 세로가 짧은 축에 맞춘다. 여기서 되면 큰 화면은 당연히 된다. */
const DEVICES = [
  { name: 'iPhone 12/13/14 (390×844)', width: 390, height: 844 },
  { name: '작은 안드로이드 (360×780)', width: 360, height: 780 },
];

let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  console.error('playwright 를 찾지 못했습니다. NODE_PATH 로 설치 경로를 지정하세요.');
  process.exit(2);
}

const browser = await chromium.launch();
const failures = [];

for (const device of DEVICES) {
  const ctx = await browser.newContext({
    viewport: { width: device.width, height: device.height },
    locale: 'ko-KR',
  });
  const page = await ctx.newPage();

  await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(async () => {
    const { accounts } = await (await fetch('/api/accounts')).json();
    const acc = accounts.find((a) => a.role === 'farmer');
    await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ userId: acc.id }),
    });
  });

  await page.goto(`${BASE}/farmer/sell.html`, { waitUntil: 'networkidle' });
  await page.setInputFiles('#photoInput', PHOTO_DIR + readdirSync(PHOTO_DIR)[0]);
  await page.waitForSelector('#stepResult', { state: 'visible', timeout: 20000 });

  const seen = await page.evaluate(() => {
    // 하단 고정 바가 있으면 그 위가 실질 접힘선이다.
    const fixed = [...document.querySelectorAll('*')].filter((e) => {
      const s = getComputedStyle(e);
      return (
        (s.position === 'fixed' || s.position === 'sticky') &&
        e.getBoundingClientRect().bottom > innerHeight - 120 &&
        e.getBoundingClientRect().height > 20
      );
    });
    const fold = fixed.length ? Math.min(...fixed.map((e) => e.getBoundingClientRect().top)) : innerHeight;
    const buttons = [...document.querySelectorAll('#candidateGrid button')];
    const visible = buttons.filter((e) => {
      const r = e.getBoundingClientRect();
      return r.top >= 0 && r.bottom <= fold;
    });
    return { total: buttons.length, visible: visible.length, fold: Math.round(fold) };
  });

  const ok = seen.visible >= Math.min(3, seen.total);
  console.log(
    `${ok ? '✅' : '❌'} ${device.name}  후보 ${seen.total}개 중 스크롤 없이 ${seen.visible}개 보임` +
      `  (접힘선 ${seen.fold}px)`,
  );
  if (!ok) {
    failures.push(
      `${device.name}: 후보 ${seen.total}개 중 ${seen.visible}개만 보인다. ` +
        '"1·2·3번 중에 고르게 한다"는 약속이 화면에서 깨진다.',
    );
  }
  await ctx.close();
}

await browser.close();

if (failures.length) {
  console.error(`\n실패 ${failures.length}건:\n  ${failures.join('\n  ')}`);
  console.error('\n결과 화면의 사진·제목 높이를 줄여 후보 3개를 접힘선 위로 올리세요(public/css/farmer.css 끝).');
  process.exit(1);
}
console.log('\n후보 3개가 모든 기준 기기에서 한 화면에 들어온다.');
