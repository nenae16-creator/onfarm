/**
 * 다중 배율 판정이 **실제 서버에서** 동작하는지 확인한다.
 *
 * 파이썬으로 흉내 낸 값이 아니라, 실행 중인 서버의 /api/ai/analyze 에
 * 브라우저가 보내는 것과 똑같은 224×224 픽셀을 보내 결과를 받는다.
 *
 *   node tools/verify_server_tta.mjs                 # 멀리서 찍은 조건
 *   node tools/verify_server_tta.mjs --cond studio
 *
 * 조건 합성(멀리서 찍기)은 tools/eval_realworld.py 의 cond_far 와 같은 방식이다:
 * 원본을 절반으로 줄여 2배 캔버스 가운데 놓는다.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const argOf = (n, d) => {
  const i = args.indexOf(n);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};

const BASE = argOf('--base', 'http://localhost:4173');
const COND = argOf('--cond', 'far');
const PER = Number(argOf('--n', '10'));
const ROOT = process.cwd();
const VALID = join(ROOT, 'data', 'onfarm_cv', 'valid');
const SIZE = 224;

let jpeg;
try {
  jpeg = require('jpeg-js');
} catch {
  console.error('jpeg-js 가 필요합니다(devDependency). npm install 후 다시 실행하세요.');
  process.exit(2);
}

/** 원본 JPEG → 224×224 RGB. 조건에 따라 피사체를 작게 만든다. */
function toPixels(file, cond) {
  const { data, width, height } = jpeg.decode(readFileSync(file), { useTArray: true });
  // fill = 피사체가 화면 '가로'에서 차지하는 비율. eval_realworld.py 의 cond_far 는 0.25 다.
  const fill = cond === 'studio' ? 1 : Number(argOf('--fill', '0.25'));
  const inner = Math.max(1, Math.round(SIZE * fill));
  const off = Math.floor((SIZE - inner) / 2);
  const out = new Uint8Array(SIZE * SIZE * 3).fill(150); // 바깥은 회색 배경
  for (let y = 0; y < inner; y += 1) {
    const sy = Math.min(height - 1, Math.floor((y / inner) * height));
    for (let x = 0; x < inner; x += 1) {
      const sx = Math.min(width - 1, Math.floor((x / inner) * width));
      const si = (sy * width + sx) * 4;
      const di = ((y + off) * SIZE + x + off) * 3;
      out[di] = data[si] ?? 0;
      out[di + 1] = data[si + 1] ?? 0;
      out[di + 2] = data[si + 2] ?? 0;
    }
  }
  return out;
}

async function login(role, name) {
  const res = await fetch(`${BASE}/api/accounts`);
  const { accounts } = await res.json();
  const acc = accounts.find((a) => a.role === role && (!name || a.name === name));
  const auth = await fetch(`${BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ userId: acc.id }),
  });
  return auth.headers.getSetCookie?.().join('; ') ?? auth.headers.get('set-cookie') ?? '';
}

const cookie = await login('farmer', '김복순');
const items = readdirSync(VALID).filter((d) => !d.startsWith('.'));
const summary = [];

for (const item of items) {
  const files = [];
  for (const grade of readdirSync(join(VALID, item))) {
    for (const f of readdirSync(join(VALID, item, grade)).slice(0, 40)) {
      files.push(join(VALID, item, grade, f));
    }
  }
  files.sort();
  const picked = files.filter((_, i) => i % Math.max(1, Math.floor(files.length / PER)) === 0)
    .slice(0, PER);

  let top1 = 0;
  let top3 = 0;
  for (const f of picked) {
    const rgb = toPixels(f, COND);
    // 서버는 사진 원본(data URL)도 함께 받아야 매물에 붙일 이미지를 만든다.
    const rgba = new Uint8Array(SIZE * SIZE * 4);
    for (let i = 0; i < SIZE * SIZE; i += 1) {
      rgba[i * 4] = rgb[i * 3] ?? 0;
      rgba[i * 4 + 1] = rgb[i * 3 + 1] ?? 0;
      rgba[i * 4 + 2] = rgb[i * 3 + 2] ?? 0;
      rgba[i * 4 + 3] = 255;
    }
    const enc = jpeg.encode({ data: Buffer.from(rgba), width: SIZE, height: SIZE }, 90);
    const image = `data:image/jpeg;base64,${Buffer.from(enc.data).toString('base64')}`;
    const pixels = Buffer.from(rgb).toString('base64');
    const res = await fetch(`${BASE}/api/ai/analyze`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', cookie },
      body: JSON.stringify({ image, pixels }),
    });
    if (!res.ok) throw new Error(`분석 실패 ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const body = await res.json();
    // 화면이 보여주는 후보 그대로다(pipeline.ts 가 상위 3개만 내려준다).
    const names = (body.candidates ?? []).map((c) => c.name);
    if (names.length === 0) throw new Error(`후보가 비어 있다: ${JSON.stringify(body).slice(0, 200)}`);
    if (names[0] === item) top1 += 1;
    if (names.slice(0, 3).includes(item)) top3 += 1;
  }
  summary.push({ item, n: picked.length, top1: top1 / picked.length, top3: top3 / picked.length });
  console.log(
    `  ${item.padEnd(4)} n=${picked.length}  top-1 ${(top1 / picked.length * 100).toFixed(0)}%` +
      `  top-3 ${(top3 / picked.length * 100).toFixed(0)}%`,
  );
}

const avg = (k) => summary.reduce((a, s) => a + s[k], 0) / summary.length;
console.log(`\n조건 '${COND}' 전체 — top-1 ${(avg('top1') * 100).toFixed(0)}% · ` +
  `top-3 ${(avg('top3') * 100).toFixed(0)}%   (실행 중인 서버 응답 기준)`);
