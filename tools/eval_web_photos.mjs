/**
 * 웹 공개 사진으로 **실행 중인 서버**를 재본다(다중 배율 판정 포함).
 *
 * ★이것은 실환경 평가가 아니다.
 *   웹 사진은 상품컷·나무에 달린 컷이 대부분이라 오히려 학습 데이터에 가깝다.
 *   고령 농가가 밭에서 찍은 폰 사진과는 다르므로 field_evaluated 를 바꾸지 않는다.
 *   '스튜디오 검증셋과 실제 폰 사진 사이의 제3의 조건' 으로만 읽는다.
 *
 * 평가 대상은 data/web_photos/선별.txt 에 사람이 눈으로 골라 적은 것만 쓴다.
 * 커먼즈 분류에서 받아도 절반 넘게 다른 것이 섞이기 때문이다.
 *
 *   node tools/eval_web_photos.mjs
 */
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const jpeg = require('jpeg-js');
let png;
try {
  png = require('pngjs').PNG;
} catch {
  png = null;
}

const BASE = 'http://localhost:4173';
const ROOT = process.cwd();
const DIR = join(ROOT, 'data', 'web_photos');
const SIZE = 224;

/** 브라우저와 같은 전처리 — 사진 전체를 224×224 로 늘린다(비율 무시). */
function toPixels(file) {
  const buf = readFileSync(file);
  let img;
  if (file.toLowerCase().endsWith('.png')) {
    if (!png) return null;
    img = png.sync.read(buf);
  } else {
    img = jpeg.decode(buf, { useTArray: true });
  }
  const { data, width, height } = img;
  const out = new Uint8Array(SIZE * SIZE * 3);
  for (let y = 0; y < SIZE; y += 1) {
    const sy = Math.min(height - 1, Math.floor((y / SIZE) * height));
    for (let x = 0; x < SIZE; x += 1) {
      const sx = Math.min(width - 1, Math.floor((x / SIZE) * width));
      const si = (sy * width + sx) * 4;
      const di = (y * SIZE + x) * 3;
      out[di] = data[si] ?? 0;
      out[di + 1] = data[si + 1] ?? 0;
      out[di + 2] = data[si + 2] ?? 0;
    }
  }
  return out;
}

/** 선별.txt — 주석(#)과 빈 줄을 빼고 '품목/파일이름' 만 읽는다. */
function selected() {
  const text = readFileSync(join(DIR, '선별.txt'), 'utf8');
  const picks = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split('#')[0].trim();
    if (!line) continue;
    const [item, stem] = line.split('/');
    if (!item || !stem) continue;
    const folder = join(DIR, item);
    const match = readdirSync(folder).find((f) => f.startsWith(stem));
    if (!match) {
      console.error(`  ⚠ 파일 없음: ${line}`);
      continue;
    }
    picks.push({ item, file: join(folder, match) });
  }
  return picks;
}

const res0 = await fetch(`${BASE}/api/accounts`);
if (!res0.ok) {
  console.error('서버가 없습니다. npm start 후 다시 실행하세요.');
  process.exit(2);
}
const { accounts } = await res0.json();
const acc = accounts.find((a) => a.role === 'farmer');
const auth = await fetch(`${BASE}/api/auth/login`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ userId: acc.id }),
});
const cookie = auth.headers.getSetCookie().join('; ');

const picks = selected();
const byItem = new Map();
for (const { item, file } of picks) {
  const rgb = toPixels(file);
  if (!rgb) {
    console.error(`  ⚠ 디코딩 실패: ${file}`);
    continue;
  }
  const rgba = new Uint8Array(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i += 1) {
    rgba[i * 4] = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  const enc = jpeg.encode({ data: Buffer.from(rgba), width: SIZE, height: SIZE }, 90);
  const body = JSON.stringify({
    image: `data:image/jpeg;base64,${Buffer.from(enc.data).toString('base64')}`,
    pixels: Buffer.from(rgb).toString('base64'),
  });
  const r = await fetch(`${BASE}/api/ai/analyze`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', cookie },
    body,
  });
  if (!r.ok) throw new Error(`분석 실패 ${r.status}`);
  const out = await r.json();
  const names = (out.candidates ?? []).map((c) => c.name);
  const rec = byItem.get(item) ?? { n: 0, top1: 0, top3: 0, wrong: [] };
  rec.n += 1;
  if (names[0] === item) rec.top1 += 1;
  else rec.wrong.push(`${item}→${names[0] ?? '?'}`);
  if (names.slice(0, 3).includes(item)) rec.top3 += 1;
  byItem.set(item, rec);
}

console.log('\n웹 공개 사진(커먼즈, 사람이 선별) · 실행 중인 서버 응답 기준\n');
let n = 0;
let t1 = 0;
let t3 = 0;
for (const [item, r] of byItem) {
  n += r.n;
  t1 += r.top1;
  t3 += r.top3;
  console.log(
    `  ${item.padEnd(4)} n=${String(r.n).padStart(2)}  top-1 ${String(Math.round((r.top1 / r.n) * 100)).padStart(3)}%` +
      `  top-3 ${String(Math.round((r.top3 / r.n) * 100)).padStart(3)}%` +
      (r.wrong.length ? `   오답: ${r.wrong.join(', ')}` : ''),
  );
}
console.log(`\n  전체 n=${n}  top-1 ${Math.round((t1 / n) * 100)}%  top-3 ${Math.round((t3 / n) * 100)}%`);
console.log('\n★ 실환경 평가가 아니다. field_evaluated 는 false 그대로다.');
console.log('  품목별 표본이 한 자릿수라 품목별 수치는 신뢰구간이 매우 넓다.');
