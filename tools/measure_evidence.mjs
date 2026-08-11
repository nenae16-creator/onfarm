/**
 * 품목별 등급 증거를 같은 프로토콜로 측정해 metadata.json 에 기록한다.
 *
 * 왜 필요한가(2차 교차검증 #4):
 *   기존 게이트는 전역 등급 정확도 0.705 > 전역 중량 기준선 0.58 만 보고 모든 품목의
 *   등급을 열었다. 그런데 양파는 모델이 중량 기준선보다 한참 나쁘다. 품목마다 따로 재야 한다.
 *
 * 공정성을 위해 지키는 것:
 *   - 모델과 중량 기준선을 **같은 개체 집합**(valid, overlap 제외)에서 잰다.
 *   - 둘 다 **개체 단위**로 집계한다(이미지 단위는 근접중복 때문에 부풀려진다).
 *   - 중량 기준선의 임계값은 train 개체에서 고르고 valid 에서 한 번만 평가한다.
 *     valid 에서 최적 임계를 찾으면 기준선이 과대평가되어 모델에 불리해진다.
 *
 * 사용: node tools/measure_evidence.mjs [--data data/onfarm_cv] [--models models] [--write]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const arg = (n, d) => {
  const i = process.argv.indexOf(`--${n}`);
  return i > -1 ? process.argv[i + 1] : d;
};
const DATA = arg('data', 'data/onfarm_cv');
const MODELS = arg('models', 'models');
const WRITE = process.argv.includes('--write');

const meta = JSON.parse(readFileSync(join(MODELS, 'metadata.json'), 'utf8'));
const GRADES = meta.grades;

/* ── 매니페스트 읽기 ───────────────────────────────────────────── */
const lines = readFileSync(join(DATA, 'manifest.csv'), 'utf8').split(/\r?\n/);
const head = lines[0].split(',');
const ix = Object.fromEntries(head.map((h, i) => [h, i]));
const rows = [];
for (let i = 1; i < lines.length; i += 1) {
  if (!lines[i]) continue;
  const c = lines[i].split(',');
  rows.push({
    split: c[ix.split],
    path: join(DATA, c[ix.path]),
    group: c[ix.group_no],
    item: c[ix.item],
    grade: c[ix.grade],
    weight: Number(c[ix.weight_g] || 0),
  });
}

// 학습에 쓴 것과 동일하게 겹친 개체는 valid 에서 뺀다
const trainGroups = new Set(rows.filter((r) => r.split === 'train').map((r) => r.group));
const validRows = rows.filter((r) => r.split === 'valid' && !trainGroups.has(r.group));
console.log(`valid 이미지 ${validRows.length.toLocaleString()} (겹친 개체 제외)`);

/** 개체 단위로 접기 — 한 개체는 라벨이 하나여야 한다 */
function foldObjects(list) {
  const byGroup = new Map();
  for (const r of list) {
    if (!byGroup.has(r.group)) byGroup.set(r.group, { ...r, images: [] });
    const g = byGroup.get(r.group);
    if (g.grade !== r.grade || g.item !== r.item) {
      throw new Error(`개체 ${r.group} 안에서 라벨이 충돌한다 — 데이터를 먼저 확인하라`);
    }
    g.images.push(r.path);
  }
  return [...byGroup.values()];
}

const trainObjs = foldObjects(rows.filter((r) => r.split === 'train'));
const validObjs = foldObjects(validRows);
console.log(`train 개체 ${trainObjs.length} / valid 개체 ${validObjs.length}`);

/* ── 중량 단독 기준선: 임계는 train 에서, 평가는 valid 에서 ────── */
function fitWeightThresholds(objs) {
  const ws = [...new Set(objs.map((o) => o.weight))].sort((a, b) => a - b);
  const truth = objs.map((o) => GRADES.indexOf(o.grade));
  let best = { acc: 0, t1: 0, t2: 0 };
  for (const t1 of ws) {
    for (const t2 of ws) {
      if (t2 < t1) continue;
      let ok = 0;
      objs.forEach((o, i) => {
        const p = o.weight <= t1 ? 0 : o.weight <= t2 ? 1 : 2;
        if (p === truth[i]) ok += 1;
      });
      if (ok > best.acc * objs.length) best = { acc: ok / objs.length, t1, t2 };
    }
  }
  return best;
}

function scoreWeight(objs, { t1, t2 }) {
  let ok = 0;
  for (const o of objs) {
    const p = o.weight <= t1 ? 0 : o.weight <= t2 ? 1 : 2;
    if (p === GRADES.indexOf(o.grade)) ok += 1;
  }
  return objs.length ? ok / objs.length : 0;
}

/* ── 모델 등급: 개체 단위 (같은 개체의 확률을 평균) ─────────────── */
const ort = require('onnxruntime-node');
const decodeJpeg = require('jpeg-js').decode;
const session = await ort.InferenceSession.create(join(MODELS, 'onfarm_qc.onnx'));
const { mean, std } = meta.normalize;
const SIZE = meta.img_size;

function tensorOf(path) {
  const img = decodeJpeg(readFileSync(path), { useTArray: true });
  const plane = SIZE * SIZE;
  const out = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[c * plane + i] = (img.data[i * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

function softmax(a) {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0) || 1;
  return e.map((v) => v / s);
}

/** 개체 하나의 모든 이미지를 평균해 등급/품목을 정한다(동점 문제 회피) */
async function predictObject(obj) {
  const gradeSum = new Array(GRADES.length).fill(0);
  const itemSum = new Array(meta.items.length).fill(0);
  for (const p of obj.images) {
    const t = new ort.Tensor('float32', tensorOf(p), [1, 3, SIZE, SIZE]);
    const out = await session.run({ image: t });
    softmax(Array.from(out.grade_logits.data)).forEach((v, i) => (gradeSum[i] += v));
    softmax(Array.from(out.item_logits.data)).forEach((v, i) => (itemSum[i] += v));
  }
  return {
    grade: GRADES[gradeSum.indexOf(Math.max(...gradeSum))],
    item: meta.items[itemSum.indexOf(Math.max(...itemSum))],
  };
}

const perItem = {};
const items = [...new Set(validObjs.map((o) => o.item))].sort();

for (const item of items) {
  const vObjs = validObjs.filter((o) => o.item === item);
  const tObjs = trainObjs.filter((o) => o.item === item);
  const thr = fitWeightThresholds(tObjs);           // 임계는 train 에서만
  const weightAcc = scoreWeight(vObjs, thr);        // 평가는 valid 에서 한 번

  let gradeOk = 0;
  let itemOk = 0;
  for (const o of vObjs) {
    const pred = await predictObject(o);
    if (pred.grade === o.grade) gradeOk += 1;
    if (pred.item === o.item) itemOk += 1;
  }
  const gradeAcc = vObjs.length ? gradeOk / vObjs.length : 0;

  perItem[item] = {
    grade_object_acc: Number(gradeAcc.toFixed(3)),
    weight_only_baseline: Number(weightAcc.toFixed(3)),
    item_object_acc: Number((vObjs.length ? itemOk / vObjs.length : 0).toFixed(3)),
    n_objects: vObjs.length,
    grade_usable: gradeAcc > weightAcc,
  };
  const mark = perItem[item].grade_usable ? '사용' : '차단';
  console.log(
    `  ${item}: 등급 ${gradeAcc.toFixed(3)} vs 중량 ${weightAcc.toFixed(3)} ` +
      `(개체 ${vObjs.length}) → ${mark}`,
  );
}

console.log('\n=== per_item ===');
console.log(JSON.stringify(perItem, null, 2));

if (WRITE) {
  meta.per_item = perItem;
  meta.field_evaluated = false; // 폰 사진 평가는 아직 하지 않았다
  meta.evidence_protocol =
    '중량 임계는 train 개체에서 적합, 평가는 valid 개체에서 1회. 모델은 개체별 확률 평균.';
  writeFileSync(join(MODELS, 'metadata.json'), JSON.stringify(meta, null, 2), 'utf8');
  console.log(`\n기록: ${join(MODELS, 'metadata.json')}`);
} else {
  console.log('\n--write 를 붙이면 metadata.json 에 기록합니다.');
}
