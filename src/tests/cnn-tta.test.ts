/**
 * 다중 배율 판정(TTA)이 실제로 동작하는지 검사한다.
 *
 * 이 기능이 없으면 "멀리서 찍은 감자"가 후보 3개 안에도 들어오지 않는다(실측 0%).
 * 그래서 여기서 지키는 것은 정확도 자체가 아니라, 정확도를 만들어 내는 **구조**다.
 *   · 중앙 크롭이 실제로 확대하는가 (같은 그림을 그대로 돌려주면 TTA 는 아무 일도 안 한다)
 *   · 배율 수만큼 배치가 만들어지는가
 *   · 합쳐진 확률이 합 1로 되돌아오는가 (안 하면 화면 신뢰도가 부풀려진다)
 *   · 등급은 원본 배율만 쓰는가
 */
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const hasModel = existsSync(`${ROOT}models/metadata.json`);

/** 가운데에만 표식이 있는 224 그림 — 확대되면 표식이 커진다. */
function pixelsWithCenterMark(size = 224): Uint8Array {
  const rgb = new Uint8Array(size * size * 3).fill(200);
  const half = Math.floor(size / 2);
  for (let y = half - 8; y < half + 8; y += 1) {
    for (let x = half - 8; x < half + 8; x += 1) {
      const i = (y * size + x) * 3;
      rgb[i] = 10;
      rgb[i + 1] = 20;
      rgb[i + 2] = 30;
    }
  }
  return rgb;
}

function countDark(rgb: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < rgb.length; i += 3) if ((rgb[i] ?? 255) < 100) n += 1;
  return n;
}

describe('다중 배율 판정 — 중앙 크롭', () => {
  it('배율 1.0 은 원본을 그대로 돌려준다', { skip: !hasModel }, async () => {
    const { CnnVisionProvider } = await import('../ai/providers/cnn.js');
    const crop = (CnnVisionProvider.prototype as any).cropCenter.bind({
      meta: { img_size: 224 },
    });
    const src = pixelsWithCenterMark();
    assert.equal(crop(src, 1), src, '1.0 에서 복사본을 만들면 헛일이다');
  });

  it('배율을 줄이면 가운데 표식이 실제로 커진다', { skip: !hasModel }, async () => {
    const { CnnVisionProvider } = await import('../ai/providers/cnn.js');
    const crop = (CnnVisionProvider.prototype as any).cropCenter.bind({
      meta: { img_size: 224 },
    });
    const src = pixelsWithCenterMark();
    const base = countDark(src);
    const half = countDark(crop(src, 0.5));
    const quarter = countDark(crop(src, 0.25));

    assert.ok(half > base * 3, `0.5 배율에서 표식이 ${base}→${half} 로 충분히 커지지 않았다`);
    assert.ok(quarter > half, `0.25 가 0.5 보다 크게 확대해야 한다 (${half} vs ${quarter})`);
  });

  it('가장자리가 잘려도 크기가 224 로 유지된다', { skip: !hasModel }, async () => {
    const { CnnVisionProvider } = await import('../ai/providers/cnn.js');
    const crop = (CnnVisionProvider.prototype as any).cropCenter.bind({
      meta: { img_size: 224 },
    });
    for (const scale of [0.7, 0.5, 0.35, 0.25, 0.05]) {
      assert.equal(crop(pixelsWithCenterMark(), scale).length, 224 * 224 * 3, `배율 ${scale}`);
    }
  });
});

describe('다중 배율 판정 — 합치기', () => {
  it('배율 목록이 1.0 을 포함한다', async () => {
    const src = await import('node:fs').then((fs) =>
      fs.readFileSync(`${ROOT}src/ai/providers/cnn.ts`, 'utf8'),
    );
    const m = src.match(/const TTA_SCALES = \[([^\]]+)\]/);
    assert.ok(m, 'TTA_SCALES 를 찾지 못했다');
    const scales = (m[1] ?? '').split(',').map((s) => Number(s.trim()));
    assert.ok(scales.includes(1), '1.0 이 빠지면 피사체가 치우쳤을 때 되레 나빠진다');
    assert.ok(scales.length >= 3, `배율이 ${scales.length}개뿐 — 2개면 치우친 구도에서 퇴행한다`);
    assert.ok(
      scales.every((s) => s > 0 && s <= 1),
      '배율은 0 초과 1 이하여야 한다',
    );
  });

  it('합친 확률을 다시 정규화한다 — 신뢰도가 부풀려지면 안 된다', async () => {
    const src = await import('node:fs').then((fs) =>
      fs.readFileSync(`${ROOT}src/ai/providers/cnn.ts`, 'utf8'),
    );
    assert.match(
      src,
      /fused\.reduce\(\(a, b\) => a \+ b, 0\)/,
      '최댓값으로 합친 뒤 합으로 나누는 코드가 없다',
    );
    assert.match(src, /fused\.map\(\(p\) => p \/ total\)/, '정규화 결과를 쓰지 않는다');
  });

  it('등급은 원본 배율만 쓴다', async () => {
    const src = await import('node:fs').then((fs) =>
      fs.readFileSync(`${ROOT}src/ai/providers/cnn.ts`, 'utf8'),
    );
    assert.match(
      src,
      /grade_logits'\]\.data as Float32Array\)\.subarray\(0, nGrades\)/,
      '등급까지 배율을 합치면 안 된다 — 크롭 조건에서 등급 성능을 잰 적이 없다',
    );
  });
});
