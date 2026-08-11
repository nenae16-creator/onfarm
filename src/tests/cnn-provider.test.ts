/**
 * CNN provider 는 모델이 없어도 앱을 깨뜨리면 안 된다.
 * (학습 결과가 나오기 전까지 서버는 기존 로컬 판정으로 돌아야 한다)
 */
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';
import { CnnVisionProvider, loadMetadata, toEvidence } from '../ai/providers/cnn.js';
import type { CnnMetadata } from '../ai/providers/cnn.js';
import { VisionProviderError } from '../ai/types.js';

const work = mkdtempSync(join(tmpdir(), 'onfarm-cnn-'));
after(() => rmSync(work, { recursive: true, force: true }));

function meta(overrides: Partial<CnnMetadata> = {}): CnnMetadata {
  return {
    items: ['배', '사과'],
    grades: ['보통', '상', '특'],
    img_size: 224,
    normalize: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
    val_object_level: { item: 0.82, grade: 0.55, n_objects: 577 },
    weight_only_grade_baseline: 0.61,
    ...overrides,
  };
}

describe('CNN provider — 모델이 없을 때', () => {
  it('메타데이터가 없으면 생성에 실패한다(앱은 폴백으로 계속 돈다)', async () => {
    await assert.rejects(
      () => CnnVisionProvider.create(join(work, 'nowhere')),
      (err: unknown) => err instanceof VisionProviderError,
    );
  });

  it('메타데이터만 있고 모델 파일이 없으면 실패한다', async () => {
    const dir = join(work, 'meta-only');
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'metadata.json'), JSON.stringify(meta()), 'utf8');
    await assert.rejects(
      () => CnnVisionProvider.create(dir),
      (err: unknown) => err instanceof VisionProviderError && /모델 파일 없음/.test(err.message),
    );
  });

  it('깨진 메타데이터를 거부한다', () => {
    const dir = join(work, 'bad-meta');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'metadata.json'), JSON.stringify({ items: [] }), 'utf8');
    assert.throws(() => loadMetadata(dir), (e: unknown) => e instanceof VisionProviderError);
  });
});

describe('CNN provider — 입력 텐서 변환', () => {
  function provider(m: CnnMetadata = meta()): CnnVisionProvider {
    // 생성자는 private 이므로 변환 로직만 떼어 검사한다.
    return Object.create(CnnVisionProvider.prototype, {
      meta: { value: m },
    }) as CnnVisionProvider;
  }

  it('픽셀 길이가 맞지 않으면 거부한다', () => {
    assert.throws(
      () => provider().toTensor(new Uint8Array(100)),
      (e: unknown) => e instanceof VisionProviderError && /픽셀 길이/.test(e.message),
    );
  });

  it('정규화가 ImageNet 기준과 일치한다', () => {
    const size = 224;
    const rgb = new Uint8Array(size * size * 3).fill(255);
    const t = provider().toTensor(rgb);
    // (1 - mean) / std
    assert.ok(Math.abs((t[0] as number) - (1 - 0.485) / 0.229) < 1e-5);
    const plane = size * size;
    assert.ok(Math.abs((t[plane] as number) - (1 - 0.456) / 0.224) < 1e-5);
    assert.ok(Math.abs((t[2 * plane] as number) - (1 - 0.406) / 0.225) < 1e-5);
  });

  it('채널 우선(NCHW) 순서로 배치한다', () => {
    const size = 224;
    const rgb = new Uint8Array(size * size * 3);
    rgb[0] = 255; // 첫 픽셀의 R 만 255
    const t = provider().toTensor(rgb);
    const plane = size * size;
    assert.ok((t[0] as number) > (t[plane] as number), 'R 평면이 G 평면보다 커야 한다');
  });
});

describe('CNN provider — 증거 전달', () => {
  it('메타데이터를 중앙 정책이 쓸 증거로 옮긴다', () => {
    const ev = toEvidence(meta({
      per_item: { 배: { grade_object_acc: 0.87, weight_only_baseline: 0.61, n_objects: 33 } },
    } as never));
    assert.equal(ev.item_object_acc, 0.82);
    assert.equal(ev.field_evaluated, false, '실환경 평가 여부를 명시하지 않으면 false 여야 한다');
    assert.ok(ev.per_item?.['배']);
  });

  it('per_item 이 없으면 증거에도 없다 — 정책이 등급을 막는다', () => {
    const ev = toEvidence(meta());
    assert.equal(ev.per_item, undefined);
  });
});
