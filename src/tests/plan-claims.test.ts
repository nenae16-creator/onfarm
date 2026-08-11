/**
 * 기획서(docs/PLAN.md)가 주장하는 것이 실제 코드·모델과 맞는지 검사한다.
 *
 * 발표에서 가장 위험한 실패는 "문서에는 그렇게 썼는데 시연에서는 다르게 동작"이다.
 * 문서와 구현이 갈라지는 순간을 테스트가 잡는다.
 */
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PLAN = readFileSync(`${ROOT}docs/PLAN.md`, 'utf8');

/** 모델이 배포돼 있을 때만 의미 있는 검사 — 없으면 건너뛴다(개발 환경 배려). */
const METADATA_PATH = `${ROOT}models/metadata.json`;
const hasModel = existsSync(METADATA_PATH);
const meta = hasModel
  ? (JSON.parse(readFileSync(METADATA_PATH, 'utf8')) as {
      items?: string[];
      field_evaluated?: boolean;
      per_item?: Record<string, { grade_usable?: boolean }>;
    })
  : null;

describe('기획서 주장 ↔ 실제 구현', () => {
  it('기획서가 실제로 읽힌다', () => {
    assert.ok(PLAN.length > 2000, '기획서가 비어 있거나 잘렸다');
    assert.match(PLAN, /Go\/No-Go/, '중단 기준이 빠졌다 — 기획력 항목의 핵심');
  });

  it('중단 기준을 먼저 쓴다는 원칙이 지켜진다', () => {
    // "확산"보다 "중단"이 먼저 나와야 한다
    const stop = PLAN.indexOf('중단 기준');
    const spread = PLAN.indexOf('Phase 3');
    assert.ok(stop > 0 && stop < spread, '중단 기준이 확산 계획보다 뒤에 있다');
  });

  it('PoC 성공 정의가 실제 metadata 필드와 일치한다', { skip: !hasModel }, () => {
    assert.match(PLAN, /field_evaluated/, '기획서에 성공 정의가 없다');
    assert.equal(
      meta?.field_evaluated,
      false,
      'metadata.field_evaluated 가 이미 true 다 — 기획서의 PoC 목표가 무의미해진다',
    );
  });

  it('기획서가 말한 학습 품목 수와 모델이 일치한다', { skip: !hasModel }, () => {
    assert.match(PLAN, /5종/, '기획서에 품목 수 표기가 없다');
    assert.equal(meta?.items?.length, 5, `모델 품목이 ${meta?.items?.length}종 — 기획서와 다르다`);
  });

  it('"양파는 자동 차단 중"이라는 서술이 사실이다', { skip: !hasModel }, () => {
    assert.match(PLAN, /양파는 자동 차단/, '기획서에 해당 서술이 없다');
    assert.equal(
      meta?.per_item?.['양파']?.grade_usable,
      false,
      '양파 등급이 열려 있다 — 기획서 서술이 거짓이 된다',
    );
  });

  it('"신뢰도 0.85 상한" 서술이 정책 상수와 일치한다', async () => {
    assert.match(PLAN, /0\.85/, '기획서에 상한 수치가 없다');
    const { POLICY_MAX_WITHOUT_FIELD_EVAL } = await import('../ai/policy.js');
    assert.equal(
      POLICY_MAX_WITHOUT_FIELD_EVAL,
      0.85,
      '정책 상수가 바뀌었는데 기획서는 0.85 라고 쓰여 있다',
    );
  });

  it('"신규 하드웨어 투자 0원" 주장의 전제(런타임 의존성)가 유지된다', () => {
    assert.match(PLAN, /신규 하드웨어 투자가 0원/);
    const pkg = JSON.parse(readFileSync(`${ROOT}package.json`, 'utf8')) as {
      dependencies?: Record<string, string>;
    };
    const deps = Object.keys(pkg.dependencies ?? {});
    assert.equal(
      deps.length,
      0,
      `필수 런타임 의존성이 생겼다(${deps.join(', ')}) — GPU·별도 서버가 필요해지면 비용 주장이 깨진다`,
    );
  });

  it('농가가 하는 일은 두 가지뿐이라는 RACI 서술이 유지된다', () => {
    assert.match(PLAN, /사진 촬영과 수량 확인 두 가지뿐/);
    // 화면에도 같은 약속이 있어야 한다
    const farmerHome = readFileSync(`${ROOT}public/farmer/index.html`, 'utf8');
    assert.match(farmerHome, /수량만 확인/, '농민 화면의 약속 문구가 사라졌다');
  });
});
