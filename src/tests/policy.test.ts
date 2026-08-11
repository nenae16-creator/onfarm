/**
 * 중앙 안전 정책 — 2차 교차검증에서 나온 결함 #4·#6·#7 을 막는 코드.
 *
 * #6: 측정 정확도가 1.0 으로 나오는 바람에 상한이 아무 일도 하지 않았다.
 * #4: 등급 게이트가 전역이라 양파(모델 0.434 < 중량 0.892)까지 등급이 나갔다.
 * #7: 상한·게이트가 CNN provider 안에만 있어 provider 를 바꾸면 우회됐다.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  confidenceCeiling,
  enforceRecognitionPolicy,
  gradeAllowedFor,
  MIN_OBJECTS_FOR_GRADE,
  POLICY_MAX_WITHOUT_FIELD_EVAL,
} from '../ai/policy.js';
import type { PolicyEvidence } from '../ai/policy.js';
import type { RecognitionResult } from '../ai/types.js';

function recognition(over: Partial<RecognitionResult> = {}): RecognitionResult {
  return {
    category: 'fruit',
    product: 'pear',
    product_ko: '배',
    variety_guess: '신고배',
    quality_hint: '상',
    confidence: 0.99,
    detected_issues: [],
    description_basis: [],
    alternatives: [{ product: 'apple', product_ko: '사과', confidence: 0.9 }],
    ...over,
  };
}

/** 실제 학습 결과를 본뜬 증거 — 품목 정확도가 1.0 으로 측정된 상황 */
function evidence(over: Partial<PolicyEvidence> = {}): PolicyEvidence {
  return {
    field_evaluated: false,
    item_object_acc: 1.0,
    per_item: {
      배: { grade_object_acc: 0.87, weight_only_baseline: 0.61, n_objects: 33, grade_usable: true },
      양파: { grade_object_acc: 0.43, weight_only_baseline: 0.89, n_objects: 80, grade_usable: false },
      감귤: { grade_object_acc: 0.76, weight_only_baseline: 0.65, n_objects: 8, grade_usable: true },
    },
    ...over,
  };
}

describe('정책 — 신뢰도 상한 (#6)', () => {
  it('측정 정확도가 1.0 이어도 실환경 미검증이면 정책값으로 잘린다', () => {
    // 이 검사가 없으면 Math.min(raw, 1.0) 이 되어 상한이 무력해진다
    assert.equal(confidenceCeiling(evidence()), POLICY_MAX_WITHOUT_FIELD_EVAL);
    const { result, applied } = enforceRecognitionPolicy(recognition(), evidence());
    assert.equal(result.confidence, POLICY_MAX_WITHOUT_FIELD_EVAL);
    assert.ok(applied.some((a) => a.includes('신뢰도')));
  });

  it('대안 후보의 신뢰도도 같은 상한을 받는다', () => {
    const { result } = enforceRecognitionPolicy(recognition(), evidence());
    assert.ok((result.alternatives?.[0]?.confidence ?? 1) <= POLICY_MAX_WITHOUT_FIELD_EVAL);
  });

  it('실환경 평가를 마치면 측정값까지 올라간다', () => {
    const ev = evidence({ field_evaluated: true, item_object_acc: 0.92 });
    assert.equal(confidenceCeiling(ev), 0.92);
  });

  it('상한보다 낮은 신뢰도는 건드리지 않는다', () => {
    const { result, applied } = enforceRecognitionPolicy(recognition({ confidence: 0.4 }), evidence());
    assert.equal(result.confidence, 0.4);
    assert.ok(!applied.some((a) => a.includes('신뢰도')));
  });

  it('증거가 없는 provider 는 보수적 상한을 받는다', () => {
    assert.equal(confidenceCeiling(null), POLICY_MAX_WITHOUT_FIELD_EVAL);
  });
});

describe('정책 — 품목별 등급 게이트 (#4)', () => {
  it('기준선을 넘는 품목은 등급을 쓴다', () => {
    assert.equal(gradeAllowedFor('배', evidence()), true);
    const { result } = enforceRecognitionPolicy(recognition(), evidence());
    assert.equal(result.quality_hint, '상');
  });

  it('중량 기준선을 못 넘는 양파는 등급이 막힌다', () => {
    assert.equal(gradeAllowedFor('양파', evidence()), false);
    const { result, applied } = enforceRecognitionPolicy(
      recognition({ product: 'onion', product_ko: '양파', quality_hint: '특' }),
      evidence(),
    );
    assert.equal(result.quality_hint, '확인필요');
    assert.ok(applied.some((a) => a.includes('양파')));
  });

  it('검증 개체가 적으면 이겨도 등급을 열지 않는다', () => {
    // 감귤은 0.76 > 0.65 로 이기지만 개체가 8개뿐이다
    assert.ok(8 < MIN_OBJECTS_FOR_GRADE);
    assert.equal(gradeAllowedFor('감귤', evidence()), false);
  });

  it('증거에 없는 품목은 등급을 쓰지 않는다', () => {
    assert.equal(gradeAllowedFor('고구마', evidence()), false);
  });

  it('전역 평균이 기준선을 넘어도 품목별로 판단한다', () => {
    // 전역만 보면 0.705 > 0.58 이라 전부 열렸다 — 그 회귀를 막는다
    const ev = evidence();
    assert.equal(gradeAllowedFor('배', ev), true);
    assert.equal(gradeAllowedFor('양파', ev), false);
  });
});

describe('정책 — provider 우회 차단 (#7)', () => {
  it('증거 없는 provider 가 confidence 1·등급 특을 불러도 깎인다', () => {
    // AI_PROVIDER=openai 로 바꿔 유효한 JSON 을 반환하는 상황
    const { result, applied } = enforceRecognitionPolicy(
      recognition({ confidence: 1, quality_hint: '특' }),
      null,
    );
    assert.equal(result.confidence, POLICY_MAX_WITHOUT_FIELD_EVAL);
    assert.equal(result.quality_hint, '확인필요');
    assert.equal(applied.length, 2);
  });

  it('이미 확인필요면 등급 관련 메시지를 덧붙이지 않는다', () => {
    const { result, applied } = enforceRecognitionPolicy(
      recognition({ confidence: 0.3, quality_hint: '확인필요' }),
      null,
    );
    assert.equal(result.quality_hint, '확인필요');
    assert.equal(applied.length, 0);
  });
});
