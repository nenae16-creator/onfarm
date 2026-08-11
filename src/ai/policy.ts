import type { QualityHint } from '../domain/types.js';
import type { RecognitionResult } from './types.js';

/**
 * 인식 결과에 적용하는 중앙 안전 정책.
 *
 * 왜 provider 밖에 있는가:
 *   상한과 등급 게이트를 CNN provider 안에 두었더니 `AI_PROVIDER=openai` 로 바꾸면
 *   통째로 우회됐다(2차 교차검증 #7). 같은 계열 프로젝트에서 이미 한 번 겪은 사고라
 *   이번에는 모든 provider 가 반드시 지나는 길목에 둔다.
 *
 * 무엇을 강제하는가:
 *   1) 실환경(폰 사진) 평가를 거치지 않은 모델은 아무리 검증 정확도가 높아도
 *      화면 신뢰도를 POLICY_MAX_WITHOUT_FIELD_EVAL 위로 올리지 못한다.
 *   2) 등급은 '그 품목에 대해 측정된 증거'가 있을 때만 쓴다. 없으면 '확인필요'.
 */

/**
 * 실환경 평가 전 표시할 수 있는 최대 신뢰도.
 *
 * 측정 정확도를 그대로 상한으로 쓰면 안 된다 — 실제로 개체 단위 품목 정확도가 1.0 으로
 * 나오는 바람에 `Math.min(raw, 1.0)` 이 되어 상한이 아무 일도 하지 않았다(2차 교차검증 #6).
 * 스튜디오 데이터로만 검증한 모델은 폰 사진에서 더 틀린다는 것이 기본 가정이므로,
 * 측정값과 별개인 보수적 정책값을 둔다.
 */
export const POLICY_MAX_WITHOUT_FIELD_EVAL = 0.85;

/** 등급을 열어주기 위해 요구하는 최소 검증 개체 수. 표본이 작으면 우연히 이길 수 있다. */
export const MIN_OBJECTS_FOR_GRADE = 20;

export interface ItemEvidence {
  /** 개체 단위 등급 정확도 */
  grade_object_acc?: number;
  /** 같은 개체 집합에서 중량만으로 맞힌 비율(사소한 기준선) */
  weight_only_baseline?: number;
  n_objects?: number;
  /** 도구가 같은 프로토콜로 비교해 내린 결론 */
  grade_usable?: boolean;
}

export interface PolicyEvidence {
  /** 폰 사진 등 실환경에서 평가했는가. 아니면 상한이 걸린다. */
  field_evaluated?: boolean;
  /** 개체 단위 품목 정확도(참고용, 상한 계산에 함께 쓴다) */
  item_object_acc?: number;
  /** 품목별 증거. 한국어 품목명이 키. */
  per_item?: Record<string, ItemEvidence>;
}

/** 이 provider 의 결과에 증거를 적용할 수 있는가(=측정된 모델인가). */
function hasEvidence(evidence: PolicyEvidence | null): evidence is PolicyEvidence {
  return Boolean(evidence);
}

export function confidenceCeiling(evidence: PolicyEvidence | null): number {
  if (!hasEvidence(evidence)) return POLICY_MAX_WITHOUT_FIELD_EVAL;
  const measured = evidence.item_object_acc ?? POLICY_MAX_WITHOUT_FIELD_EVAL;
  if (evidence.field_evaluated) return Math.min(1, Math.max(0, measured));
  return Math.min(measured, POLICY_MAX_WITHOUT_FIELD_EVAL);
}

/**
 * 이 품목의 등급을 화면에 써도 되는가.
 * 증거가 없거나, 표본이 작거나, 중량만 보는 기준선을 못 넘으면 쓰지 않는다.
 */
export function gradeAllowedFor(itemNameKo: string, evidence: PolicyEvidence | null): boolean {
  if (!hasEvidence(evidence)) return false;
  const item = evidence.per_item?.[itemNameKo];
  if (!item) return false;
  if ((item.n_objects ?? 0) < MIN_OBJECTS_FOR_GRADE) return false;
  if (item.grade_usable === false) return false;
  const acc = item.grade_object_acc;
  const base = item.weight_only_baseline;
  if (acc === undefined || base === undefined) return false;
  return acc > base;
}

export interface PolicyResult {
  result: RecognitionResult;
  /** 정책이 실제로 무엇을 깎았는지 — 로그·감사·화면 설명에 쓴다. */
  applied: string[];
}

/**
 * provider 가 무엇이든 이 함수를 통과한 결과만 파이프라인 뒤쪽으로 간다.
 *
 * @param evidence 측정 증거. 측정되지 않은 provider(휴리스틱·LLM·mock)는 null 을 넘긴다.
 */
export function enforceRecognitionPolicy(
  raw: RecognitionResult,
  evidence: PolicyEvidence | null,
): PolicyResult {
  const applied: string[] = [];
  const ceiling = confidenceCeiling(evidence);

  let confidence = raw.confidence;
  if (confidence > ceiling) {
    applied.push(`신뢰도 ${confidence.toFixed(2)} → ${ceiling.toFixed(2)} (실환경 미검증 상한)`);
    confidence = ceiling;
  }

  const alternatives = (raw.alternatives ?? []).map((alt) => ({
    ...alt,
    confidence: Math.min(alt.confidence, ceiling),
  }));

  let qualityHint: QualityHint = raw.quality_hint;
  if (qualityHint !== '확인필요' && !gradeAllowedFor(raw.product_ko, evidence)) {
    applied.push(`${raw.product_ko || '이 품목'} 등급은 근거가 없어 표시하지 않음`);
    qualityHint = '확인필요';
  }

  return {
    result: {
      ...raw,
      confidence: Number(confidence.toFixed(3)),
      quality_hint: qualityHint,
      alternatives,
    },
    applied,
  };
}
