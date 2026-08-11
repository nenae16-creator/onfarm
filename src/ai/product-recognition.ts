import { enforceRecognitionPolicy } from './policy.js';
import { resolveProvider } from './providers/index.js';
import type { ProviderSelection } from './providers/index.js';
import { unknownRecognition, validateRecognition } from './schema.js';
import type { RecognitionResult, VisionInput, VisionProvider } from './types.js';

export interface RecognitionOutcome {
  recognition: RecognitionResult;
  /** 실제로 판정에 쓰인 provider 이름 */
  source: string;
  /** 이미지가 외부로 나가지 않았는가 */
  offline: boolean;
  /** primary 실패 등으로 강등됐다면 사유 */
  degraded: string | null;
  demoMode: boolean;
  /** 중앙 안전 정책이 깎은 항목(신뢰도 상한·등급 차단). 화면과 로그에 그대로 남긴다. */
  policyApplied: string[];
}

/**
 * provider 결과에 중앙 정책을 적용한다.
 * 여기를 거치지 않고 파이프라인 뒤쪽으로 가는 경로가 있으면 안 된다.
 */
function applyPolicy(provider: VisionProvider, result: RecognitionResult) {
  return enforceRecognitionPolicy(result, provider.evidence ?? null);
}

/**
 * STEP 1. 품목 인식.
 * primary provider 가 실패하면 조용히 로컬 휴리스틱으로 폴백하고, 그 사실을 그대로 반환한다.
 * (실패를 숨기지 않는다 — 화면과 로그 양쪽에 남는다)
 */
export async function recognizeProduct(
  input: VisionInput,
  selection: ProviderSelection = resolveProvider(),
): Promise<RecognitionOutcome> {
  const { primary, fallback } = selection;
  try {
    const raw = await primary.analyzeProduct(input);
    // provider 가 직접 만든 객체라도 계약을 다시 검증한다.
    const checked = validateRecognition(raw, input.catalog);
    if (!checked.ok) throw new Error(checked.errors.join(', '));
    const policed = applyPolicy(primary, checked.value);
    return {
      recognition: policed.result,
      source: primary.name,
      offline: primary.offline,
      degraded: selection.degradedReason,
      demoMode: selection.demoMode,
      policyApplied: policed.applied,
    };
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    if (primary === fallback) {
      return {
        recognition: unknownRecognition('사진을 자동으로 확인하지 못했습니다.'),
        source: 'none',
        offline: true,
        degraded: reason,
        demoMode: selection.demoMode,
        policyApplied: [],
      };
    }
    try {
      const raw = await fallback.analyzeProduct(input);
      const checked = validateRecognition(raw, input.catalog);
      const policed = applyPolicy(
        fallback,
        checked.ok ? checked.value : unknownRecognition('사진을 자동으로 확인하지 못했습니다.'),
      );
      return {
        recognition: policed.result,
        source: fallback.name,
        offline: fallback.offline,
        degraded: `${primary.name} 실패(${reason}) → 로컬 판정으로 대체`,
        demoMode: selection.demoMode,
        policyApplied: policed.applied,
      };
    } catch (fallbackErr) {
      const reason2 = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
      return {
        recognition: unknownRecognition('사진을 자동으로 확인하지 못했습니다.'),
        source: 'none',
        offline: true,
        degraded: `${primary.name} 실패(${reason}), 로컬 판정도 실패(${reason2})`,
        demoMode: selection.demoMode,
        policyApplied: [],
      };
    }
  }
}
