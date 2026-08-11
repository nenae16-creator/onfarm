import type { Db } from '../db/index.js';
import type { Farm, Product } from '../domain/types.js';
import { todayKst } from '../lib/datetime.js';
import { recognizeProduct } from './product-recognition.js';
import type { ProviderSelection } from './providers/index.js';
import { writeProduct } from './product-writer.js';
import { analyzeQuality } from './quality-analysis.js';
import { decideFlow } from './rule-engine.js';
import type { FlowDecision } from './rule-engine.js';
import { catalog, findProductByCode, matchSkus } from './sku-matcher.js';
import type { SkuCandidate } from './sku-matcher.js';
import type { ImageFeatures, RecognitionResult } from './types.js';

export interface PipelineRequest {
  imageBase64?: string;
  mimeType?: string;
  features?: ImageFeatures;
  pixels?: Uint8Array;
  /** 사용자가 화면에서 직접 고른 품목. 있으면 인식 단계를 건너뛴다. */
  forcedProductCode?: string;
  /**
   * 모델이 원래 1순위로 뽑았던 품목.
   * 사용자가 후보를 고르면 recognition 이 그 값으로 덮여 쓰이므로,
   * '사용자가 AI 와 다른 선택을 했는가'를 나중에 알 수 없게 된다. 그래서 따로 들고 다닌다.
   */
  modelTop?: string;
  /** 모델 1순위의 원래 확신도(위와 같은 이유로 보존). */
  modelTopConfidence?: number;
  /** 최초 인식에 쓰인 provider 이름(cnn 등). 후보 선택 재분석 때 물려받는다. */
  modelSource?: string;
  farm: Farm;
  farmerName: string;
}

/** 농민에게 번호로 제시할 후보 한 칸. */
export interface Candidate {
  code: string;
  name: string;
  emoji: string | null;
  confidence: number;
  /** 가격표(SKU)가 있어 실제로 팔 수 있는 품목인가 */
  sellable: boolean;
}

export interface PipelineResult {
  recognition: RecognitionResult;
  decision: FlowDecision;
  /**
   * 화면에 번호로 띄울 후보 목록(1순위가 맨 앞).
   * 신뢰도가 높아도 비워두지 않는다 — 농민이 고르는 편이 '맞아요'만 누르는 것보다
   * 오등록을 줄이고, AI 가 단정하지 않는다는 설계와도 맞는다.
   */
  candidates: Candidate[];
  product: Product | null;
  skus: SkuCandidate[];
  selectedSku: SkuCandidate | null;
  draft: { title: string; description: string } | null;
  ai: {
    source: string;
    offline: boolean;
    degraded: string | null;
    demoMode: boolean;
    label: string;
    /** 모델의 원래 1순위(사용자 선택 전). 감사 기록용. */
    modelTop: string | null;
    modelTopConfidence: number | null;
    modelSource: string | null;
    /** 중앙 정책이 깎은 항목(신뢰도 상한·등급 차단). 비어 있지 않으면 화면·감사에 남긴다. */
    policyApplied: string[];
  };
  /** manual 모드에서 큰 버튼으로 보여줄 전체 품목 */
  catalog: Array<Pick<Product, 'code' | 'name_ko' | 'emoji'>>;
}

function sourceLabel(source: string, offline: boolean): string {
  switch (source) {
    case 'heuristic':
      return '로컬 색·질감 규칙 판정';
    case 'openai':
      return 'OpenAI 이미지 인식';
    case 'anthropic':
      return 'Claude 이미지 인식';
    case 'mock':
      return '데모 고정 응답';
    case 'cnn':
      return '학습 모델 판정(기기 내)';
    default:
      return offline ? '로컬 판정' : '외부 인식';
  }
}

/**
 * IMAGE → 품목인식 → 품질 참고신호 → SKU 매칭 → 룰엔진 → 상품문안
 * 각 단계는 별도 모듈이며, 여기서는 순서와 병합만 담당한다.
 */
export async function runPipeline(
  db: Db,
  req: PipelineRequest,
  selection?: ProviderSelection,
): Promise<PipelineResult> {
  const items = catalog(db);
  const catalogInput = items.map((p) => ({
    code: p.code,
    name_ko: p.name_ko,
    category: p.category,
    variety: p.variety,
  }));

  // STEP 1 — 품목 인식 (또는 사용자가 직접 고른 품목)
  let recognition: RecognitionResult;
  let source: string;
  let offline: boolean;
  let degraded: string | null;
  let demoMode: boolean;
  let policyApplied: string[] = [];

  const localQuality = analyzeQuality(req.features);

  if (req.forcedProductCode) {
    const picked = items.find((p) => p.code === req.forcedProductCode);
    recognition = {
      category: picked?.category ?? 'unknown',
      product: picked?.code ?? '',
      product_ko: picked?.name_ko ?? '',
      variety_guess: picked?.variety ?? null,
      quality_hint: localQuality.hint,
      confidence: 1,
      detected_issues: localQuality.issues,
      description_basis: localQuality.basis,
      alternatives: [],
    };
    source = 'manual';
    offline = true;
    degraded = null;
    demoMode = false;
    policyApplied = [];
  } else {
    const outcome = await recognizeProduct(
      {
        ...(req.imageBase64 ? { imageBase64: req.imageBase64 } : {}),
        ...(req.mimeType ? { mimeType: req.mimeType } : {}),
        ...(req.features ? { features: req.features } : {}),
        ...(req.pixels ? { pixels: req.pixels } : {}),
        catalog: catalogInput,
      },
      selection,
    );
    recognition = outcome.recognition;
    source = outcome.source;
    offline = outcome.offline;
    degraded = outcome.degraded;
    demoMode = outcome.demoMode;
    policyApplied = outcome.policyApplied;

    // STEP 2 — 로컬 품질 신호 병합. 사진 자체에 문제가 있으면 그쪽을 우선한다.
    if (req.features) {
      const mergedIssues = Array.from(
        new Set([...localQuality.issues, ...recognition.detected_issues]),
      );
      recognition = {
        ...recognition,
        detected_issues: mergedIssues,
        description_basis: Array.from(
          new Set([...recognition.description_basis, ...localQuality.basis]),
        ),
        quality_hint: localQuality.issues.length > 0 ? '확인필요' : recognition.quality_hint,
      };
    }
  }

  // STEP 3 — SKU 매칭 (가격의 유일한 출처)
  const skus = recognition.product ? matchSkus(db, recognition.product) : [];
  const selectedSku = skus[0] ?? null;
  const product = recognition.product ? findProductByCode(db, recognition.product) : null;

  // STEP 4 — 룰 엔진
  const decision = decideFlow(recognition, skus.length > 0);

  // 후보 목록: 1순위 + 대안들을 카탈로그와 맞춰 번호표로 만든다.
  // 팔 수 없는 품목(SKU 미등록)은 눌러도 진행이 막히므로 표시에서 걸러낸다.
  const seen = new Set<string>();
  const candidates: Candidate[] = [];
  const push = (code: string, confidence: number): void => {
    if (!code || seen.has(code)) return;
    const item = items.find((p) => p.code === code);
    if (!item) return;
    seen.add(code);
    candidates.push({
      code: item.code,
      name: item.name_ko,
      emoji: item.emoji,
      confidence: Number(confidence.toFixed(3)),
      sellable: matchSkus(db, item.code).length > 0,
    });
  };
  push(recognition.product, recognition.confidence);
  for (const alt of recognition.alternatives ?? []) push(alt.product, alt.confidence);
  const shown = candidates.filter((c) => c.sellable).slice(0, 3);

  // STEP 5 — 상품 문안 자동 생성
  const today = todayKst();
  const draft =
    product && selectedSku
      ? writeProduct({
          product,
          sku: selectedSku,
          farm: req.farm,
          farmerName: req.farmerName,
          recognition,
          harvestedOn: today,
          today,
          aiSourceLabel: sourceLabel(source, offline),
        })
      : null;

  return {
    recognition,
    decision,
    candidates: shown,
    product,
    skus,
    selectedSku,
    draft,
    ai: {
      source,
      offline,
      degraded,
      demoMode,
      label: sourceLabel(source, offline),
      modelTop: req.modelTop ?? (req.forcedProductCode ? null : recognition.product || null),
      modelTopConfidence:
        req.modelTopConfidence ?? (req.forcedProductCode ? null : recognition.confidence),
      modelSource: req.modelSource ?? (req.forcedProductCode ? null : source),
      policyApplied,
    },
    catalog: items.map((p) => ({ code: p.code, name_ko: p.name_ko, emoji: p.emoji })),
  };
}
