import type { QualityHint } from '../domain/types.js';
import type { PolicyEvidence } from './policy.js';

/**
 * 브라우저(canvas)에서 뽑아 서버로 보내는 이미지 특징.
 * 외부 API 없이도 품목 후보를 좁히기 위한 신호이며, 그 자체로 '판정'은 아니다.
 */
export interface ImageFeatures {
  width: number;
  height: number;
  /** 30도씩 12칸. 채도·명도로 가중해 정규화(합=1). 흰 배경/그림자는 거의 기여하지 않는다. */
  hueHistogram: number[];
  /** 0..1 */
  meanSaturation: number;
  /** 0..1 */
  meanValue: number;
  /** 인접 픽셀 밝기차 평균을 0..1로 정규화한 값(표면 요철·개수 많음의 대리 지표) */
  edgeDensity: number;
  /** 색 분포가 한 곳에 몰려 있을수록 1에 가깝다(=균일) */
  hueConcentration: number;
}

export interface CatalogItem {
  code: string;
  name_ko: string;
  category: string;
  variety: string | null;
}

export interface VisionInput {
  imageBase64?: string;
  mimeType?: string;
  features?: ImageFeatures;
  /**
   * 브라우저가 캔버스로 뽑아 보낸 224×224 RGB(0~255) 픽셀.
   * 서버에 JPEG 디코더를 들이지 않기 위한 것이며, features 와 같은 신뢰 경계에 있다.
   */
  pixels?: Uint8Array;
  /** 우리가 실제로 취급하는 품목만 후보로 준다. 없는 품목을 지어내지 않게 하기 위함. */
  catalog: CatalogItem[];
}

/** STEP 3 의 계약 스키마. 모든 provider 는 이 형태로만 응답해야 한다. */
export interface RecognitionResult {
  category: string;
  product: string;
  product_ko: string;
  variety_guess: string | null;
  quality_hint: QualityHint;
  confidence: number;
  detected_issues: string[];
  description_basis: string[];
  /** 2순위 후보. 신뢰도가 낮을 때 큰 버튼 선택지로 쓴다. */
  alternatives?: Array<{ product: string; product_ko: string; confidence: number }>;
}

export interface VisionProvider {
  readonly name: string;
  /** true 면 이미지가 외부로 나가지 않는다(로컬 판정). 화면에 그대로 표기한다. */
  readonly offline: boolean;
  /**
   * 이 provider 의 성능을 실제로 측정한 증거.
   * 중앙 안전 정책(policy.ts)이 신뢰도 상한과 등급 사용 여부를 정할 때 쓴다.
   * 측정된 적 없는 provider(휴리스틱·LLM·mock)는 넣지 않는다 → 보수적으로 취급된다.
   */
  readonly evidence?: PolicyEvidence | null;
  analyzeProduct(input: VisionInput): Promise<RecognitionResult>;
}

export class VisionProviderError extends Error {
  readonly provider: string;
  readonly reason: unknown;

  constructor(message: string, provider: string, reason?: unknown) {
    super(message);
    this.name = 'VisionProviderError';
    this.provider = provider;
    this.reason = reason;
  }
}
