import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { analyzeQuality } from '../quality-analysis.js';
import { VisionProviderError } from '../types.js';
import type { RecognitionResult, VisionInput, VisionProvider } from '../types.js';
import type { ItemEvidence, PolicyEvidence } from '../policy.js';

/**
 * AI 허브 농산물 품질(QC) 이미지로 학습한 CNN provider.
 *
 * 서버는 런타임 의존성 0 이 원칙이라 `onnxruntime-node` 를 **선택적 의존성**으로 둔다.
 * 설치돼 있으면 이 provider 가 뜨고, 없으면 생성 단계에서 거부돼 기존 로컬 판정으로 폴백한다.
 * (설치: npm i -O onnxruntime-node)
 *
 * 이미지 디코딩도 하지 않는다. JPEG 디코더를 서버에 들이면 의존성이 늘기 때문에,
 * 이미 캔버스로 사진을 다루고 있는 브라우저가 224×224 RGB 픽셀을 함께 보낸다.
 * 따라서 features 와 동일한 신뢰 경계에 있다 — DEVELOPMENT_STATUS.md 에 명시.
 */

export interface CnnMetadata {
  items: string[];
  grades: string[];
  img_size: number;
  normalize: { mean: number[]; std: number[] };
  val_object_level?: { item?: number; grade?: number; n_objects?: number };
  weight_only_grade_baseline?: number;
  mean_confidence_when_wrong?: number | null;
  /** 실환경(폰 사진) 평가를 거쳤는가. tools/verify_model.mjs 가 기록한다. */
  field_evaluated?: boolean;
  /** 품목별 등급 증거. 없으면 등급을 쓰지 않는다. */
  per_item?: Record<string, ItemEvidence>;
}

export function loadMetadata(dir: string): CnnMetadata {
  const path = join(dir, 'metadata.json');
  if (!existsSync(path)) throw new VisionProviderError(`metadata.json 없음: ${dir}`, 'cnn');
  const meta = JSON.parse(readFileSync(path, 'utf8')) as CnnMetadata;
  if (!Array.isArray(meta.items) || meta.items.length === 0) {
    throw new VisionProviderError('metadata.items 가 비어 있습니다.', 'cnn');
  }
  if (!meta.normalize?.mean || !meta.normalize?.std) {
    throw new VisionProviderError('metadata.normalize 누락', 'cnn');
  }
  return meta;
}

/**
 * 이 모델의 측정 증거를 중앙 정책이 쓸 형태로 넘긴다.
 *
 * 상한과 등급 게이트를 여기(provider 안)에서 직접 적용하지 않는다.
 * 그렇게 했더니 provider 를 바꾸면 통째로 우회됐다(2차 교차검증 #7).
 * 판단은 policy.ts 가 하고, provider 는 증거만 제공한다.
 */
export function toEvidence(meta: CnnMetadata): PolicyEvidence {
  return {
    field_evaluated: meta.field_evaluated === true,
    ...(meta.val_object_level?.item !== undefined
      ? { item_object_acc: meta.val_object_level.item }
      : {}),
    ...(meta.per_item ? { per_item: meta.per_item } : {}),
  };
}

function softmax(logits: Float32Array | number[]): number[] {
  const arr = Array.from(logits);
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

export class CnnVisionProvider implements VisionProvider {
  readonly name = 'cnn';
  readonly offline = true;
  readonly evidence: PolicyEvidence;

  private session: unknown = null;

  private constructor(
    private readonly meta: CnnMetadata,
    private readonly modelPath: string,
    private readonly ort: any,
    session: unknown,
  ) {
    this.evidence = toEvidence(meta);
    this.session = session;
  }

  /** onnxruntime-node 가 없으면 여기서 실패한다 → 팩토리가 heuristic 으로 폴백한다. */
  static async create(dir: string): Promise<CnnVisionProvider> {
    const meta = loadMetadata(dir);
    const modelPath = join(dir, 'onfarm_qc.onnx');
    if (!existsSync(modelPath)) {
      throw new VisionProviderError(`모델 파일 없음: ${modelPath}`, 'cnn');
    }
    // ONNX 가 가중치를 외부 파일로 분리해 저장하는 경우가 있다(실제로 그렇게 나왔다).
    // .onnx 만 배포하면 서버는 정상으로 뜨고 첫 요청에서야 죽는다 — 시작할 때 잡는다.
    const externalData = `${modelPath}.data`;
    if (!existsSync(externalData) && readFileSync(modelPath).includes('.onnx.data')) {
      throw new VisionProviderError(
        `외부 가중치 파일이 없습니다: ${externalData} (.onnx 와 같은 폴더에 두세요)`,
        'cnn',
      );
    }
    let ort: any;
    try {
      const moduleName = 'onnxruntime-node';
      ort = await import(moduleName);
    } catch {
      throw new VisionProviderError(
        'onnxruntime-node 가 설치돼 있지 않습니다. `npm i -O onnxruntime-node` 후 다시 시작하세요.',
        'cnn',
      );
    }
    // 세션을 여기서 만든다. lazy 로 두면 첫 동시 요청이 모델을 두 번 로딩하고(2차 교차검증 #12),
    // 로딩 실패도 서버가 뜬 뒤에야 드러난다.
    let session: unknown;
    try {
      session = await ort.InferenceSession.create(modelPath);
    } catch (err) {
      throw new VisionProviderError(
        `ONNX 세션 생성 실패: ${err instanceof Error ? err.message : String(err)}`,
        'cnn',
      );
    }
    return new CnnVisionProvider(meta, modelPath, ort, session);
  }

  /** 브라우저가 보낸 224×224 RGB(0~255) 를 정규화된 NCHW 텐서로 바꾼다. */
  toTensor(rgb: Uint8Array): Float32Array {
    const size = this.meta.img_size;
    const expected = size * size * 3;
    if (rgb.length !== expected) {
      throw new VisionProviderError(
        `픽셀 길이가 다릅니다. 기대 ${expected}, 실제 ${rgb.length}`,
        'cnn',
      );
    }
    const { mean, std } = this.meta.normalize;
    const out = new Float32Array(expected);
    const plane = size * size;
    for (let i = 0; i < plane; i += 1) {
      for (let c = 0; c < 3; c += 1) {
        const v = (rgb[i * 3 + c] ?? 0) / 255;
        out[c * plane + i] = (v - (mean[c] ?? 0)) / (std[c] ?? 1);
      }
    }
    return out;
  }

  async analyzeProduct(input: VisionInput): Promise<RecognitionResult> {
    if (!input.pixels) {
      throw new VisionProviderError('224×224 픽셀이 없습니다.', 'cnn');
    }
    const quality = analyzeQuality(input.features);
    const tensor = this.toTensor(input.pixels);

    const size = this.meta.img_size;
    const feeds = { image: new this.ort.Tensor('float32', tensor, [1, 3, size, size]) };
    const out = await (this.session as any).run(feeds);

    const itemProbs = softmax(out['item_logits'].data as Float32Array);
    const gradeProbs = softmax(out['grade_logits'].data as Float32Array);

    const ranked = itemProbs
      .map((p, i) => ({ name: this.meta.items[i] ?? '', p }))
      .sort((a, b) => b.p - a.p);
    const top = ranked[0];
    if (!top) throw new VisionProviderError('출력이 비어 있습니다.', 'cnn');

    // 학습 라벨은 한국어 품목명이므로 카탈로그의 code 로 되돌린다.
    const match = input.catalog.find((c) => c.name_ko === top.name);
    if (!match) {
      // 카탈로그에 없는 품목이면 판정하지 않는다(가격/SKU 가 없다).
      return {
        category: 'unknown',
        product: '',
        product_ko: '',
        variety_guess: null,
        quality_hint: quality.hint,
        confidence: 0,
        detected_issues: [...quality.issues, `학습 품목(${top.name})이 판매 카탈로그에 없습니다.`],
        description_basis: quality.basis,
        alternatives: [],
      };
    }

    const gradeIdx = gradeProbs.indexOf(Math.max(...gradeProbs));
    const modelGrade = this.meta.grades[gradeIdx];
    // 사진 자체에 문제가 있으면 모델 등급을 쓰지 않는다.
    // 품목별로 쓸 수 있는지는 중앙 정책(policy.ts)이 증거를 보고 판단한다.
    const hint =
      quality.issues.length === 0 && modelGrade
        ? (modelGrade as RecognitionResult['quality_hint'])
        : quality.hint;

    return {
      category: match.category,
      product: match.code,
      product_ko: match.name_ko,
      variety_guess: match.variety,
      quality_hint: hint,
      confidence: Number((top.p * (0.6 + 0.4 * quality.signalQuality)).toFixed(3)),
      detected_issues: quality.issues,
      description_basis: quality.basis,
      alternatives: ranked.slice(1, 3).flatMap((r) => {
        const alt = input.catalog.find((c) => c.name_ko === r.name);
        return alt
          ? [{ product: alt.code, product_ko: alt.name_ko, confidence: Number(r.p.toFixed(3)) }]
          : [];
      }),
    };
  }
}
