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

/**
 * 한 장을 중앙에서 여러 배율로 잘라 함께 판정한다.
 *
 * 학습 사진은 농산물이 화면을 꽉 채운 스튜디오 촬영이다(선형 차지비율 중앙값 98%).
 * 그런데 사람이 폰으로 찍으면 피사체가 화면의 절반도 안 되기 쉽고, 그 구간은
 * 모델이 학습에서 한 번도 본 적 없는 입력이라 정확도가 무너진다.
 * 실측: 화면을 꽉 채우면 5품목 전부 100%, 25%만 차지하면 감자는 top-3 에도 못 든다.
 *
 * 중앙을 잘라 확대하면 입력이 학습 분포 안으로 돌아온다. 배율을 여러 개 쓰는 이유는
 * 피사체가 얼마나 큰지 미리 알 수 없기 때문이고, 1.0 을 항상 포함하는 이유는
 * 피사체가 가장자리에 치우쳤을 때 크롭이 오히려 잘라내기 때문이다.
 *
 * 배율 수를 줄이면(예: 1.0+0.25) 중앙 조건은 같지만 치우친 구도에서 되레 나빠진다.
 */
const TTA_SCALES = [1, 0.7, 0.5, 0.35, 0.25] as const;

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

  /**
   * 224 원본에서 중앙 정사각(비율 scale)을 잘라 다시 224 로 확대한다(이중선형).
   *
   * 서버는 브라우저가 보낸 224×224 만 받는다. 원본 해상도가 오지 않으므로 화소를
   * 되살릴 수는 없지만, 잃어버린 것은 화소가 아니라 '피사체가 프레임을 채우는 정도'다.
   * 같은 56px 조각이라도 화면을 꽉 채우면 맞히고 구석에 있으면 틀린다는 것을 측정했다.
   */
  cropCenter(rgb: Uint8Array, scale: number): Uint8Array {
    const size = this.meta.img_size;
    if (scale >= 1) return rgb;
    const crop = Math.max(8, Math.round(size * scale));
    const off = Math.floor((size - crop) / 2);
    const out = new Uint8Array(size * size * 3);
    const ratio = size > 1 ? (crop - 1) / (size - 1) : 0;
    for (let y = 0; y < size; y += 1) {
      const sy = y * ratio;
      const y0 = Math.floor(sy);
      const y1 = Math.min(y0 + 1, crop - 1);
      const wy = sy - y0;
      for (let x = 0; x < size; x += 1) {
        const sx = x * ratio;
        const x0 = Math.floor(sx);
        const x1 = Math.min(x0 + 1, crop - 1);
        const wx = sx - x0;
        const i00 = ((off + y0) * size + off + x0) * 3;
        const i01 = ((off + y0) * size + off + x1) * 3;
        const i10 = ((off + y1) * size + off + x0) * 3;
        const i11 = ((off + y1) * size + off + x1) * 3;
        const o = (y * size + x) * 3;
        for (let c = 0; c < 3; c += 1) {
          const a = (rgb[i00 + c] ?? 0) * (1 - wx) + (rgb[i01 + c] ?? 0) * wx;
          const b = (rgb[i10 + c] ?? 0) * (1 - wx) + (rgb[i11 + c] ?? 0) * wx;
          out[o + c] = Math.round(a * (1 - wy) + b * wy);
        }
      }
    }
    return out;
  }

  async analyzeProduct(input: VisionInput): Promise<RecognitionResult> {
    if (!input.pixels) {
      throw new VisionProviderError('224×224 픽셀이 없습니다.', 'cnn');
    }
    const quality = analyzeQuality(input.features);
    const size = this.meta.img_size;
    const plane = size * size * 3;

    // 배율별 텐서를 한 배치로 묶어 한 번에 돌린다(전방 통과 5회, run 호출 1회).
    const batch = new Float32Array(TTA_SCALES.length * plane);
    TTA_SCALES.forEach((scale, i) => {
      batch.set(this.toTensor(this.cropCenter(input.pixels as Uint8Array, scale)), i * plane);
    });
    const feeds = {
      image: new this.ort.Tensor('float32', batch, [TTA_SCALES.length, 3, size, size]),
    };
    const out = await (this.session as any).run(feeds);

    const nItems = this.meta.items.length;
    const itemLogits = out['item_logits'].data as Float32Array;
    const perScale = TTA_SCALES.map((_, i) =>
      softmax(itemLogits.subarray(i * nItems, (i + 1) * nItems)),
    );

    // 배율끼리 최댓값으로 합친 뒤 합이 1이 되게 되돌린다.
    // 되돌리지 않으면 화면에 보이는 신뢰도가 부풀려진다 — '틀렸는데 자신 있는' 판정이 늘어난다.
    const fused = Array.from({ length: nItems }, (_, k) =>
      Math.max(...perScale.map((row) => row[k] ?? 0)),
    );
    const total = fused.reduce((a, b) => a + b, 0) || 1;
    const itemProbs = fused.map((p) => p / total);

    // 등급은 원본 배율만 쓴다. 잘라 확대한 그림에서 등급이 어떻게 되는지는 재본 적이 없다.
    const nGrades = this.meta.grades.length;
    const gradeProbs = softmax(
      (out['grade_logits'].data as Float32Array).subarray(0, nGrades),
    );

    // 배율마다 1순위가 갈리면 사진이 애매하다는 뜻이다. 화면 근거 문구로 알려준다.
    const picks = perScale.map((row) => row.indexOf(Math.max(...row)));
    const scalesAgree = picks.every((p) => p === picks[0]);

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
      description_basis: scalesAgree
        ? quality.basis
        : [...quality.basis, '가까이서 한 번 더 찍으면 더 정확합니다.'],
      alternatives: ranked.slice(1, 3).flatMap((r) => {
        const alt = input.catalog.find((c) => c.name_ko === r.name);
        return alt
          ? [{ product: alt.code, product_ko: alt.name_ko, confidence: Number(r.p.toFixed(3)) }]
          : [];
      }),
    };
  }
}
