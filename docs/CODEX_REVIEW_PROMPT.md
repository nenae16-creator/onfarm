# Codex 교차검증 프롬프트 (2차 — CV 학습·접합 범위)

1차 검토(2026-08-09)는 MVP 코어를 봤고 22건 중 17건을 반영했다.
이번 2차는 **그 이후에 추가된 CV 데이터 수집 → 학습 → 서비스 접합 → MVP 흐름 변경**이 대상이다.

## 실행 방법

```bash
codex exec --sandbox read-only -C "C:\Users\Admin\Documents\onfarm" -o "codex_review2.md" "$(cat docs/CODEX_REVIEW_PROMPT.md)"
```

읽기 전용이라 저장소를 건드리지 않는다. 30분 내외 걸린다.

---

## 여기부터가 프롬프트 본문

당신은 ON-FARM 프로젝트의 적대적 코드 검토자다. 한국어로 답하라.

### 프로젝트 개요

고령 농어민이 **사진 한 장으로 농산물을 판매 등록**하는 MVP다.
Node 24 + 내장 `node:sqlite`, 런타임 의존성 0(단 `onnxruntime-node`는 선택적 의존성),
TypeScript strict, 테스트 124개 통과. 천안시 공모전 제출용.

무시할 경로: `node_modules/`, `dist/`, `data/`, `models/*.onnx*`, `models/*.pt`, `package-lock.json`

### 이번에 검토할 범위 (1차 리뷰 이후 추가분)

| 영역 | 파일 |
|---|---|
| 데이터 수집 | `tools/aihub_ingest.py`, `tools/pack_for_colab.py`, `tools/aihub_manifest.py` |
| 학습 | `notebooks/onfarm_train_colab.py` (=.ipynb 원본) |
| 모델 접합 | `src/ai/providers/cnn.ts`, `src/ai/providers/index.ts`, `src/config.ts`, `src/server/main.ts` |
| 입력 경로 | `public/js/features.js`, `src/server/routes/ai.ts`, `src/server/analysis-store.ts`, `src/ai/pipeline.ts` |
| MVP 흐름 변경 | `public/farmer/sell.html`, `public/js/farmer-sell.js` |
| 음성 | `src/lib/korean.ts`, `src/tests/korean-speech.test.ts` |
| 검증 도구 | `tools/verify_model.mjs` |
| 문서 주장 | `docs/aihub-dataset.md`, `DEVELOPMENT_STATUS.md` |

### 배경 사실 (검토 시 전제로 삼되, 코드와 어긋나면 그것이 결함이다)

- 학습 데이터: AI 허브 「농산물 품질(QC) 이미지」(datasetkey 149). 제품 1개를 카메라 5대 ×
  4각도 × 상/하부로 찍어 **개체당 약 40장**의 근접중복이 있다. 그래서 이미지 단위 무작위 분할은
  누출이며, 공식 Train/Validation 을 쓰고 양쪽에 걸친 개체 1건(`602132129000`)만 제외했다.
- 수집 결과: 122,425장 / 5품목(감귤·감자·배·사과·양파) / train 개체 1,056 · valid 개체 201.
- 학습 결과(`models/metadata.json`): 개체 단위 품목 **1.000**, 등급 **0.705**,
  중량만 보는 기준선 0.58, 평균 확신도 0.963(오답 시 0.666).
- 서버는 JPEG 디코더가 없다. 브라우저가 224×224 RGB 픽셀을 base64 로 함께 보낸다.
- 등급은 "AI 참고 판정"이고 확정은 거점 실물 검수가 한다는 것이 제품의 핵심 주장이다.

### 검토 관점 (각각 실제 코드 근거를 대라)

**1. 실행 가능한 버그**
- 수집 도구의 이어받기(`.ingest_state.json`)와 매니페스트 append 가 **중복 행·유실**을 만들 수 있는가?
  같은 파일키를 두 번 처리하거나, 중단 시점에 따라 매니페스트와 실제 파일이 어긋나는 경로가 있는가?
- `fetch()` 의 재시도 루프에서 부분 다운로드 파일이 남아 다음 시도를 오염시키는 경우가 있는가?
- part 병합(스트리밍)이 순서를 잘못 잡는 입력이 있는가? (`.part10` vs `.part2` 정렬)
- CNN provider 의 세션 초기화가 동시 요청에서 경쟁하는가? (`this.session` lazy init)

**2. 모델 접합의 안전성 — 가장 중요**
- `capConfidence()` 와 `gradeIsUsable()` 이 **우회 가능한 경로**가 있는가?
  provider 를 바꾸거나 metadata 를 조작하면 확신도 상한이 무력화되는가?
- `pipeline.ts` 가 provider 가 준 confidence 를 재검증 없이 `decideFlow()` 에 넘기는가?
  (같은 프로젝트 계열에서 이 배선이 끊겨 안전장치가 통째로 우회된 전례가 있다)
- 학습 모델은 5품목만 아는데 카탈로그는 8품목이다. 모델이 모르는 품목(고구마·건고추·복숭아)을
  올리려는 농민이 **막다른 길**에 빠지는 경로가 있는가?
- `candidates` 에 SKU 없는 품목이 섞여 눌러도 진행이 안 되는 경우가 있는가?

**3. 클라이언트 신뢰 경계**
- 픽셀(`pixels`)과 특징(`features`)은 클라이언트가 만든다. 조작된 값으로
  **가격·품목·등급을 원하는 대로 유도**할 수 있는가? 서버가 최종적으로 무엇을 신뢰하는지 추적하라.
- `sanitizePixels()` 의 길이 검사만으로 충분한가? 메모리·CPU 관점의 남용 경로는?

**4. 학습 노트북의 방법론 결함**
- 개체 단위 정확도 계산(`by_object`)이 **다수결 동점**이나 개체가 한 장뿐일 때 올바른가?
- 중량 단독 기준선 탐색이 valid 개체 전체를 훑는데, 이는 **테스트셋에 과적합된 상한**이다.
  이 값을 모델과 비교하는 것이 공정한가? 공정하지 않다면 어떻게 고쳐야 하는가?
- `ColorJitter` 등 증강이 스튜디오→폰 사진 도메인 격차를 메우기에 충분하다고 볼 근거가 있는가?
- 품목 정확도 1.000 은 정상적인 성능인가, 아니면 **누출·단축학습(shortcut)** 의 징후인가?
  코드와 데이터 구성에서 그 원인을 찾아라. (배경·조명이 품목별로 다를 가능성 포함)

**5. 클레임 안전 — 심사 제출용이라 특히 중요**
- 화면·문서·주석에서 "AI가 등급을 확정한다 / 가격을 정한다 / 안전성을 검사한다"로
  읽힐 수 있는 문구를 모두 찾아라.
- `DEVELOPMENT_STATUS.md`·`docs/aihub-dataset.md` 의 수치 주장이 코드·데이터와 일치하는가?
  과장되거나 조건이 빠진 서술이 있는가?
- "품목 인식 100%"를 대외에 쓰면 안 되는 이유를 코드 근거로 보강하거나, 반대로 반박하라.

**6. 테스트 사각지대**
- 124개 테스트가 **실제로 무엇을 못 잡는지** 지적하라. 특히 CNN 경로는 모델 없이 도는
  테스트만 있다. 어떤 테스트를 추가해야 하는가?
- 수집 도구(Python)는 테스트가 0건이다. 어디부터 덮어야 하는가?

### 출력 형식

지적마다 반드시:
1. `파일경로:줄번호`
2. **재현 시나리오** (구체적 입력 → 잘못된 결과)
3. 확신도: 높음 / 중간 / 낮음
4. 제안 수정 방향 (1~2문장)

마지막에 **"확인된 비결함"** 절을 두어, 의심했지만 실제로는 문제가 없었던 것을 적어라.
근거 없는 일반론·스타일 지적은 쓰지 마라. 코드를 직접 읽고 확인한 것만 쓴다.
