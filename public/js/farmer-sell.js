// 농민 판매 등록 흐름 — 사진 → AI 확인 → 판매단위 → 수량 → 등록
import { $, api, money, requireRole, mountModeBanner, toast, el } from '/js/api.js';
import { prepareImage } from '/js/features.js';
import { canListen, listenQuantity, repeatLast, speak } from '/js/speak.js';
import { speakPrice, speakWeight, sinoNumber, nativeCount } from '/js/shared/korean.js';

const state = {
  session: null,
  catalog: [],
  analysisId: null,
  imagePath: null,
  analysis: null,
  product: null,
  skus: [],
  sku: null,
  quantity: 5,
  step: 'stepPhoto',
  /** 사용자가 폴백 화면에서 품목을 직접 골랐는가 (감사 기록용) */
  userPicked: false,
};

const STEPS = ['stepPhoto', 'stepLoading', 'stepResult', 'stepManual', 'stepSku', 'stepConfirm', 'stepDone'];
const STEP_PROGRESS = {
  stepPhoto: { step: '1 / 4', label: '사진 올리기', value: 25 },
  stepLoading: { step: '1 / 4', label: '사진 확인 중', value: 25 },
  stepResult: { step: '2 / 4', label: '품목 확인', value: 50 },
  stepManual: { step: '2 / 4', label: '품목 직접 선택', value: 50 },
  stepSku: { step: '3 / 4', label: '단위와 수량', value: 75 },
  stepConfirm: { step: '4 / 4', label: '마지막 확인', value: 100 },
  stepDone: { step: '완료', label: '판매 등록 완료', value: 100 },
};

function show(step) {
  state.step = step;
  for (const id of STEPS) {
    const node = document.getElementById(id);
    if (node) node.hidden = id !== step;
  }
  $('#stepDone').classList.toggle('is-active', step === 'stepDone');
  const progress = STEP_PROGRESS[step];
  const progressNode = $('#sellProgress');
  progressNode.hidden = step === 'stepDone';
  progressNode.className = `sell-progress progress-${progress.value}`;
  progressNode.setAttribute('aria-valuenow', String(progress.value));
  $('#progressStep').textContent = progress.step;
  $('#progressLabel').textContent = progress.label;
  window.scrollTo({ top: 0, behavior: 'auto' });
}

function unitWord(label) {
  if (!label) return '개';
  if (label.includes('망')) return '망';
  if (label.includes('봉')) return '봉';
  if (label.includes('상자')) return '상자';
  return '개';
}

function nameOf(code) {
  return state.catalog.find((p) => p.code === code)?.name ?? code;
}

function productMark(name) {
  return name?.trim().slice(0, 1) || '품';
}

/* ───────── 시연용 합성 이미지 ─────────
   실제 사진이 없을 때도 파이프라인 전체(특징 추출 → 판정 → SKU)를 그대로 태우기 위한 보조 수단.
   합성 이미지임을 화면에 밝힌다. */
function sampleImageFile() {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');
  // 배경은 실제 촬영 환경(회색 상자/바닥)처럼 채도가 낮게 둔다.
  ctx.fillStyle = '#b9b5ad';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < 26; i += 1) {
    const x = 70 + Math.random() * 500;
    const y = 70 + Math.random() * 340;
    const r = 46 + Math.random() * 16;
    // 신고배 표피색(#C7BA7B 근처)에 맞춘다 — 채도가 높으면 실제로 양파에 가까워진다.
    const hue = 48 + Math.random() * 6;
    const grad = ctx.createRadialGradient(x - r / 3, y - r / 3, r / 6, x, y, r);
    grad.addColorStop(0, `hsl(${hue}, 40%, 70%)`);
    grad.addColorStop(1, `hsl(${hue - 4}, 36%, 52%)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(x, y, r, r * 0.92, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(new File([blob], 'sample.jpg', { type: 'image/jpeg' })), 'image/jpeg', 0.9));
}

/* ───────── 파이프라인 호출 ───────── */
async function analyze(payload) {
  show('stepLoading');
  try {
    const result = await api('/api/ai/analyze', { body: payload });
    state.analysisId = result.analysisId;
    state.imagePath = result.imagePath;
    state.analysis = result;
    state.skus = result.skus ?? [];
    state.sku = result.selectedSku ?? state.skus[0] ?? null;
    state.product = result.recognition.product;
    route(result);
  } catch (err) {
    if (err.status === 410) {
      toast('사진 정보가 만료됐습니다. 다시 찍어 주세요.');
      show('stepPhoto');
      return;
    }
    toast(err.message ?? '분석에 실패했습니다.');
    renderManual();
    show('stepManual');
  }
}

function route(result) {
  // 신뢰도와 무관하게 후보를 번호로 보여준다.
  // '맞아요' 한 번보다 고르는 편이 오등록이 적고, AI 가 단정하지 않는다는 원칙과도 맞는다.
  const candidates = result.candidates ?? [];
  if (candidates.length === 0) {
    renderManual();
    show('stepManual');
    speak('사진을 자동으로 확인하지 못했습니다. 무엇을 파실까요?');
    return;
  }
  renderResult(result);
  show('stepResult');
  speakCandidates(candidates);
}

/**
 * 후보를 번호와 함께 읽어준다. 화면을 못 보는 상황에서도 고를 수 있게.
 * 숫자를 그대로 넣으면 TTS 가 "1번"을 '한 번(once)'으로 읽어 횟수처럼 들린다.
 * 그래서 번호는 한자어 수사로 풀어 "일번, 이번, 삼번" 이라고 말하게 한다.
 */
function speakCandidates(candidates) {
  const list = candidates.map((c, i) => `${sinoNumber(i + 1)}번 ${c.name}`).join(', ');
  speak(`사진을 확인했습니다. ${list} 중에 어느 것인가요?`, { force: true });
}

/* ───────── 각 화면 렌더 ───────── */
function renderResult(result) {
  const r = result.recognition;
  const candidates = result.candidates ?? [];
  $('#resultImage').src = result.imagePath ?? '/img/sample/placeholder.svg';
  $('#resultSub').textContent = candidates.length > 1
    ? '가장 비슷한 것부터 보여드립니다. 번호를 눌러 주세요.'
    : '사진과 같으면 눌러 주세요.';

  const grid = $('#candidateGrid');
  grid.replaceChildren();
  candidates.forEach((c, i) => {
    const first = i === 0;
    grid.append(
      el(
        'button',
        {
          type: 'button',
          class: `choice-button${first ? ' recommended' : ''}`,
          'aria-label': `${i + 1}번 ${c.name}${first ? ', 가장 비슷한 품목' : ''}`,
          onclick: () => pickProduct(c.code),
        },
        [
          el('span', { class: 'choice-number', text: `${i + 1}` }),
          el('span', { class: 'produce-mark', text: productMark(c.name), 'aria-hidden': 'true' }),
          el('span', { class: 'choice-copy' }, [
            el('strong', { text: c.name }),
            first
              ? el('small', { text: '사진과 가장 비슷합니다' })
              : null,
          ]),
        ],
      ),
    );
  });

  $('#badgeSource').textContent = result.ai.offline
    ? `${result.ai.label} · 사진 외부 전송 없음`
    : result.ai.label;
  $('#badgeQuality').textContent = `AI 품질 참고: ${r.quality_hint}`;
  $('#badgeConfidence').textContent = r.confidence
    ? `1순위 신뢰도 ${Math.round(r.confidence * 100)}%`
    : '';
  const basis = [...r.description_basis];
  if (r.detected_issues.length) basis.push(`확인 어려웠던 점: ${r.detected_issues.join(', ')}`);
  if (result.ai.degraded) basis.push(`※ ${result.ai.degraded}`);
  $('#resultBasis').textContent = basis.join(' / ');
}

function renderManual() {
  const grid = $('#manualGrid');
  grid.replaceChildren();
  for (const p of state.catalog) {
    grid.append(
      el('button', { type: 'button', class: 'choice-button manual-choice', onclick: () => pickProduct(p.code) }, [
        el('span', { class: 'produce-mark', text: productMark(p.name), 'aria-hidden': 'true' }),
        el('strong', { text: p.name }),
      ]),
    );
  }
}

async function pickProduct(code) {
  // 분석 요청 자체가 실패해 analysisId 가 없을 수 있다.
  // 그때도 사진은 손에 있으므로 사진과 함께 다시 보내 폴백이 막다른 길이 되지 않게 한다.
  if (!state.analysisId && !state.pendingImage) {
    toast('사진을 먼저 찍어 주세요.');
    show('stepPhoto');
    return;
  }
  await analyzeForced(code);
}

async function analyzeForced(code) {
  show('stepLoading');
  try {
    const payload = state.analysisId
      ? { analysisId: state.analysisId, productCode: code }
      : { image: state.pendingImage.dataUrl, features: state.pendingImage.features, pixels: state.pendingImage.pixels, productCode: code };
    const result = await api('/api/ai/analyze', { body: payload });
    if (result.analysisId) state.analysisId = result.analysisId;
    if (result.imagePath) state.imagePath = result.imagePath;
    state.analysis = result;
    state.skus = result.skus ?? [];
    state.sku = result.selectedSku ?? state.skus[0] ?? null;
    state.product = code;
    state.userPicked = true;
    if (!state.sku) {
      toast('이 품목은 아직 판매 단위가 등록되지 않았습니다.');
      renderManual();
      show('stepManual');
      return;
    }
    renderSku();
    show('stepSku');
    speakSku();
  } catch (err) {
    toast(err.message ?? '처리에 실패했습니다.');
    show('stepPhoto');
  }
}

function renderSku() {
  const name = nameOf(state.product);
  $('#skuTitle').textContent = `${name}로 확인했습니다.`;
  const box = $('#skuOptions');
  box.replaceChildren();
  if (state.skus.length > 1) {
    for (const sku of state.skus) {
      const selected = state.sku?.id === sku.id;
      box.append(
        el(
          'button',
          {
            type: 'button',
            class: `choice-button sku-choice${selected ? ' selected' : ''}`,
            'aria-pressed': String(selected),
            onclick: () => {
              state.sku = sku;
              renderSku();
              speakSku();
            },
          },
          [
            el('span', { class: 'choice-check', 'aria-hidden': 'true' }),
            el('span', { class: 'choice-copy' }, [
              el('strong', { text: sku.label }),
              el('small', { text: money(sku.price) }),
            ]),
          ],
        ),
      );
    }
  }
  $('#skuUnitLabel').textContent = state.sku ? state.sku.label : '';
  $('#skuPrice').textContent = state.sku ? money(state.sku.price) : '-';
  $('#qtyUnitWord').textContent = unitWord(state.sku?.label);
  $('#qtyValue').textContent = String(state.quantity);
  renderQuantityVisual();
}

function renderQuantityVisual() {
  const word = unitWord(state.sku?.label);
  const kind = word === '망' ? 'net' : word === '봉' ? 'bag' : 'box';
  const visibleCount = Math.min(state.quantity, 3);
  const pack = $('#quantityPack');
  pack.replaceChildren();

  for (let i = 0; i < visibleCount; i += 1) {
    pack.append(el('span', { class: `quantity-package ${kind}` }));
  }
  if (state.quantity > visibleCount) {
    pack.append(el('span', { class: 'quantity-more', text: `+${state.quantity - visibleCount}` }));
  }

  $('#qtyQuestion').textContent = `몇 ${word} 파실까요?`;
  $('#quantityCaption').textContent = `${state.quantity}${word} 선택`;
  $('#quantityVisual').setAttribute('aria-label', `선택한 수량: ${state.quantity}${word}`);
}

function speakSku() {
  if (!state.sku) return;
  const word = unitWord(state.sku.label);
  speak(
    `${nameOf(state.product)}로 확인했습니다. ${speakWeight(state.sku.weight, state.sku.unit)} 한 ${word}에 ${speakPrice(state.sku.price)}입니다. 몇 ${word} 판매하시겠습니까?`,
    { force: true },
  );
}

function setQuantity(next) {
  state.quantity = Math.max(1, Math.min(999, next));
  $('#qtyValue').textContent = String(state.quantity);
  renderQuantityVisual();
}

function renderConfirm() {
  const word = unitWord(state.sku?.label);
  $('#confirmImage').src = state.imagePath ?? '/img/sample/placeholder.svg';
  $('#cfProduct').textContent = nameOf(state.product);
  $('#cfSku').textContent = state.sku?.label ?? '-';
  $('#cfUnitPriceLabel').textContent = `한 ${word} 값`;
  $('#cfPrice').textContent = money(state.sku?.price ?? 0);
  $('#cfQty').textContent = `${state.quantity}${word}`;
  $('#cfTotal').textContent = money((state.sku?.price ?? 0) * state.quantity);
  $('#cfTitle').textContent = state.analysis?.draft?.title ?? '';
  $('#cfDescription').textContent = state.analysis?.draft?.description ?? '';
}

/* ───────── 이벤트 배선 ───────── */
async function onFile(file) {
  if (!file) return;
  show('stepLoading');
  try {
    const prepared = await prepareImage(file);
    // 분석이 실패해도 폴백 화면에서 다시 쓸 수 있도록 들고 있는다.
    state.pendingImage = prepared;
    state.analysisId = null;
    state.userPicked = false;
    $('#loadingPreview').src = prepared.dataUrl;
    $('#loadingPreview').hidden = false;
    await analyze({ image: prepared.dataUrl, features: prepared.features, pixels: prepared.pixels });
  } catch (err) {
    toast(err.message ?? '사진을 읽지 못했습니다.');
    show('stepPhoto');
  }
}

$('#photoInput').addEventListener('change', (e) => onFile(e.target.files?.[0]));
$('#sampleBtn').addEventListener('click', async () => {
  toast('시연용 합성 이미지로 진행합니다(실제 사진 아님).');
  onFile(await sampleImageFile());
});

$('#speakResult').addEventListener('click', () => repeatLast());
$('#speakSku').addEventListener('click', () => speakSku());

$('#resultNo').addEventListener('click', () => {
  renderManual();
  show('stepManual');
  speak('무엇을 파실까요?');
});
$('#resultRetake').addEventListener('click', () => show('stepPhoto'));
$('#manualRetake').addEventListener('click', () => show('stepPhoto'));

$('#qtyMinus').addEventListener('click', () => setQuantity(state.quantity - 1));
$('#qtyPlus').addEventListener('click', () => setQuantity(state.quantity + 1));

$('#voiceQty').addEventListener('click', async () => {
  const btn = $('#voiceQty');
  const label = $('#voiceQtyLabel');
  btn.disabled = true;
  btn.classList.add('is-listening');
  label.textContent = '듣고 있습니다';
  try {
    const { quantity, transcript } = await listenQuantity();
    if (quantity) {
      setQuantity(quantity);
      speak(`${nativeCount(quantity)} ${unitWord(state.sku?.label)}로 하겠습니다.`, { force: true });
    } else {
      toast(transcript ? `"${transcript}" 를 알아듣지 못했습니다.` : '잘 들리지 않았습니다.');
    }
  } catch {
    toast('음성을 듣지 못했습니다. 버튼으로 수량을 골라 주세요.');
  } finally {
    btn.disabled = false;
    btn.classList.remove('is-listening');
    label.textContent = '말로 수량 말하기';
  }
});

$('#skuNext').addEventListener('click', () => {
  renderConfirm();
  show('stepConfirm');
  speak(`${nativeCount(state.quantity)} ${unitWord(state.sku?.label)}, 모두 ${speakPrice((state.sku?.price ?? 0) * state.quantity)}입니다. 이대로 올릴까요?`, { force: true });
});
$('#confirmBack').addEventListener('click', () => {
  renderSku();
  show('stepSku');
});

$('#submitBtn').addEventListener('click', async () => {
  const btn = $('#submitBtn');
  btn.disabled = true;
  btn.textContent = '등록하는 중…';
  try {
    const res = await api('/api/farmer/listings', {
      body: {
        analysisId: state.analysisId,
        skuId: state.sku?.id,
        quantity: state.quantity,
        // 서버가 인식한 품목을 그대로 쓰는 정상 흐름에서는 보내지 않는다.
        // (항상 보내면 서버 감사 기록이 전부 '수동 선택'으로 남는다)
        ...(state.userPicked ? { productCode: state.product } : {}),
      },
    });
    const word = unitWord(state.sku?.label);
    $('#doneSummary').textContent = `${nameOf(state.product)} ${state.sku?.label} ${state.quantity}${word}`;
    if (state.session?.farm) {
      $('#doneHub').textContent = '주문이 모이면 지역 거점에 가져다 놓으시면 됩니다.';
    }
    state.lastListingId = res.listing.id;
    show('stepDone');
    speak('판매가 시작됐습니다.', { force: true });
  } catch (err) {
    toast(err.message ?? '등록에 실패했습니다.');
  } finally {
    btn.disabled = false;
    btn.textContent = '판매 등록';
  }
});

$('#doneStore').addEventListener('click', () => {
  location.href = state.lastListingId ? `/store/product?id=${state.lastListingId}` : '/';
});
$('#doneAgain').addEventListener('click', () => {
  state.analysisId = null;
  state.imagePath = null;
  state.quantity = 5;
  $('#photoInput').value = '';
  show('stepPhoto');
});
$('#doneHome').addEventListener('click', () => (location.href = '/farmer'));

$('#backBtn').addEventListener('click', () => {
  const backMap = {
    stepResult: 'stepPhoto',
    stepManual: 'stepResult',
    stepSku: 'stepResult',
    stepConfirm: 'stepSku',
  };
  const target = backMap[state.step];
  if (target) show(target);
  else location.href = '/farmer';
});

/* ───────── 부팅 ───────── */
const cfg = await mountModeBanner('#modeBanner');
state.catalog = cfg?.products ?? [];
state.session = await requireRole('farmer');
if (state.session) {
  $('#whoSub').textContent = state.session.farm?.farm_name ?? '';
}
if (canListen()) $('#voiceQty').hidden = false;
renderManual();
show('stepPhoto');
