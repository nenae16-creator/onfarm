import { findProductByCode, findSkuById } from '../../ai/sku-matcher.js';
import { db } from '../../db/index.js';
import {
  addListingInventory,
  createListing,
  getListingView,
  listByFarmer,
  setFarmerListingStatus,
} from '../../domain/listings.js';
import { listOrdersForFarmer, markFarmerOrderReady } from '../../domain/orders.js';
import { listSettlements, settlementSummary, DEMO_FEE_RATE } from '../../domain/settlements.js';
import { farmOf } from '../../domain/users.js';
import { todayKst } from '../../lib/datetime.js';
import { HttpError } from '../../lib/http.js';
import type { Router } from '../../lib/http.js';
import { requireRole } from '../../lib/session.js';
import { consumeAnalysis, findAnalysis, getAnalysis } from '../analysis-store.js';

interface CreateListingBody {
  analysisId?: string;
  skuId?: number;
  quantity?: number;
  harvestedOn?: string;
  /** 사용자가 폴백에서 품목을 바꿨을 때 */
  productCode?: string;
}

interface ManageListingBody {
  action?: string;
  quantity?: number;
}

export function registerFarmerRoutes(router: Router): void {
  /**
   * STEP 7 — [판매 등록].
   * 가격/품목/문안은 클라이언트 값을 믿지 않고 서버가 다시 조립한다.
   * (클라이언트는 skuId 와 수량만 고른다)
   */
  router.post('/api/farmer/listings', async (ctx) => {
    const user = requireRole(ctx.user, 'farmer');
    const farm = farmOf(db(), user.id);
    if (!farm) throw new HttpError(400, '농가 정보가 없습니다.', 'no_farm');

    const body = await ctx.body<CreateListingBody>();
    if (!body.analysisId) throw new HttpError(400, '분석 정보가 없습니다.', 'bad_request');

    const found = findAnalysis(body.analysisId, user.id);
    if (found?.consumed) {
      throw new HttpError(409, '이미 등록된 사진입니다. 새로 찍어주세요.', 'already_used');
    }
    const analysis = getAnalysis(body.analysisId, user.id);
    if (!analysis) {
      throw new HttpError(410, '분석 결과가 만료되었습니다. 사진을 다시 찍어주세요.', 'expired');
    }

    const quantity = Number(body.quantity);
    if (!Number.isInteger(quantity) || quantity < 1 || quantity > 999) {
      throw new HttpError(400, '수량은 1~999 사이여야 합니다.', 'bad_quantity');
    }

    const result = analysis.result;
    const productCode = body.productCode ?? result.recognition.product;
    const product = productCode ? findProductByCode(db(), productCode) : null;
    if (!product) throw new HttpError(400, '품목을 확인할 수 없습니다.', 'bad_product');

    const sku = body.skuId ? findSkuById(db(), Number(body.skuId)) : null;
    const chosenSku = sku ?? (result.selectedSku?.product_code === product.code ? result.selectedSku : null);
    if (!chosenSku) throw new HttpError(400, '판매 단위를 확인할 수 없습니다.', 'bad_sku');
    if (chosenSku.product_id !== product.id) {
      throw new HttpError(400, '품목과 판매 단위가 맞지 않습니다.', 'sku_mismatch');
    }

    const today = todayKst();
    const harvestedOn = body.harvestedOn ?? today;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(harvestedOn) || harvestedOn > today) {
      throw new HttpError(400, '수확일이 올바르지 않습니다.', 'bad_date');
    }

    // 사용자가 품목을 바꿨으면 이전 품목의 품종 추정을 그대로 끌고 가면 안 된다.
    // (배 분석 결과로 사과를 올렸는데 제목이 '신고배'가 되는 사고)
    // 모델의 원래 1순위와 실제 등록 품목을 비교한다.
    // (사용자가 후보를 고르면 recognition 은 이미 그 값으로 바뀌어 있다)
    const modelTop = result.ai.modelTop ?? result.recognition.product;
    // 후보 선택 재분석으로 source 가 'manual' 로 덮여도 최초 인식기의 이름을 남긴다.
    const modelSource = result.ai.modelSource ?? result.ai.source;
    const userOverrode = Boolean(modelTop) && modelTop !== product.code;
    const productChanged = result.recognition.product !== product.code;
    const recognition = productChanged
      ? {
          ...result.recognition,
          product: product.code,
          product_ko: product.name_ko,
          variety_guess: product.variety,
          category: product.category,
          // 품목 자체가 틀렸다면 그 사진 근거로 매긴 등급도 근거를 잃는다.
          quality_hint: '확인필요' as const,
          description_basis: [],
          alternatives: [],
        }
      : result.recognition;

    // 문안은 분석 시점 것을 쓰되, 품목·SKU·수확일 중 하나라도 바뀌면 다시 만든다.
    let title = result.draft?.title ?? '';
    let description = result.draft?.description ?? '';
    const skuChanged = result.selectedSku?.id !== chosenSku.id;
    const dateChanged = harvestedOn !== today;
    if (!title || skuChanged || productChanged || dateChanged) {
      const { writeProduct } = await import('../../ai/product-writer.js');
      const redraft = writeProduct({
        product,
        sku: chosenSku,
        farm,
        farmerName: user.name,
        recognition,
        harvestedOn,
        today,
        aiSourceLabel: productChanged ? '농민이 직접 선택' : result.ai.label,
      });
      title = redraft.title;
      description = redraft.description;
    }

    // 여기서 분석을 소진한다. 재시도·동시요청으로 같은 사진이 두 매물이 되는 것을 막는다.
    if (!consumeAnalysis(body.analysisId, user.id)) {
      throw new HttpError(409, '이미 등록된 사진입니다. 새로 찍어주세요.', 'already_used');
    }

    const listing = createListing(db(), {
      farmerId: user.id,
      farmId: farm.id,
      productId: product.id,
      skuId: chosenSku.id,
      title,
      description,
      imagePath: analysis.imagePath,
      quantity,
      unitPrice: chosenSku.price,
      harvestedOn,
      aiAnalysis: {
        recognition,
        decision: result.decision,
        source: result.ai.source,
        offline: result.ai.offline,
        // '맞아요'를 누른 정상 흐름까지 수동 개입으로 기록하면 감사 로그가 거짓이 된다.
        // 실제로 품목이 바뀐 경우만 override 로 남긴다.
        modelTop: modelTop || null,
        modelSource,
        userOverride: userOverrode ? { from: modelTop, to: product.code } : null,
      },
      // 확신도는 '모델이 이 품목에 대해 냈던 값'만 기록한다.
      // 사용자가 고른 뒤의 confidence(=1)를 쓰면 화면에 'AI 신뢰도 100%'로 둔갑한다.
      // 사용자가 모델과 다른 품목을 골랐다면 그 확신도는 이 품목 것이 아니므로 남기지 않는다.
      aiConfidence: userOverrode ? null : (result.ai.modelTopConfidence ?? null),
      aiSource: userOverrode ? `${modelSource}+manual` : modelSource,
      qualityHint: recognition.quality_hint,
    });

    ctx.json({ listing }, 201);
  });

  router.get('/api/farmer/listings', (ctx) => {
    const user = requireRole(ctx.user, 'farmer');
    ctx.json({ listings: listByFarmer(db(), user.id) });
  });

  router.post('/api/farmer/listings/:id/manage', async (ctx) => {
    const user = requireRole(ctx.user, 'farmer');
    const id = Number(ctx.params['id']);
    if (!Number.isInteger(id)) throw new HttpError(400, '잘못된 상품입니다.', 'bad_request');

    const listing = getListingView(db(), id);
    if (!listing || listing.farmer_id !== user.id) {
      throw new HttpError(404, '상품을 찾을 수 없습니다.', 'not_found');
    }
    if (listing.inspection_status !== 'ai_checked' || listing.has_rejection) {
      throw new HttpError(409, '주문 또는 거점 검수가 시작된 상품은 변경할 수 없습니다.', 'locked_listing');
    }

    const body = await ctx.body<ManageListingBody>();
    let changed = false;
    if (body.action === 'pause') {
      changed = listing.status === 'active' && setFarmerListingStatus(db(), id, user.id, 'closed');
    } else if (body.action === 'resume') {
      changed = listing.status === 'closed' && setFarmerListingStatus(db(), id, user.id, 'active');
    } else if (body.action === 'add_stock') {
      const quantity = Number(body.quantity);
      if (!Number.isInteger(quantity) || quantity < 1 || listing.quantity + quantity > 999) {
        throw new HttpError(400, '추가 수량은 전체 999개 이하여야 합니다.', 'bad_quantity');
      }
      changed = addListingInventory(db(), id, user.id, quantity);
    } else {
      throw new HttpError(400, '지원하지 않는 관리 작업입니다.', 'bad_action');
    }

    if (!changed) throw new HttpError(409, '현재 상태에서는 처리할 수 없습니다.', 'bad_transition');
    ctx.json({ listing: getListingView(db(), id) });
  });

  router.get('/api/farmer/orders', (ctx) => {
    const user = requireRole(ctx.user, 'farmer');
    ctx.json({ orders: listOrdersForFarmer(db(), user.id) });
  });

  router.post('/api/farmer/order-items/:id/ready', (ctx) => {
    const user = requireRole(ctx.user, 'farmer');
    const id = Number(ctx.params['id']);
    if (!Number.isInteger(id)) throw new HttpError(400, '잘못된 주문 상품입니다.', 'bad_request');
    if (!markFarmerOrderReady(db(), id, user.id)) {
      throw new HttpError(409, '이미 처리했거나 준비할 수 없는 주문입니다.', 'bad_transition');
    }
    ctx.json({ ok: true, status: 'ready_for_hub' });
  });

  router.get('/api/farmer/settlements', (ctx) => {
    const user = requireRole(ctx.user, 'farmer');
    ctx.json({
      summary: settlementSummary(db(), user.id),
      rows: listSettlements(db(), user.id),
      feeRate: DEMO_FEE_RATE,
    });
  });
}
