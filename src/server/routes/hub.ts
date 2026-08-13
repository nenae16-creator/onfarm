import { db } from '../../db/index.js';
import { listInspections, recordInspection } from '../../domain/inspections.js';
import { belongsToHub, getListingView } from '../../domain/listings.js';
import {
  advanceHubOrderItem,
  getOrderItemForHub,
  listOrderItemsForHub,
  markHubOrderReceived,
} from '../../domain/orders.js';
import type { User } from '../../domain/types.js';
import { HttpError } from '../../lib/http.js';
import type { Ctx, Router } from '../../lib/http.js';
import { requireRole } from '../../lib/session.js';

/** 실물 검수로 확정할 수 있는 등급. 임의 문자열이 소비자 화면에 걸리면 안 된다. */
const GRADES = ['특', '상', '보통'] as const;

/** 담당자는 자기 거점, 관리자는 전체. */
function hubScope(user: User): number | null {
  if (user.role === 'admin') return null;
  if (user.hub_id === null) {
    throw new HttpError(403, '소속 거점이 지정되지 않은 계정입니다.', 'no_hub');
  }
  return user.hub_id;
}

function assertInScope(ctx: Ctx, user: User, listingId: number): void {
  const scope = hubScope(user);
  if (scope === null) return; // 관리자
  if (!belongsToHub(db(), listingId, scope)) {
    throw new HttpError(403, '다른 거점의 상품입니다.', 'other_hub');
  }
}

export function registerHubRoutes(router: Router): void {
  router.get('/api/hub/dashboard', (ctx) => {
    const user = requireRole(ctx.user, 'hub_operator');
    const scope = hubScope(user);
    const items = listOrderItemsForHub(db(), scope);
    const count = (status: string): number => items.filter((item) => item.fulfillment_status === status).length;
    ctx.json({
      hubId: scope,
      counters: {
        incoming: count('ready_for_hub'),
        needInspection: count('hub_received'),
        readyToShip: count('hub_passed'),
        shipping: count('ready_to_ship'),
        delivered: count('delivered'),
      },
      items,
      grades: GRADES,
    });
  });

  router.get('/api/hub/listings/:id/inspections', (ctx) => {
    const user = requireRole(ctx.user, 'hub_operator');
    const id = Number(ctx.params['id']);
    if (!Number.isInteger(id)) throw new HttpError(400, '잘못된 상품입니다.', 'bad_request');
    assertInScope(ctx, user, id);
    ctx.json({ inspections: listInspections(db(), id) });
  });

  /**
   * 실물 검수 결과 입력 — 여기서 확정된 등급만 '확정'이다.
   * AI 참고값을 그대로 승격시키지 않도록, 통과/조정 모두 담당자가 등급을 명시해야 한다.
   */
  router.post('/api/hub/inspections', async (ctx) => {
    const user = requireRole(ctx.user, 'hub_operator');
    const body = await ctx.body<{
      orderItemId?: number;
      result?: string;
      gradedQuality?: string;
      note?: string;
    }>();

    const orderItemId = Number(body.orderItemId);
    if (!Number.isInteger(orderItemId)) throw new HttpError(400, '주문 상품을 선택해주세요.', 'bad_request');
    const item = getOrderItemForHub(db(), orderItemId, hubScope(user));
    if (!item) throw new HttpError(404, '주문 상품을 찾을 수 없습니다.', 'not_found');
    const listingId = item.listing_id;

    const listing = getListingView(db(), listingId);
    if (!listing) throw new HttpError(404, '상품을 찾을 수 없습니다.', 'not_found');
    if (listing.status === 'closed') {
      throw new HttpError(409, '이미 반려된 상품입니다.', 'closed');
    }
    if (item.fulfillment_status !== 'hub_received') {
      throw new HttpError(409, '입고 확인된 주문 상품만 검수할 수 있습니다.', 'not_received');
    }
    if (listing.inspection_status !== 'ai_checked' && listing.inspection_status !== 'hub_pending') {
      throw new HttpError(409, '이미 검수가 끝난 상품입니다.', 'inspection_finished');
    }

    const result = body.result;
    if (result !== 'pass' && result !== 'downgrade' && result !== 'reject') {
      throw new HttpError(400, '검수 결과 값이 올바르지 않습니다.', 'bad_request');
    }

    let gradedQuality: string | null = null;
    if (result !== 'reject') {
      const grade = typeof body.gradedQuality === 'string' ? body.gradedQuality.trim() : '';
      if (!(GRADES as readonly string[]).includes(grade)) {
        throw new HttpError(
          400,
          `확정 등급을 ${GRADES.join('/')} 중에서 선택해주세요.`,
          'bad_grade',
        );
      }
      gradedQuality = grade;
    }

    const note = typeof body.note === 'string' ? body.note.trim().slice(0, 200) || null : null;
    if (result === 'reject' && !note) {
      throw new HttpError(400, '반려 사유를 입력해주세요.', 'missing_rejection_note');
    }

    const inspection = recordInspection(db(), {
      listingId,
      orderItemId,
      hubId: user.hub_id,
      inspector: user.name,
      result,
      gradedQuality,
      note,
    });
    ctx.json({ inspection }, 201);
  });

  router.post('/api/hub/order-items/:id/status', async (ctx) => {
    const user = requireRole(ctx.user, 'hub_operator');
    const id = Number(ctx.params['id']);
    if (!Number.isInteger(id)) throw new HttpError(400, '잘못된 주문 상품입니다.', 'bad_request');

    const body = await ctx.body<{ status?: string }>();
    const scope = hubScope(user);
    if (!getOrderItemForHub(db(), id, scope)) {
      throw new HttpError(404, '주문 상품을 찾을 수 없습니다.', 'not_found');
    }
    const status = body.status;
    const changed = status === 'hub_received'
      ? markHubOrderReceived(db(), id, scope)
      : status === 'ready_to_ship' || status === 'delivered'
        ? advanceHubOrderItem(db(), id, scope, status)
        : false;
    if (!changed) {
      throw new HttpError(409, '순서대로만 처리할 수 있습니다.', 'bad_transition');
    }
    ctx.json({ ok: true, status });
  });
}
