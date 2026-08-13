import { all, one, run, tx } from '../db/index.js';
import type { Db } from '../db/index.js';
import { setConfirmedQuality, setInspectionStatus } from './listings.js';
import type { HubInspection } from './types.js';

export interface InspectionInput {
  listingId: number;
  orderItemId?: number;
  hubId: number | null;
  inspector: string;
  result: 'pass' | 'downgrade' | 'reject';
  gradedQuality?: string | null;
  note?: string | null;
}

/**
 * 거점 실물 검수 기록.
 * 여기서 확정된 등급만이 '확정 등급'이며, AI 참고 판정과 별도 컬럼으로 남는다.
 */
export function recordInspection(db: Db, input: InspectionInput): HubInspection {
  // 기록과 상태 변경은 함께 성립하거나 함께 실패해야 한다.
  // (검수 기록만 남고 상태는 그대로인 불일치를 막는다)
  return tx(db, () => {
    if (input.orderItemId !== undefined) {
      const item = one<{ listing_id: number; fulfillment_status: string }>(
        db,
        'SELECT listing_id, fulfillment_status FROM order_items WHERE id = ?',
        input.orderItemId,
      );
      if (item?.listing_id !== input.listingId || item.fulfillment_status !== 'hub_received') {
        throw new Error('입고 확인된 주문 상품만 검수할 수 있습니다.');
      }
    }

    const res = run(
      db,
      `INSERT INTO hub_inspections (listing_id, hub_id, inspector, result, graded_quality, note)
       VALUES (?,?,?,?,?,?)`,
      input.listingId,
      input.hubId,
      input.inspector,
      input.result,
      input.gradedQuality ?? null,
      input.note ?? null,
    );

    if (input.result === 'reject') {
      run(db, "UPDATE listings SET status = 'closed' WHERE id = ?", input.listingId);
    } else {
      setInspectionStatus(db, input.listingId, 'hub_passed');
      if (input.gradedQuality) setConfirmedQuality(db, input.listingId, input.gradedQuality);
      run(
        db,
        `UPDATE order_items SET fulfillment_status = 'hub_passed'
          WHERE listing_id = ? AND fulfillment_status = 'hub_received'`,
        input.listingId,
      );
    }

    const created = one<HubInspection>(
      db,
      'SELECT * FROM hub_inspections WHERE id = ?',
      res.lastInsertRowid,
    );
    if (!created) throw new Error('검수 기록 저장 실패');
    return created;
  });
}

export function listInspections(db: Db, listingId: number): HubInspection[] {
  return all<HubInspection>(
    db,
    'SELECT * FROM hub_inspections WHERE listing_id = ? ORDER BY created_at DESC, id DESC',
    listingId,
  );
}

export interface HubCounters {
  incoming: number;
  needInspection: number;
  readyToShip: number;
  delivered: number;
  soldOut: number;
}

/** @param hubId null 이면 전체(관리자). */
export function hubCounters(db: Db, hubId: number | null = null): HubCounters {
  const scope = hubId === null ? '' : 'AND f.hub_id = ?';
  const params = hubId === null ? [] : [hubId];
  const row = one<HubCounters>(
    db,
    `SELECT
       SUM(CASE WHEN l.inspection_status = 'ai_checked'     THEN 1 ELSE 0 END) AS incoming,
       SUM(CASE WHEN l.inspection_status = 'hub_pending'    THEN 1 ELSE 0 END) AS needInspection,
       SUM(CASE WHEN l.inspection_status IN ('hub_passed','ready_to_ship') THEN 1 ELSE 0 END) AS readyToShip,
       SUM(CASE WHEN l.inspection_status = 'delivered'      THEN 1 ELSE 0 END) AS delivered,
       SUM(CASE WHEN l.status = 'sold_out'                  THEN 1 ELSE 0 END) AS soldOut
     FROM listings l JOIN farms f ON f.id = l.farm_id
     WHERE l.status != 'closed' ${scope}`,
    ...params,
  );
  return {
    incoming: row?.incoming ?? 0,
    needInspection: row?.needInspection ?? 0,
    readyToShip: row?.readyToShip ?? 0,
    delivered: row?.delivered ?? 0,
    soldOut: row?.soldOut ?? 0,
  };
}
