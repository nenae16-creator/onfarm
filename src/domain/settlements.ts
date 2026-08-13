import { all, one, run } from '../db/index.js';
import type { Db } from '../db/index.js';
import type { OrderItem, Settlement } from './types.js';

/**
 * 데모용 수수료율. 실제 요율은 운영 정책/계약으로 정해지며 코드가 정하지 않는다.
 * (기획서·화면에도 '가정값'으로 표기한다)
 */
export const DEMO_FEE_RATE = 0.08;

export function createSettlement(db: Db, item: OrderItem, feeRate = DEMO_FEE_RATE): Settlement {
  const gross = item.amount;
  const fee = Math.round(gross * feeRate);
  const net = gross - fee;
  const res = run(
    db,
    `INSERT INTO settlements (farmer_id, order_item_id, gross, fee, net, status)
     VALUES (?,?,?,?,?, 'pending')`,
    item.farmer_id,
    item.id,
    gross,
    fee,
    net,
  );
  const created = one<Settlement>(db, 'SELECT * FROM settlements WHERE id = ?', res.lastInsertRowid);
  if (!created) throw new Error('정산 레코드 생성 실패');
  return created;
}

export interface SettlementRow extends Settlement {
  order_no: string;
  title: string;
  quantity: number;
  has_rejection: 0 | 1;
}

export interface PayableSettlementRow extends Settlement {
  order_no: string;
  title: string;
  quantity: number;
  farm_name: string;
}

export function listSettlements(db: Db, farmerId: number): SettlementRow[] {
  return all<SettlementRow>(
    db,
    `SELECT st.*, o.order_no, l.title, oi.quantity,
            EXISTS (
              SELECT 1 FROM hub_inspections hi
               WHERE hi.listing_id = oi.listing_id AND hi.result = 'reject'
            ) AS has_rejection
       FROM settlements st
       JOIN order_items oi ON oi.id = st.order_item_id
       JOIN orders o       ON o.id = oi.order_id
       JOIN listings l     ON l.id = oi.listing_id
      WHERE st.farmer_id = ?
      ORDER BY st.created_at DESC, st.id DESC`,
    farmerId,
  );
}

export function listPayableSettlements(db: Db): PayableSettlementRow[] {
  return all<PayableSettlementRow>(
    db,
    `SELECT st.*, o.order_no, l.title, oi.quantity, f.farm_name
       FROM settlements st
       JOIN order_items oi ON oi.id = st.order_item_id
       JOIN orders o       ON o.id = oi.order_id
       JOIN listings l     ON l.id = oi.listing_id
       JOIN farms f        ON f.id = l.farm_id
      WHERE st.status = 'pending' AND oi.fulfillment_status = 'delivered'
        AND st.due_on IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM hub_inspections hi
                         WHERE hi.listing_id = oi.listing_id AND hi.result = 'reject')
      ORDER BY st.due_on, st.id`,
  );
}

export function markSettlementPaid(db: Db, id: number, reference: string): 'paid' | 'already_paid' | 'conflict' | 'not_found' | 'not_ready' | 'rejected' {
  const row = one<SettlementRow & { fulfillment_status: string }>(
    db,
    `SELECT st.*, o.order_no, l.title, oi.quantity, oi.fulfillment_status,
            EXISTS (SELECT 1 FROM hub_inspections hi
                     WHERE hi.listing_id = oi.listing_id AND hi.result = 'reject') AS has_rejection
       FROM settlements st JOIN order_items oi ON oi.id = st.order_item_id
       JOIN orders o ON o.id = oi.order_id JOIN listings l ON l.id = oi.listing_id
      WHERE st.id = ?`,
    id,
  );
  if (!row) return 'not_found';
  if (row.status === 'paid') return row.payment_reference === reference ? 'already_paid' : 'conflict';
  if (row.has_rejection) return 'rejected';
  if (row.fulfillment_status !== 'delivered') return 'not_ready';
  run(
    db,
    "UPDATE settlements SET status = 'paid', paid_at = datetime('now'), payment_reference = ? WHERE id = ? AND status = 'pending'",
    reference,
    id,
  );
  return 'paid';
}

export interface SettlementSummary {
  processingNet: number;
  pendingNet: number;
  paidNet: number;
  totalGross: number;
  totalFee: number;
  count: number;
}

export function settlementSummary(db: Db, farmerId: number): SettlementSummary {
  const row = one<{
    processingNet: number | null;
    pendingNet: number | null;
    paidNet: number | null;
    totalGross: number | null;
    totalFee: number | null;
    count: number;
  }>(
    db,
    `SELECT
       SUM(CASE WHEN st.status = 'pending' AND st.due_on IS NULL THEN st.net ELSE 0 END) AS processingNet,
       SUM(CASE WHEN st.status = 'pending' AND st.due_on IS NOT NULL THEN st.net ELSE 0 END) AS pendingNet,
       SUM(CASE WHEN st.status = 'paid'    THEN st.net ELSE 0 END) AS paidNet,
       SUM(st.gross) AS totalGross,
       SUM(st.fee)   AS totalFee,
       COUNT(*)   AS count
     FROM settlements st
     JOIN order_items oi ON oi.id = st.order_item_id
    WHERE st.farmer_id = ?
      AND (st.status = 'paid' OR NOT EXISTS (
        SELECT 1 FROM hub_inspections hi
         WHERE hi.listing_id = oi.listing_id AND hi.result = 'reject'
      ))`,
    farmerId,
  );
  return {
    processingNet: row?.processingNet ?? 0,
    pendingNet: row?.pendingNet ?? 0,
    paidNet: row?.paidNet ?? 0,
    totalGross: row?.totalGross ?? 0,
    totalFee: row?.totalFee ?? 0,
    count: row?.count ?? 0,
  };
}
