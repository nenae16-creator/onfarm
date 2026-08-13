import { all, one, run, tx } from '../db/index.js';
import type { Db } from '../db/index.js';
import { orderNo } from '../lib/datetime.js';
import { decrementInventory } from './listings.js';
import { createSettlement } from './settlements.js';
import type { FulfillmentStatus, Listing, Order, OrderItem } from './types.js';

export interface OrderLine {
  listingId: number;
  quantity: number;
}

export interface CreateOrderInput {
  consumerId: number;
  lines: OrderLine[];
  receiverName: string;
  receiverPhone: string;
  address: string;
  memo?: string;
}

export class OrderError extends Error {
  constructor(
    message: string,
    readonly code: 'EMPTY' | 'NOT_FOUND' | 'OUT_OF_STOCK' | 'INVALID',
  ) {
    super(message);
    this.name = 'OrderError';
  }
}

export interface CreatedOrder {
  order: Order;
  items: OrderItem[];
}

/**
 * 주문 생성. 재고 차감 → 주문/주문항목 → 정산 예정 레코드까지 한 트랜잭션으로 처리한다.
 * 재고가 모자라면 전체를 롤백한다(부분 성공 없음).
 */
export function createOrder(db: Db, input: CreateOrderInput): CreatedOrder {
  const quantities = new Map<number, number>();
  for (const line of input.lines) {
    if (!Number.isSafeInteger(line.listingId) || !Number.isSafeInteger(line.quantity)
        || line.listingId <= 0 || line.quantity <= 0) {
      throw new OrderError('수량이 올바르지 않습니다.', 'INVALID');
    }
    const quantity = (quantities.get(line.listingId) ?? 0) + line.quantity;
    if (!Number.isSafeInteger(quantity)) throw new OrderError('수량이 올바르지 않습니다.', 'INVALID');
    quantities.set(line.listingId, quantity);
  }
  const lines = [...quantities].map(([listingId, quantity]) => ({ listingId, quantity }));
  if (lines.length === 0) throw new OrderError('주문할 상품이 없습니다.', 'EMPTY');
  const phone = input.receiverPhone.trim();
  const address = input.address.trim();
  if (!input.receiverName.trim() || !phone || !address
      || phone === '010-0000-0000' || address === '충남 천안시 동남구 ...') {
    throw new OrderError('받는 분 정보가 필요합니다.', 'INVALID');
  }

  return tx(db, () => {
    let total = 0;
    const prepared: Array<{ listing: Listing; quantity: number; amount: number }> = [];

    for (const line of lines) {
      const listing = one<Listing>(db, 'SELECT * FROM listings WHERE id = ?', line.listingId);
      if (!listing) throw new OrderError('상품을 찾을 수 없습니다.', 'NOT_FOUND');
      if (!decrementInventory(db, listing.id, line.quantity)) {
        throw new OrderError(`재고가 부족합니다: ${listing.title}`, 'OUT_OF_STOCK');
      }
      const amount = listing.unit_price * line.quantity;
      total += amount;
      prepared.push({ listing, quantity: line.quantity, amount });
    }

    // 주문번호는 무작위라 드물게 충돌한다. 충돌 때문에 정상 주문이 실패하면 안 된다.
    let no = orderNo();
    for (let attempt = 0; attempt < 5; attempt += 1) {
      if (!one(db, 'SELECT id FROM orders WHERE order_no = ?', no)) break;
      no = orderNo();
    }
    const orderRes = run(
      db,
      `INSERT INTO orders (consumer_id, order_no, total_amount, receiver_name, receiver_phone, address, memo, status)
       VALUES (?,?,?,?,?,?,?, 'paid')`,
      input.consumerId,
      no,
      total,
      input.receiverName.trim(),
      phone,
      address,
      input.memo?.trim() ?? null,
    );
    const orderId = orderRes.lastInsertRowid;

    const items: OrderItem[] = [];
    for (const p of prepared) {
      const itemRes = run(
        db,
        `INSERT INTO order_items (order_id, listing_id, sku_id, farmer_id, unit_price, quantity, amount)
         VALUES (?,?,?,?,?,?,?)`,
        orderId,
        p.listing.id,
        p.listing.sku_id,
        p.listing.farmer_id,
        p.listing.unit_price,
        p.quantity,
        p.amount,
      );
      const item = one<OrderItem>(db, 'SELECT * FROM order_items WHERE id = ?', itemRes.lastInsertRowid);
      if (item) {
        items.push(item);
        createSettlement(db, item);
      }
      // 주문이 들어오면 거점 검수 대기로 넘어간다.
      run(
        db,
        `UPDATE listings SET inspection_status = 'hub_pending'
          WHERE id = ? AND inspection_status = 'ai_checked'`,
        p.listing.id,
      );
    }

    const order = one<Order>(db, 'SELECT * FROM orders WHERE id = ?', orderId);
    if (!order) throw new OrderError('주문 생성에 실패했습니다.', 'INVALID');
    return { order, items };
  });
}

export interface FarmerOrderRow extends OrderItem {
  order_no: string;
  order_status: string;
  ordered_at: string;
  title: string;
  sku_label: string;
  inspection_status: string;
  hub_name: string | null;
  hub_address: string | null;
  has_rejection: 0 | 1;
  rejection_note: string | null;
}

export function listOrdersForFarmer(db: Db, farmerId: number): FarmerOrderRow[] {
  return all<FarmerOrderRow>(
    db,
    `SELECT oi.*, o.order_no, o.status AS order_status, o.created_at AS ordered_at,
            l.title, l.inspection_status, s.label AS sku_label,
            h.name AS hub_name, h.address AS hub_address,
            EXISTS (
              SELECT 1 FROM hub_inspections hi
               WHERE hi.listing_id = l.id AND hi.result = 'reject'
            ) AS has_rejection,
            (SELECT hi.note FROM hub_inspections hi
              WHERE hi.listing_id = l.id AND hi.result = 'reject'
              ORDER BY hi.created_at DESC, hi.id DESC LIMIT 1) AS rejection_note
       FROM order_items oi
       JOIN orders o   ON o.id = oi.order_id
       JOIN listings l ON l.id = oi.listing_id
       JOIN skus s     ON s.id = oi.sku_id
       LEFT JOIN farms f ON f.id = l.farm_id
       LEFT JOIN hubs h  ON h.id = f.hub_id
      WHERE oi.farmer_id = ?
      ORDER BY o.created_at DESC, oi.id DESC`,
    farmerId,
  );
}

export function markFarmerOrderReady(db: Db, orderItemId: number, farmerId: number): boolean {
  return run(
    db,
    `UPDATE order_items
        SET fulfillment_status = 'ready_for_hub'
      WHERE id = ? AND farmer_id = ? AND fulfillment_status = 'farmer_preparing'
        AND NOT EXISTS (
          SELECT 1 FROM hub_inspections hi
           WHERE hi.listing_id = order_items.listing_id AND hi.result = 'reject'
        )`,
    orderItemId,
    farmerId,
  ).changes === 1;
}

export interface HubOrderItemRow extends OrderItem {
  order_no: string;
  title: string;
  sku_label: string;
  quality_hint: string | null;
  confirmed_quality: string | null;
  ai_confidence: number | null;
  farm_name: string;
  region_sigungu: string;
  hub_name: string | null;
  has_rejection: 0 | 1;
  help_requested: 0 | 1;
}

const HUB_ORDER_ITEMS_SELECT = `
  SELECT oi.*, o.order_no, l.title,
         l.quality_hint, l.confirmed_quality, l.ai_confidence,
         s.label AS sku_label, f.farm_name, f.region_sigungu, h.name AS hub_name,
         EXISTS (
           SELECT 1 FROM hub_inspections hi
            WHERE hi.listing_id = l.id AND hi.result = 'reject'
         ) AS has_rejection,
         EXISTS (
           SELECT 1 FROM refund_help_requests rh WHERE rh.order_item_id = oi.id
         ) AS help_requested
    FROM order_items oi
    JOIN orders o   ON o.id = oi.order_id
    JOIN listings l ON l.id = oi.listing_id
    JOIN skus s     ON s.id = oi.sku_id
    JOIN farms f    ON f.id = l.farm_id
    LEFT JOIN hubs h ON h.id = f.hub_id
`;

export function listOrderItemsForHub(db: Db, hubId: number | null): HubOrderItemRow[] {
  const scope = hubId === null
    ? "WHERE oi.fulfillment_status != 'farmer_preparing' OR EXISTS (SELECT 1 FROM refund_help_requests rh WHERE rh.order_item_id = oi.id)"
    : "WHERE f.hub_id = ? AND (oi.fulfillment_status != 'farmer_preparing' OR EXISTS (SELECT 1 FROM refund_help_requests rh WHERE rh.order_item_id = oi.id))";
  const params = hubId === null ? [] : [hubId];
  return all<HubOrderItemRow>(
    db,
    `${HUB_ORDER_ITEMS_SELECT} ${scope}
      ORDER BY CASE oi.fulfillment_status
        WHEN 'ready_for_hub' THEN 0 WHEN 'hub_received' THEN 1
        WHEN 'hub_passed' THEN 2 WHEN 'ready_to_ship' THEN 3
        WHEN 'farmer_preparing' THEN 4 WHEN 'delivered' THEN 5 ELSE 6 END,
        o.created_at, oi.id`,
    ...params,
  );
}

export function getOrderItemForHub(
  db: Db,
  orderItemId: number,
  hubId: number | null,
): HubOrderItemRow | null {
  const scope = hubId === null ? '' : 'AND f.hub_id = ?';
  const params = hubId === null ? [orderItemId] : [orderItemId, hubId];
  return one<HubOrderItemRow>(
    db,
    `${HUB_ORDER_ITEMS_SELECT} WHERE oi.id = ? ${scope}`,
    ...params,
  );
}

export function markHubOrderReceived(db: Db, orderItemId: number, hubId: number | null): boolean {
  const scope = hubId === null ? '' : 'AND f.hub_id = ?';
  const params = hubId === null ? [orderItemId] : [orderItemId, hubId];
  return run(
    db,
    `UPDATE order_items
        SET fulfillment_status = CASE
              WHEN EXISTS (
                SELECT 1 FROM listings l
                 WHERE l.id = order_items.listing_id AND l.confirmed_quality IS NOT NULL
              ) THEN 'hub_passed'
              ELSE 'hub_received'
            END
      WHERE id = ? AND fulfillment_status = 'ready_for_hub'
        AND EXISTS (
          SELECT 1 FROM listings l JOIN farms f ON f.id = l.farm_id
           WHERE l.id = order_items.listing_id ${scope}
        )`,
    ...params,
  ).changes === 1;
}

export function advanceHubOrderItem(
  db: Db,
  orderItemId: number,
  hubId: number | null,
  status: Extract<FulfillmentStatus, 'ready_to_ship' | 'delivered'>,
): boolean {
  return tx(db, () => {
    const current = status === 'ready_to_ship' ? 'hub_passed' : 'ready_to_ship';
    const scope = hubId === null ? '' : 'AND f.hub_id = ?';
    const params = hubId === null
      ? [status, orderItemId, current]
      : [status, orderItemId, current, hubId];
    const changed = run(
      db,
      `UPDATE order_items
          SET fulfillment_status = ?
        WHERE id = ? AND fulfillment_status = ?
          AND EXISTS (
            SELECT 1 FROM listings l JOIN farms f ON f.id = l.farm_id
             WHERE l.id = order_items.listing_id ${scope}
          )`,
      ...params,
    ).changes === 1;
    if (changed && status === 'delivered') {
      run(
        db,
        "UPDATE settlements SET due_on = date('now', '+9 hours', '+7 days') WHERE order_item_id = ? AND due_on IS NULL",
        orderItemId,
      );
    }
    return changed;
  });
}

export interface ConsumerOrderRow {
  order_no: string;
  status: string;
  created_at: string;
  total_amount: number;
  items: Array<{
    id: number;
    listing_id: number;
    title: string;
    quantity: number;
    amount: number;
    inspection_status: string;
    fulfillment_status: FulfillmentStatus;
    has_rejection: 0 | 1;
    rejection_note: string | null;
    refund_amount: number | null;
    help_requested: 0 | 1;
  }>;
}

export interface DeliveryProfile {
  receiver_name: string;
  receiver_phone: string;
  address: string;
}

export function lastDeliveryForConsumer(db: Db, consumerId: number): DeliveryProfile | null {
  return one<DeliveryProfile>(
    db,
    `SELECT receiver_name, receiver_phone, address
       FROM orders
      WHERE consumer_id = ? AND status != 'cancelled'
        AND receiver_phone != '010-0000-0000'
        AND address != '충남 천안시 동남구 ...'
       ORDER BY created_at DESC, id DESC LIMIT 1`,
    consumerId,
  );
}

export function requestRefundHelp(
  db: Db,
  orderItemId: number,
  consumerId: number,
): 'created' | 'existing' | 'not_found' | 'not_rejected' {
  const item = one<{ listing_id: number }>(
    db,
    `SELECT oi.listing_id FROM order_items oi
       JOIN orders o ON o.id = oi.order_id
      WHERE oi.id = ? AND o.consumer_id = ?`,
    orderItemId,
    consumerId,
  );
  if (!item) return 'not_found';
  if (!one(db, "SELECT id FROM hub_inspections WHERE listing_id = ? AND result = 'reject' LIMIT 1", item.listing_id)) {
    return 'not_rejected';
  }
  const result = run(
    db,
    'INSERT OR IGNORE INTO refund_help_requests (order_item_id, consumer_id) VALUES (?,?)',
    orderItemId,
    consumerId,
  );
  return result.changes === 1 ? 'created' : 'existing';
}

export function listOrdersForConsumer(db: Db, consumerId: number): ConsumerOrderRow[] {
  const orders = all<{
    id: number;
    order_no: string;
    status: string;
    created_at: string;
    total_amount: number;
  }>(
    db,
    'SELECT id, order_no, status, created_at, total_amount FROM orders WHERE consumer_id = ? ORDER BY created_at DESC, id DESC',
    consumerId,
  );
  return orders.map((o) => ({
    order_no: o.order_no,
    status: o.status,
    created_at: o.created_at,
    total_amount: o.total_amount,
    items: all<ConsumerOrderRow['items'][number]>(
      db,
      `SELECT oi.id, oi.listing_id, l.title, oi.quantity, oi.amount,
              l.inspection_status, oi.fulfillment_status,
              EXISTS (
                SELECT 1 FROM hub_inspections hi
                 WHERE hi.listing_id = l.id AND hi.result = 'reject'
              ) AS has_rejection,
              (SELECT hi.note FROM hub_inspections hi
                WHERE hi.listing_id = l.id AND hi.result = 'reject'
                ORDER BY hi.created_at DESC, hi.id DESC LIMIT 1) AS rejection_note,
              CASE WHEN EXISTS (
                SELECT 1 FROM hub_inspections hi
                 WHERE hi.listing_id = l.id AND hi.result = 'reject'
              ) THEN oi.amount ELSE NULL END AS refund_amount,
              EXISTS (
                SELECT 1 FROM refund_help_requests rh WHERE rh.order_item_id = oi.id
              ) AS help_requested
         FROM order_items oi JOIN listings l ON l.id = oi.listing_id
        WHERE oi.order_id = ?`,
      o.id,
    ),
  }));
}

export function setOrderStatus(db: Db, orderId: number, status: Order['status']): void {
  run(db, 'UPDATE orders SET status = ? WHERE id = ?', status, orderId);
}
