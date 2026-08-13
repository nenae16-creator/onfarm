import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { describe, it } from 'node:test';
import { all, one, openDb, run } from '../db/index.js';
import { seed } from '../db/seed.js';
import { recordInspection, hubCounters } from '../domain/inspections.js';
import { getListingView, listStoreListings } from '../domain/listings.js';
import {
  advanceHubOrderItem,
  createOrder,
  listOrdersForConsumer,
  listOrdersForFarmer,
  markFarmerOrderReady,
  markHubOrderReceived,
  OrderError,
  requestRefundHelp,
} from '../domain/orders.js';
import { listSettlements, markSettlementPaid, settlementSummary } from '../domain/settlements.js';
import type { ListingView } from '../domain/types.js';
import { consumerNamed, freshDb } from './helpers.js';

function firstTwoListings(db: ReturnType<typeof freshDb>): [ListingView, ListingView] {
  const rows = listStoreListings(db);
  assert.ok(rows.length >= 2, '시드 매물이 2개 이상이어야 한다');
  return [rows[0] as ListingView, rows[1] as ListingView];
}

describe('주문 생성', () => {
  it('주문하면 재고가 줄고 정산 예정이 생긴다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    const before = listing.remaining_quantity;

    const { order, items } = createOrder(db, {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 2 }],
      receiverName: '최수민',
      receiverPhone: '010-1234-5678',
      address: '충남 천안시 동남구',
    });

    assert.equal(order.total_amount, listing.unit_price * 2);
    assert.equal(items.length, 1);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, before - 2);

    const settlements = listSettlements(db, listing.farmer_id);
    assert.equal(settlements.length, 1);
    assert.equal(settlements[0]?.gross, listing.unit_price * 2);
    assert.equal(settlements[0]?.net, settlements[0]!.gross - settlements[0]!.fee);
    assert.equal(settlements[0]?.status, 'pending');
  });

  it('주문이 들어오면 거점 검수 대기로 넘어간다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    assert.equal(listing.inspection_status, 'ai_checked');

    createOrder(db, {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 1 }],
      receiverName: '최수민',
      receiverPhone: '010-1234-5678',
      address: '충남 천안시',
    });

    assert.equal(getListingView(db, listing.id)?.inspection_status, 'hub_pending');
    assert.equal(hubCounters(db).needInspection, 1);
  });

  it('여러 상품을 한 주문에 담을 수 있다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [a, b] = firstTwoListings(db);
    const { order, items } = createOrder(db, {
      consumerId: consumer.id,
      lines: [
        { listingId: a.id, quantity: 1 },
        { listingId: b.id, quantity: 2 },
      ],
      receiverName: '최수민',
      receiverPhone: '010-1234-5678',
      address: '서울시',
    });
    assert.equal(items.length, 2);
    assert.equal(order.total_amount, a.unit_price + b.unit_price * 2);
  });

  it('같은 상품 줄은 하나로 합쳐 재고와 정산을 한 번만 처리한다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    const before = listing.remaining_quantity;
    const { items } = createOrder(db, {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 1 }, { listingId: listing.id, quantity: 2 }],
      receiverName: '최수민',
      receiverPhone: '010-1234-5678',
      address: '서울시',
    });

    assert.equal(items.length, 1);
    assert.equal(items[0]?.quantity, 3);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, before - 3);
    assert.equal(listSettlements(db, listing.farmer_id).length, 1);
  });

  it('배송 완료 뒤 예정일을 잡고 지급 완료 기록을 한 번만 남긴다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    const { items } = createOrder(db, {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 1 }],
      receiverName: '최수민',
      receiverPhone: '010-1234-5678',
      address: '서울시',
    });
    const item = items[0]!;
    assert.equal(markSettlementPaid(db, 1, 'PAY-EARLY'), 'not_ready');
    run(db, "UPDATE order_items SET fulfillment_status = 'ready_to_ship' WHERE id = ?", item.id);
    assert.equal(advanceHubOrderItem(db, item.id, null, 'delivered'), true);
    const due = one<{ due_on: string | null }>(db, 'SELECT due_on FROM settlements WHERE order_item_id = ?', item.id);
    assert.match(due?.due_on ?? '', /^\d{4}-\d{2}-\d{2}$/);
    assert.equal(markSettlementPaid(db, 1, 'PAY-001'), 'paid');
    const paid = one<{ paid_at: string | null; payment_reference: string | null }>(db, 'SELECT paid_at, payment_reference FROM settlements WHERE id = 1');
    assert.ok(paid?.paid_at);
    assert.equal(paid?.payment_reference, 'PAY-001');
    assert.equal(markSettlementPaid(db, 1, 'PAY-001'), 'already_paid');
    assert.equal(markSettlementPaid(db, 1, 'PAY-CHANGED'), 'conflict');
    assert.deepEqual(one(db, 'SELECT paid_at, payment_reference FROM settlements WHERE id = 1'), paid);
  });

  it('재고가 모자라면 전부 되돌린다 — 부분 성공은 없다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [a, b] = firstTwoListings(db);
    const beforeA = a.remaining_quantity;

    assert.throws(
      () =>
        createOrder(db, {
          consumerId: consumer.id,
          lines: [
            { listingId: a.id, quantity: 1 },
            { listingId: b.id, quantity: b.remaining_quantity + 5 },
          ],
          receiverName: '최수민',
          receiverPhone: '010-1234-5678',
          address: '서울시',
        }),
      (err: unknown) => err instanceof OrderError && err.code === 'OUT_OF_STOCK',
    );

    assert.equal(getListingView(db, a.id)?.remaining_quantity, beforeA, 'A 재고가 되돌아와야 한다');
    assert.equal(all(db, 'SELECT id FROM orders').length, 0, '주문이 남으면 안 된다');
    assert.equal(all(db, 'SELECT id FROM order_items').length, 0);
    assert.equal(all(db, 'SELECT id FROM settlements').length, 0);
  });

  it('없는 상품·빈 주문·받는 분 누락을 막는다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const base = { consumerId: consumer.id, receiverName: '최수민', receiverPhone: '010', address: '서울' };
    assert.throws(() => createOrder(db, { ...base, lines: [] }), /주문할 상품/);
    assert.throws(() => createOrder(db, { ...base, lines: [{ listingId: 99999, quantity: 1 }] }), /찾을 수 없/);
    const [a] = firstTwoListings(db);
    assert.throws(
      () => createOrder(db, { ...base, address: '   ', lines: [{ listingId: a.id, quantity: 1 }] }),
      /받는 분/,
    );
    assert.throws(
      () => createOrder(db, { ...base, receiverPhone: '010-0000-0000', lines: [{ listingId: a.id, quantity: 1 }] }),
      /받는 분/,
    );
    assert.throws(
      () => createOrder(db, { ...base, address: '충남 천안시 동남구 ...', lines: [{ listingId: a.id, quantity: 1 }] }),
      /받는 분/,
    );
  });

  it('농민 주문 목록과 정산 요약이 맞아떨어진다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    createOrder(db, {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 3 }],
      receiverName: '최수민',
      receiverPhone: '010-1234-5678',
      address: '충남 천안시',
    });

    const orders = listOrdersForFarmer(db, listing.farmer_id);
    assert.equal(orders.length, 1);
    assert.equal(orders[0]?.quantity, 3);
    assert.equal(orders[0]?.title, listing.title);
    assert.equal(orders[0]?.hub_name, listing.hub_name);
    assert.ok(orders[0]?.hub_address?.startsWith(listing.region_sido));

    const summary = settlementSummary(db, listing.farmer_id);
    assert.equal(summary.count, 1);
    assert.equal(summary.totalGross, listing.unit_price * 3);
    assert.equal(summary.processingNet, summary.totalGross - summary.totalFee);
    assert.equal(summary.pendingNet, 0);
  });

  it('같은 상품을 다시 주문해도 주문 상품별 진행 상태는 섞이지 않는다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    const input = {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 1 }],
      receiverName: '최수민', receiverPhone: '010', address: '천안',
    };
    const first = createOrder(db, input).items[0]!;
    const second = createOrder(db, input).items[0]!;

    assert.equal(markFarmerOrderReady(db, first.id, listing.farmer_id), true);
    assert.equal(markHubOrderReceived(db, first.id, null), true);
    recordInspection(db, {
      listingId: listing.id,
      orderItemId: first.id,
      hubId: null,
      inspector: '담당자',
      result: 'pass',
      gradedQuality: '상',
    });
    assert.equal(advanceHubOrderItem(db, first.id, null, 'ready_to_ship'), true);
    assert.equal(advanceHubOrderItem(db, first.id, null, 'delivered'), true);

    assert.equal(one<{ fulfillment_status: string }>(db, 'SELECT fulfillment_status FROM order_items WHERE id = ?', first.id)?.fulfillment_status, 'delivered');
    assert.equal(one<{ fulfillment_status: string }>(db, 'SELECT fulfillment_status FROM order_items WHERE id = ?', second.id)?.fulfillment_status, 'farmer_preparing');

    assert.equal(markFarmerOrderReady(db, second.id, listing.farmer_id), true);
    assert.equal(markHubOrderReceived(db, second.id, null), true);
    assert.equal(one<{ fulfillment_status: string }>(db, 'SELECT fulfillment_status FROM order_items WHERE id = ?', second.id)?.fulfillment_status, 'hub_passed', '이미 검수된 배치는 재검수하지 않는다');
  });

  it('검수 반려는 환불 예정으로 보이고 미지급 정산에서만 제외된다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    const other = listStoreListings(db).find(
      (candidate) => candidate.farmer_id === listing.farmer_id && candidate.id !== listing.id,
    );
    assert.ok(other, '같은 농민의 다른 상품이 있어야 한다');
    createOrder(db, {
      consumerId: consumer.id,
      lines: [
        { listingId: listing.id, quantity: 1 },
        { listingId: other.id, quantity: 1 },
      ],
      receiverName: '최수민', receiverPhone: '010', address: '천안',
    });

    run(db, "UPDATE listings SET status = 'closed' WHERE id = ?", listing.id);
    assert.equal(listOrdersForFarmer(db, listing.farmer_id)[0]?.has_rejection, 0);
    assert.equal(listOrdersForFarmer(db, listing.farmer_id)[0]?.rejection_note, null);
    assert.equal(listOrdersForConsumer(db, consumer.id)[0]?.items[0]?.has_rejection, 0);
    run(db, "UPDATE listings SET status = 'active' WHERE id = ?", listing.id);

    recordInspection(db, {
      listingId: listing.id, hubId: null, inspector: '담당자', result: 'reject', note: '이전 사유',
    });
    recordInspection(db, {
      listingId: listing.id, hubId: null, inspector: '담당자', result: 'reject', note: '표면 부패 확인',
    });

    const farmerOrder = listOrdersForFarmer(db, listing.farmer_id).find((order) => order.title === listing.title);
    assert.equal(farmerOrder?.has_rejection, 1);
    assert.equal(farmerOrder?.rejection_note, '표면 부패 확인');
    const consumerItems = listOrdersForConsumer(db, consumer.id)[0]?.items ?? [];
    const rejectedItem = consumerItems.find((item) => item.title === listing.title);
    const payableItem = consumerItems.find((item) => item.title === other.title);
    assert.equal(rejectedItem?.has_rejection, 1);
    assert.equal(rejectedItem?.rejection_note, '표면 부패 확인');
    assert.equal(rejectedItem?.refund_amount, rejectedItem?.amount);
    assert.equal(payableItem?.has_rejection, 0);
    assert.equal(payableItem?.refund_amount, null);

    assert.equal(requestRefundHelp(db, rejectedItem!.id, consumer.id), 'created');
    assert.equal(requestRefundHelp(db, rejectedItem!.id, consumer.id), 'existing');
    assert.equal(requestRefundHelp(db, payableItem!.id, consumer.id), 'not_rejected');
    assert.equal(requestRefundHelp(db, rejectedItem!.id, consumerNamed(db, '최수민').id), 'not_found');
    assert.equal(all(db, 'SELECT id FROM refund_help_requests').length, 1);
    assert.equal(listOrdersForConsumer(db, consumer.id)[0]?.items.find((item) => item.id === rejectedItem?.id)?.help_requested, 1);

    const settlements = listSettlements(db, listing.farmer_id);
    const rejectedSettlement = settlements.find((row) => row.title === listing.title);
    const payableSettlement = settlements.find((row) => row.title === other.title);
    assert.equal(rejectedSettlement?.has_rejection, 1);
    assert.equal(payableSettlement?.has_rejection, 0);
    assert.equal(all(db, 'SELECT id FROM settlements').length, 2, '정산 원본 이력은 보존한다');

    const summary = settlementSummary(db, listing.farmer_id);
    assert.equal(summary.count, 1);
    assert.equal(summary.totalGross, payableSettlement?.gross);
    assert.equal(summary.processingNet, payableSettlement?.net);
    assert.equal(summary.pendingNet, 0);
  });

  it('이미 지급한 정산은 뒤늦은 반려 기록에도 숨기지 않는다', () => {
    const db = freshDb();
    const consumer = consumerNamed(db);
    const [listing] = firstTwoListings(db);
    createOrder(db, {
      consumerId: consumer.id,
      lines: [{ listingId: listing.id, quantity: 1 }],
      receiverName: '최수민', receiverPhone: '010', address: '천안',
    });
    run(db, "UPDATE settlements SET status = 'paid'");
    recordInspection(db, {
      listingId: listing.id, hubId: null, inspector: '담당자', result: 'reject', note: '사후 확인',
    });

    const row = listSettlements(db, listing.farmer_id)[0];
    const summary = settlementSummary(db, listing.farmer_id);
    assert.equal(row?.has_rejection, 1);
    assert.equal(summary.pendingNet, 0);
    assert.equal(summary.paidNet, row?.net);
    assert.equal(summary.totalGross, row?.gross);
    assert.equal(summary.count, 1);
  });
});

describe('주문 상품 상태 마이그레이션', () => {
  it('기존 DB 주문의 진행 상태를 보존하고 제약을 적용한다', () => {
    const dir = mkdtempSync(join(tmpdir(), 'onfarm-migrate-'));
    const path = join(dir, 'legacy.db');
    let current: ReturnType<typeof openDb> | null = null;
    let legacy: DatabaseSync | null = null;
    let migrated: ReturnType<typeof openDb> | null = null;
    try {
      current = openDb(path);
      seed(current);
      const consumer = consumerNamed(current);
      const [listing, pendingListing] = firstTwoListings(current);
      const item = createOrder(current, {
        consumerId: consumer.id,
        lines: [{ listingId: listing.id, quantity: 1 }],
        receiverName: '최수민', receiverPhone: '010', address: '천안',
      }).items[0]!;
      const pendingItem = createOrder(current, {
        consumerId: consumer.id,
        lines: [{ listingId: pendingListing.id, quantity: 1 }],
        receiverName: '최수민', receiverPhone: '010', address: '천안',
      }).items[0]!;
      run(current, "UPDATE listings SET inspection_status = 'delivered' WHERE id = ?", listing.id);
      current.close();
      current = null;

      legacy = new DatabaseSync(path);
      legacy.exec('PRAGMA foreign_keys = OFF');
      legacy.exec(`
        CREATE TABLE legacy_order_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          order_id INTEGER NOT NULL REFERENCES orders(id),
          listing_id INTEGER NOT NULL REFERENCES listings(id),
          sku_id INTEGER NOT NULL REFERENCES skus(id),
          farmer_id INTEGER NOT NULL REFERENCES users(id),
          unit_price INTEGER NOT NULL,
          quantity INTEGER NOT NULL CHECK (quantity > 0),
          amount INTEGER NOT NULL
        );
        INSERT INTO legacy_order_items
          SELECT id, order_id, listing_id, sku_id, farmer_id, unit_price, quantity, amount
            FROM order_items;
        DROP TABLE order_items;
        ALTER TABLE legacy_order_items RENAME TO order_items;
        CREATE TABLE legacy_settlements (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          farmer_id INTEGER NOT NULL REFERENCES users(id),
          order_item_id INTEGER NOT NULL REFERENCES order_items(id),
          gross INTEGER NOT NULL,
          fee INTEGER NOT NULL,
          net INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','paid')),
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO legacy_settlements (id, farmer_id, order_item_id, gross, fee, net, status, created_at)
          SELECT id, farmer_id, order_item_id, gross, fee, net, status, created_at FROM settlements;
        DROP TABLE settlements;
        ALTER TABLE legacy_settlements RENAME TO settlements;
      `);
      legacy.close();
      legacy = null;

      const upgraded = openDb(path);
      migrated = upgraded;
      assert.equal(one<{ fulfillment_status: string }>(upgraded, 'SELECT fulfillment_status FROM order_items WHERE id = ?', item.id)?.fulfillment_status, 'delivered');
      assert.equal(one<{ fulfillment_status: string }>(upgraded, 'SELECT fulfillment_status FROM order_items WHERE id = ?', pendingItem.id)?.fulfillment_status, 'ready_for_hub');
      assert.match(one<{ due_on: string }>(upgraded, 'SELECT due_on FROM settlements WHERE order_item_id = ?', item.id)?.due_on ?? '', /^\d{4}-\d{2}-\d{2}$/);
      assert.equal(one<{ due_on: string | null }>(upgraded, 'SELECT due_on FROM settlements WHERE order_item_id = ?', pendingItem.id)?.due_on, null);
      assert.throws(
        () => run(upgraded, "UPDATE order_items SET fulfillment_status = 'invalid' WHERE id = ?", item.id),
        /CHECK constraint failed/,
      );
      migrated.close();
      migrated = null;
    } finally {
      current?.close();
      legacy?.close();
      migrated?.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('거점 실물 검수', () => {
  it('검수를 통과시키면 상태가 넘어가고 기록이 남는다', () => {
    const db = freshDb();
    const [listing] = firstTwoListings(db);
    recordInspection(db, {
      listingId: listing.id,
      hubId: null,
      inspector: '성환거점 담당자',
      result: 'pass',
      gradedQuality: '상',
    });
    const after = getListingView(db, listing.id);
    assert.equal(after?.inspection_status, 'hub_passed');
    assert.equal(all(db, 'SELECT id FROM hub_inspections').length, 1);
  });

  it('확정 등급은 AI 참고값을 덮어쓰지 않고 별도로 기록된다', () => {
    const db = freshDb();
    const [listing] = firstTwoListings(db);
    const aiHint = listing.quality_hint;
    recordInspection(db, {
      listingId: listing.id,
      hubId: null,
      inspector: '담당자',
      result: 'downgrade',
      gradedQuality: '보통',
      note: '표면 흠집',
    });
    const after = getListingView(db, listing.id);
    assert.equal(after?.confirmed_quality, '보통', '확정 등급이 기록돼야 한다');
    assert.equal(after?.quality_hint, aiHint, 'AI 참고값은 그대로 남아야 감사 추적이 가능하다');
    const row = one<{ result: string; graded_quality: string }>(
      db,
      'SELECT result, graded_quality FROM hub_inspections WHERE listing_id = ?',
      listing.id,
    );
    assert.equal(row?.result, 'downgrade');
  });

  it('반려하면 판매가 닫히고 매장에서 사라진다', () => {
    const db = freshDb();
    const [listing] = firstTwoListings(db);
    recordInspection(db, { listingId: listing.id, hubId: null, inspector: '담당자', result: 'reject', note: '부패' });
    assert.equal(getListingView(db, listing.id)?.status, 'closed');
    assert.equal(listStoreListings(db).find((l) => l.id === listing.id), undefined);
  });
});
