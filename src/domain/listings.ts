import { all, one, run } from '../db/index.js';
import type { Db } from '../db/index.js';
import type { InspectionStatus, Listing, ListingView, QualityHint } from './types.js';

const VIEW_SELECT = `
  SELECT l.*,
         p.code AS product_code, p.name_ko AS product_name, p.emoji AS product_emoji,
         s.label AS sku_label, s.weight AS sku_weight, s.unit AS sku_unit,
         f.farm_name, f.region_sido, f.region_sigungu, f.region_detail,
         u.name AS farmer_name,
         h.name AS hub_name,
         EXISTS (
           SELECT 1 FROM hub_inspections hi
            WHERE hi.listing_id = l.id AND hi.result = 'reject'
         ) AS has_rejection
    FROM listings l
    JOIN products p ON p.id = l.product_id
    JOIN skus s     ON s.id = l.sku_id
    JOIN farms f    ON f.id = l.farm_id
    JOIN users u    ON u.id = l.farmer_id
    LEFT JOIN hubs h ON h.id = f.hub_id
`;

export interface CreateListingInput {
  farmerId: number;
  farmId: number;
  productId: number;
  skuId: number;
  title: string;
  description: string;
  imagePath: string | null;
  quantity: number;
  unitPrice: number;
  harvestedOn: string;
  aiAnalysis: unknown;
  aiConfidence: number | null;
  aiSource: string | null;
  qualityHint: QualityHint | null;
}

export function createListing(db: Db, input: CreateListingInput): Listing {
  if (!Number.isInteger(input.quantity) || input.quantity <= 0) {
    throw new Error('수량은 1 이상의 정수여야 합니다.');
  }
  if (!Number.isInteger(input.unitPrice) || input.unitPrice <= 0) {
    throw new Error('단가는 표준 SKU 에서 와야 합니다.');
  }
  const res = run(
    db,
    `INSERT INTO listings
       (farmer_id, farm_id, product_id, sku_id, title, description, image_path,
        quantity, remaining_quantity, unit_price, harvested_on,
        ai_analysis, ai_confidence, ai_source, quality_hint, inspection_status, status)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ai_checked','active')`,
    input.farmerId,
    input.farmId,
    input.productId,
    input.skuId,
    input.title,
    input.description,
    input.imagePath,
    input.quantity,
    input.quantity,
    input.unitPrice,
    input.harvestedOn,
    input.aiAnalysis ? JSON.stringify(input.aiAnalysis) : null,
    input.aiConfidence,
    input.aiSource,
    input.qualityHint,
  );
  const created = one<Listing>(db, 'SELECT * FROM listings WHERE id = ?', res.lastInsertRowid);
  if (!created) throw new Error('상품 등록에 실패했습니다.');
  return created;
}

export interface StoreFilter {
  productCode?: string;
  region?: string;
  limit?: number;
}

export function listStoreListings(db: Db, filter: StoreFilter = {}): ListingView[] {
  const where: string[] = ["l.status = 'active'", 'l.remaining_quantity > 0'];
  const params: Array<string | number> = [];
  if (filter.productCode) {
    where.push('p.code = ?');
    params.push(filter.productCode);
  }
  if (filter.region) {
    where.push('(f.region_sigungu LIKE ? OR f.region_sido LIKE ?)');
    params.push(`%${filter.region}%`, `%${filter.region}%`);
  }
  // 문자열 보간으로 들어가는 값이라 정수임을 여기서도 한 번 더 보장한다.
  const requested = Number.isFinite(filter.limit) ? Math.trunc(filter.limit as number) : 60;
  const limit = Math.min(Math.max(requested, 1), 200);
  return all<ListingView>(
    db,
    `${VIEW_SELECT} WHERE ${where.join(' AND ')} ORDER BY l.created_at DESC, l.id DESC LIMIT ${limit}`,
    ...params,
  );
}

export function getListingView(db: Db, id: number): ListingView | null {
  return one<ListingView>(db, `${VIEW_SELECT} WHERE l.id = ?`, id);
}

export function listByFarmer(db: Db, farmerId: number): ListingView[] {
  return all<ListingView>(
    db,
    `${VIEW_SELECT} WHERE l.farmer_id = ? ORDER BY l.created_at DESC, l.id DESC`,
    farmerId,
  );
}

export function setFarmerListingStatus(
  db: Db,
  listingId: number,
  farmerId: number,
  status: 'active' | 'closed',
): boolean {
  const current = status === 'active' ? 'closed' : 'active';
  const hasStock = status === 'active' ? 'AND remaining_quantity > 0' : '';
  return run(
    db,
    `UPDATE listings SET status = ?
      WHERE id = ? AND farmer_id = ? AND status = ?
        AND inspection_status = 'ai_checked'
        AND NOT EXISTS (
          SELECT 1 FROM hub_inspections hi
           WHERE hi.listing_id = listings.id AND hi.result = 'reject'
        ) ${hasStock}`,
    status,
    listingId,
    farmerId,
    current,
  ).changes === 1;
}

export function addListingInventory(
  db: Db,
  listingId: number,
  farmerId: number,
  quantity: number,
): boolean {
  if (!Number.isInteger(quantity) || quantity <= 0) return false;
  return run(
    db,
    `UPDATE listings
        SET quantity = quantity + ?,
            remaining_quantity = remaining_quantity + ?,
            status = CASE WHEN status = 'sold_out' THEN 'active' ELSE status END
      WHERE id = ? AND farmer_id = ? AND inspection_status = 'ai_checked'
        AND NOT EXISTS (
          SELECT 1 FROM hub_inspections hi
           WHERE hi.listing_id = listings.id AND hi.result = 'reject'
        )
        AND quantity + ? <= 999`,
    quantity,
    quantity,
    listingId,
    farmerId,
    quantity,
  ).changes === 1;
}

/** @param hubId null 이면 전체(관리자). 담당자는 자기 거점 물량만 본다. */
export function listForHub(db: Db, hubId: number | null = null): ListingView[] {
  const scope = hubId === null ? '' : 'AND f.hub_id = ?';
  const params = hubId === null ? [] : [hubId];
  return all<ListingView>(
    db,
    `${VIEW_SELECT} WHERE l.status != 'closed' ${scope} ORDER BY
       CASE l.inspection_status
         WHEN 'hub_pending' THEN 0 WHEN 'ai_checked' THEN 1
         WHEN 'hub_passed' THEN 2 WHEN 'ready_to_ship' THEN 3 ELSE 4 END,
       l.created_at DESC`,
    ...params,
  );
}

/** 이 매물이 해당 거점 소관인지. */
export function belongsToHub(db: Db, listingId: number, hubId: number): boolean {
  const row = one<{ hub_id: number | null }>(
    db,
    'SELECT f.hub_id FROM listings l JOIN farms f ON f.id = l.farm_id WHERE l.id = ?',
    listingId,
  );
  return row?.hub_id === hubId;
}

/**
 * 재고 차감. 남은 수량이 부족하면 아무것도 바꾸지 않고 false 를 돌려준다.
 * (조건을 UPDATE 문 안에 두어 동시 주문에서도 초과판매가 생기지 않는다)
 */
export function decrementInventory(db: Db, listingId: number, quantity: number): boolean {
  if (!Number.isInteger(quantity) || quantity <= 0) return false;
  const res = run(
    db,
    `UPDATE listings
        SET remaining_quantity = remaining_quantity - ?,
            status = CASE WHEN remaining_quantity - ? <= 0 THEN 'sold_out' ELSE status END
      WHERE id = ? AND status = 'active' AND remaining_quantity >= ?`,
    quantity,
    quantity,
    listingId,
    quantity,
  );
  return res.changes === 1;
}

export function setInspectionStatus(db: Db, listingId: number, status: InspectionStatus): void {
  run(db, 'UPDATE listings SET inspection_status = ? WHERE id = ?', status, listingId);
}

/**
 * 거점 실물 검수로 확정된 등급을 기록한다.
 * AI 참고값(quality_hint)은 그대로 남긴다 — 둘을 같은 칸에 쓰면
 * 소비자 화면에서 '확정 등급'이 'AI 참고'로 둔갑한다.
 */
export function setConfirmedQuality(db: Db, listingId: number, grade: string): void {
  run(db, 'UPDATE listings SET confirmed_quality = ? WHERE id = ?', grade, listingId);
}
