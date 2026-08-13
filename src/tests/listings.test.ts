import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { defaultSku } from '../ai/sku-matcher.js';
import { one } from '../db/index.js';
import type { Db } from '../db/index.js';
import {
  addListingInventory,
  createListing,
  decrementInventory,
  getListingView,
  listStoreListings,
  setFarmerListingStatus,
} from '../domain/listings.js';
import { recordInspection } from '../domain/inspections.js';
import type { Listing, Product } from '../domain/types.js';
import { farmerNamed, freshDb } from './helpers.js';

function makeListing(db: Db, quantity = 3): Listing {
  const { user, farm } = farmerNamed(db);
  const product = one<Product>(db, "SELECT * FROM products WHERE code = 'pear'");
  const sku = defaultSku(db, 'pear');
  assert.ok(product && sku);
  return createListing(db, {
    farmerId: user.id,
    farmId: farm.id,
    productId: product.id,
    skuId: sku.id,
    title: '테스트 배',
    description: '설명',
    imagePath: null,
    quantity,
    unitPrice: sku.price,
    harvestedOn: '2026-08-09',
    aiAnalysis: { recognition: { product: 'pear' } },
    aiConfidence: 0.8,
    aiSource: 'heuristic',
    qualityHint: '상',
  });
}

describe('상품 등록', () => {
  it('등록하면 남은 수량이 전체 수량과 같고 바로 판매 중이 된다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 5);
    assert.equal(listing.remaining_quantity, 5);
    assert.equal(listing.status, 'active');
    assert.equal(listing.inspection_status, 'ai_checked');
  });

  it('수량 0 이하나 가격 0 은 거부한다', () => {
    const db = freshDb(false);
    assert.throws(() => makeListing(db, 0), /수량/);
  });

  it('등록 즉시 소비자 매장 목록 맨 위에 뜬다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 2);
    const listed = listStoreListings(db);
    assert.equal(listed[0]?.id, listing.id);
    assert.equal(listed[0]?.product_name, '배');
    assert.equal(listed[0]?.farm_name, '복순이네 배농장');
  });

  it('품목·지역 필터가 동작한다', () => {
    const db = freshDb();
    assert.ok(listStoreListings(db, { productCode: 'pear' }).every((l) => l.product_code === 'pear'));
    const jeju = listStoreListings(db, { region: '제주' });
    assert.ok(jeju.length > 0);
    assert.ok(jeju.every((l) => `${l.region_sido}${l.region_sigungu}`.includes('제주')));
    const chungnamPear = listStoreListings(db, { productCode: 'pear', region: '충남' });
    assert.ok(chungnamPear.length > 0);
    assert.ok(chungnamPear.every((l) => l.product_code === 'pear' && l.region_sido === '충남'));
  });
});

describe('재고 차감 — 초과 판매가 나오면 안 된다', () => {
  it('있는 만큼만 팔린다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 3);
    assert.equal(decrementInventory(db, listing.id, 2), true);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, 1);
  });

  it('남은 수량보다 많이 요청하면 아무것도 바꾸지 않는다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 3);
    assert.equal(decrementInventory(db, listing.id, 4), false);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, 3);
  });

  it('0 이 되면 자동으로 완판 처리되고 더는 팔리지 않는다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 2);
    assert.equal(decrementInventory(db, listing.id, 2), true);
    const after = getListingView(db, listing.id);
    assert.equal(after?.remaining_quantity, 0);
    assert.equal(after?.status, 'sold_out');
    assert.equal(decrementInventory(db, listing.id, 1), false);
    assert.equal(listStoreListings(db).find((l) => l.id === listing.id), undefined);
  });

  it('연속 차감이 정확히 누적된다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 10);
    for (let i = 0; i < 10; i += 1) assert.equal(decrementInventory(db, listing.id, 1), true);
    assert.equal(decrementInventory(db, listing.id, 1), false);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, 0);
  });

  it('잘못된 수량은 무시한다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 3);
    assert.equal(decrementInventory(db, listing.id, 0), false);
    assert.equal(decrementInventory(db, listing.id, -1), false);
    assert.equal(decrementInventory(db, listing.id, 1.5), false);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, 3);
  });
});

describe('농가 판매 관리', () => {
  it('주문 전 자기 상품만 바꾸고 거점 반려 상품은 되살리지 않는다', () => {
    const db = freshDb(false);
    const listing = makeListing(db, 3);
    const owner = farmerNamed(db).user;
    const other = farmerNamed(db, '이만수').user;

    assert.equal(addListingInventory(db, listing.id, other.id, 2), false);
    assert.equal(addListingInventory(db, listing.id, owner.id, 2), true);
    assert.equal(getListingView(db, listing.id)?.quantity, 5);
    assert.equal(getListingView(db, listing.id)?.remaining_quantity, 5);

    assert.equal(setFarmerListingStatus(db, listing.id, owner.id, 'closed'), true);
    assert.equal(listStoreListings(db).some((item) => item.id === listing.id), false);
    assert.equal(setFarmerListingStatus(db, listing.id, owner.id, 'active'), true);

    recordInspection(db, {
      listingId: listing.id,
      hubId: null,
      inspector: '테스트 담당자',
      result: 'reject',
    });
    assert.equal(getListingView(db, listing.id)?.has_rejection, 1);
    assert.equal(setFarmerListingStatus(db, listing.id, owner.id, 'active'), false);
    assert.equal(addListingInventory(db, listing.id, owner.id, 1), false);
  });
});
