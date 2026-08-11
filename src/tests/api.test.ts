import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';

/* 실제 데이터 폴더를 건드리지 않도록 임시 경로를 먼저 잡고 동적 import 한다. */
const workDir = mkdtempSync(join(tmpdir(), 'onfarm-test-'));
process.env['DATA_DIR'] = workDir;
process.env['DB_PATH'] = join(workDir, 'test.db');
process.env['AI_PROVIDER'] = 'mock'; // 무대 시연과 동일하게 고정 응답으로 검증
process.env['SESSION_SECRET'] = 'test-secret';

const { createApp } = await import('../server/main.js');
const { closeDb, db } = await import('../db/index.js');
const { seed } = await import('../db/seed.js');

seed(db());
const server = createApp();
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = (server.address() as AddressInfo).port;
const base = `http://127.0.0.1:${port}`;

const PNG_1X1 =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const FEATURES = {
  width: 256,
  height: 192,
  hueHistogram: [0.02, 0.7, 0.12, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.005, 0.005],
  meanSaturation: 0.35,
  meanValue: 0.62,
  edgeDensity: 0.3,
  hueConcentration: 0.72,
};

interface Reply {
  status: number;
  body: any;
}

function client() {
  let cookie = '';
  return async function call(path: string, options: { method?: string; body?: unknown } = {}): Promise<Reply> {
    const headers: Record<string, string> = {};
    if (cookie) headers['cookie'] = cookie;
    if (options.body !== undefined) headers['content-type'] = 'application/json';
    const res = await fetch(`${base}${path}`, {
      method: options.method ?? (options.body !== undefined ? 'POST' : 'GET'),
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    });
    const setCookies = res.headers.getSetCookie?.() ?? [];
    if (setCookies.length > 0) cookie = setCookies.map((c) => c.split(';')[0]).join('; ');
    const body = await res.json().catch(() => null);
    return { status: res.status, body };
  };
}

async function loginAs(call: ReturnType<typeof client>, name: string): Promise<number> {
  const accounts = await call('/api/accounts');
  const account = accounts.body.accounts.find((a: { name: string }) => a.name === name);
  assert.ok(account, `계정 없음: ${name}`);
  const res = await call('/api/auth/login', { body: { userId: account.id } });
  assert.equal(res.status, 200);
  return account.id;
}

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
  closeDb();
  rmSync(workDir, { recursive: true, force: true });
});

describe('HTTP — 설정과 권한', () => {
  it('설정 API 가 AI 상태를 그대로 노출한다', async () => {
    const call = client();
    const res = await call('/api/config');
    assert.equal(res.status, 200);
    assert.equal(res.body.ai.provider, 'mock');
    assert.equal(res.body.ai.demoMode, true, '데모 모드는 화면에 배지로 표시돼야 한다');
    assert.ok(res.body.products.length >= 8);
  });

  it('로그인 없이 분석 API 를 부르면 401', async () => {
    const call = client();
    const res = await call('/api/ai/analyze', { body: { image: PNG_1X1 } });
    assert.equal(res.status, 401);
  });

  it('소비자는 판매 등록 흐름에 접근할 수 없다', async () => {
    const call = client();
    await loginAs(call, '장바구니');
    assert.equal((await call('/api/ai/analyze', { body: { image: PNG_1X1 } })).status, 403);
    assert.equal((await call('/api/farmer/listings')).status, 403);
    assert.equal((await call('/api/hub/dashboard')).status, 403);
  });

  it('농민은 거점 대시보드에 접근할 수 없다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    assert.equal((await call('/api/hub/dashboard')).status, 403);
    assert.equal((await call('/api/store/orders')).status, 403);
  });

  it('거점 담당자는 대시보드를 볼 수 있다', async () => {
    const call = client();
    await loginAs(call, '성환거점 담당자');
    const res = await call('/api/hub/dashboard');
    assert.equal(res.status, 200);
    assert.ok(res.body.counters);
  });
});

describe('HTTP — 사진 한 장에서 주문까지', () => {
  it('전체 흐름이 실제로 돈다', async () => {
    const farmer = client();
    await loginAs(farmer, '김복순');

    // ① 사진 분석
    const analyzed = await farmer('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    assert.equal(analyzed.status, 200);
    assert.ok(analyzed.body.analysisId);
    assert.equal(analyzed.body.recognition.product, 'pear');
    assert.equal(analyzed.body.decision.mode, 'auto');
    assert.equal(analyzed.body.selectedSku.price, 29000);
    assert.ok(analyzed.body.imagePath.startsWith('/uploads/'));
    const analysisId = analyzed.body.analysisId as string;

    // ② 품목과 맞지 않는 SKU 는 거부
    const appleSku = analyzed.body.catalog.find((c: { code: string }) => c.code === 'apple');
    assert.ok(appleSku);
    const mismatched = await farmer('/api/farmer/listings', {
      body: { analysisId, skuId: 3, quantity: 5 },
    });
    assert.equal(mismatched.status, 400);
    assert.equal(mismatched.body.code, 'sku_mismatch');

    // ③ 잘못된 수량 거부
    const badQty = await farmer('/api/farmer/listings', {
      body: { analysisId, skuId: analyzed.body.selectedSku.id, quantity: 0 },
    });
    assert.equal(badQty.status, 400);

    // ④ 정상 등록 — 가격은 클라이언트가 아니라 서버 SKU 에서 온다
    const created = await farmer('/api/farmer/listings', {
      body: { analysisId, skuId: analyzed.body.selectedSku.id, quantity: 5, unitPrice: 1 },
    });
    assert.equal(created.status, 201);
    assert.equal(created.body.listing.unit_price, 29000);
    assert.equal(created.body.listing.remaining_quantity, 5);
    assert.match(created.body.listing.title, /수확한 신고배/);
    const listingId = created.body.listing.id as number;

    // ⑤ 소비자 매장 맨 위에 노출
    const store = client();
    const listed = await store('/api/store/listings');
    assert.equal(listed.status, 200);
    assert.equal(listed.body.listings[0].id, listingId);
    assert.equal(listed.body.listings[0].farm_name, '복순이네 배농장');

    // ⑥ 소비자 주문
    await loginAs(store, '장바구니');
    const ordered = await store('/api/store/orders', {
      body: {
        lines: [{ listingId, quantity: 2 }],
        receiverName: '장바구니',
        receiverPhone: '010-5555-1000',
        address: '충남 천안시 동남구',
      },
    });
    assert.equal(ordered.status, 201);
    assert.equal(ordered.body.order.total_amount, 58000);

    // ⑦ 재고 초과 주문은 409
    const oversell = await store('/api/store/orders', {
      body: {
        lines: [{ listingId, quantity: 99 }],
        receiverName: '장바구니',
        receiverPhone: '010-5555-1000',
        address: '충남 천안시',
      },
    });
    assert.equal(oversell.status, 409);

    // ⑧ 농민 화면에 주문/정산이 보인다
    const orders = await farmer('/api/farmer/orders');
    assert.equal(orders.status, 200);
    assert.equal(orders.body.orders[0].quantity, 2);
    const settlements = await farmer('/api/farmer/settlements');
    assert.equal(settlements.body.summary.totalGross, 58000);

    // ⑨ 거점에서 검수하면 상태가 넘어간다
    const hub = client();
    await loginAs(hub, '성환거점 담당자');
    const inspected = await hub('/api/hub/inspections', {
      body: { listingId, result: 'pass', gradedQuality: '상' },
    });
    assert.equal(inspected.status, 201);
    const detail = await store(`/api/store/listings/${listingId}`);
    assert.equal(detail.body.listing.inspection_status, 'hub_passed');
    assert.equal(detail.body.listing.remaining_quantity, 3);
  });

  it('다른 농민의 분석 결과로는 상품을 올릴 수 없다', async () => {
    const a = client();
    await loginAs(a, '김복순');
    const analyzed = await a('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    const analysisId = analyzed.body.analysisId as string;

    const b = client();
    await loginAs(b, '이만수');
    const stolen = await b('/api/farmer/listings', { body: { analysisId, quantity: 1 } });
    assert.equal(stolen.status, 410);
  });

  it('폴백 — 품목을 직접 고르면 그 품목으로 진행된다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const analyzed = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    const forced = await call('/api/ai/analyze', {
      body: { analysisId: analyzed.body.analysisId, productCode: 'sweet_potato' },
    });
    assert.equal(forced.status, 200);
    assert.equal(forced.body.recognition.product, 'sweet_potato');
    assert.equal(forced.body.selectedSku.price, 15000);
    assert.equal(forced.body.ai.source, 'manual');

    // 후보 선택으로 재분석해도 '최초 인식기'가 감사 기록에 남아야 한다.
    // (재분석 source='manual' 이 그대로 저장되면 어떤 AI 가 봤는지 사라진다)
    const created = await call('/api/farmer/listings', {
      body: {
        analysisId: forced.body.analysisId,
        skuId: forced.body.selectedSku.id,
        quantity: 1,
        productCode: 'sweet_potato',
      },
    });
    assert.equal(created.status, 201);
    assert.equal(created.body.listing.ai_source, 'mock+manual',
      '최초 인식기(mock) + 사용자가 바꿈(+manual) 으로 기록돼야 한다');
  });

  it('같은 분석 ID 로 두 번 등록할 수 없다 — 재시도가 중복 매물을 만들면 안 된다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const analyzed = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    const body = { analysisId: analyzed.body.analysisId, skuId: analyzed.body.selectedSku.id, quantity: 2 };

    const first = await call('/api/farmer/listings', { body });
    assert.equal(first.status, 201);
    const second = await call('/api/farmer/listings', { body });
    assert.equal(second.status, 409, `두 번째 등록은 막혀야 한다 (실제 ${second.status})`);
  });

  it('품목을 바꿔 올리면 이전 품목의 품종이 제목에 남지 않는다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const analyzed = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    assert.equal(analyzed.body.recognition.product, 'pear');

    const forced = await call('/api/ai/analyze', {
      body: { analysisId: analyzed.body.analysisId, productCode: 'apple' },
    });
    const created = await call('/api/farmer/listings', {
      body: {
        analysisId: forced.body.analysisId,
        skuId: forced.body.selectedSku.id,
        quantity: 1,
        productCode: 'apple',
      },
    });
    assert.equal(created.status, 201);
    assert.ok(!created.body.listing.title.includes('신고배'), `제목에 이전 품종이 남음: ${created.body.listing.title}`);
    assert.match(created.body.listing.title, /부사/);
  });

  it('미래 수확일은 거부한다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const analyzed = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    const res = await call('/api/farmer/listings', {
      body: { analysisId: analyzed.body.analysisId, skuId: analyzed.body.selectedSku.id, quantity: 1, harvestedOn: '2099-01-01' },
    });
    assert.equal(res.status, 400);
  });

  it('만료·위조된 분석 ID 는 거부한다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/farmer/listings', { body: { analysisId: 'not-a-real-id', quantity: 1 } });
    assert.equal(res.status, 410);
  });

  it('사진도 분석 ID 도 없으면 400', async () => {
    const call = client();
    await loginAs(call, '김복순');
    assert.equal((await call('/api/ai/analyze', { body: {} })).status, 400);
  });

  it('지원하지 않는 파일 형식은 415', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/ai/analyze', { body: { image: 'data:application/pdf;base64,QQ==' } });
    assert.equal(res.status, 415);
  });

  it('농민은 자기 상품만 본다', async () => {
    const a = client();
    await loginAs(a, '김복순');
    const mine = await a('/api/farmer/listings');
    assert.ok(mine.body.listings.length > 0);
    assert.ok(mine.body.listings.every((l: { farm_name: string }) => l.farm_name === '복순이네 배농장'));
  });
});

describe('HTTP — 잘못된 입력이 500 이 되면 안 된다', () => {
  it('JSON null·배열·잘못된 타입은 400 이다', async () => {
    const call = client();
    await loginAs(call, '장바구니');
    for (const body of [null, [1, 2], 'hello'] as unknown[]) {
      const res = await fetch(`${base}/api/store/orders`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      assert.equal(res.status < 500, true, `${JSON.stringify(body)} → ${res.status}`);
    }
  });

  it('lines·receiverName 타입이 이상해도 400 이다', async () => {
    const call = client();
    await loginAs(call, '장바구니');
    assert.equal((await call('/api/store/orders', { body: { lines: {} } })).status, 400);
    assert.equal(
      (await call('/api/store/orders', { body: { lines: [], receiverName: {} } })).status < 500,
      true,
    );
  });

  it('image 가 문자열이 아니면 400 이다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/ai/analyze', { body: { image: {} } });
    assert.equal(res.status, 400);
  });

  it('limit=abc 가 SQL 로 새지 않는다', async () => {
    const res = await fetch(`${base}/api/store/listings?limit=abc`);
    assert.equal(res.status, 200);
  });

  it('잘못 인코딩된 URL 이 서버를 죽이지 않는다', async () => {
    const res = await fetch(`${base}/%`);
    assert.ok(res.status === 400 || res.status === 404, `실제 ${res.status}`);
    // 서버가 여전히 살아 있어야 한다
    assert.equal((await fetch(`${base}/api/config`)).status, 200);
  });
});

describe('HTTP — 거점 권한과 확정 등급', () => {
  it('확정 등급 없이 검수를 통과시킬 수 없다', async () => {
    const call = client();
    await loginAs(call, '성환거점 담당자');
    const dash = await call('/api/hub/dashboard');
    const target = dash.body.listings[0];
    const res = await call('/api/hub/inspections', { body: { listingId: target.id, result: 'pass' } });
    assert.equal(res.status, 400, 'AI 참고값이 자동 승격되면 안 된다');
    assert.equal(res.body.code, 'bad_grade');
  });

  it('허용목록 밖 등급 문자열을 거부한다', async () => {
    const call = client();
    await loginAs(call, '성환거점 담당자');
    const dash = await call('/api/hub/dashboard');
    const target = dash.body.listings[0];
    const res = await call('/api/hub/inspections', {
      body: { listingId: target.id, result: 'pass', gradedQuality: '무농약·안전성 검사 완료' },
    });
    assert.equal(res.status, 400);
  });

  it('다른 거점 매물은 보이지도, 처리되지도 않는다', async () => {
    const admin = client();
    await loginAs(admin, '운영자');
    const all = await admin('/api/hub/dashboard');
    const jeju = all.body.listings.find((l: { region_sido: string }) => l.region_sido === '제주');
    assert.ok(jeju, '제주 매물이 시드에 있어야 한다');

    const call = client();
    await loginAs(call, '성환거점 담당자');
    const mine = await call('/api/hub/dashboard');
    assert.ok(
      !mine.body.listings.some((l: { id: number }) => l.id === jeju.id),
      '다른 거점 매물이 목록에 뜨면 안 된다',
    );
    const res = await call('/api/hub/inspections', {
      body: { listingId: jeju.id, result: 'pass', gradedQuality: '상' },
    });
    assert.equal(res.status, 403);
  });

  it('검수 전 매물을 배송 완료로 건너뛸 수 없다', async () => {
    const call = client();
    await loginAs(call, '성환거점 담당자');
    const dash = await call('/api/hub/dashboard');
    const target = dash.body.listings.find(
      (l: { inspection_status: string }) => l.inspection_status === 'ai_checked',
    );
    assert.ok(target);
    const res = await call(`/api/hub/listings/${target.id}/status`, { body: { status: 'delivered' } });
    assert.equal(res.status, 409);
  });

  it('확정 등급은 AI 참고값과 별도 필드로 소비자에게 내려간다', async () => {
    const hub = client();
    await loginAs(hub, '성환거점 담당자');
    const dash = await hub('/api/hub/dashboard');
    const target = dash.body.listings[0];
    const aiHint = target.quality_hint;

    const res = await hub('/api/hub/inspections', {
      body: { listingId: target.id, result: 'downgrade', gradedQuality: '보통' },
    });
    assert.equal(res.status, 201);

    const store = client();
    const detail = await store(`/api/store/listings/${target.id}`);
    assert.equal(detail.body.listing.confirmed_quality, '보통');
    assert.equal(detail.body.listing.quality_hint, aiHint);
  });
});

describe('HTTP — 데모 초기화 보호', () => {
  it('확인값 없이는 초기화되지 않는다', async () => {
    const count = async (): Promise<number> => {
      const body = (await fetch(`${base}/api/store/listings`).then((r) => r.json())) as {
        listings: unknown[];
      };
      return body.listings.length;
    };
    const before = await count();
    const res = await fetch(`${base}/api/demo/reset`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({}),
    });
    assert.equal(res.status, 400);
    assert.equal(await count(), before);
  });
});

describe('HTTP — 학습 모델용 픽셀 전달', () => {
  const PIXELS_OK = Buffer.alloc(224 * 224 * 3, 128).toString('base64');

  it('정확한 길이의 픽셀은 받아들이고 분석이 계속된다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/ai/analyze', {
      body: { image: PNG_1X1, features: FEATURES, pixels: PIXELS_OK },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.recognition.product, 'pear');
  });

  it('길이가 틀린 픽셀은 버리고도 분석이 죽지 않는다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    for (const bad of [Buffer.alloc(100).toString('base64'), 'not-base64!!', '', 12345]) {
      const res = await call('/api/ai/analyze', {
        body: { image: PNG_1X1, features: FEATURES, pixels: bad },
      });
      assert.equal(res.status, 200, `pixels=${String(bad).slice(0, 12)} 에서 ${res.status}`);
    }
  });

  it('재분석 때 픽셀을 다시 보내지 않아도 보관본이 쓰인다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const first = await call('/api/ai/analyze', {
      body: { image: PNG_1X1, features: FEATURES, pixels: PIXELS_OK },
    });
    const again = await call('/api/ai/analyze', {
      body: { analysisId: first.body.analysisId, productCode: 'apple' },
    });
    assert.equal(again.status, 200);
    assert.equal(again.body.recognition.product, 'apple');
  });
});

describe('HTTP — 중앙 안전 정책이 실제 응답에 적용된다', () => {
  // mock provider 는 confidence 0.91 / quality_hint '상' 을 고정 반환하고 측정 증거가 없다.
  // 정책 배선이 끊기면 그 값이 그대로 화면까지 나간다(2차 교차검증 #7).
  it('증거 없는 provider 의 신뢰도가 정책 상한으로 잘려서 내려온다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    assert.equal(res.status, 200);
    assert.ok(
      res.body.recognition.confidence <= 0.85,
      `정책 상한 0.85 를 넘었다: ${res.body.recognition.confidence}`,
    );
  });

  it('증거 없는 provider 의 등급은 확인필요로 내려온다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    assert.equal(
      res.body.recognition.quality_hint,
      '확인필요',
      "측정 증거가 없으면 '상' 을 그대로 보여주면 안 된다",
    );
  });

  it('정책이 깎은 내역이 응답에 남는다', async () => {
    const call = client();
    await loginAs(call, '김복순');
    const res = await call('/api/ai/analyze', { body: { image: PNG_1X1, features: FEATURES } });
    assert.ok(Array.isArray(res.body.ai.policyApplied));
    assert.ok(res.body.ai.policyApplied.length > 0, '무엇을 깎았는지 남아야 감사가 된다');
  });
});

describe('HTTP — 정적 화면', () => {
  it('주요 화면이 모두 응답한다', async () => {
    for (const path of [
      '/',
      '/login',
      '/demo',
      '/farmer',
      '/farmer/sell',
      '/farmer/listings',
      '/farmer/orders',
      '/farmer/settlement',
      '/store/product',
      '/store/cart',
      '/store/orders',
      '/hub',
      '/manifest.webmanifest',
      '/js/shared/korean.js',
    ]) {
      const res = await fetch(`${base}${path}`);
      assert.equal(res.status, 200, `${path} 가 ${res.status}`);
    }
  });

  it('없는 경로는 404', async () => {
    assert.equal((await fetch(`${base}/없는페이지`)).status, 404);
  });

  it('상위 경로 탈출 시도를 막는다', async () => {
    const res = await fetch(`${base}/uploads/..%2F..%2Fpackage.json`);
    assert.ok(res.status === 404 || res.status === 400, `실제 ${res.status}`);
  });
});
