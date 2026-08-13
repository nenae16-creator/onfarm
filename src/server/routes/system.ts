import { readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { config, uploadsDir } from '../../config.js';
import { resolveProvider } from '../../ai/providers/index.js';
import { catalog } from '../../ai/sku-matcher.js';
import { db, tx } from '../../db/index.js';
import { seed } from '../../db/seed.js';
import { listPayableSettlements, markSettlementPaid } from '../../domain/settlements.js';
import { HttpError } from '../../lib/http.js';
import type { Router } from '../../lib/http.js';
import { requireRole } from '../../lib/session.js';
import { clearAnalyses } from '../analysis-store.js';

export const BRAND = {
  name: 'ON-FARM',
  nameKo: '온팜',
  tagline: '사진 찍고, 수량 확인하면 끝.',
} as const;

/** 초기화가 비우는 표. 마스터(products·skus·users·farms·hubs)는 seed 가 다시 채운다. */
const RESET_TABLES = [
  'settlements',
  'refund_help_requests',
  'order_items',
  'orders',
  'hub_inspections',
  'listings',
  'farms',
  'users',
  'skus',
  'products',
  'hubs',
];

export function registerSystemRoutes(router: Router): void {
  router.get('/api/config', (ctx) => {
    const provider = resolveProvider();
    ctx.json({
      brand: BRAND,
      ai: {
        provider: provider.primary.name,
        offline: provider.primary.offline,
        demoMode: provider.demoMode,
        degradedReason: provider.degradedReason,
        configured: config.ai.provider,
      },
      products: catalog(db()).map((p) => ({
        code: p.code,
        name: p.name_ko,
        emoji: p.emoji,
        category: p.category,
      })),
    });
  });

  router.get('/api/admin/settlements', (ctx) => {
    requireRole(ctx.user, 'admin');
    ctx.json({ settlements: listPayableSettlements(db()) });
  });

  router.post('/api/admin/settlements/:id/pay', async (ctx) => {
    requireRole(ctx.user, 'admin');
    const id = Number(ctx.params['id']);
    if (!Number.isInteger(id)) throw new HttpError(400, '잘못된 정산입니다.', 'bad_request');
    const body = await ctx.body<{ reference?: unknown }>();
    const reference = typeof body.reference === 'string' ? body.reference.trim() : '';
    if (!reference || reference.length > 80 || /[\u0000-\u001f\u007f]/.test(reference)) {
      throw new HttpError(400, '지급 참조 번호를 입력해주세요.', 'invalid_reference');
    }
    const result = markSettlementPaid(db(), id, reference);
    if (result === 'not_found') throw new HttpError(404, '정산을 찾을 수 없습니다.', 'not_found');
    if (result === 'not_ready') throw new HttpError(409, '배송 완료 후 지급할 수 있습니다.', 'not_delivered');
    if (result === 'rejected') throw new HttpError(409, '반려된 상품은 지급할 수 없습니다.', 'rejected');
    if (result === 'conflict') throw new HttpError(409, '이미 다른 참조 번호로 지급 기록되었습니다.', 'payment_conflict');
    ctx.json({ paid: true, alreadyPaid: result === 'already_paid' });
  });

  /**
   * 시연 직전에 데이터를 초기 상태로 되돌린다.
   *
   * 되돌릴 수 없는 동작이라 두 겹으로 막는다.
   *  - loopback 요청만 허용 (원격에서 남의 데모를 지울 수 없게)
   *  - 본문에 confirm 문자열 요구 (다른 사이트가 자동 제출하는 폼 한 방에 지워지지 않게)
   * DB 파일을 지우지 않고 표만 비운다 — 진행 중인 요청이 닫힌 핸들을 잡는 사고를 없애기 위함.
   */
  router.post('/api/demo/reset', async (ctx) => {
    const remote = ctx.req.socket.remoteAddress ?? '';
    const isLocal = remote === '127.0.0.1' || remote === '::1' || remote === '::ffff:127.0.0.1';
    if (!isLocal) {
      throw new HttpError(403, '데모 초기화는 로컬에서만 가능합니다.', 'forbidden');
    }

    const body = await ctx.body<{ confirm?: string }>();
    if (body.confirm !== 'RESET') {
      throw new HttpError(400, '초기화 확인값이 필요합니다.', 'confirm_required');
    }

    const handle = db();
    tx(handle, () => {
      for (const table of RESET_TABLES) handle.exec(`DELETE FROM ${table}`);
      handle.exec("DELETE FROM sqlite_sequence WHERE name IN ('listings','orders','order_items')");
    });

    clearAnalyses();

    // 업로드된 사진도 함께 정리한다(안 지우면 디스크에 계속 쌓인다).
    try {
      for (const file of readdirSync(uploadsDir())) {
        rmSync(join(uploadsDir(), file), { force: true });
      }
    } catch {
      /* 업로드 폴더가 아직 없으면 무시 */
    }

    seed(handle);
    ctx.json({ ok: true, message: '데모 데이터를 초기화했습니다.' });
  });
}
