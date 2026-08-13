/**
 * 클레임 안전 검사 — 제출물 전체에서 금지 표현이 되살아나지 못하게 막는다.
 *
 * 심사 관점에서 두 종류의 표현이 위험하다.
 *  1) 과제4 밖으로 밀어내는 말 ("직거래 플랫폼", "쇼핑몰") → 주제 적합성 감점
 *  2) 근거 없는 단정 ("전국 최초", "AI가 등급을 확정") → 발표장에서 반박당함
 *
 * 근거: docs/POSITIONING.md 의 금지/대체 표현표.
 * 사람이 문서를 고칠 때마다 잊기 쉬워 테스트로 고정한다.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));

/** 검사 대상: 심사위원·사용자가 실제로 보는 것들 */
const TARGET_DIRS = ['public'];
const TARGET_FILES = ['README.md'];
const TARGET_EXT = ['.html', '.md'];

/** 금지 표현과 그 이유. docs/POSITIONING.md 와 같이 관리한다. */
const FORBIDDEN: Array<{ pattern: RegExp; why: string; allow?: RegExp }> = [
  { pattern: /전국\s*최초/, why: '근거 없는 최초 주장' },
  { pattern: /중간\s*유통\s*마진|유통마진\s*절감/, why: '역대 수상작·기존 사업과 겹치는 표현' },
  { pattern: /쇼핑몰/, why: '새 쇼핑몰 구축으로 읽히면 시 정책과 경쟁 구도가 된다' },
  {
    pattern: /AI가?\s*(품질\s*)?등급을?\s*(확정|판정)한다/,
    why: 'AI는 참고 판정만 한다 — 확정은 거점 실물 검수',
  },
  { pattern: /안전성을?\s*검사한다/, why: '식품 안전성은 검사하지 않는다' },
  { pattern: /적정\s*가격을?\s*(결정|산출)/, why: '가격은 운영자가 등록한 표준 SKU 가 유일 출처' },
];

function collectFiles(): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) {
        walk(full);
      } else if (TARGET_EXT.some((e) => full.endsWith(e))) {
        out.push(full);
      }
    }
  };
  for (const d of TARGET_DIRS) walk(join(ROOT, d));
  for (const f of TARGET_FILES) out.push(join(ROOT, f));
  return out;
}

describe('클레임 안전 — 금지 표현', () => {
  const files = collectFiles();

  it('검사 대상 파일이 실제로 잡힌다', () => {
    // 대상이 0개면 아래 검사가 전부 무의미하게 통과한다
    assert.ok(files.length >= 10, `검사 대상이 ${files.length}개뿐 — 경로를 확인하라`);
    assert.ok(files.some((f) => f.endsWith('README.md')));
    assert.ok(files.some((f) => f.includes('farmer')));
  });

  for (const { pattern, why } of FORBIDDEN) {
    it(`"${pattern.source}" 가 없다 — ${why}`, () => {
      const hits: string[] = [];
      for (const file of files) {
        const text = readFileSync(file, 'utf8');
        text.split(/\r?\n/).forEach((line, i) => {
          if (pattern.test(line)) hits.push(`${file.replace(ROOT, '')}:${i + 1}  ${line.trim().slice(0, 70)}`);
        });
      }
      assert.equal(hits.length, 0, `금지 표현 발견:\n  ${hits.join('\n  ')}`);
    });
  }
});

/**
 * 문구가 아니라 **약속**을 검사한다.
 *
 * 디자인을 새로 하면 같은 뜻을 다른 말로 쓴다. 실제로 화면이 바뀌면서
 * "참고 판정" 이 "최종 품질은 농산물을 직접 보고 확정합니다" 로 바뀌었다.
 * 한 문장에 못을 박아 두면 이런 정상적인 개선까지 막고, 그러다 보면
 * 테스트를 지우게 된다 — 그러면 약속이 진짜로 사라져도 아무도 모른다.
 * 그래서 표현은 여러 갈래를 받아들이되, 약속이 없으면 반드시 깨지게 한다.
 */
function keepsPromise(html: string, alternatives: RegExp[]): boolean {
  return alternatives.some((p) => p.test(html));
}

describe('클레임 안전 — 있어야 하는 약속', () => {
  it('소비자 상세 화면: AI 가 등급·안전성을 확정하지 않는다고 밝힌다', () => {
    const html = readFileSync(join(ROOT, 'public/store/product.html'), 'utf8');
    assert.ok(
      keepsPromise(html, [
        /등급을 확정하지 않으며/,
        /등급이나 식품 안전성을 확정하지 않습니다/,
        /AI[^<]{0,40}확정하지 않습니다/,
      ]),
      'AI 가 확정하지 않는다는 고지가 사라졌다',
    );
    assert.ok(
      keepsPromise(html, [/거점 실물 검수/, /거점의 실물 확인/, /거점[^<]{0,20}실물/]),
      '확정 주체(거점 실물 확인) 표기가 사라졌다',
    );
  });

  it('농민 화면: 최종 품질은 사람이 실물로 정한다고 밝힌다', () => {
    const html = readFileSync(join(ROOT, 'public/farmer/index.html'), 'utf8');
    assert.ok(
      keepsPromise(html, [
        /참고 판정/,
        /최종 품질은[^<]{0,30}확정/,
        /품질은 거점에서 확인/,
      ]),
      'AI 판정이 참고값일 뿐이라는 표시가 사라졌다',
    );
    assert.ok(
      keepsPromise(html, [/AI가 가격을[^<]{0,20}않습니다/, /가격[^<]{0,20}미리 정한/]),
      'AI 가 가격을 정하지 않는다는 표시가 사라졌다',
    );
  });

  it('README 가 과제4 프레임(출하 손실)으로 시작한다', () => {
    const md = readFileSync(join(ROOT, 'README.md'), 'utf8');
    assert.match(md, /출하/, '출하 프레임이 빠졌다 — 유통업으로 읽힌다');
  });
});
