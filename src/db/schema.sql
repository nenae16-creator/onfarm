-- ON-FARM 스키마 (SQLite / node:sqlite)
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  role        TEXT NOT NULL CHECK (role IN ('farmer','consumer','hub_operator','admin')),
  name        TEXT NOT NULL,
  phone       TEXT,
  -- 거점 담당자가 소속된 거점. 담당자는 자기 거점 물량만 다룬다.
  hub_id      INTEGER,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 지역 집하·검수 거점 (로컬푸드 직매장/가공센터를 상정)
CREATE TABLE IF NOT EXISTS hubs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  region      TEXT NOT NULL,
  address     TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS farms (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id       INTEGER NOT NULL REFERENCES users(id),
  farm_name     TEXT NOT NULL,
  region_sido   TEXT NOT NULL,
  region_sigungu TEXT NOT NULL,
  region_detail TEXT,
  hub_id        INTEGER REFERENCES hubs(id),
  payout_bank   TEXT,
  payout_account TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 표준 품목 마스터
CREATE TABLE IF NOT EXISTS products (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  code        TEXT NOT NULL UNIQUE,           -- pear, apple, sweet_potato ...
  name_ko     TEXT NOT NULL,                  -- 배
  category    TEXT NOT NULL,                  -- fruit | vegetable | root | seafood
  variety     TEXT,                           -- 신고배
  emoji       TEXT,
  sample_image TEXT,                          -- 데모/시드용 일러스트 경로
  active      INTEGER NOT NULL DEFAULT 1
);

-- 표준 SKU(운영자가 미리 등록하는 가격/판매단위). AI 는 가격을 만들지 않는다.
CREATE TABLE IF NOT EXISTS skus (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id  INTEGER NOT NULL REFERENCES products(id),
  code        TEXT NOT NULL UNIQUE,           -- pear_shingo_5kg
  label       TEXT NOT NULL,                  -- 5kg 한 상자
  weight      REAL NOT NULL,
  unit        TEXT NOT NULL DEFAULT 'kg',
  price       INTEGER NOT NULL,               -- 원 단위 정수
  is_default  INTEGER NOT NULL DEFAULT 0,
  active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS listings (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  farmer_id          INTEGER NOT NULL REFERENCES users(id),
  farm_id            INTEGER NOT NULL REFERENCES farms(id),
  product_id         INTEGER NOT NULL REFERENCES products(id),
  sku_id             INTEGER NOT NULL REFERENCES skus(id),
  title              TEXT NOT NULL,
  description        TEXT NOT NULL,
  image_path         TEXT,
  quantity           INTEGER NOT NULL CHECK (quantity > 0),
  remaining_quantity INTEGER NOT NULL CHECK (remaining_quantity >= 0),
  unit_price         INTEGER NOT NULL,
  harvested_on       TEXT,
  ai_analysis        TEXT,                    -- JSON 원문 보관(감사·재현용)
  ai_confidence      REAL,
  ai_source          TEXT,                    -- heuristic | openai | anthropic | mock | manual
  quality_hint       TEXT,                    -- 특 | 상 | 보통 | 확인필요  (AI 참고 판정, 덮어쓰지 않는다)
  confirmed_quality  TEXT,                    -- 거점 실물 검수로 확정된 등급 (이것만이 '확정')
  inspection_status  TEXT NOT NULL DEFAULT 'ai_checked'
                     CHECK (inspection_status IN ('ai_checked','hub_pending','hub_passed','ready_to_ship','delivered')),
  status             TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active','sold_out','closed')),
  created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_listings_farmer ON listings(farmer_id, created_at DESC);

CREATE TABLE IF NOT EXISTS orders (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  consumer_id    INTEGER NOT NULL REFERENCES users(id),
  order_no       TEXT NOT NULL UNIQUE,
  total_amount   INTEGER NOT NULL,
  receiver_name  TEXT NOT NULL,
  receiver_phone TEXT NOT NULL,
  address        TEXT NOT NULL,
  memo           TEXT,
  status         TEXT NOT NULL DEFAULT 'paid'
                 CHECK (status IN ('paid','preparing','shipped','done','cancelled')),
  created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS order_items (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id    INTEGER NOT NULL REFERENCES orders(id),
  listing_id  INTEGER NOT NULL REFERENCES listings(id),
  sku_id      INTEGER NOT NULL REFERENCES skus(id),
  farmer_id   INTEGER NOT NULL REFERENCES users(id),
  unit_price  INTEGER NOT NULL,
  quantity    INTEGER NOT NULL CHECK (quantity > 0),
  amount      INTEGER NOT NULL,
  fulfillment_status TEXT NOT NULL DEFAULT 'farmer_preparing'
                    CHECK (fulfillment_status IN (
                      'farmer_preparing','ready_for_hub','hub_received',
                      'hub_passed','ready_to_ship','delivered'
                    ))
);

CREATE INDEX IF NOT EXISTS idx_order_items_farmer ON order_items(farmer_id);

CREATE TABLE IF NOT EXISTS refund_help_requests (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  order_item_id INTEGER NOT NULL UNIQUE REFERENCES order_items(id),
  consumer_id   INTEGER NOT NULL REFERENCES users(id),
  created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS hub_inspections (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  listing_id     INTEGER NOT NULL REFERENCES listings(id),
  hub_id         INTEGER REFERENCES hubs(id),
  inspector      TEXT NOT NULL,
  result         TEXT NOT NULL CHECK (result IN ('pass','downgrade','reject')),
  graded_quality TEXT,                        -- 실물 검수로 확정된 등급
  note           TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settlements (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  farmer_id     INTEGER NOT NULL REFERENCES users(id),
  order_item_id INTEGER NOT NULL REFERENCES order_items(id),
  gross         INTEGER NOT NULL,
  fee           INTEGER NOT NULL,
  net           INTEGER NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','paid')),
  due_on        TEXT,
  paid_at       TEXT,
  payment_reference TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_settlements_farmer ON settlements(farmer_id, status);
