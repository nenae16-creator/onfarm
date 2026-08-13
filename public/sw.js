/*
  아주 얇은 서비스워커.
  - 화면 껍데기(HTML/CSS/JS/아이콘)만 캐시해 통신이 느린 산지에서도 화면이 뜨게 한다.
  - API 응답은 절대 캐시하지 않는다(재고·주문이 과거 값으로 보이면 안 되므로).
*/
const CACHE = 'onfarm-shell-v3';
const SHELL = [
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
  '/css/base.css',
  '/css/farmer.css',
  '/css/store.css',
  '/js/api.js',
  '/js/cart.js',
  '/js/store.js',
  '/js/speak.js',
  '/js/features.js',
  '/js/farmer-sell.js',
  '/js/shared/korean.js',
  '/img/icon.svg',
  '/img/sample/placeholder.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET') return;
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/uploads/')) return;

  event.respondWith(
    fetch(event.request)
      .then((res) => {
        if (res.ok && url.origin === location.origin) {
          const copy = res.clone();
          caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        }
        return res;
      })
      .catch(() =>
        caches
          .match(event.request)
          .then((hit) => hit ?? caches.match(url.pathname.startsWith('/farmer') ? '/farmer' : '/')),
      ),
  );
});
