// 소비자 매장 메인
import { $, api, el, imageOf, money } from '/js/api.js';
import { mountCartCount } from '/js/cart.js';

mountCartCount();

const cfg = await api('/api/config').catch(() => ({ products: [] }));
const state = { product: '', region: '', query: '' };

const FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'region:충남', label: '충남' },
  { key: 'region:충북', label: '충북' },
  { key: 'region:제주', label: '제주' },
];

function renderFilters() {
  const box = $('#filters');
  box.replaceChildren();
  const options = [
    ...FILTERS,
    ...cfg.products.map((product) => ({ key: `product:${product.code}`, label: product.name })),
  ];

  for (const option of options) {
    const active =
      (option.key === 'all' && !state.product && !state.region) ||
      (option.key === `product:${state.product}` && state.product) ||
      (option.key === `region:${state.region}` && state.region);

    box.append(
      el('button', {
        type: 'button',
        'aria-pressed': String(Boolean(active)),
        text: option.label,
        onclick: () => {
          if (option.key === 'all') {
            state.product = '';
            state.region = '';
          } else if (option.key.startsWith('product:')) {
            state.product = option.key.slice(8) === state.product ? '' : option.key.slice(8);
          } else {
            state.region = option.key.slice(7) === state.region ? '' : option.key.slice(7);
          }
          renderFilters();
          load();
        },
      }),
    );
  }
}

function card(listing) {
  const region = `${listing.region_sigungu}${listing.region_detail ? ` ${listing.region_detail}` : ''}`;
  return el('a', { class: 'product-card', href: `/store/product?id=${listing.id}` }, [
    el('div', { class: 'product-card-media' }, [
      el('img', { class: 'thumb', src: imageOf(listing), alt: listing.title, loading: 'lazy' }),
      el('span', { class: 'product-card-status', text: '거점 확인 과정 공개' }),
    ]),
    el('div', { class: 'body' }, [
      el('div', { class: 'farm', text: `${region} · ${listing.farm_name}` }),
      el('div', { class: 'name', text: listing.title }),
      el('div', { class: 'price', text: money(listing.unit_price) }),
      el('div', { class: 'meta' }, [
        el('span', { text: listing.sku_label }),
        el('span', { text: `남은 수량 ${listing.remaining_quantity}` }),
      ]),
    ]),
  ]);
}

async function load() {
  const params = new URLSearchParams();
  if (state.product) params.set('product', state.product);
  if (state.region) params.set('region', state.region);
  const { listings: allListings } = await api(`/api/store/listings?${params.toString()}`);

  const keyword = state.query.toLocaleLowerCase('ko-KR');
  const listings = keyword
    ? allListings.filter((listing) =>
        [listing.title, listing.product_name, listing.farm_name, listing.region_sido, listing.region_sigungu]
          .filter(Boolean)
          .join(' ')
          .toLocaleLowerCase('ko-KR')
          .includes(keyword),
      )
    : allListings;

  const grid = $('#grid');
  grid.replaceChildren();
  for (const listing of listings) grid.append(card(listing));

  $('#empty').hidden = listings.length > 0;
  $('#listCount').textContent = `${listings.length}건`;
  const productName = cfg.products.find((product) => product.code === state.product)?.name ?? '';
  const filterName = [state.region, productName].filter(Boolean).join(' ');
  $('#listTitle').textContent = state.query
    ? `“${state.query}” 검색 결과`
    : filterName
      ? `${filterName} 상품`
      : '오늘 올라온 농산물';
}

$('#searchForm').addEventListener('submit', (event) => {
  event.preventDefault();
  state.query = $('#searchInput').value.trim();
  load();
  $('#listTitle').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

renderFilters();
await load();
