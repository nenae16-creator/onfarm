// 장바구니는 브라우저 localStorage 에만 둔다(주문 확정 시 서버가 가격·재고를 다시 검증한다).
const KEY = 'onfarm.cart.v1';

function normalize(items) {
  const quantities = new Map();
  for (const item of Array.isArray(items) ? items : []) {
    if (!Number.isSafeInteger(item?.listingId) || !Number.isSafeInteger(item?.quantity) || item.listingId <= 0 || item.quantity <= 0) continue;
    const quantity = (quantities.get(item.listingId) ?? 0) + item.quantity;
    if (Number.isSafeInteger(quantity)) quantities.set(item.listingId, quantity);
  }
  return [...quantities].map(([listingId, quantity]) => ({ listingId, quantity }));
}

export function readCart() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return normalize(raw);
  } catch {
    return [];
  }
}

export function writeCart(items) {
  localStorage.setItem(KEY, JSON.stringify(normalize(items)));
  document.dispatchEvent(new CustomEvent('cart:changed'));
}

export function addToCart(listingId, quantity = 1) {
  const items = readCart();
  const found = items.find((i) => i.listingId === listingId);
  if (found) found.quantity += quantity;
  else items.push({ listingId, quantity });
  writeCart(items);
  return items;
}

export function setQuantity(listingId, quantity) {
  const items = readCart()
    .map((i) => (i.listingId === listingId ? { ...i, quantity } : i))
    .filter((i) => i.quantity > 0);
  writeCart(items);
  return items;
}

export function clearCart() {
  writeCart([]);
}

export function cartCount() {
  return readCart().reduce((sum, i) => sum + i.quantity, 0);
}

export function mountCartCount(selector = '#cartCount') {
  const render = () => {
    const node = document.querySelector(selector);
    if (node) node.textContent = String(cartCount());
  };
  document.addEventListener('cart:changed', render);
  render();
}
