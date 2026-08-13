export type Role = 'farmer' | 'consumer' | 'hub_operator' | 'admin';

export interface User {
  id: number;
  role: Role;
  name: string;
  phone: string | null;
  /** 거점 담당자만 값이 있다. 이 거점의 물량만 처리할 수 있다. */
  hub_id: number | null;
  created_at: string;
}

export interface Hub {
  id: number;
  name: string;
  region: string;
  address: string | null;
}

export interface Farm {
  id: number;
  user_id: number;
  farm_name: string;
  region_sido: string;
  region_sigungu: string;
  region_detail: string | null;
  hub_id: number | null;
}

export type ProductCategory = 'fruit' | 'vegetable' | 'root' | 'seafood';

export interface Product {
  id: number;
  code: string;
  name_ko: string;
  category: ProductCategory;
  variety: string | null;
  emoji: string | null;
  sample_image: string | null;
  active: number;
}

export interface Sku {
  id: number;
  product_id: number;
  code: string;
  label: string;
  weight: number;
  unit: string;
  price: number;
  is_default: number;
  active: number;
}

/** 실물 검수 전 단계까지 포함한 진행 상태. */
export type InspectionStatus =
  | 'ai_checked'
  | 'hub_pending'
  | 'hub_passed'
  | 'ready_to_ship'
  | 'delivered';

export type ListingStatus = 'active' | 'sold_out' | 'closed';

/** AI 가 확정하지 않는 '참고' 등급. UI 문구도 항상 참고 판정으로 표기한다. */
export type QualityHint = '특' | '상' | '보통' | '확인필요';

export interface Listing {
  id: number;
  farmer_id: number;
  farm_id: number;
  product_id: number;
  sku_id: number;
  title: string;
  description: string;
  image_path: string | null;
  quantity: number;
  remaining_quantity: number;
  unit_price: number;
  harvested_on: string | null;
  ai_analysis: string | null;
  ai_confidence: number | null;
  ai_source: string | null;
  /** AI 참고 판정. 확정 등급이 아니다. */
  quality_hint: QualityHint | null;
  /** 거점 실물 검수로 확정된 등급. 이 값이 있으면 이것이 최종이다. */
  confirmed_quality: string | null;
  inspection_status: InspectionStatus;
  status: ListingStatus;
  created_at: string;
}

/** 목록/상세 화면에 그대로 내려보내는 조인 뷰. */
export interface ListingView extends Listing {
  product_code: string;
  product_name: string;
  product_emoji: string | null;
  sku_label: string;
  sku_weight: number;
  sku_unit: string;
  farm_name: string;
  region_sido: string;
  region_sigungu: string;
  region_detail: string | null;
  farmer_name: string;
  hub_name: string | null;
  has_rejection: 0 | 1;
}

export type OrderStatus = 'paid' | 'preparing' | 'shipped' | 'done' | 'cancelled';

export type FulfillmentStatus =
  | 'farmer_preparing'
  | 'ready_for_hub'
  | 'hub_received'
  | 'hub_passed'
  | 'ready_to_ship'
  | 'delivered';

export interface Order {
  id: number;
  consumer_id: number;
  order_no: string;
  total_amount: number;
  receiver_name: string;
  receiver_phone: string;
  address: string;
  memo: string | null;
  status: OrderStatus;
  created_at: string;
}

export interface OrderItem {
  id: number;
  order_id: number;
  listing_id: number;
  sku_id: number;
  farmer_id: number;
  unit_price: number;
  quantity: number;
  amount: number;
  fulfillment_status: FulfillmentStatus;
}

export interface Settlement {
  id: number;
  farmer_id: number;
  order_item_id: number;
  gross: number;
  fee: number;
  net: number;
  status: 'pending' | 'paid';
  due_on: string | null;
  paid_at: string | null;
  payment_reference: string | null;
  created_at: string;
}

export interface HubInspection {
  id: number;
  listing_id: number;
  hub_id: number | null;
  inspector: string;
  result: 'pass' | 'downgrade' | 'reject';
  graded_quality: string | null;
  note: string | null;
  created_at: string;
}
