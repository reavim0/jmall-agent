"""Per-category query schemas.

Single source of truth for what fields a category supports, what enum values
each field accepts, and how user surface forms map to canonical values.

Design:
- One category, one schema dict. Schemas are static data; new category =
  new dict + register in CATEGORY_SCHEMAS.
- Field types are minimal: `enum` (closed value set), `range` (numeric min/max
  with unit), `freetext` (model passes through user wording).
- `soft_buckets` carries any preference the schema fields cannot express
  (free-form natural language). Goes into semantic search only, never into
  lexical hard filters.
- `alias_map` translates user-language surface forms to canonical enum values.
  Applied as a code-level post-processing step on LLM output, NOT exposed in
  the prompt — the LLM only sees the enum values list.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# cell_phone
# ---------------------------------------------------------------------------

CELL_PHONE_SCHEMA: dict[str, Any] = {
    "category": "cell_phone",
    "fields": {
        # Identity
        "brand": {
            "type": "enum",
            "values": [
                "Apple", "Samsung", "Huawei", "Honor", "Xiaomi", "OPPO", "Vivo",
                "OnePlus", "Realme", "Motorola", "Nokia", "Google", "Sony",
                "ASUS", "Nubia", "Black Shark", "REDMAGIC", "iQOO",
            ],
            # Tier groups for "general flagship vs gaming-dedicated" distinction.
            # gaming_dedicated: brands whose entire lineup targets gamers
            #   (chassis with shoulder triggers, RGB, gaming optimization).
            # general_flagship: mainstream brands whose flagships are universal
            #   daily-use phones that also happen to game well.
            # ASUS is intentionally NOT in gaming_dedicated — they ship ROG
            # (gaming) AND ZenFone (general). Series-level filtering handles
            # ROG via the freetext `series` field (must_not.series=["ROG Phone"]).
            # iQOO has a sub-brand Neo line that's gaming-leaning but the
            # parent brand is general-purpose; left out of both for now.
            "tier_groups": {
                "gaming_dedicated":  ["Black Shark", "REDMAGIC", "Nubia"],
                "general_flagship":  ["Apple", "Samsung", "Huawei", "Honor",
                                      "Xiaomi", "OPPO", "Vivo", "OnePlus",
                                      "Google", "Sony"],
            },
        },
        "series": {
            "type": "freetext",
            "description": "产品系列名（Galaxy S23 / iPhone 14 / Mate 60 等）。LLM 直接采用用户提到的写法即可，无需归一。",
        },
        "model": {
            "type": "freetext",
            "description": "具体型号（iPhone 14 Pro Max / Galaxy S23 Ultra 等）。LLM 直接采用用户提到的写法。",
        },

        # Hardware specs (enum)
        "processor": {
            "type": "enum",
            "values": [
                "Snapdragon 8 Gen 2", "Snapdragon 8+ Gen 1", "Snapdragon 8 Gen 1",
                "Snapdragon 888", "Snapdragon 870", "Snapdragon 865",
                "Snapdragon 778G", "Snapdragon 695",
                "Apple A16 Bionic", "Apple A15 Bionic", "Apple A14 Bionic",
                "Dimensity 9200", "Dimensity 9000", "Dimensity 8200", "Dimensity 6020",
                "Kirin 9000S", "Kirin 9000", "Kirin 980",
                "Exynos 2200", "Exynos 990",
            ],
            # Abstract tier labels → concrete chip subset. LLM can write
            # `{"tier": "flagship_2023"}` instead of listing chips by name;
            # `expand_tier()` resolves it to the value list before retrieval.
            "tier_groups": {
                "flagship_2023":   ["Snapdragon 8 Gen 2", "Snapdragon 8+ Gen 1",
                                    "Apple A16 Bionic", "Dimensity 9200", "Kirin 9000S"],
                "upper_mid_2023":  ["Snapdragon 8 Gen 1", "Apple A15 Bionic",
                                    "Snapdragon 888", "Snapdragon 870", "Dimensity 8200"],
                "mid_2023":        ["Snapdragon 778G", "Snapdragon 865",
                                    "Dimensity 6020", "Kirin 980", "Apple A14 Bionic"],
                "entry_2023":      ["Snapdragon 695", "Exynos 990"],
            },
        },
        "ram":     {"type": "enum", "values": ["4GB", "6GB", "8GB", "12GB", "16GB", "18GB"]},
        "rom":     {"type": "enum", "values": ["64GB", "128GB", "256GB", "512GB", "1TB"]},
        "network": {"type": "enum", "values": ["5G", "4G LTE"]},

        # Hardware specs (range)
        "battery_mah":       {"type": "range", "min": 1000, "max": 10000, "unit": "mAh"},
        "screen_size_inch":  {"type": "range", "min": 4.0,  "max": 8.0,   "unit": "inch"},
        "screen_refresh_hz": {"type": "range", "min": 60,   "max": 165,   "unit": "Hz"},
        "charging_w":        {"type": "range", "min": 5,    "max": 240,   "unit": "W"},
        "camera_main_mp":    {"type": "range", "min": 8,    "max": 200,   "unit": "MP"},
        "weight_g":          {"type": "range", "min": 100,  "max": 300,   "unit": "g"},

        # Note: `use_case` is intentionally NOT a schema field. The catalog has
        # no structured use-case attribute on products — title-keyword matching
        # ("gaming" in title) is unreliable and excludes real flagship phones
        # whose marketing copy doesn't use the word. User intent like
        # "适合玩游戏" / "拍照好" / "长续航" belongs in soft_buckets, where it
        # joins semantic_query and influences ranking without hard-filtering.

        # Negative-list helpers (mostly for must_not)
        "condition": {"type": "enum", "values": ["翻新机", "二手机"]},

        # Budget travels here (range field), so it's no longer a separate top-level Budget dict.
        "budget": {"type": "range", "min": 0, "max": 50000, "unit": "RMB"},
    },
    "soft_buckets": {
        "must_have":    "schema 字段不能表达的必要约束（如指定 OS 版本、特殊接口、礼盒装、海外版等）",
        "nice_to_have": "schema 字段不能表达的软偏好（如颜色喜好、外观风格、手感、品牌情怀、送礼场景等）",
        "avoid":        "schema 字段不能表达的排除项（如外形元素、设计风格等）",
    },
    "alias_map": {
        "brand": {
            "苹果": "Apple", "iphone": "Apple", "果子": "Apple",
            "华为": "Huawei", "鸿蒙": "Huawei", "华子": "Huawei",
            "小米": "Xiaomi", "redmi": "Xiaomi", "红米": "Xiaomi", "粗粮": "Xiaomi",
            "三星": "Samsung", "galaxy": "Samsung", "星星星": "Samsung",
            "一加": "OnePlus", "1+": "OnePlus",
            "荣耀": "Honor",
            "谷歌": "Google", "pixel": "Google",
            "黑鲨": "Black Shark",
            "红魔": "Nubia",
            "iqoo": "iQOO",
        },
        "processor": {
            "SD8G2": "Snapdragon 8 Gen 2", "8 gen 2": "Snapdragon 8 Gen 2",
            "骁龙8 Gen 2": "Snapdragon 8 Gen 2", "骁龙8gen2": "Snapdragon 8 Gen 2",
            "SD8+G1": "Snapdragon 8+ Gen 1", "8+ gen 1": "Snapdragon 8+ Gen 1",
            "骁龙8+ Gen 1": "Snapdragon 8+ Gen 1", "骁龙8+gen1": "Snapdragon 8+ Gen 1",
            "A16": "Apple A16 Bionic", "苹果A16": "Apple A16 Bionic",
            "A15": "Apple A15 Bionic", "苹果A15": "Apple A15 Bionic",
            "A14": "Apple A14 Bionic",
            "天玑9200": "Dimensity 9200", "D9200": "Dimensity 9200",
            "天玑9000": "Dimensity 9000",
            "天玑8200": "Dimensity 8200",
            "麒麟9000S": "Kirin 9000S", "麒麟9000s": "Kirin 9000S",
            "麒麟9000": "Kirin 9000",
            "麒麟980": "Kirin 980",
        },
        "ram": {"6G": "6GB", "8G": "8GB", "12G": "12GB", "16G": "16GB", "18G": "18GB"},
        "rom": {"128G": "128GB", "256G": "256GB", "512G": "512GB", "1T": "1TB", "1024GB": "1TB"},
        "network": {"5g": "5G", "4g": "4G LTE", "lte": "4G LTE"},
        "condition": {
            "翻新": "翻新机", "renewed": "翻新机", "refurbished": "翻新机",
            "二手": "二手机", "used": "二手机",
        },
    },
}


# ---------------------------------------------------------------------------
# accessories
# ---------------------------------------------------------------------------

ACCESSORIES_SCHEMA: dict[str, Any] = {
    "category": "accessories",
    "fields": {
        "sub_category": {
            "type": "enum",
            "values": ["耳机", "手机壳", "膜", "充电器", "充电宝", "数据线", "支架", "转接头", "others"],
        },
        "brand": {
            "type": "freetext",
            "description": "配件品牌；catalog 中长尾，不限定 enum。LLM 直接写用户提到的品牌即可。",
        },
        "compatibility": {
            "type": "freetext",
            "description": "适配的设备/接口/平台（如 'iPhone 15 Pro' / 'USB-C' / '华为 Mate 60'）。",
        },
        "budget": {"type": "range", "min": 0, "max": 5000, "unit": "RMB"},
    },
    "soft_buckets": {
        "must_have":    "schema 不能表达的必要约束（如认证、特殊功能等）",
        "nice_to_have": "schema 不能表达的软偏好（颜色、外观、手感等）",
        "avoid":        "schema 不能表达的排除项",
    },
    "alias_map": {
        "sub_category": {
            "耳塞": "耳机", "蓝牙耳机": "耳机", "tws": "耳机", "无线耳机": "耳机", "有线耳机": "耳机",
            "保护壳": "手机壳", "壳子": "手机壳", "保护套": "手机壳", "手机套": "手机壳",
            "钢化膜": "膜", "贴膜": "膜", "保护膜": "膜", "屏幕膜": "膜",
            "充电头": "充电器", "适配器": "充电器", "电源适配器": "充电器",
            "移动电源": "充电宝",
            "充电线": "数据线", "数据线缆": "数据线",
            "手机支架": "支架", "桌面支架": "支架", "车载支架": "支架",
            "转换器": "转接头", "转换头": "转接头", "扩展坞": "转接头",
        },
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CATEGORY_SCHEMAS: dict[str, dict[str, Any]] = {
    "cell_phone": CELL_PHONE_SCHEMA,
    "accessories": ACCESSORIES_SCHEMA,
}

# Closed enum used by route_turn to constrain the categories it may emit.
CATEGORIES_ENUM: list[str] = list(CATEGORY_SCHEMAS.keys())


def normalize_category(category: str | None) -> str | None:
    text = str(category or "").strip()
    return text if text in CATEGORY_SCHEMAS else None


def get_schema_for_category(category: str | None) -> dict[str, Any] | None:
    """Return a deep copy of the schema for `category`, or None if unknown."""
    normalized = normalize_category(category)
    if not normalized:
        return None
    return deepcopy(CATEGORY_SCHEMAS[normalized])


def llm_visible_schema(category: str | None) -> dict[str, Any] | None:
    """Return the slimmed-down schema view the LLM should see in its prompt.

    Strips `alias_map` (LLM should never see surface-form translations — those
    are handled deterministically post-LLM) but keeps fields, soft_buckets,
    and category name.
    """
    schema = get_schema_for_category(category)
    if schema is None:
        return None
    schema.pop("alias_map", None)
    return schema


def normalize_alias(category: str, field: str, surface: Any) -> str | None:
    """Translate a surface form to its canonical enum value for `(category, field)`.

    Returns None when:
    - category is unknown
    - field is not an enum (range/freetext have no canonical translation)
    - surface form has no alias entry AND isn't itself an enum value
    - alias would resolve to a value not in the enum's `values`
    """
    schema = CATEGORY_SCHEMAS.get(normalize_category(category) or "")
    if not schema:
        return None
    field_def = (schema.get("fields") or {}).get(field) or {}
    if field_def.get("type") != "enum":
        return None
    text = str(surface or "").strip()
    if not text:
        return None
    enum_values = field_def.get("values") or []
    if text in enum_values:
        return text
    alias_map = (schema.get("alias_map") or {}).get(field) or {}
    surface_lower = text.lower()
    for alias, canonical in alias_map.items():
        if str(alias).strip().lower() == surface_lower and canonical in enum_values:
            return canonical
    return None


def coerce_enum_values(category: str, field: str, raw_values: Any) -> list[str]:
    """Filter+normalize a list-like input to canonical enum values.

    Also accepts the `{"tier": "<tier_name>"}` shortcut: it expands via
    tier_groups to the canonical value list. Drops anything that doesn't
    translate cleanly.
    """
    # Tier shortcut: {"tier": "flagship_2023"} → expand to value list.
    if isinstance(raw_values, dict) and "tier" in raw_values:
        return expand_tier(category, field, str(raw_values.get("tier") or ""))
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    if not isinstance(raw_values, (list, tuple, set)):
        return []
    seen: set[str] = set()
    output: list[str] = []
    for value in raw_values:
        canonical = normalize_alias(category, field, value)
        if canonical and canonical not in seen:
            output.append(canonical)
            seen.add(canonical)
    return output


def expand_tier(category: str, field: str, tier_name: str) -> list[str]:
    """Resolve `{"tier": tier_name}` shortcut to the canonical value list.

    Returns [] if the category / field / tier_name is unknown, or if the
    field has no `tier_groups`. Output is filtered to values that still
    appear in the field's `values` enum (so a stale tier_group entry never
    leaks a value that's no longer canonical).
    """
    schema = CATEGORY_SCHEMAS.get(normalize_category(category) or "")
    if not schema:
        return []
    field_def = (schema.get("fields") or {}).get(field) or {}
    tier_groups = field_def.get("tier_groups") or {}
    raw = tier_groups.get(tier_name) or []
    if not isinstance(raw, list):
        return []
    enum_values = set(field_def.get("values") or [])
    seen: set[str] = set()
    output: list[str] = []
    for v in raw:
        if v in enum_values and v not in seen:
            output.append(v)
            seen.add(v)
    return output


def coerce_range_value(category: str, field: str, raw_value: Any) -> dict[str, float] | None:
    """Validate a `{min, max}`-shaped range against the field's bounds.

    Returns the cleaned dict (with only present bounds), or None if the field
    isn't a range or the input is unusable.
    """
    schema = CATEGORY_SCHEMAS.get(normalize_category(category) or "")
    if not schema:
        return None
    field_def = (schema.get("fields") or {}).get(field) or {}
    if field_def.get("type") != "range":
        return None
    if not isinstance(raw_value, dict):
        return None
    cleaned: dict[str, float] = {}
    for key in ("min", "max"):
        if key not in raw_value or raw_value[key] is None:
            continue
        try:
            cleaned[key] = float(raw_value[key])
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return None
    # Clip to the field's declared bounds.
    f_min, f_max = float(field_def.get("min", float("-inf"))), float(field_def.get("max", float("inf")))
    if "min" in cleaned:
        cleaned["min"] = max(f_min, min(cleaned["min"], f_max))
    if "max" in cleaned:
        cleaned["max"] = max(f_min, min(cleaned["max"], f_max))
    if "min" in cleaned and "max" in cleaned and cleaned["min"] > cleaned["max"]:
        cleaned["min"], cleaned["max"] = cleaned["max"], cleaned["min"]
    return cleaned


def field_type(category: str, field: str) -> str | None:
    """Return 'enum' | 'range' | 'freetext' | None for `(category, field)`."""
    schema = CATEGORY_SCHEMAS.get(normalize_category(category) or "")
    if not schema:
        return None
    field_def = (schema.get("fields") or {}).get(field) or {}
    type_ = field_def.get("type")
    return type_ if isinstance(type_, str) else None


# ---------------------------------------------------------------------------
# CLI inspection (python -m agent_v3.query_schemas)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    summary = {
        "categories": CATEGORIES_ENUM,
        "schemas": {
            cat: {
                "field_count": len(schema.get("fields") or {}),
                "field_types": {f: (d.get("type") if isinstance(d, dict) else "?")
                                for f, d in (schema.get("fields") or {}).items()},
                "alias_field_keys": list((schema.get("alias_map") or {}).keys()),
            }
            for cat, schema in CATEGORY_SCHEMAS.items()
        },
        "examples": {
            "normalize_alias_brand_华为":     normalize_alias("cell_phone", "brand", "华为"),
            "normalize_alias_processor_SD8G2": normalize_alias("cell_phone", "processor", "SD8G2"),
            "normalize_alias_use_case_打游戏": normalize_alias("cell_phone", "use_case", "打游戏"),
            "coerce_range_budget":             coerce_range_value("cell_phone", "budget", {"min": 3000, "max": 5500}),
            "coerce_enum_processor_mixed":     coerce_enum_values("cell_phone", "processor", ["骁龙8 Gen 2", "8+ gen 1", "Junk Chip"]),
            "llm_visible_schema_keys":         list((llm_visible_schema("cell_phone") or {}).keys()),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
