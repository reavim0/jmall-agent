from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_v3.embeddings import sentence_transformer_embedding as _shared_st_embedding
from agent_v3.query_schemas import CATEGORY_SCHEMAS, normalize_category as normalize_query_category
from agent_v3.state import QueryPlan, RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHONE_CATALOG_PATH = PROJECT_ROOT / "processed" / "v3" / "idx_cell_phone.jsonl"
PHONE_BM25_PATH = PROJECT_ROOT / "processed" / "v3" / "bm25" / "cell_phone.sqlite"
PHONE_FAISS_INDEX_PATH = PROJECT_ROOT / "processed" / "v3" / "faiss" / "cell_phone.index"
PHONE_FAISS_META_PATH = PROJECT_ROOT / "processed" / "v3" / "faiss" / "cell_phone.meta.jsonl"
PHONE_FAISS_MANIFEST_PATH = PROJECT_ROOT / "processed" / "v3" / "faiss" / "cell_phone.manifest.json"
HASH_EMBED_DIM = 1024
RECENCY_REFERENCE_DATE = date(2024, 1, 1)
RECENCY_WINDOW_DAYS = 365 * 6
RECENCY_MAX_SCORE = 18.0

BRAND_ALIASES = {
    "苹果": "Apple",
    "Apple": "Apple",
    "华为": "Huawei",
    "Huawei": "Huawei",
    "小米": "Xiaomi",
    "Xiaomi": "Xiaomi",
    "Redmi": "Xiaomi",
    "三星": "Samsung",
    "Samsung": "Samsung",
    "Google": "Google",
    "谷歌": "Google",
    "OnePlus": "OnePlus",
    "一加": "OnePlus",
}


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _normalize_retrieval_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text or "")
        if len(token) > 1
    ]


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _recency_score(product: dict[str, Any]) -> tuple[float, list[str]]:
    product_date = _parse_date(product.get("date_first_available"))
    if product_date is None:
        return 0.0, []
    distance_days = abs((RECENCY_REFERENCE_DATE - product_date).days)
    closeness = max(0.0, 1.0 - (distance_days / RECENCY_WINDOW_DAYS))
    score = closeness * RECENCY_MAX_SCORE
    return score, [f"date {product_date.isoformat()}"]


def _dedupe_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [] if values is None else [values]
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _hash_tokens(text: str) -> list[str]:
    normalized = _normalize_retrieval_text(text).lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    grams: list[str] = []
    grams.extend(words)
    compact_cjk = "".join(token for token in words if re.fullmatch(r"[\u4e00-\u9fff]", token))
    for n in (2, 3):
        grams.extend(compact_cjk[i : i + n] for i in range(max(0, len(compact_cjk) - n + 1)))
    for word in [token for token in words if re.fullmatch(r"[a-z0-9]+", token)]:
        if len(word) > 4:
            grams.extend(word[i : i + 4] for i in range(len(word) - 3))
    return [gram for gram in grams if gram]


def _hash_embedding(text: str, dim: int = HASH_EMBED_DIM) -> Any:
    # Catalog-specific tokenization (`_hash_tokens` adds 4-grams on long latin
    # words and CJK n-grams) differs from `agent_v3.embeddings._tokenize`, so
    # we still build the bag-of-tokens here; the deterministic blake2b → signed
    # column projection itself is shared via embeddings.hash_embedding when the
    # manifest doesn't request catalog-specific tokens.
    import numpy as np
    import hashlib

    vector = np.zeros((1, dim), dtype="float32")
    for token in _hash_tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        column = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[0, column] += sign
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vector / norm


def _query_values(plan: QueryPlan, field: str) -> list[str]:
    lexical = plan.get("lexical_query") or {}
    values: list[str] = []
    for bucket in ("must", "should"):
        bucket_values = (lexical.get(bucket) or {}).get(field) or []
        if bucket_values is None:
            continue
        if not isinstance(bucket_values, list):
            bucket_values = [bucket_values]
        values.extend(_normalize_retrieval_text(value) for value in bucket_values if str(value).strip())
    return list(dict.fromkeys(values))


def _bucket_values(plan: QueryPlan, bucket: str, field: str) -> list[str]:
    lexical = plan.get("lexical_query") or {}
    raw_values = (lexical.get(bucket) or {}).get(field)
    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raw_values = [raw_values]
    return list(dict.fromkeys(_normalize_retrieval_text(value) for value in raw_values if str(value).strip()))


def _bucket_fields(plan: QueryPlan, bucket: str) -> dict[str, list[str]]:
    lexical = plan.get("lexical_query") or {}
    raw_bucket = lexical.get(bucket) or {}
    if not isinstance(raw_bucket, dict):
        return {}
    fields: dict[str, list[str]] = {}
    for field, raw_values in raw_bucket.items():
        if field == "category" or raw_values is None:
            continue
        values = _bucket_values(plan, bucket, str(field))
        if values:
            fields[str(field)] = values
    return fields


def _strict_anchor_terms(plan: QueryPlan) -> list[str]:
    if plan.get("query_type") not in {"product_ref", "exact_model"}:
        return []
    return [*_query_values(plan, "model"), *_query_values(plan, "series")]


def _fts_phrase(value: str) -> str:
    text = re.sub(r'"', " ", str(value or "")).strip()
    return f'"{text}"' if text else ""


def _fts_or(values: list[str]) -> str:
    phrases = [_fts_phrase(value) for value in values if str(value).strip()]
    return " OR ".join(phrases)


def _field_match(field: str, values: list[str]) -> str:
    phrases = [_fts_phrase(value) for value in values if str(value).strip()]
    if not phrases:
        return ""
    terms = [f"{field}:{phrase}" for phrase in phrases]
    if len(terms) == 1:
        return terms[0]
    return f"({' OR '.join(terms)})"


def _schema_fields(category: str) -> set[str]:
    schema = CATEGORY_SCHEMAS.get(category) or {}
    fields_def = schema.get("fields") or {}
    return set(fields_def.keys()) if isinstance(fields_def, dict) else set()


def _bm25_match_query(plan: QueryPlan) -> str:
    category = normalize_query_category(plan.get("category")) or "cell_phone"
    fields = _schema_fields(category)
    parts: list[str] = []

    anchors = _strict_anchor_terms(plan)
    for field in ("model", "series"):
        values = _query_values(plan, field)
        if values:
            match = _field_match(field, values)
            if match:
                parts.append(match)

    if anchors:
        return " ".join(parts)

    brand_values = _wanted_brands(plan)
    brand_match = _field_match("brand", brand_values)
    if brand_match:
        parts.append(brand_match)

    for field in ("model", "series", "processor", "ram", "rom", "storage", "camera", "battery", "charging", "screen", "network", "os", "weight", "color"):
        if field not in fields:
            continue
        values = _query_values(plan, field)
        match = _field_match(field, values)
        if match:
            parts.append(match)

    query_text = _normalize_retrieval_text(plan.get("query") or (plan.get("semantic_query") or {}).get("text") or "")
    query_tokens = [token for token in _tokens(query_text) if token not in {"手机", "cell", "phone"}]
    if query_tokens:
        parts.append(_field_match("all_text", query_tokens[:8]))

    if not parts:
        brand_terms = _wanted_brands(plan)
        if brand_terms:
            parts.append(_field_match("brand", brand_terms))
    return " OR ".join(part for part in parts if part)


@lru_cache(maxsize=1)
def load_phone_catalog(path: str | None = None) -> list[dict[str, Any]]:
    catalog_path = Path(path) if path else PHONE_CATALOG_PATH
    if not catalog_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with catalog_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _wanted_brands(plan: QueryPlan) -> list[str]:
    """Brands the user wants. Per the all-must design these now live in
    `must.brand`; `should.brand` is read as a fallback for legacy plans."""
    lexical = plan.get("lexical_query") or {}
    must_brands = ((lexical.get("must") or {}).get("brand") or [])
    should_brands = ((lexical.get("should") or {}).get("brand") or [])
    seen: set[str] = set()
    out: list[str] = []
    for brand in [*must_brands, *should_brands]:
        canonical = BRAND_ALIASES.get(str(brand), str(brand))
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _blocked_conditions(plan: QueryPlan) -> set[str]:
    must_not = ((plan.get("lexical_query") or {}).get("must_not") or {})
    return {str(value) for value in must_not.get("condition") or []}


def _category_matches(plan: QueryPlan, product: dict[str, Any]) -> bool:
    category = normalize_query_category(plan.get("category"))
    product_category = product.get("category")
    if category == "cell_phone":
        return product_category == "cell_phone"
    return True


def _budget_matches(plan: QueryPlan, product: dict[str, Any]) -> bool:
    budget = plan.get("budget") or {}
    price = product.get("price")
    if price is None:
        return True
    min_price = budget.get("min")
    max_price = budget.get("max")
    if min_price is not None and price < min_price:
        return False
    if max_price is not None and price > max_price:
        return False
    return True


def _condition_matches(plan: QueryPlan, product: dict[str, Any]) -> bool:
    blocked = _blocked_conditions(plan)
    signals = set(product.get("condition_signals") or [])
    if "翻新机" in blocked and "renewed" in signals:
        return False
    if "二手机" in blocked and "used" in signals:
        return False
    return True


def _brand_matches(plan: QueryPlan, product: dict[str, Any]) -> bool:
    brands = _wanted_brands(plan)
    if not brands:
        return True
    product_brand = str(product.get("brand") or "")
    return any(brand.lower() == product_brand.lower() for brand in brands)


def _forbidden_brands(plan: QueryPlan) -> list[str]:
    must_not = ((plan.get("lexical_query") or {}).get("must_not") or {})
    return [BRAND_ALIASES.get(str(brand), str(brand)) for brand in (must_not.get("brand") or [])]


def _brand_not_excluded(plan: QueryPlan, product: dict[str, Any]) -> bool:
    """Hard filter for must_not.brand. Populated by `disliked_brands` —
    typically from a user clarify-option choice — and must be respected
    regardless of BM25 scoring."""
    forbidden = _forbidden_brands(plan)
    if not forbidden:
        return True
    product_brand = str(product.get("brand") or "").lower()
    if not product_brand:
        return True
    return all(brand.lower() != product_brand for brand in forbidden)


def _field_values_match_product(product: dict[str, Any], field: str, values: list[str]) -> bool:
    if not values:
        return True
    if field == "brand":
        product_brand = str(product.get("brand") or "").lower()
        canonical_values = [BRAND_ALIASES.get(str(value), str(value)) for value in values]
        return any(str(value).lower() == product_brand for value in canonical_values)
    product_text = _product_field_text(product, field)
    product_compact = re.sub(r"[\s_\-]+", "", product_text.lower())
    if field in {"ram", "rom", "storage"}:
        if not _has_reliable_capacity_evidence(product, field, values):
            return False
        product_caps = _normalized_capacity_values(product_text)
        return any(product_caps & _normalized_capacity_values(value) for value in values)
    for value in values:
        normalized_value = _norm_text(value)
        compact_value = re.sub(r"[\s_\-]+", "", normalized_value.lower())
        if normalized_value and (normalized_value in product_text or compact_value in product_compact):
            return True
    return False


def _must_fields_match(plan: QueryPlan, product: dict[str, Any]) -> bool:
    """Apply `lexical_query.must` as real structured filters.

    Handles three value shapes per field:
      - list[str]      : enum / freetext values, substring-match against product
      - {"min", "max"} : range filter, product spec must fall in window
      - other          : ignored
    Category and condition keep dedicated filters elsewhere; brand has its
    own _brand_matches; this covers the remaining schema fields.
    """
    must_bucket = (plan.get("lexical_query") or {}).get("must") or {}
    if not isinstance(must_bucket, dict):
        return True
    for field, raw_value in must_bucket.items():
        if field in {"category", "condition", "brand"}:
            continue
        # Range value: dict with min/max → product spec must fall in window.
        if isinstance(raw_value, dict) and ("min" in raw_value or "max" in raw_value):
            product_value = _product_range_value(product, str(field))
            if product_value is None:
                # Spec missing from this product — can't verify, treat as fail.
                return False
            if not _range_overlaps(raw_value, product_value):
                return False
            continue
        # Enum / freetext: list-like values, substring match.
        values = _bucket_values(plan, "must", str(field))
        if not values:
            continue
        if not _field_values_match_product(product, str(field), values):
            return False
    return True


def _must_not_fields_exclude(plan: QueryPlan, product: dict[str, Any]) -> bool:
    for field, values in _bucket_fields(plan, "must_not").items():
        if field in {"brand", "condition"}:
            continue
        if _field_values_match_product(product, field, values):
            return False
    return True


def _strict_anchor_matches(plan: QueryPlan, product: dict[str, Any]) -> bool:
    anchors = _strict_anchor_terms(plan)
    if not anchors:
        return True
    haystack = _norm_text(
        " ".join(
            [
                product.get("title") or "",
                product.get("model_number") or "",
                str((product.get("raw_attributes") or {}).get("Model Name") or ""),
                " ".join((product.get("rag_texts") or {}).values()),
            ]
        )
    )
    collapsed = re.sub(r"[\s_\-]+", "", haystack)
    return any(_norm_text(anchor) in haystack or re.sub(r"[\s_\-]+", "", _norm_text(anchor)) in collapsed for anchor in anchors)


def _semantic_text_for_plan(plan: QueryPlan) -> str:
    semantic = plan.get("semantic_query") or {}
    lexical = plan.get("lexical_query") or {}
    must = lexical.get("must") or {}
    should = lexical.get("should") or {}
    parts = [
        plan.get("query"),
        semantic.get("text"),
        " ".join(_wanted_brands(plan)),
        " ".join(str(item) for item in (semantic.get("positive_aspects") or [])),
        " ".join(str(item) for values in must.values() for item in (values or []) if isinstance(values, list)),
        " ".join(str(item) for values in should.values() for item in (values or []) if isinstance(values, list)),
    ]
    return _normalize_retrieval_text(" ".join(str(part) for part in parts if part))


def _compact_field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_compact_field_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_compact_field_text(item) for key, item in value.items() if not str(key).startswith("_"))
    return _norm_text(value)


def _normalized_capacity_values(value: Any) -> set[str]:
    values: set[str] = set()
    text = _compact_field_text(value)
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        try:
            numeric = float(number)
        except ValueError:
            continue
        if numeric.is_integer():
            values.add(f"{int(numeric)}gb")
        values.add(f"{numeric:g}gb")
    for match in re.findall(r"\d+(?:\.\d+)?\s*(?:gb|g)", text, flags=re.I):
        normalized = re.sub(r"\s+", "", match.lower()).replace("g", "gb")
        values.add(normalized)
    return values


def _capacity_numbers(values: Any) -> set[float]:
    numbers: set[float] = set()
    for value in values:
        for capacity in _normalized_capacity_values(value):
            match = re.match(r"(\d+(?:\.\d+)?)gb$", capacity)
            if not match:
                continue
            try:
                numbers.add(float(match.group(1)))
            except ValueError:
                continue
    return numbers


def _has_reliable_capacity_evidence(product: dict[str, Any], field: str, values: list[str]) -> bool:
    if field not in {"ram", "rom", "storage"}:
        return True
    requested = _capacity_numbers(values)
    if not requested:
        return True

    specs = product.get("specs") or {}
    evidence_items = ((specs.get("_evidence") or {}).get("memory") or [])
    wanted_kind = "ram" if field == "ram" else "storage"
    reliable_confidence = {"high", "medium"}
    for item in evidence_items:
        if item.get("kind") != wanted_kind:
            continue
        try:
            value_gb = float(item.get("value_gb"))
        except (TypeError, ValueError):
            continue
        if value_gb not in requested:
            continue
        confidence = str(item.get("confidence") or "").lower()
        evidence_text = _norm_text(item.get("evidence") or "")
        source_text = _norm_text(item.get("source") or "")
        if confidence in reliable_confidence:
            return True
        if field == "ram" and ("ram" in evidence_text or "ram" in source_text):
            return True
        if field in {"rom", "storage"} and any(token in evidence_text or token in source_text for token in ("rom", "storage", "capacity")):
            return True
    return False


def _product_field_text(product: dict[str, Any], field: str) -> str:
    specs = product.get("specs") or {}
    raw = product.get("raw_attributes") or {}
    rag_texts = product.get("rag_texts") or {}
    if field == "processor":
        return _compact_field_text([specs.get("processor"), raw.get("Processor"), raw.get("CPU Model"), product.get("title")])
    if field == "ram":
        # RAM/ROM are commonly confused in raw Amazon details. For retrieval
        # scoring, trust the cleaned v3 specs rather than raw detail keys.
        return _compact_field_text([specs.get("ram"), specs.get("ram_gb")])
    if field in {"rom", "storage"}:
        return _compact_field_text([specs.get("storage"), specs.get("storage_gb"), raw.get("Memory Storage Capacity"), raw.get("Hard Disk Size")])
    if field == "screen":
        return _compact_field_text([specs.get("screen_size"), raw.get("Screen Size"), product.get("title")])
    if field == "battery":
        return _compact_field_text([specs.get("battery"), raw.get("Battery Capacity"), product.get("title")])
    if field == "camera":
        return _compact_field_text([specs.get("camera_rear"), specs.get("camera_front"), specs.get("camera_rear_mp"), specs.get("camera_front_mp"), rag_texts.get("features"), product.get("title")])
    if field in {"charging", "network", "os", "color", "condition"}:
        return _compact_field_text([specs.get(field), raw.get(field), product.get("title")])
    return _compact_field_text([specs.get(field), raw.get(field), product.get("title")])


def _normalized_model_text(value: Any) -> str:
    text = _normalize_retrieval_text(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _model_aliases_for_anchor(anchor: str, product: dict[str, Any]) -> set[str]:
    normalized = _normalized_model_text(anchor)
    aliases = {normalized} if normalized else set()
    brand = _normalized_model_text(product.get("brand"))
    if brand and normalized and not normalized.startswith(f"{brand} "):
        aliases.add(f"{brand} {normalized}")
    return aliases


def _product_model_names(product: dict[str, Any]) -> list[str]:
    raw = product.get("raw_attributes") or {}
    values = [
        raw.get("Model Name"),
        raw.get("Item model number"),
        product.get("model_number"),
        *(product.get("compatibility") or []),
    ]
    return [_normalized_model_text(value) for value in values if _normalized_model_text(value)]


def _model_exactness_score(plan: QueryPlan, product: dict[str, Any]) -> tuple[float, list[str]]:
    anchors = [*_query_values(plan, "model"), *_query_values(plan, "series")]
    if not anchors:
        return 0.0, []

    score = 0.0
    reasons: list[str] = []
    model_names = _product_model_names(product)
    title = _normalized_model_text(product.get("title") or "")
    strict = bool(_strict_anchor_terms(plan))

    for anchor in anchors:
        aliases = _model_aliases_for_anchor(anchor, product)
        if not aliases:
            continue
        exact_name = any(name in aliases for name in model_names)
        suffix_name = any(
            any(name.startswith(f"{alias} ") and name not in aliases for alias in aliases)
            for name in model_names
        )
        title_exact = any(re.search(rf"(^|\s){re.escape(alias)}($|\s)", title) for alias in aliases)
        if exact_name:
            score += 45.0 if strict else 18.0
            reasons.append(f"exact_model {anchor}")
        elif title_exact:
            score += 20.0 if strict else 8.0
            reasons.append(f"title_model {anchor}")
        if suffix_name:
            score -= 35.0 if strict else 12.0
            reasons.append(f"model_suffix_mismatch {anchor}")

    return score, reasons


# Per-field scoring cap: a field either matches (full bonus) or doesn't (zero).
# Multiple values inside the same field don't stack — preventing the LLM from
# inflating the score by listing many enum values for one field.
#
# `should` bucket is intentionally not scored here. The current design routes
# all schema-expressible preferences into `must` (hard filter); soft preferences
# go into the `soft` block of the QueryPlan and into semantic_query, never into
# the lexical layer. We keep the `_SHOULD_FIELD_BONUS` symbol but never apply
# it — kept only so legacy code that references it stays importable.
_MUST_FIELD_BONUS = 10.0
_SHOULD_FIELD_BONUS = 0.0   # retired (was 5.0); kept for backward import compat


def _field_has_match(plan_values: list[str], product: dict[str, Any], field: str) -> str | None:
    """Return the first plan value that matches the product's field text, or None.

    For enum/freetext fields. Range fields use `_field_has_range_match` instead.
    """
    if not plan_values:
        return None
    product_text = _product_field_text(product, field)
    product_compact = re.sub(r"[\s_\-]+", "", product_text.lower())
    if field in {"ram", "rom", "storage"}:
        if not _has_reliable_capacity_evidence(product, field, plan_values):
            return None
        product_caps = _normalized_capacity_values(product_text)
        for value in plan_values:
            query_caps = _normalized_capacity_values(value)
            if query_caps and product_caps & query_caps:
                return str(value)
        return None
    for value in plan_values:
        normalized = _norm_text(value)
        compact = re.sub(r"[\s_\-]+", "", normalized.lower())
        if normalized and (normalized in product_text or compact in product_compact):
            return str(value)
    return None


# Range fields → product spec extractor. Returns the numeric value found in
# the product's relevant spec (e.g. battery_mah=5000 from "5000mAh"), or None.
_RANGE_FIELD_EXTRACTORS = {
    "battery_mah":      r"(\d{3,5})\s*mah",
    "screen_size_inch": r"(\d+(?:\.\d+)?)\s*(?:inch|inches|\")",
    "screen_refresh_hz": r"(\d{2,3})\s*hz",
    "charging_w":       r"(\d{1,3})\s*w(?:att)?",
    "camera_main_mp":   r"(\d{2,3})\s*mp",
    "weight_g":         r"(\d{2,4})\s*g(?:ram)?",
    "budget":           None,  # price extracted from product.price separately
}


def _product_range_value(product: dict[str, Any], field: str) -> float | None:
    """Pull a single numeric value from product specs/title for a range field."""
    if field == "budget":
        try:
            return float(product.get("price")) if product.get("price") is not None else None
        except (TypeError, ValueError):
            return None
    pattern = _RANGE_FIELD_EXTRACTORS.get(field)
    if not pattern:
        return None
    text = _product_field_text(product, field).lower()
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _range_overlaps(range_spec: dict[str, Any], product_value: float) -> bool:
    """Check whether `product_value` falls inside the {min, max} range."""
    lo = range_spec.get("min")
    hi = range_spec.get("max")
    if lo is not None:
        try:
            if product_value < float(lo):
                return False
        except (TypeError, ValueError):
            pass
    if hi is not None:
        try:
            if product_value > float(hi):
                return False
        except (TypeError, ValueError):
            pass
    return True


def _lexical_field_score(plan: QueryPlan, product: dict[str, Any]) -> tuple[float, list[str]]:
    """Per-field cap scoring (must bucket only).

      must.field hit   → +10  (one-shot per field, regardless of how many values)
      brand / category → not scored here (brand has its own +20 in _score_product;
                          category is a hard filter, no soft contribution).

    `should` bucket has been retired by the schema-driven all-must design;
    soft preferences live in plan.soft (which feeds semantic_query) and never
    enter lexical scoring.
    """
    lexical = plan.get("lexical_query") or {}
    score = 0.0
    reasons: list[str] = []

    bucket = lexical.get("must") or {}
    if not isinstance(bucket, dict):
        return score, reasons

    for field, raw_values in bucket.items():
        field_str = str(field)
        if field_str in {"brand", "category"}:
            continue
        # Range value: check if product's spec falls in [min, max] window.
        if isinstance(raw_values, dict) and ("min" in raw_values or "max" in raw_values):
            product_value = _product_range_value(product, field_str)
            if product_value is None:
                continue
            if _range_overlaps(raw_values, product_value):
                score += _MUST_FIELD_BONUS
                reasons.append(f"{field_str} {product_value:g}∈range")
            continue
        # Enum / freetext value: text-match against product's spec text.
        values = _dedupe_values(raw_values)
        matched_value = _field_has_match(values, product, field_str)
        if matched_value is None:
            continue
        score += _MUST_FIELD_BONUS
        reasons.append(f"{field_str} {matched_value}")
    return score, reasons


# Soft-scoring caps for the non-anchor signals. Hard filters (brand exclusion,
# category match, must_not.condition, etc) are applied separately upstream.
_BRAND_BONUS = 20.0
_TOKEN_HITS_PER_TOKEN = 0.4
_TOKEN_HITS_CAP = 3.0
_RATING_COUNT_DIVISOR = 1000.0
_RATING_COUNT_NUMERATOR_CAP = 5000.0
_PRICE_PRESENT_BONUS = 0.5


def _score_product(plan: QueryPlan, product: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    # ---- Brand (binary, capped at +20) ----
    brands = _wanted_brands(plan)
    product_brand = str(product.get("brand") or "")
    if brands:
        if any(brand.lower() == product_brand.lower() for brand in brands):
            score += _BRAND_BONUS
            reasons.append(f"brand {product_brand}")
        else:
            return -1.0, []

    # ---- Title token hits (+0.4 per token, total cap +3) ----
    query_text = _norm_text(plan.get("query") or (plan.get("semantic_query") or {}).get("text") or "")
    product_text = _norm_text(" ".join([
        product.get("title") or "",
        product.get("brand") or "",
        " ".join((product.get("rag_texts") or {}).values()),
    ]))
    query_tokens = [tok for tok in _tokens(query_text) if tok not in {"手机", "cell", "phone"}]
    token_hits = sum(1 for tok in query_tokens if tok in product_text)
    if token_hits:
        capped = min(token_hits * _TOKEN_HITS_PER_TOKEN, _TOKEN_HITS_CAP)
        score += capped
        reasons.append(f"keyword_hits {token_hits} (+{capped:.1f})")

    # ---- Schema field matches (must +10/field, should +5/field, capped per field) ----
    lexical_score, lexical_reasons = _lexical_field_score(plan, product)
    score += lexical_score
    reasons.extend(lexical_reasons[:5])

    # ---- Model exact / suffix anchor (+18~+45 / -12~-35) ----
    model_score, model_reasons = _model_exactness_score(plan, product)
    score += model_score
    reasons.extend(model_reasons[:3])

    # ---- Quality / freshness signals ----
    rating = product.get("rating_avg")
    if rating:
        score += float(rating)
    rating_count = product.get("rating_count") or 0
    if rating_count:
        score += min(float(rating_count), _RATING_COUNT_NUMERATOR_CAP) / _RATING_COUNT_DIVISOR
    if product.get("price") is not None:
        score += _PRICE_PRESENT_BONUS
    recency, recency_reasons = _recency_score(product)
    score += recency
    reasons.extend(recency_reasons)

    return score, reasons


def _slice_results(results: list[RetrievalResult], *, offset: int, limit: int) -> list[RetrievalResult]:
    """Return `results[offset:offset+limit]` with rank renumbered to absolute
    position so the frontend waterfall can show a single increasing index."""
    page = results[offset : offset + limit]
    for index, item in enumerate(page, start=offset + 1):
        item["rank"] = index
    return page


def search_catalog(plan: QueryPlan, *, limit: int = 5, offset: int = 0) -> list[RetrievalResult]:
    from agent_v3.tracing import timed_tool

    category = normalize_query_category(plan.get("category"))
    if category not in {None, "cell_phone"}:
        return []

    with timed_tool("search_catalog", metadata={
        "plan_id": plan.get("plan_id"),
        "query_type": plan.get("query_type"),
        "category": category,
        "limit": limit,
        "offset": offset,
    }) as md:
        results = _search_catalog_impl(plan, limit=limit, offset=offset, category=category)
        md["result_count"] = len(results)
        return results


def _search_catalog_impl(plan: QueryPlan, *, limit: int, offset: int, category: str | None) -> list[RetrievalResult]:
    fetch_limit = max(limit + offset, limit)
    if PHONE_BM25_PATH.exists() and PHONE_FAISS_INDEX_PATH.exists() and PHONE_FAISS_META_PATH.exists():
        full = search_catalog_hybrid(plan, limit=fetch_limit)
        return _slice_results(full, offset=offset, limit=limit)
    if PHONE_BM25_PATH.exists():
        full = search_catalog_bm25(plan, limit=fetch_limit)
        return _slice_results(full, offset=offset, limit=limit)
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for product in load_phone_catalog():
        if not _category_matches(plan, product):
            continue
        if not _budget_matches(plan, product):
            continue
        if not _condition_matches(plan, product):
            continue
        if not _brand_matches(plan, product):
            continue
        if not _brand_not_excluded(plan, product):
            continue
        if not _must_fields_match(plan, product):
            continue
        if not _must_not_fields_exclude(plan, product):
            continue
        if not _strict_anchor_matches(plan, product):
            continue
        score, reasons = _score_product(plan, product)
        if score <= 0:
            continue
        scored.append((score, product, reasons))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[RetrievalResult] = []
    for rank, (score, product, reasons) in enumerate(scored[: fetch_limit], 1):
        results.append(
            {
                "rank": rank,
                "title": product.get("title") or "",
                "price": product.get("price"),
                "raw_id": product.get("product_id") or "",
                "url": "",
                "brief_reason": "; ".join(reasons[:4]),
                "extra": {
                    "score": round(score, 3),
                    "brand": product.get("brand"),
                    "category": product.get("category"),
                    "specs": {k: v for k, v in (product.get("specs") or {}).items() if not str(k).startswith("_")},
                    "images": (product.get("images") or [])[:3],
                    "condition_signals": product.get("condition_signals") or [],
                },
            }
        )
    return _slice_results(results, offset=offset, limit=limit)


def _bm25_candidate_rows(plan: QueryPlan, *, limit: int) -> list[tuple[float, dict[str, Any]]]:
    match_query = _bm25_match_query(plan)
    brands = _wanted_brands(plan)
    forbidden = _forbidden_brands(plan)
    params: list[Any] = []
    where = ["p.category = ?"]
    params.append(normalize_query_category(plan.get("category")) or "cell_phone")
    if brands:
        where.append(f"lower(p.brand) IN ({', '.join(['?'] * len(brands))})")
        params.extend(brand.lower() for brand in brands)
    if forbidden:
        where.append(f"lower(p.brand) NOT IN ({', '.join(['?'] * len(forbidden))})")
        params.extend(brand.lower() for brand in forbidden)

    con = sqlite3.connect(f"file:{PHONE_BM25_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        if match_query:
            sql = f"""
                SELECT p.product_json, bm25(products_fts, 1.0, 8.0, 12.0, 12.0, 4.0, 3.0, 3.0, 3.0, 5.0, 3.0, 2.5, 2.5, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 10.0, 1.0) AS bm25_score
                FROM products_fts
                JOIN products p ON p.rowid = products_fts.rowid
                WHERE products_fts MATCH ? AND {" AND ".join(where)}
                ORDER BY bm25_score ASC
                LIMIT ?
            """
            rows = con.execute(sql, [match_query, *params, limit]).fetchall()
        else:
            sql = f"""
                SELECT p.product_json, 0.0 AS bm25_score
                FROM products p
                WHERE {" AND ".join(where)}
                LIMIT ?
            """
            rows = con.execute(sql, [*params, limit]).fetchall()
    finally:
        con.close()
    return [(float(row["bm25_score"] or 0.0), json.loads(row["product_json"])) for row in rows]


def _filtered_bm25_candidates(
    plan: QueryPlan,
    *,
    limit: int,
    expand_limit: bool = True,
) -> list[tuple[float, dict[str, Any], list[str]]]:
    rows = _bm25_candidate_rows(plan, limit=min(max(limit * 30, 100), 1000) if expand_limit else limit)
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for bm25_score, product in rows:
        if not _category_matches(plan, product):
            continue
        if not _budget_matches(plan, product):
            continue
        if not _condition_matches(plan, product):
            continue
        if not _brand_matches(plan, product):
            continue
        if not _brand_not_excluded(plan, product):
            continue
        if not _must_fields_match(plan, product):
            continue
        if not _must_not_fields_exclude(plan, product):
            continue
        if not _strict_anchor_matches(plan, product):
            continue
        structural_score, reasons = _score_product(plan, product)
        if structural_score <= 0 and not _strict_anchor_terms(plan):
            continue
        score = structural_score + max(0.0, -bm25_score)
        if _strict_anchor_terms(plan):
            score += 50
            reasons.insert(0, "strict_model_or_series_anchor")
        scored.append((score, product, reasons))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def _format_results(scored: list[tuple[float, dict[str, Any], list[str]]], *, retriever: str, limit: int) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for rank, (score, product, reasons) in enumerate(scored[:limit], 1):
        results.append(
            {
                "rank": rank,
                "title": product.get("title") or "",
                "price": product.get("price"),
                "raw_id": product.get("product_id") or "",
                "url": "",
                "brief_reason": "; ".join(reasons[:4]),
                "extra": {
                    "score": round(score, 3),
                    "brand": product.get("brand"),
                    "category": product.get("category"),
                    "retriever": retriever,
                    "specs": {k: v for k, v in (product.get("specs") or {}).items() if not str(k).startswith("_")},
                    "images": (product.get("images") or [])[:3],
                    "condition_signals": product.get("condition_signals") or [],
                },
            }
        )
    return results


def search_catalog_bm25(plan: QueryPlan, *, limit: int = 5) -> list[RetrievalResult]:
    return _format_results(_filtered_bm25_candidates(plan, limit=limit), retriever="bm25", limit=limit)


def warm_catalog_retriever(*, warm_embedding: bool = True) -> dict[str, Any]:
    """Load catalog retrieval artifacts once at process startup.

    The underlying loaders still use `lru_cache`; this function just moves the
    first-hit FAISS/SentenceTransformer cost out of the user's first search.
    It returns a compact status object for the server debug surface.
    """
    status: dict[str, Any] = {
        "catalog_rows": 0,
        "bm25": "missing",
        "faiss": "missing",
        "embedding": "skipped",
        "errors": [],
    }
    try:
        status["catalog_rows"] = len(load_phone_catalog())
    except Exception as exc:
        status["errors"].append(f"catalog: {exc}")

    if PHONE_BM25_PATH.exists():
        try:
            con = sqlite3.connect(f"file:{PHONE_BM25_PATH}?mode=ro", uri=True)
            try:
                con.execute("SELECT count(*) FROM products").fetchone()
            finally:
                con.close()
            status["bm25"] = "ready"
        except Exception as exc:
            status["bm25"] = "error"
            status["errors"].append(f"bm25: {exc}")

    if PHONE_FAISS_INDEX_PATH.exists() and PHONE_FAISS_META_PATH.exists():
        try:
            _index, _meta, manifest = _load_faiss_artifacts()
            status["faiss"] = "ready"
            provider = str(manifest.get("embedding_provider") or "local_hashing")
            if warm_embedding and provider == "sentence_transformers":
                _sentence_transformer_embedding("catalog retriever warmup", manifest)
                status["embedding"] = "ready"
            elif provider == "local_hashing":
                _hash_embedding("catalog retriever warmup", int(manifest.get("dimensions") or HASH_EMBED_DIM))
                status["embedding"] = "ready"
            else:
                status["embedding"] = f"skipped:{provider}"
        except Exception as exc:
            status["faiss"] = "error"
            status["errors"].append(f"faiss: {exc}")

    return status


@lru_cache(maxsize=1)
def _load_faiss_artifacts() -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import faiss

    index = faiss.read_index(str(PHONE_FAISS_INDEX_PATH))
    meta: list[dict[str, Any]] = []
    with PHONE_FAISS_META_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    manifest: dict[str, Any] = {"model": "local_hashing_v1", "embedding_provider": "local_hashing", "dimensions": HASH_EMBED_DIM}
    if PHONE_FAISS_MANIFEST_PATH.exists():
        try:
            manifest.update(json.loads(PHONE_FAISS_MANIFEST_PATH.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return index, meta, manifest


def _sentence_transformer_embedding(text: str, manifest: dict[str, Any]) -> Any:
    return _shared_st_embedding(text, manifest, device_env_keys=("AGENT_V3_EMBED_DEVICE",))


def _faiss_candidate_rows(plan: QueryPlan, *, limit: int) -> list[tuple[float, dict[str, Any]]]:
    from agent_v3.tracing import timed_tool

    try:
        with timed_tool("faiss:load_artifacts") as md:
            index, meta, manifest = _load_faiss_artifacts()
            md["meta_count"] = len(meta) if meta else 0
    except Exception:
        return []
    query_text = _semantic_text_for_plan(plan)
    if not query_text:
        return []
    provider = str(manifest.get("embedding_provider") or "local_hashing")

    with timed_tool("faiss:embed_query", metadata={"provider": provider, "query_len": len(query_text)}):
        if provider == "local_hashing":
            vector = _hash_embedding(query_text, int(manifest.get("dimensions") or HASH_EMBED_DIM))
        elif provider == "sentence_transformers":
            try:
                vector = _sentence_transformer_embedding(query_text, manifest)
            except Exception:
                return []
        else:
            return []

    with timed_tool("faiss:index_search", metadata={"limit": limit}) as md:
        scores, indices = index.search(vector.astype("float32"), max(limit, 1))
        md["raw_hit_count"] = int((indices[0] >= 0).sum()) if hasattr(indices[0], "sum") else 0

    rows: list[tuple[float, dict[str, Any]]] = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0 or idx >= len(meta):
            continue
        rows.append((float(score), meta[int(idx)]))
    return rows


def _filtered_faiss_candidates(plan: QueryPlan, *, limit: int) -> list[tuple[float, dict[str, Any], list[str]]]:
    rows = _faiss_candidate_rows(plan, limit=max(limit * 30, 100))
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for vector_score, product in rows:
        if not _category_matches(plan, product):
            continue
        if not _budget_matches(plan, product):
            continue
        if not _condition_matches(plan, product):
            continue
        if not _brand_matches(plan, product):
            continue
        if not _brand_not_excluded(plan, product):
            continue
        if not _must_fields_match(plan, product):
            continue
        if not _must_not_fields_exclude(plan, product):
            continue
        if not _strict_anchor_matches(plan, product):
            continue
        structural_score, reasons = _score_product(plan, product)
        if structural_score <= 0 and not _strict_anchor_terms(plan):
            continue
        score = structural_score + max(0.0, vector_score) * 10
        reasons.insert(0, "faiss_semantic")
        if _strict_anchor_terms(plan):
            score += 50
            reasons.insert(0, "strict_model_or_series_anchor")
        scored.append((score, product, reasons))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def _filtered_structured_candidates(plan: QueryPlan, *, limit: int) -> list[tuple[float, dict[str, Any], list[str]]]:
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    has_structured_should = bool(((plan.get("lexical_query") or {}).get("should") or {}))
    for product in load_phone_catalog():
        if not _category_matches(plan, product):
            continue
        if not _budget_matches(plan, product):
            continue
        if not _condition_matches(plan, product):
            continue
        if not _brand_matches(plan, product):
            continue
        if not _brand_not_excluded(plan, product):
            continue
        if not _must_fields_match(plan, product):
            continue
        if not _must_not_fields_exclude(plan, product):
            continue
        if not _strict_anchor_matches(plan, product):
            continue

        lexical_score, lexical_reasons = _lexical_field_score(plan, product)
        if has_structured_should and lexical_score <= 0 and not _strict_anchor_terms(plan):
            continue
        score, reasons = _score_product(plan, product)
        if score <= 0:
            continue
        if lexical_reasons:
            reasons = list(dict.fromkeys([*lexical_reasons, *reasons]))
        scored.append((score, product, reasons))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def _product_key(product: dict[str, Any]) -> str:
    return str(product.get("product_id") or product.get("title") or "")


def _rrf_fuse(
    rankings: list[list[tuple[float, dict[str, Any], list[str]]]],
    *,
    k: int = 60,
) -> list[tuple[float, dict[str, Any], list[str]]]:
    fused: dict[str, tuple[float, dict[str, Any], list[str]]] = {}
    for ranking in rankings:
        for rank, (candidate_score, product, reasons) in enumerate(ranking, 1):
            key = _product_key(product)
            if not key:
                continue
            contribution = 1.0 / (k + rank)
            old_score, old_product, old_reasons = fused.get(key, (0.0, product, []))
            merged_reasons = list(dict.fromkeys([*old_reasons, *reasons]))
            structural_score = max(old_score, candidate_score)
            fused[key] = (structural_score + contribution, old_product, merged_reasons)
    return sorted(fused.values(), key=lambda item: item[0], reverse=True)


def search_catalog_hybrid(plan: QueryPlan, *, limit: int = 5) -> list[RetrievalResult]:
    from agent_v3.tracing import timed_tool

    # SQLite FTS on the WSL-mounted catalog can raise disk I/O errors when a
    # broad OR query pulls hundreds of large JSON rows. Keep first-stage BM25
    # recall bounded; FAISS provides the second recall branch.
    candidate_limit = min(max(limit * 20, 100), 100)

    with timed_tool("bm25_recall", metadata={"limit": candidate_limit}) as md:
        bm25 = _filtered_bm25_candidates(plan, limit=candidate_limit, expand_limit=False)
        md["candidate_count"] = len(bm25)

    with timed_tool("structured_recall", metadata={"limit": max(limit * 10, 50)}) as md:
        structured = _filtered_structured_candidates(plan, limit=max(limit * 10, 50))
        md["candidate_count"] = len(structured)

    with timed_tool("faiss_recall", metadata={"limit": max(limit * 10, 50)}) as md:
        faiss = _filtered_faiss_candidates(plan, limit=max(limit * 10, 50))
        md["candidate_count"] = len(faiss)

    if not faiss and not structured:
        return _format_results(bm25, retriever="bm25", limit=limit)

    with timed_tool("rrf_fuse", metadata={
        "bm25_n": len(bm25), "structured_n": len(structured), "faiss_n": len(faiss),
    }) as md:
        fused = _rrf_fuse([bm25, structured, faiss])
        md["fused_count"] = len(fused)

    results = _format_results(fused, retriever="hybrid_bm25_faiss", limit=limit)
    for result in results:
        reasons = result.get("brief_reason") or ""
        if "faiss_semantic" in reasons and "bm25" not in reasons:
            result["brief_reason"] = reasons
    return results
