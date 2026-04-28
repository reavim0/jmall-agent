from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_v3.embeddings import hash_embedding as _shared_hash_embedding
from agent_v3.embeddings import sentence_transformer_embedding as _shared_st_embedding

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = DATA_DIR / "knowledge"
KB_ROOT = DATA_DIR / "knowledge_bases"
BM25_ROOT = PROJECT_ROOT / "processed" / "v3" / "knowledge_bm25"
FAISS_ROOT = PROJECT_ROOT / "processed" / "v3" / "knowledge_faiss"
KB_FILES = (
    "aliases.jsonl",
    "categories.jsonl",
    "tech_terms.jsonl",
    "models.jsonl",
    "review_facts.jsonl",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)

TYPE_MAP = {
    "brand_alias": "brand",
    "category_alias": "category",
    "tech_term": "tech_term",
    "model_series": "model",
    "review_fact": "market_fact",
    "shopping_guide": "market_fact",
}

BUILT_TYPE_MAP = {
    ("alias", "brand_alias"): "brand",
    ("category", "category_alias"): "category",
    ("tech_term", "tech_term"): "tech_term",
    ("model_fact", "model_series"): "model",
    ("review", "review_fact"): "market_fact",
    ("shopping_guide", "shopping_guide"): "market_fact",
}


def normalize(text: str) -> str:
    return re.sub(r"[\s_\-/,.;:，。？！、（）()\[\]{}]+", "", (text or "").casefold())


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).casefold()
        if not token:
            continue
        tokens.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in (2, 3):
                if len(token) > size:
                    tokens.update(token[index : index + size] for index in range(len(token) - size + 1))
    return tokens


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("_file", str(path.relative_to(DATA_DIR)))
            record.setdefault("_line", lineno)
            rows.append(record)
    return rows


def _knowledge_bm25_path(category: str | None) -> Path | None:
    if not category:
        return None
    path = BM25_ROOT / f"{category}.sqlite"
    return path if path.exists() else None


def _knowledge_faiss_paths(category: str | None) -> tuple[Path, Path, Path] | None:
    if not category:
        return None
    index_path = FAISS_ROOT / f"{category}.index"
    meta_path = FAISS_ROOT / f"{category}.meta.jsonl"
    manifest_path = FAISS_ROOT / f"{category}.manifest.json"
    if index_path.exists() and meta_path.exists():
        return index_path, meta_path, manifest_path
    return None


def _fts_query(query: str) -> str:
    tokens = sorted(tokenize(query), key=lambda value: (-len(value), value))
    useful: list[str] = []
    for token in tokens:
        if not token or '"' in token:
            continue
        if len(token) == 1 and not re.fullmatch(r"[a-z0-9]", token, flags=re.IGNORECASE):
            continue
        useful.append(f'"{token}"')
        if len(useful) >= 16:
            break
    return " OR ".join(useful)


def _bm25_candidate_records(
    query: str,
    *,
    category: str | None,
    knowledge_types: list[str] | tuple[str, ...] | set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    path = _knowledge_bm25_path(category)
    match_query = _fts_query(query)
    if not path or not match_query:
        return []

    type_values = {str(value) for value in knowledge_types or [] if str(value)}
    if "model" in type_values:
        type_values.update({"model_fact", "model_series"})
    if "market_fact" in type_values:
        type_values.update({"review", "review_fact", "shopping_guide"})
    if "brand" in type_values:
        type_values.add("brand_alias")
    if "category" in type_values:
        type_values.add("category_alias")
    where = ["documents_fts MATCH ?"]
    params: list[Any] = [match_query]
    if type_values:
        placeholders = ",".join("?" for _ in type_values)
        where.append(f"(d.knowledge_type IN ({placeholders}) OR d.record_type IN ({placeholders}))")
        params.extend(type_values)
        params.extend(type_values)

    sql = f"""
        SELECT d.doc_json
        FROM documents_fts
        JOIN documents d ON d.rowid = documents_fts.rowid
        WHERE {' AND '.join(where)}
        ORDER BY bm25(documents_fts, 8.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 2.0, 4.0, 6.0) ASC
        LIMIT ?
    """
    params.append(limit)
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [json.loads(row["doc_json"]) for row in rows]


@lru_cache(maxsize=8)
def _documents_by_id(category: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in load_knowledge_records():
        if not _matches_category(record, category):
            continue
        doc_id = str(record.get("doc_id") or record.get("legacy_id") or record.get("id") or "")
        if doc_id:
            output[doc_id] = record
    return output


@lru_cache(maxsize=4)
def _load_knowledge_faiss_artifacts(category: str) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    paths = _knowledge_faiss_paths(category)
    if paths is None:
        raise FileNotFoundError(f"No knowledge FAISS index for category {category!r}")
    index_path, meta_path, manifest_path = paths
    import faiss

    index = faiss.read_index(str(index_path))
    meta: list[dict[str, Any]] = []
    with meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    manifest: dict[str, Any] = {
        "model": "local_hashing_v1",
        "embedding_provider": "local_hashing",
        "dimensions": 1024,
    }
    if manifest_path.exists():
        try:
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return index, meta, manifest


def _hash_embedding(text: str, dim: int) -> Any:
    return _shared_hash_embedding(text, dim)


def _sentence_transformer_embedding(text: str, manifest: dict[str, Any]) -> Any:
    return _shared_st_embedding(
        text,
        manifest,
        device_env_keys=("AGENT_V3_KNOWLEDGE_EMBED_DEVICE", "AGENT_V3_EMBED_DEVICE"),
    )


def _faiss_candidate_records(
    query: str,
    *,
    category: str | None,
    knowledge_types: list[str] | tuple[str, ...] | set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not category:
        return []
    try:
        index, meta, manifest = _load_knowledge_faiss_artifacts(category)
    except Exception:
        return []
    provider = str(manifest.get("embedding_provider") or "local_hashing")
    if provider == "local_hashing":
        vector = _hash_embedding(query, int(manifest.get("dimensions") or 1024))
    elif provider == "sentence_transformers":
        try:
            vector = _sentence_transformer_embedding(query, manifest)
        except Exception:
            return []
    else:
        return []
    scores, indices = index.search(vector.astype("float32"), max(limit, 1))
    docs_by_id = _documents_by_id(category)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx in indices[0]:
        if idx < 0 or idx >= len(meta):
            continue
        chunk = meta[int(idx)]
        doc_id = str(chunk.get("parent_doc_id") or "")
        record = docs_by_id.get(doc_id)
        if not record:
            continue
        key = str(record.get("doc_id") or record.get("legacy_id") or record.get("id") or doc_id)
        if key in seen:
            continue
        if not _matches_knowledge_types(record, knowledge_types):
            continue
        seen.add(key)
        output.append(record)
    return output


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("doc_id") or record.get("legacy_id") or record.get("id") or record.get("title") or "")


def _query_years(query: str) -> set[str]:
    return set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", query or ""))


def _record_all_text(record: dict[str, Any]) -> str:
    return " ".join(_record_strings(record))


def _temporal_mismatch(query: str, record: dict[str, Any]) -> bool:
    """Detect query/record year mismatch using *content* signals, not file metadata.

    `updated_at` is when the JSONL was last regenerated, not the era of the
    content — relying on it would make every record look fresh forever. We
    instead trust `valid_years`, `tags` (historical_YYYY), and the in-record
    text itself.
    """
    query_years = _query_years(query)
    if not query_years:
        return False
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}

    text = _record_all_text(record)
    if any(year in text for year in query_years):
        return False

    valid_years = {str(y) for y in (record.get("valid_years") or raw.get("valid_years") or [])}
    if valid_years and any(year in valid_years for year in query_years):
        return False

    tags = {str(tag).lower() for tag in [*(record.get("tags") or []), *(raw.get("tags") or [])]}
    historical_years: set[str] = set()
    for tag in tags:
        if tag.startswith("historical_") and len(tag) > len("historical_"):
            historical_years.add(tag.split("historical_", 1)[1])
    if historical_years and any(year in historical_years for year in query_years):
        return False

    # If the record carries explicit era markers (valid_years or historical_*)
    # that disagree with the query year(s), it's a mismatch.
    if valid_years or historical_years:
        return True

    # No era signals at all — be lenient (not a mismatch).
    return False


def _quality_penalty(record: dict[str, Any], reason: str) -> float:
    if reason.startswith("exact:") or reason.startswith("alias_exact:"):
        return 1.0
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    confidence = str(record.get("confidence") or raw.get("confidence") or "").lower()
    source_quality = str(record.get("source_quality") or raw.get("source_quality") or "").lower()
    tags = {str(tag) for tag in [*(record.get("tags") or []), *(raw.get("tags") or [])]}
    if "do_not_use_as_current_fact" in tags or "leak" in tags or "leak" in source_quality:
        return 0.25
    if confidence == "low" or "low_confidence" in source_quality:
        return 0.55
    return 1.0


def _merge_candidate_records(
    bm25_records: list[dict[str, Any]],
    faiss_records: list[dict[str, Any]],
    fallback_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
    merged: dict[str, dict[str, Any]] = {}
    rank_bonus: dict[str, float] = {}
    rank_reason: dict[str, str] = {}
    for source_name, records in (("bm25", bm25_records), ("faiss", faiss_records)):
        for rank, record in enumerate(records, start=1):
            key = _record_key(record)
            if not key:
                continue
            merged.setdefault(key, record)
            contribution = 1.0 / (60 + rank)
            rank_bonus[key] = rank_bonus.get(key, 0.0) + contribution
            old_reason = rank_reason.get(key)
            rank_reason[key] = f"{old_reason}+{source_name}" if old_reason else source_name
    for record in fallback_records:
        key = _record_key(record)
        if key:
            merged.setdefault(key, record)
    return list(merged.values()), rank_bonus, rank_reason


@lru_cache(maxsize=1)
def load_sources_index() -> dict[str, dict[str, Any]]:
    """source_id -> {title, publisher, url, source_quality} from every
    knowledge base's built/sources.jsonl. Used to render clickable source
    links in the tool-trail UI without making the LLM hallucinate URLs."""
    index: dict[str, dict[str, Any]] = {}
    if KB_ROOT.exists():
        for path in sorted(KB_ROOT.glob("*/built/sources.jsonl")):
            for record in _read_jsonl(path):
                sid = str(record.get("id") or "").strip()
                if not sid:
                    continue
                index[sid] = {
                    "id": sid,
                    "title": record.get("title") or sid,
                    "publisher": record.get("publisher"),
                    "url": record.get("url"),
                    "source_quality": record.get("source_quality"),
                }
    return index


def match_to_source_digest(match: dict[str, Any]) -> dict[str, Any]:
    """Compact, UI-safe digest of one knowledge match for the tool trail.

    Pulls the displayable fields out of the verbose `record` blob and
    resolves `source_ids` against the sources index so the popover can show
    real publisher names + URLs (no LLM citation guessing involved).
    """
    record = match.get("record") if isinstance(match.get("record"), dict) else {}
    sources_index = load_sources_index()
    resolved_sources: list[dict[str, Any]] = []
    seen_sids: set[str] = set()
    for sid in record.get("source_ids") or []:
        sid_str = str(sid).strip()
        if not sid_str or sid_str in seen_sids:
            continue
        seen_sids.add(sid_str)
        src = sources_index.get(sid_str)
        if src:
            resolved_sources.append(src)
        else:
            resolved_sources.append({"id": sid_str, "title": sid_str, "url": None})
    snippet = (
        record.get("summary")
        or record.get("text")
        or record.get("normalized")
        or ""
    )
    if isinstance(snippet, str) and len(snippet) > 280:
        snippet = snippet[:277].rstrip() + "…"
    return {
        "id": match.get("id"),
        "title": record.get("title") or record.get("normalized") or str(match.get("id") or ""),
        "type": match.get("type") or match.get("result_type"),
        "score": match.get("score"),
        "match_reason": match.get("match_reason"),
        "snippet": snippet,
        "sources": resolved_sources,
    }


@lru_cache(maxsize=1)
def load_knowledge_records() -> tuple[dict[str, Any], ...]:
    built_records: list[dict[str, Any]] = []
    if KB_ROOT.exists():
        for path in sorted(KB_ROOT.glob("*/built/documents.jsonl")):
            built_records.extend(_read_jsonl(path))
    if built_records:
        return tuple(built_records)

    records: list[dict[str, Any]] = []
    for file_name in KB_FILES:
        path = KB_DIR / file_name
        records.extend(_read_jsonl(path))
    return tuple(records)


def _record_strings(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("text", "normalized", "title", "summary", "retrieval_text", "knowledge_type", "record_type", "category"):
        value = record.get(key)
        if value:
            values.append(str(value))
    for key in ("aliases", "tags", "best_for", "keywords", "source_ids", "valid_years", "aspects", "brands", "models"):
        for value in record.get(key) or []:
            values.append(str(value))
    for key in ("candidate_models", "market", "catalog_availability"):
        value = record.get(key)
        if value:
            values.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    raw = record.get("raw")
    if isinstance(raw, dict):
        values.extend(_record_strings({**raw, "raw": None}))
    entities = record.get("entities")
    if isinstance(entities, dict):
        for entity_values in entities.values():
            if isinstance(entity_values, list):
                values.extend(str(value) for value in entity_values)
    facts = record.get("facts")
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, dict) and fact.get("claim"):
                values.append(str(fact["claim"]))
    return values


def _score_record(query: str, record: dict[str, Any]) -> tuple[float, str]:
    q_norm = normalize(query)
    if not q_norm:
        return 0.0, "empty_query"

    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    primary_strings = [
        str(record.get(key) or raw.get(key) or "")
        for key in ("text", "normalized", "title")
        if record.get(key) or raw.get(key)
    ]
    for value in primary_strings:
        value_norm = normalize(value)
        if q_norm and q_norm == value_norm:
            return 1.2, f"exact:{value}"

    aliases = [str(alias) for alias in record.get("aliases") or []]
    aliases.extend(str(alias) for alias in raw.get("aliases") or [])
    for alias in aliases:
        alias_norm = normalize(alias)
        if alias_norm and q_norm == alias_norm:
            return 0.95, f"alias_exact:{alias}"
        if alias_norm and (alias_norm in q_norm or q_norm in alias_norm):
            return 0.9, f"alias:{alias}"

    strings = _record_strings(record)
    text_norm = normalize(str(record.get("text") or raw.get("text") or ""))
    normalized_norm = normalize(str(record.get("normalized") or raw.get("normalized") or ""))
    for label, value_norm in (("text", text_norm), ("normalized", normalized_norm)):
        if value_norm and (value_norm in q_norm or q_norm in value_norm):
            return 0.86, f"substring:{label}"

    haystack = " ".join(strings)
    haystack_norm = normalize(haystack)
    if haystack_norm and (q_norm in haystack_norm):
        return 0.72, "substring:record"

    q_tokens = tokenize(query)
    r_tokens = tokenize(haystack)
    if q_tokens and r_tokens:
        overlap = q_tokens & r_tokens
        if overlap:
            coverage = len(overlap) / max(1, len(q_tokens))
            if coverage >= 0.66:
                return 0.68 + min(0.12, coverage * 0.12), "token_overlap"
            if coverage >= 0.4:
                return 0.42 + coverage * 0.2, "weak_token_overlap"

    return 0.0, "no_match"


def _result_type(record: dict[str, Any]) -> str:
    if "knowledge_type" in record or "record_type" in record:
        return BUILT_TYPE_MAP.get(
            (str(record.get("knowledge_type") or ""), str(record.get("record_type") or "")),
            str(record.get("knowledge_type") or record.get("record_type") or "unknown"),
        )
    return TYPE_MAP.get(str(record.get("type") or ""), str(record.get("type") or "unknown"))


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "doc_id",
        "legacy_id",
        "id",
        "type",
        "knowledge_type",
        "record_type",
        "title",
        "text",
        "normalized",
        "brand",
        "category",
        "entities",
        "facts",
        "retrieval_text",
        "keywords",
        "aliases",
        "confidence",
        "summary",
        "tags",
        "source",
        "source_ids",
        "source_quality",
        "updated_at",
        "ttl_days",
        "best_for",
        "signals",
        "caveats",
        "key_specs",
        "valid_years",
        "aspects",
        "brands",
        "models",
        "candidate_models",
        "market",
        "catalog_availability",
    }
    public = {key: record[key] for key in allowed if key in record}
    raw = record.get("raw")
    if isinstance(raw, dict):
        public["raw"] = {key: raw[key] for key in allowed if key in raw}
    return public


def _matches_category(record: dict[str, Any], category: str | None) -> bool:
    if not category:
        return True
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    return str(record.get("category") or raw.get("category") or "") == category


def _matches_knowledge_types(record: dict[str, Any], knowledge_types: list[str] | tuple[str, ...] | set[str] | None) -> bool:
    if not knowledge_types:
        return True
    requested = {str(value) for value in knowledge_types}
    result_type = _result_type(record)
    knowledge_type = str(record.get("knowledge_type") or "")
    record_type = str(record.get("record_type") or record.get("type") or "")
    return bool(requested & {result_type, knowledge_type, record_type})


def _public_value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record.get(key)
    raw = record.get("raw")
    if isinstance(raw, dict) and key in raw:
        return raw.get(key)
    return default


def lookup_knowledge(
    query: str,
    *,
    limit: int = 5,
    category: str | None = None,
    knowledge_types: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    from agent_v3.tracing import timed_tool

    query = (query or "").strip()
    if not query:
        return {"found": False, "query": query, "type": None, "result": {}, "matches": []}

    with timed_tool("lookup_knowledge", metadata={
        "query": query,
        "category": category,
        "knowledge_types": list(knowledge_types) if knowledge_types else None,
        "limit": limit,
    }) as md:
        result = _lookup_knowledge_impl(query, limit=limit, category=category, knowledge_types=knowledge_types)
        md["found"] = bool(result.get("found"))
        matches = result.get("matches") or []
        md["match_count"] = len(matches)
        # Compact digests so the UI can render clickable source rows.
        # Capped at 5 — the trail is a research log, not the answer body.
        md["sources"] = [match_to_source_digest(m) for m in matches[:5] if isinstance(m, dict)]
        return result


def _lookup_knowledge_impl(
    query: str,
    *,
    limit: int = 5,
    category: str | None = None,
    knowledge_types: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    bm25_records = _bm25_candidate_records(
        query,
        category=category,
        knowledge_types=knowledge_types,
        limit=max(limit * 20, 50),
    )
    faiss_records = _faiss_candidate_records(
        query,
        category=category,
        knowledge_types=knowledge_types,
        limit=max(limit * 20, 50),
    )
    records, rank_bonus, rank_reason = _merge_candidate_records(
        bm25_records,
        faiss_records,
        list(load_knowledge_records()),
    )

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for record in records:
        if not _matches_category(record, category):
            continue
        if not _matches_knowledge_types(record, knowledge_types):
            continue
        score, reason = _score_record(query, record)
        key = _record_key(record)
        if key in rank_bonus:
            source_reason = rank_reason.get(key, "")
            if source_reason == "bm25+faiss":
                rank_score = 0.52 + min(0.10, rank_bonus[key] * 3)
            elif source_reason == "faiss":
                rank_score = 0.40 + min(0.08, rank_bonus[key] * 3)
            else:
                rank_score = 0.46 + min(0.06, rank_bonus[key] * 2)
            # Don't let pure BM25/FAISS recall lift a record that has no
            # semantic signal in `_score_record` straight to a passing score.
            # Cap the boost so noisy keyword overlap (e.g. "手机" in every
            # smartphone record) cannot masquerade as a real match.
            if score <= 0.0:
                rank_score = min(rank_score, 0.45)
            elif score < 0.4:
                rank_score = min(rank_score, score + 0.15)
            if rank_score > score:
                score = rank_score
                reason = source_reason or "indexed_recall"
        if _temporal_mismatch(query, record):
            score *= 0.25
            reason = f"{reason}; temporal_mismatch"
        penalty = _quality_penalty(record, reason)
        if penalty < 1.0:
            score *= penalty
            reason = f"{reason}; low_confidence"
        # Brand/category alias records are pure synonym mappings — they tell
        # the agent "iQOO is a brand alias" or "手机 means smartphone" but
        # carry no spec/model knowledge. Useful as fallback context but should
        # not dominate top matches over real model/review records.
        raw_type = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        record_type_str = str(
            record.get("record_type")
            or record.get("knowledge_type")
            or raw_type.get("type")
            or record.get("type")
            or ""
        )
        if record_type_str in {"brand_alias", "category_alias", "brand", "category"}:
            score = min(score, 0.5)
            reason = f"{reason}; alias_only_demote"
        if score > 0:
            scored.append((score, reason, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [
        {
            "id": record.get("legacy_id") or record.get("id") or record.get("doc_id"),
            "type": record.get("record_type") or record.get("type") or record.get("knowledge_type"),
            "result_type": _result_type(record),
            "score": round(score, 4),
            "match_reason": reason,
            "record": _public_record(record),
        }
        for score, reason, record in scored[:limit]
        if score >= 0.55
    ]

    if not matches:
        return {"found": False, "query": query, "type": None, "result": {}, "matches": []}

    best = matches[0]
    record = best["record"]
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    result = {
        "id": record.get("legacy_id") or record.get("id") or record.get("doc_id"),
        "doc_id": record.get("doc_id"),
        "normalized": _public_value(record, "normalized") or raw.get("normalized") or record.get("title") or _public_value(record, "text"),
        "confidence": _public_value(record, "confidence", "medium"),
        "summary": _public_value(record, "summary", ""),
        "aliases": _public_value(record, "aliases", []),
        "source": _public_value(record, "source"),
        "source_ids": _public_value(record, "source_ids", []),
        "source_quality": _public_value(record, "source_quality"),
        "updated_at": _public_value(record, "updated_at"),
        "ttl_days": _public_value(record, "ttl_days"),
        "knowledge_type": record.get("knowledge_type"),
        "record_type": record.get("record_type") or raw.get("type"),
        "entities": record.get("entities", {}),
        "facts": record.get("facts", []),
    }
    for key in (
        "brand",
        "category",
        "best_for",
        "signals",
        "caveats",
        "key_specs",
        "tags",
        "keywords",
        "valid_years",
        "aspects",
        "brands",
        "models",
        "candidate_models",
        "market",
        "catalog_availability",
    ):
        value = _public_value(record, key)
        if value is not None:
            result[key] = value

    return {
        "found": True,
        "query": query,
        "category": category,
        "knowledge_types": list(knowledge_types) if knowledge_types else None,
        "type": best["result_type"],
        "result": result,
        "matches": matches,
    }
