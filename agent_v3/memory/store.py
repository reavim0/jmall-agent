from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent_v3.state import BehaviorEvent, UserMemoryItem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_DB_PATH = PROJECT_ROOT / "agent_v3" / "data" / "memory.sqlite"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS behavior_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    category TEXT,
    scope TEXT NOT NULL,
    signal_strength TEXT NOT NULL,
    confidence REAL NOT NULL,
    target_id TEXT,
    raw_id TEXT,
    title TEXT,
    brand TEXT,
    key TEXT,
    value TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_behavior_events_user_time
ON behavior_events(user_id, created_at);

CREATE TABLE IF NOT EXISTS user_memory (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT,
    scope TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    decayed_score REAL NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_memory_lookup
ON user_memory(user_id, category, scope, memory_type, confidence);

CREATE TABLE IF NOT EXISTS aggregate_cooccurrence (
    edge_id TEXT PRIMARY KEY,
    category TEXT,
    left_key TEXT NOT NULL,
    left_value TEXT NOT NULL,
    right_key TEXT NOT NULL,
    right_value TEXT NOT NULL,
    weight REAL NOT NULL,
    support_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_aggregate_cooccurrence_left
ON aggregate_cooccurrence(category, left_key, left_value, weight);
"""


class MemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_MEMORY_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA_SQL)

    def upsert_behavior_event(self, event: BehaviorEvent) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO behavior_events (
                    event_id, user_id, session_id, turn_id, event_type, category, scope,
                    signal_strength, confidence, target_id, raw_id, title, brand, key, value,
                    source, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    event.get("event_id"),
                    event.get("user_id"),
                    event.get("session_id"),
                    event.get("turn_id"),
                    event.get("event_type"),
                    event.get("category"),
                    event.get("scope"),
                    event.get("signal_strength"),
                    event.get("confidence"),
                    event.get("target_id"),
                    event.get("raw_id"),
                    event.get("title"),
                    event.get("brand"),
                    event.get("key"),
                    event.get("value"),
                    event.get("source"),
                    event.get("created_at"),
                    json.dumps(event.get("metadata") or {}, ensure_ascii=False),
                ],
            )

    def list_behavior_events(self, user_id: str, *, limit: int = 50) -> list[BehaviorEvent]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM behavior_events
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [user_id, limit],
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def upsert_user_memory(self, item: UserMemoryItem) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO user_memory (
                    memory_id, user_id, category, scope, memory_type, key, value,
                    confidence, evidence_count, positive_count, negative_count,
                    decayed_score, first_seen_at, last_seen_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    item.get("memory_id"),
                    item.get("user_id"),
                    item.get("category"),
                    item.get("scope"),
                    item.get("memory_type"),
                    item.get("key"),
                    item.get("value"),
                    item.get("confidence"),
                    item.get("evidence_count"),
                    item.get("positive_count"),
                    item.get("negative_count"),
                    item.get("decayed_score"),
                    item.get("first_seen_at"),
                    item.get("last_seen_at"),
                    json.dumps(item.get("metadata") or {}, ensure_ascii=False),
                ],
            )

    def get_user_memory(self, memory_id: str) -> UserMemoryItem | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM user_memory WHERE memory_id = ?", [memory_id]).fetchone()
        return self._row_to_memory(row) if row else None

    def list_user_memory(
        self,
        user_id: str,
        *,
        category: str | None = None,
        scope: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 30,
    ) -> list[UserMemoryItem]:
        where = ["user_id = ?", "confidence >= ?"]
        params: list[Any] = [user_id, min_confidence]
        if category:
            where.append("(category = ? OR category IS NULL)")
            params.append(category)
        if scope:
            where.append("scope = ?")
            params.append(scope)
        params.append(limit)
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT * FROM user_memory
                WHERE {" AND ".join(where)}
                ORDER BY confidence DESC, evidence_count DESC, last_seen_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def upsert_cooccurrence_edge(self, edge: dict[str, Any]) -> None:
        with self.connect() as con:
            old = con.execute(
                "SELECT weight, support_count FROM aggregate_cooccurrence WHERE edge_id = ?",
                [edge.get("edge_id")],
            ).fetchone()
            weight = float(edge.get("weight") or 0.0)
            support_count = int(edge.get("support_count") or 1)
            if old:
                weight += float(old["weight"] or 0.0)
                support_count += int(old["support_count"] or 0)
            con.execute(
                """
                INSERT OR REPLACE INTO aggregate_cooccurrence (
                    edge_id, category, left_key, left_value, right_key, right_value,
                    weight, support_count, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    edge.get("edge_id"),
                    edge.get("category"),
                    edge.get("left_key"),
                    edge.get("left_value"),
                    edge.get("right_key"),
                    edge.get("right_value"),
                    weight,
                    support_count,
                    edge.get("updated_at"),
                    json.dumps(edge.get("metadata") or {}, ensure_ascii=False),
                ],
            )

    def list_cooccurrence_edges(
        self,
        *,
        category: str | None,
        left_key: str,
        left_value: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [left_key, left_value]
        where = ["left_key = ?", "left_value = ?"]
        if category:
            where.append("(category = ? OR category IS NULL)")
            params.append(category)
        params.append(limit)
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT *
                FROM aggregate_cooccurrence
                WHERE {" AND ".join(where)}
                ORDER BY weight DESC, support_count DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_cooccurrence(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> BehaviorEvent:
        return {
            "event_id": row["event_id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "turn_id": row["turn_id"],
            "event_type": row["event_type"],
            "category": row["category"],
            "scope": row["scope"],
            "signal_strength": row["signal_strength"],
            "confidence": float(row["confidence"]),
            "target_id": row["target_id"],
            "raw_id": row["raw_id"],
            "title": row["title"],
            "brand": row["brand"],
            "key": row["key"],
            "value": row["value"],
            "source": row["source"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> UserMemoryItem:
        return {
            "memory_id": row["memory_id"],
            "user_id": row["user_id"],
            "category": row["category"],
            "scope": row["scope"],
            "memory_type": row["memory_type"],
            "key": row["key"],
            "value": row["value"],
            "confidence": float(row["confidence"]),
            "evidence_count": int(row["evidence_count"]),
            "positive_count": int(row["positive_count"]),
            "negative_count": int(row["negative_count"]),
            "decayed_score": float(row["decayed_score"]),
            "first_seen_at": row["first_seen_at"],
            "last_seen_at": row["last_seen_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    @staticmethod
    def _row_to_cooccurrence(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "edge_id": row["edge_id"],
            "category": row["category"],
            "left_key": row["left_key"],
            "left_value": row["left_value"],
            "right_key": row["right_key"],
            "right_value": row["right_value"],
            "weight": float(row["weight"]),
            "support_count": int(row["support_count"]),
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }
