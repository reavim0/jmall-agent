from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Any


_CJK_RANGE = "一-鿿"
_TOKEN_RE = re.compile(rf"[a-z0-9]+|[{_CJK_RANGE}]+", re.IGNORECASE)
_CJK_FULLMATCH = re.compile(rf"[{_CJK_RANGE}]+")


def _tokenize(text: str) -> list[str]:
    """Tokenization shared with `lookup_knowledge.tokenize`. Splits on
    latin/digit runs and CJK runs, then expands CJK runs into 2- and 3-grams."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).casefold()
        if not token:
            continue
        tokens.append(token)
        if _CJK_FULLMATCH.fullmatch(token):
            for size in (2, 3):
                if len(token) > size:
                    tokens.extend(token[index : index + size] for index in range(len(token) - size + 1))
    return tokens


def hash_embedding(text: str, dim: int) -> Any:
    """Deterministic local hashing embedding shared by knowledge FAISS."""
    import numpy as np

    matrix = np.zeros((1, dim), dtype="float32")
    for token in _tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        column = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        matrix[0, column] += sign
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return matrix / norm


@lru_cache(maxsize=4)
def load_sentence_transformer(model_name: str, device: str, max_seq_length: int | None) -> Any:
    """Cache shared by all callers so the same SentenceTransformer model is
    only loaded into memory once per process — even when both catalog and
    knowledge retrieval ask for the same model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    return model


def sentence_transformer_embedding(
    text: str,
    manifest: dict[str, Any],
    *,
    device_env_keys: tuple[str, ...] = ("AGENT_V3_EMBED_DEVICE",),
) -> Any:
    """Encode `text` using the SentenceTransformer model declared in
    `manifest`. Manifest fields: model, device, max_seq_length, normalized."""
    model_name = str(manifest.get("model") or "BAAI/bge-m3")
    device = ""
    for key in device_env_keys:
        if os.getenv(key):
            device = os.getenv(key) or ""
            break
    if not device:
        device = str(manifest.get("device") or "cpu")
    max_seq_length_value = manifest.get("max_seq_length")
    max_seq_length = int(max_seq_length_value) if max_seq_length_value else None
    model = load_sentence_transformer(model_name, device, max_seq_length)
    vector = model.encode(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=bool(manifest.get("normalized", True)),
        show_progress_bar=False,
    )
    return vector.astype("float32")
