"""Incremental fragment extractor for streamed router JSON.

The router LLM emits a single JSON object like:

    {
      "type": "clarify_user",
      "reason": "...",
      "categories": ["cell_phone"],
      "question": "你想要哪种类型的手机？",
      "options": [
        {"id": "1", "label": "...", "description": "..."},
        {"id": "2", "label": "...", "description": "..."}
      ]
    }

The full JSON only becomes parseable once the closing `}` arrives. To make
the UI feel "live" (Claude-style), we extract two kinds of fragments
incrementally as tokens stream in:

  - the `question` string (emitted once when its closing quote arrives)
  - each `options[i]` object (emitted once when its closing brace arrives)

The router prompt is responsible for guaranteeing `question` lands in the
output stream before `options[]` so the user sees the question first.
"""

from __future__ import annotations

import json
import re
from typing import Any


_QUESTION_VALUE_RE = re.compile(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"')


class StreamingRouterFragmenter:
    """Stateful parser that ingests JSON tokens and yields fragment events.

    Usage:
        f = StreamingRouterFragmenter()
        for delta in stream:
            for event in f.feed(delta):
                ...  # event is {"kind": "question", "text": ...} or
                     #          {"kind": "option", "option": {...}}
    """

    def __init__(self) -> None:
        self.buffer: str = ""
        self.question_emitted: bool = False
        self.emitted_option_keys: set[str] = set()

    def feed(self, delta: str) -> list[dict[str, Any]]:
        if not delta:
            return []
        self.buffer += delta
        out: list[dict[str, Any]] = []
        out.extend(self._extract_question())
        out.extend(self._extract_options())
        return out

    def _extract_question(self) -> list[dict[str, Any]]:
        if self.question_emitted:
            return []
        m = _QUESTION_VALUE_RE.search(self.buffer)
        if not m:
            return []
        # Decode JSON string escapes (\n, \", \\, \uXXXX) by feeding the
        # captured value back through json.loads after re-quoting.
        try:
            text = json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            return []
        self.question_emitted = True
        return [{"kind": "question", "text": text}]

    def _extract_options(self) -> list[dict[str, Any]]:
        # Find the start of the options array. We don't care if it's been
        # emitted before — _emitted_option_keys dedups individual entries.
        opt_label_idx = self.buffer.find('"options"')
        if opt_label_idx < 0:
            return []
        arr_open = self.buffer.find('[', opt_label_idx)
        if arr_open < 0:
            return []

        out: list[dict[str, Any]] = []
        i = arr_open + 1
        n = len(self.buffer)
        while i < n:
            # Skip whitespace and commas between options.
            while i < n and self.buffer[i] in " \t\n\r,":
                i += 1
            if i >= n or self.buffer[i] == "]":
                break
            if self.buffer[i] != "{":
                # Unexpected — bail out and let next feed retry.
                break
            end = self._scan_balanced_object(i)
            if end < 0:
                # Object not yet complete; stop and wait for more bytes.
                break
            chunk = self.buffer[i : end + 1]
            try:
                option = json.loads(chunk)
            except json.JSONDecodeError:
                # Malformed (shouldn't happen if scan is correct); skip past it.
                i = end + 1
                continue
            key = self._option_key(option, position=len(self.emitted_option_keys))
            if key not in self.emitted_option_keys:
                self.emitted_option_keys.add(key)
                out.append({"kind": "option", "option": option})
            i = end + 1
        return out

    def _scan_balanced_object(self, start: int) -> int:
        """Return index of `}` that closes the object starting at `start`,
        or -1 if not yet in buffer. Respects strings and escapes."""
        depth = 0
        in_string = False
        escape = False
        i = start
        n = len(self.buffer)
        while i < n:
            c = self.buffer[i]
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif in_string:
                if c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    @staticmethod
    def _option_key(option: dict[str, Any], *, position: int) -> str:
        opt_id = str(option.get("id") or "").strip()
        if opt_id:
            return f"id:{opt_id}"
        label = str(option.get("label") or "").strip()
        if label:
            return f"label:{label}"
        return f"pos:{position}"
