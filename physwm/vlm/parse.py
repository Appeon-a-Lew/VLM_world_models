"""Parse model answers into {object id: [x, y, z]} (camera frame, metres)."""

from __future__ import annotations

import json
import re

from physwm.schema import Episode

_SYNONYMS = {"sphere": "ball", "cube": "box"}
_ENTRY_RE = re.compile(
    r'"label"\s*:\s*"([^"]+)"[^{}]*?"(?:center|bbox_3d)"\s*:\s*\[([^\]]+)\]'
    r'|"(?:center|bbox_3d)"\s*:\s*\[([^\]]+)\][^{}]*?"label"\s*:\s*"([^"]+)"'
)


def _norm(label: str) -> str:
    words = label.lower().replace("_", " ").strip().strip(".").split()
    return " ".join(_SYNONYMS.get(w, w) for w in words if w not in ("the", "a"))


def _entries(text: str) -> list[tuple[str, list[float]]]:
    text = re.sub(r"```(?:json)?", "", text)
    lo, hi = text.find("["), text.rfind("]")
    if lo != -1 and hi > lo:
        try:
            data = json.loads(text[lo:hi + 1])
            if isinstance(data, dict):
                data = [data]
            out = []
            for e in data:
                if not isinstance(e, dict) or "label" not in e:
                    continue
                v = e.get("center", e.get("bbox_3d"))
                if isinstance(v, list) and len(v) >= 3:
                    out.append((str(e["label"]), [float(x) for x in v[:3]]))
            if out:
                return out
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # fallback: regex over possibly truncated / malformed JSON
    out = []
    for m in _ENTRY_RE.finditer(text):
        label, nums = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        try:
            vals = [float(x) for x in nums.split(",")[:3]]
        except ValueError:
            continue
        if len(vals) == 3:
            out.append((label, vals))
    return out


def parse_answer(text: str, ep: Episode) -> tuple[dict[str, list[float] | None], bool]:
    """Returns (objects, parsed). `parsed` is False when nothing usable was found."""
    by_name = {_norm(o.name): o.id for o in ep.objects}
    objects: dict[str, list[float] | None] = {str(o.id): None for o in ep.objects}
    entries = _entries(text)
    for label, xyz in entries:
        oid = by_name.get(_norm(label))
        if oid is not None and objects[str(oid)] is None:
            objects[str(oid)] = xyz
    return objects, any(v is not None for v in objects.values())
