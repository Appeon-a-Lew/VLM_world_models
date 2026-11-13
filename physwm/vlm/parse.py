"""Parse model answers into {object id: [x, y, z]} (camera frame, metres).

:func:`parse_box3d` additionally extracts full body-axis ``extents`` per object for the
phase-2 centre + extents answer format.
"""

from __future__ import annotations

import json
import math
import re

from physwm.schema import Episode

_SYNONYMS = {"sphere": "ball", "cube": "box"}
_BOX_RE = re.compile(r"\{[^{}]*\}")
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
        if not all(math.isfinite(v) for v in xyz):
            continue
        oid = by_name.get(_norm(label))
        if oid is not None and objects[str(oid)] is None:
            objects[str(oid)] = xyz
    return objects, any(v is not None for v in objects.values())


def _finite3(values) -> list[float] | None:
    try:
        vals = [float(v) for v in values[:3]]
    except (TypeError, ValueError):
        return None
    return vals if len(vals) == 3 and all(math.isfinite(v) for v in vals) else None


def _fields(chunk: str) -> tuple[str | None, list[float] | None, list[float] | None]:
    """Pull (label, centre, extents) out of one ``{...}`` entry, tolerating extra keys."""
    label = re.search(r'"label"\s*:\s*"([^"]+)"', chunk)
    centre = re.search(r'"(?:center|bbox_3d)"\s*:\s*\[([^\]]*)\]', chunk)
    extents = re.search(r'"extents"\s*:\s*\[([^\]]*)\]', chunk)

    def nums(m):
        if m is None:
            return None
        try:
            return _finite3(m.group(1).split(","))
        except ValueError:
            return None

    return (label.group(1) if label else None), nums(centre), nums(extents)


def _box_entries(text: str) -> list[tuple[str, list[float] | None, list[float] | None]]:
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
                out.append((str(e["label"]), _finite3(e.get("center", e.get("bbox_3d")) or []),
                            _finite3(e.get("extents") or [])))
            if out:
                return out
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    out = []
    chunks = _BOX_RE.findall(text)
    if not chunks:  # truncated answer without the closing brace: split on labels instead
        starts = [m.start() for m in re.finditer(r'"label"\s*:', text)]
        chunks = [text[s:e] for s, e in zip(starts, starts[1:] + [len(text)])]
    for chunk in chunks:
        label, centre, extents = _fields(chunk)
        if label is not None and (centre is not None or extents is not None):
            out.append((label, centre, extents))
    return out


def parse_box3d(text: str, ep: Episode) -> tuple[dict[str, list[float] | None],
                                                dict[str, list[float] | None], bool]:
    """Centre + extents variant of :func:`parse_answer`.

    Returns ``(centres, extents, parsed)`` keyed by ``str(object id)``; missing or
    non-finite fields stay ``None``. ``parsed`` is False when no centre could be matched.
    """
    by_name = {_norm(o.name): o.id for o in ep.objects}
    centres: dict[str, list[float] | None] = {str(o.id): None for o in ep.objects}
    extents: dict[str, list[float] | None] = {str(o.id): None for o in ep.objects}
    for label, centre, ext in _box_entries(text):
        oid = by_name.get(_norm(label))
        if oid is None:
            continue
        if centre is not None and centres[str(oid)] is None:
            centres[str(oid)] = centre
        if ext is not None and extents[str(oid)] is None:
            extents[str(oid)] = ext
    return centres, extents, any(v is not None for v in centres.values())
