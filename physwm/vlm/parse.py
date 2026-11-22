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


def _match_entries(entries, ep: Episode):
    by_name = {_norm(o.name): o.id for o in ep.objects}
    centres: dict[str, list[float] | None] = {str(o.id): None for o in ep.objects}
    extents: dict[str, list[float] | None] = {str(o.id): None for o in ep.objects}
    for label, centre, ext in entries:
        oid = by_name.get(_norm(label))
        if oid is None:
            continue
        if centre is not None and centres[str(oid)] is None:
            centres[str(oid)] = centre
        if ext is not None and extents[str(oid)] is None:
            extents[str(oid)] = ext
    return centres, extents, any(v is not None for v in centres.values())


def parse_chain3d(text: str, ep: Episode):
    """Parse ``{"current": [...], "future": [...]}`` Stage-1 chained 3D box predictions.

    Returns ``(cur_centres, cur_extents, fut_centres, fut_extents, parsed_cur, parsed_fut)``.
    """
    cleaned = re.sub(r"```(?:json)?", "", text)
    lo, hi = cleaned.find("{"), cleaned.rfind("}")
    if lo != -1 and hi > lo:
        try:
            data = json.loads(cleaned[lo:hi + 1])
            if isinstance(data, dict) and ("current" in data or "future" in data):
                def _from_list(items):
                    if not isinstance(items, list):
                        return []
                    out = []
                    for e in items:
                        if not isinstance(e, dict) or "label" not in e:
                            continue
                        out.append((
                            str(e["label"]),
                            _finite3(e.get("center", e.get("bbox_3d")) or []),
                            _finite3(e.get("extents") or []),
                        ))
                    return out
                c_c, c_e, c_ok = _match_entries(_from_list(data.get("current")), ep)
                f_c, f_e, f_ok = _match_entries(_from_list(data.get("future")), ep)
                return c_c, c_e, f_c, f_e, c_ok, f_ok
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: split on "current" / "future" keys if output was truncated
    m_cur = re.search(r'"current"\s*:', cleaned)
    m_fut = re.search(r'"future"\s*:', cleaned)
    cur_txt = cleaned[m_cur.end():m_fut.start() if m_fut else len(cleaned)] if m_cur else ""
    fut_txt = cleaned[m_fut.end():] if m_fut else ""
    c_c, c_e, c_ok = _match_entries(_box_entries(cur_txt), ep) if cur_txt else _match_entries([], ep)
    f_c, f_e, f_ok = _match_entries(_box_entries(fut_txt), ep) if fut_txt else _match_entries([], ep)
    return c_c, c_e, f_c, f_e, c_ok, f_ok


def parse_ctx3d(text: str, ep: Episode):
    """Parse a ctx3d answer: current boxes, future boxes, the emitted action context and
    the emitted per-object support.

    Returns ``(cur_centres, cur_extents, fut_centres, fut_extents, ctx, support, parsed_cur,
    parsed_fut)``; ``ctx`` / ``support`` are raw dicts or ``None`` when absent or malformed
    (the current/future parsing matches :func:`parse_chain3d`; the extra keys are ignored
    there and scored here instead).
    """
    cleaned = re.sub(r"```(?:json)?", "", text)
    ctx = None
    m = re.search(r'"action_context"\s*:\s*\{', cleaned)
    if m:
        depth, start = 0, m.end() - 1
        for k in range(start, len(cleaned)):
            if cleaned[k] == "{":
                depth += 1
            elif cleaned[k] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        ctx = json.loads(cleaned[start:k + 1])
                    except (json.JSONDecodeError, ValueError):
                        ctx = None
                    break
    support = _support_entries(cleaned, ep)
    c_c, c_e, f_c, f_e, c_ok, f_ok = parse_chain3d(text, ep)
    return c_c, c_e, f_c, f_e, ctx, support, c_ok, f_ok


def _support_entries(cleaned: str, ep: Episode) -> dict[str, str] | None:
    """Per-object ``support`` strings emitted in the "current" list: {str(object id):
    value}, or None when the answer emits none."""
    by_name = {_norm(o.name): o.id for o in ep.objects}
    data = None
    lo, hi = cleaned.find("{"), cleaned.rfind("}")
    if lo != -1 and hi > lo:
        try:
            loaded = json.loads(cleaned[lo:hi + 1])
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, ValueError):
            pass
    out: dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("current"), list):
        for e in data["current"]:
            if isinstance(e, dict) and isinstance(e.get("support"), str):
                oid = by_name.get(_norm(str(e.get("label", ""))))
                if oid is not None and str(oid) not in out:
                    out[str(oid)] = e["support"].strip().lower()
        return out or None
    for m in re.finditer(r'"label"\s*:\s*"([^"]+)"[^{}]*?"support"\s*:\s*"([^"]*)"', cleaned):
        oid = by_name.get(_norm(m.group(1)))
        if oid is not None and str(oid) not in out:
            out[str(oid)] = m.group(2).strip().lower()
    return out or None


# Vocabulary duplicated from physwm.vlm.prompts (kept local so parse stays import-light;
# tests/test_idp.py asserts the two stay in sync).
_IDP_DIRECTIONS = ("left", "right", "away", "toward")
_IDP_STRENGTHS = ("weak", "medium", "strong")
_NO_MOTION = "no clear motion"


def parse_idp(text: str, ep: Episode) -> tuple[dict | None, bool]:
    """Parse an IDP answer: the target/direction/strength triple, the explicit no-motion
    class, or nothing usable.

    Returns ``(answer, parsed)``. On a full triple ``answer["target"]`` is the matched
    object id; the no-motion class returns ``{"action": "no clear motion"}``.
    """
    cleaned = re.sub(r"```(?:json)?", "", text)
    data: dict = {}
    lo, hi = cleaned.find("{"), cleaned.rfind("}")
    if lo != -1 and hi > lo:
        try:
            loaded = json.loads(cleaned[lo:hi + 1])
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, ValueError):
            pass
    if not data:  # fallback: pull the fields individually out of truncated output
        m = re.search(r'"action"\s*:\s*"([^"]+)"', cleaned)
        if m:
            data = {"action": m.group(1)}
        for key in ("target", "direction", "strength"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', cleaned)
            if m:
                data[key] = m.group(1)
    if str(data.get("action", "")).strip().lower() == _NO_MOTION:
        return {"action": _NO_MOTION}, True
    target = data.get("target")
    by_name = {_norm(o.name): o.id for o in ep.objects}
    oid = by_name.get(_norm(str(target))) if target is not None else None
    direction = str(data.get("direction", "")).strip().lower()
    strength = str(data.get("strength", "")).strip().lower()
    if oid is None or direction not in _IDP_DIRECTIONS or strength not in _IDP_STRENGTHS:
        return None, False
    return {"target": oid, "direction": direction, "strength": strength}, True
