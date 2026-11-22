"""Simulator-provided physical context: the pre-action geometry behind the ctx3d style.

The ``action_context`` payload emitted by the ctx3d answer style (and injected verbatim
into prompts by ``infer.py --oracle-context``) is built here from episode ground truth:

- ``edge_distance_m`` -- how far the target centre is from the table boundary along the
  action direction (world frame, metres);
- ``obstacles`` -- other objects inside the push corridor, nearest surface first, with
  the surface-to-surface gap along the direction (0.0 when already touching);
- per-object ``support`` -- the name of the supporting object, "table", or "none"
  (world-frame rest-pose analysis; vacuous for the push family by construction, meaningful
  from the stack/drop families onward).

The payload format is frozen for the whole M0/M1 ladder: the ctx3d answer schema does not
change when the new interaction families arrive, so no answer-format confound can enter
the staged comparison.
"""

from __future__ import annotations

import numpy as np

from physwm.schema import Episode

SUPPORT_TOL = 0.005  # m; rest-contact tolerance for support detection
MAX_OBSTACLES = 3


def _footprint_radius(spec) -> float:
    """Approximate horizontal reach of an upright body around its centre."""
    return float(max(spec.half_extents[0], spec.half_extents[1]))


def support_map(ep: Episode, t: float = 0.0) -> dict[int, str]:
    """Object id -> what it rests on at time ``t``: a supporter's name, "table", or "none"."""
    states = ep.state_at(t)
    out: dict[int, str] = {}
    for i, (spec, st) in enumerate(zip(ep.objects, states)):
        bottom = st.pos[2] - spec.half_extents[2]
        best: tuple[float, str] | None = None
        for j, (other, ost) in enumerate(zip(ep.objects, states)):
            if j == i:
                continue
            top = ost.pos[2] + other.half_extents[2]
            if not (bottom - SUPPORT_TOL <= top <= bottom + SUPPORT_TOL):
                continue
            if (abs(st.pos[0] - ost.pos[0]) < spec.half_extents[0] + other.half_extents[0]
                    and abs(st.pos[1] - ost.pos[1]) < spec.half_extents[1] + other.half_extents[1]):
                # a nested stack rests on its highest valid supporter
                if best is None or top > best[0]:
                    best = (top, other.name)
        if best is not None:
            out[spec.id] = best[1]
        elif abs(bottom) < SUPPORT_TOL and st.on_table:
            out[spec.id] = "table"
        else:
            out[spec.id] = "none"
    return out


def edge_distance(ep: Episode, origin_xy, direction_xy) -> float:
    """Distance (m) from ``origin_xy`` to the table boundary along ``direction_xy``."""
    d = np.asarray(direction_xy, dtype=float)
    if d.shape != (2,) or not np.all(np.isfinite(d)) or float(np.linalg.norm(d)) < 1e-9:
        raise ValueError("direction must be a finite non-zero 2D vector")
    d = d / np.linalg.norm(d)
    p = np.asarray(origin_xy, dtype=float)
    best = float("inf")
    for axis, half in zip((0, 1), ep.table_half_size[:2]):
        if abs(d[axis]) < 1e-12:
            continue
        for bound in (half, -half):
            t_hit = (bound - p[axis]) / d[axis]
            if t_hit > 1e-9:
                best = min(best, t_hit)
    if not np.isfinite(best):
        raise ValueError("degenerate action direction: ray never leaves the table")
    return float(best)


def obstacles(ep: Episode, t: float = 0.0) -> list[dict]:
    """Other objects inside the action corridor at time ``t``, nearest surface first."""
    states = ep.state_at(t)
    target = ep.action.target
    tpos = np.asarray(states[target].pos[:2], dtype=float)
    d = np.asarray(ep.action.direction_world[:2], dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    r_t = _footprint_radius(ep.objects[target])
    out = []
    for j, (spec, st) in enumerate(zip(ep.objects, states)):
        if j == target:
            continue
        rel = np.asarray(st.pos[:2], dtype=float) - tpos
        along = float(rel @ d)
        if along <= 0:
            continue
        perp = float(np.linalg.norm(rel - along * d))
        if perp > r_t + _footprint_radius(spec):
            continue
        gap = max(0.0, along - r_t - _footprint_radius(spec))
        out.append((along, {"label": spec.name, "distance_m": round(gap, 3)}))
    out.sort(key=lambda x: x[0])
    return [item for _, item in out[:MAX_OBSTACLES]]


def action_context(ep: Episode) -> dict:
    """The frozen ctx3d action-context payload (pre-action, world frame)."""
    states = ep.state_at(0.0)
    return {
        "target": ep.objects[ep.action.target].name,
        "direction": ep.action.direction_label,
        "edge_distance_m": round(edge_distance(
            ep, states[ep.action.target].pos[:2], ep.action.direction_world[:2]), 3),
        "obstacles": obstacles(ep, 0.0),
    }


def oracle_payload(ep: Episode, t: float = 0.0) -> dict:
    """Simulator-provided context for ``--oracle-context``: GT per-object support alongside
    the frozen action payload.

    Injecting support too keeps the oracle diagnostic honest: a flat oracle result cannot
    hide a support-prediction failure, because predicted support is scored separately
    (``idpmetrics.summarize_ctx``) rather than silently replaced.
    """
    sup = support_map(ep, t)
    return {"support": {o.name: sup[o.id] for o in ep.objects},
            "action_context": action_context(ep)}
