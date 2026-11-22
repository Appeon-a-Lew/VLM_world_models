"""Accuracy reporting for the auxiliary M0 channels.

Neither channel is a gate: research/PREREG.md gates on the Task-B paired effect error
only. These tables keep the screen honest about whether the auxiliary heads actually
learned their jobs:

- IDP answers (task I): target/direction/strength accuracy against the GT triple, plus
  the explicit no-motion class;
- ctx3d emitted action contexts: the *predicted-context* diagnostic that pairs with
  ``infer.py --oracle-context`` (predicted vs simulator-provided context localizes the
  bottleneck to context prediction or context use).
"""

from __future__ import annotations

from collections import defaultdict

from physwm.schema import Episode
from physwm.vlm.context import action_context, support_map
from physwm.vlm.prompts import IDP_TASK, NO_MOTION_LABEL, idp_target


def summarize_idp(preds: list[dict], episodes: dict[str, Episode]) -> list[dict]:
    """Per-split IDP accuracy rows (one row per split present in the predictions)."""
    by_split: dict[str, list] = defaultdict(list)
    for p in preds:
        ep = episodes.get(p.get("episode_id"))
        if ep is not None and p.get("task") == IDP_TASK:
            by_split[ep.split].append((p, ep))
    out = []
    for split, pairs in sorted(by_split.items()):
        rows = []
        for p, ep in pairs:
            gt = idp_target(ep)
            ans = p.get("idp")
            if not isinstance(ans, dict):
                rows.append({"parsed": False, "exact": 0, "target": None, "direction": None,
                             "strength": None, "gt_no_motion": False})
                continue
            if gt.get("action") == NO_MOTION_LABEL:
                rows.append({"parsed": bool(p.get("parsed")),
                             "exact": int(ans.get("action") == NO_MOTION_LABEL),
                             "target": None, "direction": None, "strength": None,
                             "gt_no_motion": True})
                continue
            want_id = ep.objects[ep.action.target].id
            rows.append({
                "parsed": bool(p.get("parsed")),
                "exact": int(ans.get("target") == want_id
                             and ans.get("direction") == gt["direction"]
                             and ans.get("strength") == gt["strength"]),
                "target": int(ans.get("target") == want_id),
                "direction": int(ans.get("direction") == gt["direction"]),
                "strength": int(ans.get("strength") == gt["strength"]),
                "gt_no_motion": False,
            })
        action_rows = [r for r in rows if not r["gt_no_motion"]]
        nomotion_rows = [r for r in rows if r["gt_no_motion"]]

        def mean(sel, key):
            vals = [r[key] for r in sel if r[key] is not None]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        out.append({
            "split": split, "task": IDP_TASK, "episodes": len(pairs),
            "parse_rate": mean(rows, "parsed"),
            "exact_acc": mean(rows, "exact"),
            "target_acc": mean(action_rows, "target"),
            "direction_acc": mean(action_rows, "direction"),
            "strength_acc": mean(action_rows, "strength"),
            "no_motion_acc": mean(nomotion_rows, "exact"),
            "n_action_gt": len(action_rows), "n_no_motion_gt": len(nomotion_rows),
        })
    return out


def summarize_ctx(preds: list[dict], episodes: dict[str, Episode]) -> list[dict]:
    """Emitted vs simulator action context on ctx3d forward-prediction records.

    Model output is untrusted: malformed fields (wrong types, ``"obstacles": null``) count
    as missing and never raise, so one bad generation cannot take the arm's report down,
    while the Task-B forecast of the same record stays scored by the main pipeline.
    Predicted per-object support is scored separately from the oracle payload, which
    supplies it (research/PREREG.md addendum: oracle mode must not leave support untested).
    """
    by_split: dict[str, list] = defaultdict(list)
    for p in preds:
        ep = episodes.get(p.get("episode_id"))
        if ep is not None and "action_context" in p and p.get("task") in ("B", "N"):
            by_split[ep.split].append((p, ep))
    out = []
    for split, pairs in sorted(by_split.items()):
        n = len(pairs)
        target = direction = obstacles = parsed = 0
        edge_errs: list[float] = []
        sup_hits = sup_total = 0
        for p, ep in pairs:
            ctx = p.get("action_context")
            if not isinstance(ctx, dict):
                continue
            parsed += 1
            gt = action_context(ep)

            def _text(v):
                return v.strip().lower() if isinstance(v, str) else ""

            target += int(_text(ctx.get("target")) == gt["target"].lower())
            direction += int(_text(ctx.get("direction")) == gt["direction"])
            try:
                edge_errs.append(abs(float(ctx.get("edge_distance_m")) - gt["edge_distance_m"]))
            except (TypeError, ValueError):
                pass  # malformed distance counts as missing, not as an error
            obs = ctx.get("obstacles")
            got = ({str(o.get("label", "")).lower() for o in obs if isinstance(o, dict)}
                   if isinstance(obs, list) else set())
            want = {o["label"].lower() for o in gt["obstacles"]}
            obstacles += int(got == want)
            sup = p.get("support")
            if isinstance(sup, dict):
                try:
                    t_cur = float(ep.frame_times[(p.get("frames") or [0])[-1]])
                except (IndexError, KeyError, TypeError, ValueError):
                    t_cur = 0.0
                gt_sup = support_map(ep, t_cur)
                for o in ep.objects:
                    got_sup = sup.get(str(o.id))
                    if isinstance(got_sup, str):
                        sup_total += 1
                        sup_hits += int(got_sup.strip().lower() == gt_sup[o.id].lower())
        out.append({
            "split": split, "episodes": n,
            "ctx_parse_rate": parsed / n if n else float("nan"),
            "ctx_target_acc": target / n if n else float("nan"),
            "ctx_direction_acc": direction / n if n else float("nan"),
            "ctx_obstacles_acc": obstacles / n if n else float("nan"),
            "ctx_edge_mae_m": float(sum(edge_errs) / len(edge_errs)) if edge_errs else float("nan"),
            "ctx_support_acc": (sup_hits / sup_total) if sup_total else float("nan"),
            "ctx_support_scored": sup_total,
        })
    return out
