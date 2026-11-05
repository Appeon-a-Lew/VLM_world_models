"""Endpoint and paired-effect measurements for matched counterfactual families.

All differences are computed in world coordinates. Confidence intervals resample
families, not individual objects or branches. Endpoint accuracy accompanies
invariance: an unresponsive model can have perfect invariance and poor physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physwm.geometry import cam_to_world, world_to_cam
from physwm.schema import Episode, load_split
from physwm.eval.metrics import MISSING_PENALTY, load_predictions, validate_coverage

VARIANTS = ('original', 'reverse', 'half', 'double', 'recolor', 'elevation')
VISUAL = ('recolor', 'elevation')


def load_families(items: list[tuple[Path, Episode]]) -> dict:
    families = defaultdict(dict)
    for path, ep in items:
        meta = json.loads((path / 'counterfactual.json').read_text())
        variant, fid = meta['variant'], meta['family_id']
        if variant not in VARIANTS or variant in families[fid]:
            raise ValueError(f'unknown or duplicate variant in family {fid}: {variant}')
        if meta['split'] != ep.split or meta['seed'] != ep.seed:
            raise ValueError(f'counterfactual identity mismatch: {ep.episode_id}')
        families[fid][variant] = (ep, meta)
    if not families:
        raise ValueError('no counterfactual families')
    for fid, family in families.items():
        if set(family) != set(VARIANTS):
            raise ValueError(f'incomplete family: {fid}')
        base, base_meta = family['original']
        ids = [o.id for o in base.objects]
        for variant, (ep, meta) in family.items():
            if ep.split != base.split:
                raise ValueError(f'family leaked across splits: {fid}')
            if (meta['reference_episode_id'] != base.episode_id
                    or meta['state_sha256'] != base_meta['state_sha256']
                    or meta['model_sha256'] != base_meta['model_sha256']):
                raise ValueError(f'inconsistent reference/state/model: {fid}/{variant}')
            if [o.id for o in ep.objects] != ids or ep.action.target != base.action.target:
                raise ValueError(f'object or target identity changed: {fid}/{variant}')
            for obj, original in zip(ep.objects, base.objects):
                for attr in ('shape', 'half_extents', 'mass', 'density', 'friction'):
                    if getattr(obj, attr) != getattr(original, attr):
                        raise ValueError(f'physical object parameters changed: {fid}/{variant}/{attr}')
            factor = .5 if variant == 'half' else 2. if variant == 'double' else 1.
            sign = -1 if variant == 'reverse' else 1
            if (not np.allclose(ep.action.direction_world, sign * np.asarray(base.action.direction_world))
                    or not np.isclose(ep.action.impulse, factor * base.action.impulse)
                    or not np.isclose(ep.action.speed, factor * base.action.speed)):
                raise ValueError(f'action does not match intervention: {fid}/{variant}')
            if variant != 'elevation' and ep.camera != base.camera:
                raise ValueError(f'unintended camera intervention: {fid}/{variant}')
            if ep.times != base.times:
                raise ValueError(f'observation schedule changed: {fid}/{variant}')
            for state, ref in zip(ep.state_at(0), base.state_at(0)):
                if not np.allclose(state.pos + state.quat, ref.pos + ref.quat, rtol=0, atol=1e-10):
                    raise ValueError(f'initial physical state changed: {fid}/{variant}')
            if variant in VISUAL:
                if (not np.allclose(ep.action.direction_world, base.action.direction_world)
                        or not np.isclose(ep.action.impulse, base.action.impulse)):
                    raise ValueError(f'visual variant changed action: {fid}/{variant}')
                for t in base.times:
                    for state, ref in zip(ep.state_at(t), base.state_at(t)):
                        if not np.allclose(state.pos + state.quat, ref.pos + ref.quat, rtol=0, atol=1e-10):
                            raise ValueError(f'visual variant changed physics: {fid}/{variant}')
    return dict(families)


def _world_prediction(pred: dict, ep: Episode, oid: int):
    raw = pred.get('objects', {}).get(str(oid))
    try:
        xyz = np.asarray(raw, dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            return None
    except (ValueError, TypeError):
        return None
    return cam_to_world(xyz, ep.camera.cam2world)


def _summary(values, n_boot, rng):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return dict(n_families=0, mean=None, median=None, ci95=None)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = a[rng.integers(0, len(a), len(a))].mean()
    return dict(n_families=len(a), mean=float(a.mean()), median=float(np.median(a)),
                ci95=np.quantile(means, [.025, .975]).tolist())


def analyze(families, predictions, n_boot=2000, seed=0):
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    episodes = {ep.episode_id: ep for f in families.values() for ep, _ in f.values()}
    b_predictions = [p for p in predictions if p['task'] == 'B']
    validate_coverage(b_predictions, episodes, ('B',))
    lookup = {p['episode_id']: p for p in b_predictions if p['episode_id'] in episodes}
    rng = np.random.default_rng(seed)
    per_variant = {}
    for variant in VARIANTS:
        scopes = {}
        for scope in ('target', 'all'):
            endpoint, effect, effect_valid, gt_effect, disagreement = [], [], [], [], []
            n_objects = missing_endpoint = missing_pair = 0
            for fid in sorted(families):
                family = families[fid]
                ep, _ = family[variant]
                base, _ = family['original']
                selected = [ep.action.target] if scope == 'target' else [o.id for o in ep.objects]
                ep_gt = {o.id: np.asarray(s.pos) for o, s in zip(ep.objects, ep.state_at(1))}
                base_gt = {o.id: np.asarray(s.pos) for o, s in zip(base.objects, base.state_at(1))}
                ends, effects, observed, truth, pred_delta = [], [], [], [], []
                for oid in selected:
                    p = _world_prediction(lookup[ep.episode_id], ep, oid)
                    p0 = _world_prediction(lookup[base.episode_id], base, oid)
                    n_objects += 1
                    missing_endpoint += p is None
                    ends.append(MISSING_PENALTY if p is None else float(np.linalg.norm(p - ep_gt[oid])))
                    if variant == 'original':
                        continue
                    delta = ep_gt[oid] - base_gt[oid]
                    truth.append(float(np.linalg.norm(delta)))
                    if p is None or p0 is None:
                        missing_pair += 1
                        effects.append(MISSING_PENALTY)
                    else:
                        err = float(np.linalg.norm((p - p0) - delta))
                        effects.append(err)
                        observed.append(err)
                        pred_delta.append(float(np.linalg.norm(p - p0)))
                endpoint.append(float(np.mean(ends)))
                if variant != 'original':
                    effect.append(float(np.mean(effects)))
                    gt_effect.append(float(np.mean(truth)))
                    # Only fully observed families enter valid-only sensitivity summaries.
                    if len(observed) == len(selected):
                        effect_valid.append(float(np.mean(observed)))
                        disagreement.append(float(np.mean(pred_delta)))
            record = dict(endpoint_l2_m=_summary(endpoint, n_boot, rng), n_objects=n_objects,
                          missing_endpoint_rate=missing_endpoint / n_objects)
            if variant != 'original':
                record.update(effect_l2_m=_summary(effect, n_boot, rng),
                              effect_l2_valid_families_m=_summary(effect_valid, n_boot, rng),
                              gt_effect_m=_summary(gt_effect, n_boot, rng),
                              predicted_change_valid_families_m=_summary(disagreement, n_boot, rng),
                              missing_pair_rate=missing_pair / n_objects)
            scopes[scope] = record
        per_variant[variant] = scopes
    return dict(n_families=len(families), n_episodes=len(episodes), variants=per_variant)


def static_predictions(families):
    records = []
    for family in families.values():
        for ep, _ in family.values():
            pos = world_to_cam([s.pos for s in ep.state_at(0)], ep.camera.cam2world)
            records.append(dict(episode_id=ep.episode_id, task='B', parsed=True,
                                objects={str(o.id): p.tolist() for o, p in zip(ep.objects, pos)}))
    return records


def render_report(results):
    lines = ['# Matched counterfactual measurements', '',
             'World-space errors in metres. CIs resample complete scene families; all-object scores first average objects within each family.', '',
             '| Method | Variant | Target endpoint | Target effect error | Pair missing |',
             '|---|---|---:|---:|---:|']
    for method, data in results['methods'].items():
        for variant, scopes in data['variants'].items():
            r = scopes['target']
            effect = r.get('effect_l2_m', {}).get('mean')
            effect_text = '—' if effect is None else f'{effect:.4f}'
            missing = r.get('missing_pair_rate')
            missing_text = '—' if missing is None else f'{missing:.1%}'
            lines.append(f"| {method} | {variant} | {r['endpoint_l2_m']['mean']:.4f} | {effect_text} | {missing_text} |")
    lines += ['', 'See JSON for confidence intervals, all-object scores, ground-truth effect sizes, endpoint missingness and valid-family sensitivity summaries.', '',
              'Missing endpoints or pairs receive the existing finite 1 m penalty; it is not a worst-case bound. Valid-only summaries can be selected toward easier families.',
              'Recolor/elevation effect error measures invariance because their ground-truth world trajectories are identical. Invariance without endpoint accuracy does not establish physical understanding.',
              'One checkpoint and a small pilot do not establish training-seed robustness or broad causal generalization.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--splits', nargs='+', required=True)
    parser.add_argument('--pred', nargs='*', default=[], help='name=prediction_file')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not args.pred and not args.static:
        parser.error('provide --pred or --static')
    items = [item for split in args.splits for item in load_split(args.data, split)]
    families = load_families(items)
    methods = {}
    prediction_hashes = {}
    for item in args.pred:
        name, path = item.split('=', 1)
        if name in methods or (args.static and name == 'static'):
            parser.error(f'duplicate/reserved method: {name}')
        methods[name] = analyze(families, load_predictions(path), args.n_boot, args.seed)
        prediction_hashes[name] = dict(path=str(Path(path).resolve()),
                                      sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if args.static:
        methods['static'] = analyze(families, static_predictions(families), args.n_boot, args.seed)
    data_digest = hashlib.sha256()
    for path, ep in sorted(items, key=lambda item: item[1].episode_id):
        data_digest.update(ep.episode_id.encode() + b'\0')
        data_digest.update((path / 'meta.json').read_bytes())
        data_digest.update((path / 'counterfactual.json').read_bytes())
    result = dict(units='metres', bootstrap_unit='scene_family', bootstrap_samples=args.n_boot,
                  bootstrap_seed=args.seed, methods=methods,
                  provenance=dict(data_root=str(Path(args.data).resolve()), splits=args.splits,
                                  metadata_sha256=data_digest.hexdigest(), predictions=prediction_hashes,
                                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'counterfactual.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'counterfactual.md').write_text(render_report(result))
    print(render_report(result))


if __name__ == '__main__':
    main()
