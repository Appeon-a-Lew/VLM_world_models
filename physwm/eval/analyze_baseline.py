"""CPU-only paired analysis of saved A/B predictions; no model calls.

python -m physwm.eval.analyze_baseline --data DATA --pred predictions.jsonl --out analysis
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from physwm.eval.metrics import MISSING_PENALTY, gt_centers_cam, load_predictions, validate_coverage
from physwm.schema import load_split

PRIMARY_OOD = ('val_ood_objects', 'val_ood_forces', 'val_ood_config')
SPLITS = ('val_id', 'val_ood_objects', 'val_ood_forces', 'val_ood_config',
          'val_diag_elev', 'val_diag_count', 'val_diag_corner', 'val_diag_elev_corner')


def xyz(record, object_id):
    try:
        value = np.asarray(record.get('objects', {}).get(str(object_id)), dtype=float)
        if value.shape == (3,) and np.isfinite(value).all():
            return value
    except (TypeError, ValueError):
        pass
    return None


def episode_row(ep, a, b):
    """Target metrics; paired displacement uses B minus the same episode's A."""
    i = ep.action.target
    p, q = xyz(a, i), xyz(b, i)
    g0, g1 = gt_centers_cam(ep, 0)[i], gt_centers_cam(ep, 1)[i]
    delta = None if p is None or q is None else q - p - (g1 - g0)
    err = None if q is None else q - g1
    pos = np.asarray(ep.state_at(0)[i].pos)
    edge = float(np.min(np.asarray(ep.table_half_size)[:2] - np.abs(pos[:2])))
    row = dict(episode_id=ep.episode_id, split=ep.split, shape=ep.objects[i].shape,
               speed=ep.action.speed, speed_group=('weak' if ep.action.speed < .3 else
                   'strong' if ep.action.speed > 1.6 else 'in_range'),
               displacement=float(np.linalg.norm(g1-g0)), edge_distance=edge,
               elevation=ep.camera.elevation_deg, missing_A=p is None, missing_B=q is None,
               missing_displacement=delta is None,
               A_error=MISSING_PENALTY if p is None else float(np.linalg.norm(p-g0)),
               B_error=MISSING_PENALTY if q is None else float(np.linalg.norm(err)),
               static_error=float(np.linalg.norm(g1-g0)),
               own_static_error=MISSING_PENALTY if p is None else float(np.linalg.norm(p-g1)),
               displacement_error=None if delta is None else float(np.linalg.norm(delta)))
    for j, axis in enumerate('xyz'):
        row['B_abs_'+axis] = None if err is None else float(abs(err[j]))
        row['A_abs_'+axis] = None if p is None else float(abs(p[j]-g0[j]))
        row['displacement_abs_'+axis] = None if delta is None else float(abs(delta[j]))
    row['displacement_group'] = '<0.01' if row['displacement'] < .01 else '0.01–0.1' if row['displacement'] < .1 else '>=0.1'
    row['edge_group'] = '<0.1' if edge < .1 else '0.1–0.25' if edge < .25 else '>=0.25'
    row['elevation_group'] = '<30' if row['elevation'] < 30 else '30–60' if row['elevation'] < 60 else '>=60'
    return row


METRICS = ('A_error', 'B_error', 'static_error', 'own_static_error', 'displacement_error') + tuple(
    prefix+axis for prefix in ('A_abs_', 'B_abs_', 'displacement_abs_') for axis in 'xyz')


def summarize_rows(rows):
    result = {'episodes': len(rows)}
    for name in METRICS:
        values = [r[name] for r in rows if r[name] is not None]
        result[name] = dict(n=len(values), missing_rate=1-len(values)/len(rows) if rows else None,
                            mean=float(np.mean(values)) if values else None,
                            median=float(np.median(values)) if values else None)
    for name in ('missing_A', 'missing_B', 'missing_displacement'):
        result[name] = float(np.mean([r[name] for r in rows])) if rows else None
    return result


def paired_bootstrap(groups, baseline, repetitions=10000, seed=0):
    """Resample whole episodes within split; average split means with equal weight.

    Negative differences favor B. Both errors use the fixed missing penalty.
    """
    if repetitions < 1:
        raise ValueError('bootstrap repetitions must be positive')
    rng = np.random.default_rng(seed)
    arrays = [np.asarray([r['B_error']-r[baseline] for r in group]) for group in groups]
    if not arrays or any(len(a) == 0 for a in arrays):
        raise ValueError('bootstrap requires nonempty splits')
    draws = np.zeros(repetitions)
    # Bound memory even for large cohorts and repetition counts.
    for a in arrays:
        for start in range(0, repetitions, 256):
            stop = min(start+256, repetitions)
            draws[start:stop] += a[rng.integers(len(a), size=(stop-start, len(a)))].mean(axis=1)/len(arrays)
    return dict(difference=float(np.mean([a.mean() for a in arrays])),
                ci95=[float(x) for x in np.quantile(draws, [.025, .975])],
                bootstrap_repetitions=repetitions, seed=seed, episodes=sum(map(len, arrays)),
                split_count=len(arrays), weighting='equal split means', unit='episode')


def analyze(episodes, predictions, repetitions=10000, seed=0):
    validate_coverage(predictions, episodes, ('A', 'B'))
    indexed = {(p['episode_id'], p['task']): p for p in predictions if p['episode_id'] in episodes}
    rows = [episode_row(ep, indexed[eid, 'A'], indexed[eid, 'B']) for eid, ep in sorted(episodes.items())]
    splits = sorted({r['split'] for r in rows})
    grouped = [[r for r in rows if r['split'] == split] for split in splits]
    result = {'scope': 'target object only', 'splits': {s: summarize_rows(g) for s, g in zip(splits, grouped)},
              'overall_descriptive': {b: paired_bootstrap(grouped, b, repetitions, seed) for b in ('static_error', 'own_static_error')},
              'paired_by_split': {s: {b: paired_bootstrap([g], b, repetitions, seed) for b in ('static_error', 'own_static_error')} for s, g in zip(splits, grouped)}}
    by_split = dict(zip(splits, grouped))
    primary_groups = [by_split[s] for s in PRIMARY_OOD if s in by_split]
    primary_available = len(primary_groups) == len(PRIMARY_OOD)
    result['primary_ood'] = {
        'available': primary_available, 'splits': list(PRIMARY_OOD),
        'missing_splits': [s for s in PRIMARY_OOD if s not in by_split],
        'paired': {b: paired_bootstrap(primary_groups, b, repetitions, seed) for b in
                   ('static_error', 'own_static_error')} if primary_available else None,
        'equal_split_means': {m: float(np.mean([result['splits'][s][m]['mean'] for s in PRIMARY_OOD]))
            if all(result['splits'][s][m]['mean'] is not None for s in PRIMARY_OOD) else None
            for m in METRICS} if primary_available else None}
    result['complete_pair_sensitivity'] = {}
    for baseline in ('static_error', 'own_static_error'):
        complete = [[r for r in group if not r['missing_B'] and
                     (baseline == 'static_error' or not r['missing_A'])] for group in grouped]
        result['complete_pair_sensitivity'][baseline] = {
            'counts_by_split': {s: len(g) for s, g in zip(splits, complete)},
            'overall_descriptive': paired_bootstrap(complete, baseline, repetitions, seed) if all(complete) else None,
            'primary_ood': paired_bootstrap([complete[splits.index(s)] for s in PRIMARY_OOD], baseline, repetitions, seed)
                if primary_available and all(complete[splits.index(s)] for s in PRIMARY_OOD) else None,
            'warning': 'Conditional on observed valid predictions; selection can bias this sensitivity.'}
    result['overall_descriptive_equal_split_means'] = {m: float(np.mean([result['splits'][s][m]['mean'] for s in splits]))
        if all(result['splits'][s][m]['mean'] is not None for s in splits) else None for m in METRICS}
    result['strata_by_split'] = {s: {key: {str(value): summarize_rows([r for r in group if r[key] == value])
        for value in sorted({r[key] for r in group})} for key in
        ('speed_group', 'shape', 'displacement_group', 'edge_group', 'elevation_group')} for s, group in zip(splits, grouped)}
    return result


def report(result):
    lines = ['# Saved baseline analysis', '', 'Target errors in metres. The primary estimate weights the three main OOD splits equally; all requested splits are separately descriptive.',
             'Negative paired differences favor model B. CIs resample episodes within splits.',
             'A and B comparisons are descriptive; no additive causal perception/dynamics decomposition.',
             'L2 endpoint errors use the 1 m missing penalty. Axis and displacement metrics exclude missing pairs; see counts and missing rates.', '',
             '| Split | N | A mean | B mean | Static mean | Own static mean | B missing |', '|---|---:|---:|---:|---:|---:|---:|']
    for split, s in result['splits'].items():
        lines.append(f"| {split} | {s['episodes']} | {s['A_error']['mean']:.4f} | {s['B_error']['mean']:.4f} | {s['static_error']['mean']:.4f} | {s['own_static_error']['mean']:.4f} | {s['missing_B']:.3f} |")
    if result['primary_ood']['available']:
        for baseline, values in result['primary_ood']['paired'].items():
            lines.append(f"\nPrimary main-OOD B minus {baseline}: {values['difference']:.4f} m, 95% CI [{values['ci95'][0]:.4f}, {values['ci95'][1]:.4f}].")
    else:
        lines.append('\nPrimary main-OOD estimate unavailable; missing splits: '+', '.join(result['primary_ood']['missing_splits'])+'.')
    for baseline, values in result['overall_descriptive'].items():
        lines.append(f"\nOverall descriptive B minus {baseline}: {values['difference']:.4f} m, 95% CI [{values['ci95'][0]:.4f}, {values['ci95'][1]:.4f}].")
    lines.extend(['', 'Detailed means, medians, axis errors, counts, missing rates and within-split strata are in analysis.json.',
                  'Strata are descriptive; small strata have no confidence intervals or significance claims. No multiple-comparison adjustment is implied.',
                  'Primary L2 estimates retain missing predictions at the fixed penalty; complete-pair sensitivity is also provided and can suffer selection bias.',
                  'All metrics concern the action target. The n for penalized endpoint errors includes missing predictions; missing_A and missing_B report validity separately.',
                  'Speed strata: weak <0.3 m/s, strong >1.6 m/s, otherwise in_range. Displacement: <0.01, 0.01–0.1, >=0.1 m.',
                  'Edge distance is initial target centre distance to the nearest table edge: <0.1, 0.1–0.25, >=0.25 m. Elevation bins: <30, 30–60, >=60 degrees.'])
    lines.extend(['', '## Target displacement and camera axes', '',
                  '| Split | Displacement n | Displacement mean | Displacement median | B abs x | B abs y | B abs z |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for split, summary in result['splits'].items():
        displacement = summary['displacement_error']
        values = [displacement['mean'], displacement['median']] + [summary['B_abs_'+axis]['mean'] for axis in 'xyz']
        lines.append('| '+split+' | '+str(displacement['n'])+' | '+' | '.join(map(fmt, values))+' |')
    lines.extend(['', '## Within-split descriptive strata', '',
                  '| Split | Stratum | Level | N | B mean | B median | Static mean | Own static mean | B missing |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for split, dimensions in result['strata_by_split'].items():
        for dimension, levels in dimensions.items():
            for level, summary in levels.items():
                values = [summary['B_error']['mean'], summary['B_error']['median'],
                          summary['static_error']['mean'], summary['own_static_error']['mean'], summary['missing_B']]
                lines.append('| '+split+' | '+dimension+' | '+level+' | '+str(summary['episodes'])+' | '+' | '.join(map(fmt, values))+' |')
    return '\n'.join(lines)+'\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', nargs='+', default=SPLITS)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.bootstrap < 1:
        ap.error('--bootstrap must be positive')
    episodes = {ep.episode_id: ep for s in args.splits for _, ep in load_split(args.data, s)}
    result = analyze(episodes, load_predictions(args.pred), args.bootstrap, args.seed)
    result['provenance'] = dict(analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_sha256=hashlib.sha256(Path(args.pred).read_bytes()).hexdigest(),
        splits=list(args.splits), seed=args.seed, bootstrap=args.bootstrap,
        definition='target only; B endpoint minus comparator endpoint error; episode paired percentile bootstrap',
        metadata_sha256=hashlib.sha256(json.dumps([ep.to_dict() for _, ep in sorted(episodes.items())], sort_keys=True).encode()).hexdigest())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (out/'report.md').write_text(report(result))
    print(report(result))


if __name__ == '__main__':
    main()
