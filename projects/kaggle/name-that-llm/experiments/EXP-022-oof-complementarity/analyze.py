"""Analyze frozen, previously accepted OOF caches; no fitting or remote actions.

--prepare validates/binds the inputs and code. --run requires that frozen
manifest and computes descriptive overlap for dense + 15 sparse scenarios.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from validation.protocol import load_split, select_sparse_rows, validate_score_cache
from src.oof_complementarity import analyze_scenario

STYLE = ROOT / 'experiments/EXP-010-012-stylegram-share'
SVC = ROOT / 'cloud/svc4-raw-public-r1'
STYLE_RESULTS = STYLE / 'download-v2/results'
SVC_RESULTS = SVC / 'outputs-v4/results'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = path.resolve(strict=True)
        require(path.is_relative_to(ROOT.resolve()), 'Input outside project')
        digest = sha(path)
        require(expected is None or digest == expected, 'Accepted input changed: ' + str(path))
        key = path.relative_to(ROOT.resolve()).as_posix()
        require(key not in self.hashes or self.hashes[key] == digest, 'Input changed while loading')
        self.hashes[key] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.track(path, expected).read_text(encoding='utf-8'))

    def unchanged(self):
        for name, digest in self.hashes.items():
            require(sha(ROOT / name) == digest, 'Input changed during analysis: ' + name)


def route_checks(z, truth, labels, plain_scores=None):
    n = len(truth)
    require(z['scores'].shape == (n, len(labels)) and np.isfinite(z['scores']).all(), 'Invalid style scores')
    require(z['true'].shape == (n,) and z['true'].dtype.kind in 'iu', 'Invalid true-label indices')
    require(((z['true'] >= 0) & (z['true'] < len(labels))).all(), 'Out-of-range true labels')
    require(np.array(labels)[z['true']].tolist() == truth.model.tolist(), 'Style truth/order mismatch')
    g, mask = z['group_ids'], z['centred_mask']
    require(g.shape == mask.shape == (n,) and g.dtype.kind in 'iu' and mask.dtype == bool, 'Invalid route metadata')
    counts = Counter(g.tolist())
    eligible = np.array([2 <= counts[x] <= 18 for x in g], dtype=bool)
    require(np.array_equal(mask, eligible), 'Centre mask differs from eligible inferred group sizes')
    if plain_scores is not None:
        require(plain_scores.shape == (n, len(labels)) and np.isfinite(plain_scores).all(), 'Invalid plain scores')
        require(np.array_equal(z['scores'][~mask], plain_scores[~mask]), 'Plain-route scores changed')
    return np.array([counts[x] for x in g]), mask


def load_inputs():
    inputs = Inputs()
    for name in ('train.csv', 'validation/prompt-group-v1/manifest.json', 'validation/prompt-group-v1/folds.csv'):
        inputs.track(ROOT / name)
    train, folds, labels = load_split(ROOT)
    config = inputs.json(HERE / 'config.json')
    require(config['members']['stylegram'] == {'experiment_id': 'EXP-010', 'public_share': 0.5}, 'Changed Stylegram member')
    require(config['members']['svc4'] == {'experiment_id': 'EXP-018', 'public_row_weight': 0.35}, 'Changed svc4 member')
    require(config['validation']['sparse_per_prompt'] == [2, 3, 6] and config['validation']['sparse_repeats'] == 5, 'Changed scenarios')
    for key in ('training', 'fusion', 'calibration', 'threshold_tuning', 'class_specific_routing', 'official_baseline_changed'):
        require(config['analysis'][key] is False, 'Unexpected scope: ' + key)
    style_accept = inputs.json(STYLE / 'acceptance-20261101/acceptance-status.json')
    style_verified = inputs.json(STYLE / 'analysis/verified.json')
    svc_verified = inputs.json(SVC / 'analysis/verified.json')
    require(style_accept['status'] == style_verified['status'] == svc_verified['status'] == 'PASS', 'Parents not accepted')
    style_receipt = inputs.json(STYLE / 'acceptance-20261101/download-manifest.json')
    style_hashes = {x['path']: x['sha256'] for x in style_receipt['files']}
    require(len(style_hashes) == len(style_receipt['files']), 'Duplicate Stylegram receipt paths')
    svc_hashes = svc_verified['input_result_hashes']
    style_protocol = inputs.json(STYLE / 'payload/protocol.json')
    svc_protocol = inputs.json(SVC / 'payload/protocol.json', svc_verified['protocol_sha256'])
    require(style_protocol['labels'] == svc_protocol['labels'] == labels, 'Member label orders differ')
    require(style_protocol['train_sha256'] == svc_protocol['train_sha256'] == sha(ROOT / 'train.csv'), 'Training version mismatch')

    def style_path(name):
        require('results/' + name in style_hashes, 'Stylegram file absent from accepted receipt')
        return inputs.track(STYLE_RESULTS / name, style_hashes['results/' + name])

    def svc_path(name):
        require(name in svc_hashes, 'svc4 file absent from accepted receipt')
        return inputs.track(SVC_RESULTS / name, svc_hashes[name])

    require(inputs.json(style_path('environment.json'))['protocol'] == style_protocol, 'Stylegram environment binding')
    require(inputs.json(svc_path('environment.json'))['protocol'] == svc_protocol, 'svc4 environment binding')
    require(inputs.json(style_path('result.json'))['status'] == inputs.json(svc_path('result.json'))['status'] == 'COMPLETE', 'Incomplete parents')
    csv_a = pd.read_csv(style_path('oof-share0.5.csv'), dtype={'id': str, 'prompt_id': str})
    score_a = np.load(style_path('oof-share0.5.npy'), allow_pickle=False)
    csv_b = pd.read_csv(svc_path('EXP-018/oof.csv'), dtype={'id': str, 'prompt_id': str})
    score_b = np.load(svc_path('EXP-018/oof_decision.npy'), allow_pickle=False)
    validate_score_cache(train, folds, csv_a, score_a, labels, labels)
    validate_score_cache(train, folds, csv_b, score_b, labels, labels)
    dense = folds[['id', 'prompt_id', 'model', 'fold']].copy()
    dense['stylegram_prediction'] = np.array(labels)[score_a.argmax(1)]
    dense['svc4_prediction'] = np.array(labels)[score_b.argmax(1)]
    dense['stylegram_centred'] = False
    dense['stylegram_group_size'] = 0
    dense['stylegram_plain_prediction'] = ''
    dense_plain = np.full_like(score_a, np.nan)
    by_id = train.set_index('id', drop=False)
    positions = dict(zip(train.id, range(len(train))))
    for fold in range(5):
        ix = np.flatnonzero(folds.fold.to_numpy() == fold)
        with np.load(style_path(f'fold{fold}-share0.5.npz'), allow_pickle=False) as z:
            require(z['ids'].tolist() == train.id.iloc[ix].tolist() and z['labels'].tolist() == labels, 'Dense fold identity mismatch')
            sizes, mask = route_checks(z, train.iloc[ix], labels, z['plain_scores'])
            require(np.array_equal(z['scores'], score_a[ix]), 'Dense fold scores differ from full OOF')
            dense.loc[ix, 'stylegram_centred'] = mask
            dense.loc[ix, 'stylegram_group_size'] = sizes
            dense.loc[ix, 'stylegram_plain_prediction'] = np.array(labels)[z['plain_scores'].argmax(1)]
            dense_plain[ix] = z['plain_scores']
        with np.load(svc_path(f'EXP-018/fold{fold}.npz'), allow_pickle=False) as z:
            require(z['ids'].tolist() == train.id.iloc[ix].tolist() and z['labels'].tolist() == labels, 'svc4 fold identity mismatch')
            require(np.array_equal(z['scores'], score_b[ix]) and np.array(labels)[z['true']].tolist() == train.model.iloc[ix].tolist(), 'svc4 fold cache mismatch')
    require(np.isfinite(dense_plain).all(), 'Incomplete dense plain cache')
    scenarios = [('dense', None, None, dense)]
    for m in config['validation']['sparse_per_prompt']:
        for repeat in range(config['validation']['sparse_repeats']):
            parts = []
            for fold in range(5):
                expected = select_sparse_rows(folds[folds.fold.eq(fold)], m, repeat, 42)
                with np.load(style_path(f'sparse-f{fold}-a0.5-m{m}-r{repeat}.npz'), allow_pickle=False) as z:
                    ids = z['ids'].tolist()
                    require(len(ids) == len(set(ids)) and set(ids) == set(expected.id), 'Wrong sparse selected IDs')
                    ix = np.array([positions[x] for x in ids])
                    truth = by_id.loc[ids]
                    sizes, mask = route_checks(z, truth, labels, dense_plain[ix])
                    frame = folds.iloc[ix][['id', 'prompt_id', 'model', 'fold']].copy()
                    require(frame.fold.eq(fold).all(), 'Sparse cache crosses folds')
                    frame['stylegram_prediction'] = np.array(labels)[z['scores'].argmax(1)]
                    frame['svc4_prediction'] = np.array(labels)[score_b[ix].argmax(1)]
                    frame['stylegram_centred'] = mask
                    frame['stylegram_group_size'] = sizes
                    frame['stylegram_plain_prediction'] = np.array(labels)[dense_plain[ix].argmax(1)]
                    parts.append(frame)
            frame = pd.concat(parts).sort_values('id').reset_index(drop=True)
            require(frame.id.is_unique and len(frame) == 150*m and frame.groupby('prompt_id').size().eq(m).all(), 'Sparse scenario population mismatch')
            require(frame.id.tolist() == select_sparse_rows(folds, m, repeat, 42).id.tolist(), 'Sparse pooled order differs from canonical')
            scenarios.append((f'm{m}-repeat{repeat}', m, repeat, frame))
    inputs.unchanged()
    return inputs, config, labels, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    require(args.prepare != args.run, 'Choose exactly one of --prepare/--run')
    start = time.monotonic()
    inputs, config, labels, scenarios = load_inputs()
    code = [HERE / 'analyze.py', ROOT / 'src/oof_complementarity.py', ROOT / 'validation/protocol.py']
    frozen = {'config': {'path': (HERE/'config.json').relative_to(ROOT).as_posix(), 'sha256': sha(HERE/'config.json')},
              'inputs': [{'path': name, 'sha256': digest} for name, digest in sorted(inputs.hashes.items())],
              'code': [{'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)} for path in code]}
    manifest_path = HERE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if args.prepare:
        require(manifest['freeze_status'] == 'DRAFT', 'Refusing to refreeze an existing frozen experiment')
        manifest.update(frozen, freeze_status='FROZEN', frozen_at=stamp(), note='Read-only cached prediction analysis; input bytes and consumed scenario identity checked before analysis.', environment={'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__})
        save(manifest_path, manifest)
        save(HERE/'metrics/input-preflight.json', {'status': 'PASS', 'observed_at': stamp(), 'inputs': len(inputs.hashes), 'scenarios': len(scenarios), 'rows_by_scenario': {name: len(frame) for name, m, repeat, frame in scenarios}, 'stylegram_sparse_npz': 75, 'stylegram_dense_fold_npz': 5, 'svc4_fold_npz': 5, 'source_hashes_bound_to_prior_acceptance': True, 'group_mask_checked': True, 'uncentred_scores_match_plain': True, 'input_manifest_sha256': sha(manifest_path)})
        print(json.dumps({'status': 'PREPARED', 'inputs': len(inputs.hashes), 'scenarios': len(scenarios)}))
        return
    require(manifest['freeze_status'] == 'FROZEN', 'Experiment not frozen')
    for key in frozen:
        require(manifest[key] == frozen[key], 'Frozen ' + key + ' changed')
    require(not (HERE/'metrics/result.json').exists(), 'Results already exist; use verification, not rerun')
    tables = {key: [] for key in ('summary', 'per_class', 'per_fold', 'per_prompt', 'per_class_fold')}
    all_rows = []
    for name, m, repeat, frame in scenarios:
        require(time.monotonic()-start < config['budget']['wall_seconds'], 'Analysis time budget reached')
        analyzed = analyze_scenario(frame, labels, name, m=m, repeat=repeat, seed=42, bootstrap_draws=config['validation']['bootstrap_draws'])
        tables['summary'].append(analyzed['summary'])
        for key in tables:
            if key != 'summary':
                tables[key].extend({'scenario': name, 'm': m, 'repeat': repeat, **row} for row in analyzed[key])
        rows = frame.copy(); rows.insert(0, 'scenario', name)
        sa, sb = rows.stylegram_prediction.eq(rows.model), rows.svc4_prediction.eq(rows.model)
        rows['outcome'] = np.select([sa & sb, sa & ~sb, ~sa & sb], ['both_correct', 'stylegram_only_correct', 'svc4_only_correct'], default='both_wrong')
        all_rows.append(rows)
        print('ANALYZED ' + name, flush=True)
    inputs.unchanged()
    for entry in manifest['code']:
        require(sha(ROOT/entry['path']) == entry['sha256'], 'Code changed during analysis')
    for key, rows in tables.items():
        flat = [{k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                 for k, v in row.items()} for row in rows]
        pd.DataFrame(flat).to_csv(HERE/'metrics'/f'{key}.csv', index=False, lineterminator='\n')
    pd.concat(all_rows).to_csv(HERE/'oof/paired_predictions.csv', index=False, lineterminator='\n')
    result = {'status': 'COMPLETE', 'observed_at': stamp(), 'scope': config['analysis']['scope'], 'members': config['members'], 'scenarios': 16, 'summaries': tables['summary'], 'elapsed_seconds': time.monotonic()-start, 'training': False, 'fusion': False, 'calibration': False, 'manifest_sha256': sha(manifest_path), 'limits': ['Error complementarity is descriptive; it does not identify which model is right on a new row or measure a deployable ensemble gain.', 'No group-specific routing, thresholds or blend weights were optimized.', 'Sparse repeats reuse fitted models and prompts and are not independent runs.', 'Both upstream datasets retain their documented generation/selection and unknown semantic overlap limitations.']}
    save(HERE/'metrics/result.json', result)
    paths = sorted((HERE/'metrics').glob('*.csv')) + [HERE/'oof/paired_predictions.csv', HERE/'metrics/result.json', HERE/'metrics/input-preflight.json']
    save(HERE/'metrics/output-manifest.json', {'files': [{'path': f.relative_to(HERE).as_posix(), 'sha256': sha(f), 'bytes': f.stat().st_size} for f in paths]})
    print(json.dumps({'status': 'COMPLETE', 'scenarios': 16, 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
