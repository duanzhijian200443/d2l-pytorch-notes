"""Frozen local ATLAS20 replay. Never train, call a provider or fill missing ATLAS rows."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / 'cloud/full-stack-capture-r1/outputs-v2/results'
PRIOR = ROOT / 'experiments/EXP-032-sinkhorn-fixed-cache'
ARTIFACT_SHA = '1732849196220d023716a2dc912d660282c94022c2df370c69d9a6c2899bac63'
PARQUET_PYTHON = Path('C:/Users/34331/AppData/Local/uv/cache/environments-v2/audit-e5542b9a7a231d97/Scripts/python.exe')
DESCRIPTIONS = {
    'A0': '原完整ATLAS20缓存基线',
    'A1': '仅替换为已隔离公共池的Stylegram配方',
    'A2': 'A1的Stylegram分数再逐行标准化',
    'A3': '标准化后加入10% raw svc4',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def entry(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def prepare():
    assert not (HERE / 'outputs').exists(), 'Existing calculation; do not overwrite or retune'
    review = read(HERE / 'metrics/input-review.json')
    assert review['status'] == 'PASS_BOUND_REORDERED_PUBLIC_CANDIDATE_MARGINS'
    for name, item in review['inputs'].items():
        assert sha(ROOT / name) == item['sha256'], name
    assert sha(ROOT / review['output']['path']) == review['output']['sha256']
    assert sha(CACHE / 'artifacts.json') == ARTIFACT_SHA
    protocol = read(PRIOR / 'protocol.json')
    paths = [HERE / 'metrics/input-review.json', ROOT / review['output']['path'], PRIOR / 'protocol.json',
             ROOT / 'experiments/EXP-031-full-stack-capture-r1/metrics/download-receipt.json',
             CACHE / 'artifacts.json', ROOT / 'cloud/fixed-blend-multiclass/outputs-v2/results/artifacts.json']
    paths += [ROOT / item['path'] for item in protocol['inputs'].values()]
    config = {'schema_version': 1, 'experiment_id': 'EXP-045', 'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'exploratory_external_full_stack_public_data_replay', 'arms': DESCRIPTIONS,
              'primary_comparisons': ['A1-A0', 'A2-A1', 'A3-A2', 'A3-A0'],
              'parameters': {'stylegram_weight': 1.516, 'raw_svc_internal_alpha': .1, 'comp_weight': .15,
                             'atlas_weight': .2, 'atlas_margin': .3, 'sinkhorn': .45, 'target_mix': .35,
                             'typography_scale': 1.5, 'typography_cap': .3},
              'preserve': ['all other main/comp members and weights', 'original family gate training and raw scores',
                           'original main svc4 and main divisor', 'all 18 predicted classes', 'original dynamic ATLAS route'],
              'coverage_policy': 'Score only arms whose newly requested ATLAS rows all have verified cached predictions; otherwise null score.',
              'actual_capture_public_pool': {'rows': 10469, 'groups': 716, 'evidence': 'EXP031 logs/comp.log'},
              'candidate_stylegram_public_pool': {'rows': 10073, 'groups': 694},
              'recipe_difference_limit': 'EXP009 10325 rows was a six-source projection, not the actual EXP031 run. A1 is a recipe replacement, not a pure deletion of252 rows.',
              'fixed_mask_diagnostic_scoring': False, 'refit': False, 'self_generated_data': False, 'retune': False,
              'validation': {'rows': 729, 'prompt_groups': 56, 'true_classes': 17, 'predicted_classes': 18,
                             'metric': 'covered-17-class macro recall', 'bootstrap_draws': 2000, 'bootstrap_seed': 42,
                             'already_inspected_panel': True, 'full_stack_training_independence_certified': False,
                             'is_clean_oof': False},
              'budget': {'gpu_seconds': 0, 'api_usd': 0, 'colab_cu': 0, 'remote_runs': 0},
              'source_capture_artifact_sha256': ARTIFACT_SHA, 'inputs': [entry(path) for path in paths]}
    save(HERE / 'config.json', config)
    codes = [Path(__file__), HERE / 'math_replay.py', HERE / 'test_math_replay.py', PRIOR / 'compare_cached.py',
             ROOT / 'src/full_stack_replay.py', ROOT / 'src/margin_blend.py', ROOT / 'src/sinkhorn_compare.py']
    save(HERE / 'manifest.json', {'experiment_id': 'EXP-045', 'status': 'FROZEN_BEFORE_SCORING',
                                'config': entry(HERE / 'config.json'), 'code': [entry(p) for p in codes],
                                'inputs': config['inputs'], 'original_cache_unchanged': True})
    print('EXP045 frozen before score computation; no models or cloud run')


def check_frozen():
    manifest = read(HERE / 'manifest.json')
    for item in [manifest['config']] + manifest['code'] + manifest['inputs']:
        assert sha(ROOT / item['path']) == item['sha256'], 'Frozen file changed: ' + item['path']
    review = read(HERE / 'metrics/input-review.json')
    for name, item in review['inputs'].items():
        assert sha(ROOT / name) == item['sha256'], name
    return read(HERE / 'config.json')


def score_pair(truth, baseline, candidate):
    classes = sorted(truth.model.unique())
    assert len(classes) == 17 and 'gemini-3.7-flash' not in classes
    labels = {v: i for i, v in enumerate(classes)}
    y = truth.model.map(labels).to_numpy(dtype=int)
    support = np.bincount(y, minlength=17)
    a = baseline == truth.model.to_numpy()
    b = candidate == truth.model.to_numpy()
    correct_a = np.bincount(y[a], minlength=17)
    correct_b = np.bincount(y[b], minlength=17)
    groups = sorted(truth.prompt_id.unique())
    gi = truth.prompt_id.map({v: i for i, v in enumerate(groups)}).to_numpy(dtype=int)
    n = np.zeros((len(groups), 17), dtype=np.int64)
    d = np.zeros_like(n)
    np.add.at(n, (gi, y), 1)
    np.add.at(d, (gi, y), b.astype(int)-a.astype(int))
    draws = np.random.default_rng(42).integers(0, len(groups), (2000, len(groups)))
    count = np.zeros((2000, len(groups)), dtype=np.int64)
    np.add.at(count, (np.arange(2000)[:, None], draws), 1)
    den = count @ n
    invalid = int((den == 0).any(axis=1).sum())
    ci = None if invalid else np.quantile((count @ d / den).mean(axis=1), [.025, .975]).tolist()
    return {'reference': {'macro_recall': float((correct_a/support).mean()), 'accuracy': float(a.mean()), 'correct': int(a.sum())},
            'candidate': {'macro_recall': float((correct_b/support).mean()), 'accuracy': float(b.mean()), 'correct': int(b.sum())},
            'delta_macro_recall': float(((correct_b-correct_a)/support).mean()),
            'fixed_errors': int((~a & b).sum()), 'new_errors': int((a & ~b).sum()), 'net_correct': int(b.sum()-a.sum()),
            'changed_predictions': int((baseline != candidate).sum()),
            'ci95': ci, 'bootstrap': {'unit': 'prompt_id', 'groups': len(groups), 'draws': 2000, 'seed': 42,
                                    'invalid_draws': invalid, 'missing_class_policy': 'null entire CI if any draw misses a class',
                                    'retraining_uncertainty': False, 'multiple_comparison_adjustment': 'none'},
            'per_class': [{'label': label, 'support': int(support[i]), 'reference_correct': int(correct_a[i]),
                           'candidate_correct': int(correct_b[i]), 'reference_recall': float(correct_a[i]/support[i]),
                           'candidate_recall': float(correct_b[i]/support[i])} for i, label in enumerate(classes)]}


def calculate():
    config = check_frozen()
    out = HERE / 'outputs'
    assert not out.exists(), 'Existing outputs; preserve prior calculation'
    out.mkdir()
    started = time.monotonic()
    cmp = load(PRIOR / 'compare_cached.py', 'exp045_capture_acceptance')
    protocol = cmp.read_json(PRIOR / 'protocol.json')
    cmp.validate_protocol(protocol)
    accepted = cmp.verify_real_artifacts(CACHE, protocol, ARTIFACT_SHA)
    panel = pd.read_csv(cmp.bound_input(protocol['inputs']['blind_panel']))
    train = pd.read_csv(cmp.bound_input(protocol['inputs']['official_train']))
    panel['text'] = panel.text.fillna('')
    train['text'] = train.text.fillna('')
    with np.load(HERE / 'oof/aligned-candidate-margins.npz', allow_pickle=False) as data:
        assert set(data.files) == {'ids', 'source_ids', 'text_sha256', 'labels', 'stylegram_scores', 'svc4_scores'}
        assert data['ids'].tolist() == panel.id.tolist()
        assert data['text_sha256'].tolist() == [hashlib.sha256(t.encode()).hexdigest() for t in panel.text]
        labels = data['labels'].tolist()
        style, svc = data['stylegram_scores'].copy(), data['svc4_scores'].copy()
    math = load(HERE / 'math_replay.py', 'exp045_math')
    with cmp.captured_helpers(CACHE, accepted['plan']) as helpers:
        snapshots = {name: helpers['full_stack_cache'].read_cache(CACHE / 'cache', name, panel.id.tolist(), panel.text.tolist(), labels,
                     required_notes=['event-index'], manifest_sha256=accepted['receipts'][name]['manifest_sha256']) for name in ('main', 'comp')}
        events = {name: helpers['full_stack_observer'].decode_events(snap) for name, snap in snapshots.items()}
        replay = math.replay_four_arms(events['main'], events['comp'], style, svc, mode='strict',
                     expected_main_members=accepted['plan']['expected_main_members'], expected_comp_members=accepted['plan']['expected_comp_members'])
        post = load(ROOT / 'src/sinkhorn_compare.py', 'exp045_post')
        scores, routes, missing_union = {}, {}, set()
        for name, arm in replay['arms'].items():
            requested = np.asarray(arm['requested_rows'], dtype=int)
            missing = list(arm['missing_ids'])
            missing_union.update(missing)
            routes[name] = {'requested_count': len(requested), 'covered_count': len(requested)-len(missing),
                            'missing_count': len(missing), 'missing_ids': missing, 'requested_ids': panel.id.iloc[requested].tolist(),
                            'scored': arm['pre_sinkhorn'] is not None}
            if arm['pre_sinkhorn'] is None:
                continue
            computed = post.compute_sinkhorn_variant(arm['pre_sinkhorn'], train.model.tolist(), train.text.tolist(), panel.text.tolist(), variant='control', labels=labels)
            if name == 'A0':
                parquet = cmp.read_parquet_predictions(CACHE / 'submission.parquet', PARQUET_PYTHON)
                save(out / 'baseline-parity.json', cmp.verify_control(computed, events['main'], parquet, panel.id.tolist(), labels))
            scores[name] = computed['final_logits']
        assert 'A0' in scores
        # Freeze all usable predictions and coverage before accessing evaluation truth.
        np.savez_compressed(out / 'scores.npz', ids=panel.id.to_numpy(dtype=str), labels=np.array(labels), **scores)
        save(out / 'coverage.json', routes)
        missing_panel = panel[panel.id.isin(missing_union)].copy()
        missing_panel.to_csv(out / 'atlas-missing-panel.csv', index=False, lineterminator='\n')
        save(out / 'atlas-completion-plan.json', {'status': 'NEEDS_ADDITIONAL_MODEL_INFERENCE' if missing_union else 'READY',
             'rows': len(missing_panel), 'panel_sha256': sha(out / 'atlas-missing-panel.csv'), 'labels': labels,
             'source_run_id': 'exp031-20261101T110842Z', 'same_atlas_model_and_inference_settings_required': True,
             'new_gpu_or_cloud_run_started': False, 'model_args_path': (CACHE / 'model-args.json').relative_to(ROOT).as_posix(),
             'source_captured_input_inventory': (CACHE / 'input-availability-check.json').relative_to(ROOT).as_posix(),
             'affected_arms': [name for name in routes if routes[name]['missing_count']],
             'no_zero_fill_no_old_mask_fallback': True})
    truth = pd.read_csv(cmp.bound_input(protocol['inputs']['truth_sidecar']), dtype=str, keep_default_na=False)
    assert truth.id.tolist() == panel.id.tolist() and truth.prompt_id.nunique() == 56
    assert truth.text_sha256.tolist() == [hashlib.sha256(t.encode()).hexdigest() for t in panel.text]
    assert len(truth.model.unique()) == 17
    predictions = truth[['id', 'source_id', 'prompt_id', 'model']].copy()
    for name, values in scores.items():
        predictions[name] = np.array(labels)[values.argmax(axis=1)]
    predictions.to_csv(out / 'predictions.csv', index=False)
    comparisons = {}
    for pair in config['primary_comparisons']:
        candidate, reference = pair.split('-')
        if candidate in scores and reference in scores:
            comparisons[pair] = score_pair(truth, predictions[reference].to_numpy(), predictions[candidate].to_numpy())
        else:
            comparisons[pair] = {'status': 'NOT_SCORED_MISSING_ATLAS', 'ci95': None, 'delta_macro_recall': None}
    summary = {'status': 'PARTIAL_ATLAS_COVERAGE' if missing_union else 'COMPLETE', 'experiment_id': 'EXP-045',
               'observed_at': datetime.now(timezone.utc).isoformat(), 'local_wall_seconds': time.monotonic()-started,
               'config_sha256': sha(HERE / 'config.json'), 'manifest_sha256': sha(HERE / 'manifest.json'),
               'scope': config['scope'], 'rows': 729, 'prompt_groups': 56, 'covered_true_classes': 17, 'output_classes': 18,
               'routes': routes, 'scored_arms': list(scores), 'comparisons': comparisons,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
               'baseline_unchanged': True, 'independence_certified': False, 'is_clean_oof': False,
               'model_inference_calls': 0, 'gpu_seconds': 0, 'new_self_generated_rows': 0,
               'source_capture_artifact_sha256': ARTIFACT_SHA}
    cmp.verify_artifact_inventory(CACHE, ARTIFACT_SHA)
    check_frozen()
    save(out / 'summary.json', summary)
    save(out / 'artifacts.json', {'status': summary['status'], 'files': [entry(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'status': summary['status'], 'scored_arms': list(scores), 'missing_atlas_rows': len(missing_union), 'seconds': summary['local_wall_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else calculate()
