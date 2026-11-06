"""Independent anchored-artifact and OOF arithmetic audit. No training/network."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EX = HERE.parent
DOWNLOAD = ROOT / 'cloud/format120-044/outputs-v2'
B = DOWNLOAD / 'results'
ZIP_SHA = '3fcad3730b537d9ec2af517da285bab0587350e8ba5c2ec0fb5c75a8c11b8de5'
ARMS = ('raw120', 'clean120')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def closed_inventory(folder, files, excluded=()):
    names = set()
    for entry in files:
        name = entry['path']
        assert name not in names and name not in excluded
        p = folder / name
        assert p.resolve().is_relative_to(folder.resolve())
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], name
        names.add(name)
    assert names == {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                     if p.is_file() and p.relative_to(folder).as_posix() not in excluded}


def metric(y, prediction):
    den = np.bincount(y, minlength=18)
    correct = np.bincount(y[prediction == y], minlength=18)
    assert (den > 0).all()
    return {'rows': len(y), 'correct': int(correct.sum()),
            'accuracy': float(correct.sum() / len(y)), 'balanced_accuracy': float((correct / den).mean())}


def comparison(y, raw, clean):
    a, b = metric(y, raw), metric(y, clean)
    ac, bc = raw == y, clean == y
    return {'raw120': a, 'clean120': b, 'delta_ba': b['balanced_accuracy'] - a['balanced_accuracy'],
            'delta_correct': b['correct'] - a['correct'], 'fixed_errors': int((~ac & bc).sum()),
            'new_errors': int((ac & ~bc).sum()), 'both_correct': int((ac & bc).sum()),
            'both_wrong': int((~ac & ~bc).sum()), 'changed_predictions': int((raw != clean).sum())}


archive_path = DOWNLOAD / 'exp044-results.zip'
assert sha(archive_path) == ZIP_SHA and archive_path.stat().st_size == 8498985
with zipfile.ZipFile(archive_path) as archive:
    entries = archive.infolist()
    assert len(entries) == len({x.filename for x in entries}) == 348
    assert {x.filename for x in entries} == {p.relative_to(B).as_posix() for p in B.rglob('*') if p.is_file()}
    for info in entries:
        p = B / info.filename
        assert p.resolve().is_relative_to(B.resolve())
        assert p.stat().st_size == info.file_size and p.read_bytes() == archive.read(info)
manifest = read(EX / 'manifest.json')
for item in [manifest['config'], manifest['protocol']] + manifest['inputs'] + manifest['code']:
    assert sha(ROOT / item['path']) == item['sha256'], item['path']
global_inventory = read(B / 'artifacts.json')
closed_inventory(B, global_inventory['files'], {'artifacts.json', '.guard.lock', '.claim.lock', 'guard.json', 'state.json'})
assert not list(B.rglob('failure.json'))
result, guard, state = [read(B / (name + '.json')) for name in ('result', 'guard', 'state')]
assert all(x['status'] == 'COMPLETE' for x in (result, guard, state, global_inventory))
identity = result['identity']
assert identity == read(B / 'run.json')['identity']
assert identity['run_id'] == manifest['run_id'] == global_inventory['run_id']
assert identity['config_sha256'] == sha(EX / 'config.json')
for name in ('runner', 'guard'):
    assert identity[name + '_sha256'] == sha(EX / (name + '.py'))
for name, digest in identity['input_hashes'].items():
    assert sha(ROOT / 'cloud/format120-044/payload' / name) == digest
assert identity['strict_environment'] is True
assert identity['environment'] == {'numpy': '2.1.3', 'pandas': '2.3.3', 'scipy': '1.16.3', 'sklearn': '1.6.1', 'python': '3.13.15'}
assert guard['child_exit'] == 0 and guard['termination_reason'] is None and guard['tree_stopped'] is True
assert 0 < guard['cumulative_elapsed_seconds'] < guard['hard_seconds'] == 6600
assert guard['peak_rss_bytes'] < guard['rss_limit_bytes'] == 20 * 1024**3
budget = read(B / 'budget.json')
assert len(budget['attempts']) == 1 and budget['attempts'][0]['status'] == 'COMPLETE'
assert budget['attempts'][0]['charged_seconds'] == guard['cumulative_elapsed_seconds']
attempt = guard['attempt']
scheduler = read(B / 'attempts' / attempt / 'scheduler.json')
assert scheduler == {'status': 'COMPLETE', 'max_active_workers': 2}
selftest = read(B / 'selftests' / attempt / 'selftest.json')
assert selftest['status'] == 'PASS' and selftest['parallel_workers_observed'] == 2 and selftest['max_abs_score_difference'] == 0
assert selftest['environment'] == identity['environment'] and selftest['completed_resume_reused'] and selftest['corruption_rejected']

fold_path = ROOT / 'validation/prompt-group-v1/folds.csv'
assert sha(fold_path) == 'e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551'
official = list(csv.DictReader(fold_path.open(encoding='utf-8', newline='')))
labels = identity['labels']
ids = np.array([row['id'] for row in official])
folds = np.array([int(row['fold']) for row in official])
prompts = np.array([row['prompt_id'] for row in official])
truth = np.array([labels.index(row['model']) for row in official])
assert len(ids) == len(set(ids)) == 2699 and len(set(prompts)) == 150
aggregate = B / result['aggregate_path']
oof = {arm: load(aggregate / f'oof-{arm}.npz') for arm in ARMS}
pred = {}
for arm, data in oof.items():
    assert np.array_equal(data['ids'], ids) and np.array_equal(data['true'], truth) and np.array_equal(data['fold'], folds)
    assert data['labels'].tolist() == labels and data['scores'].shape == (2699, 18) and np.isfinite(data['scores']).all()
    assert data['run_id'].item() == identity['run_id'] and data['identity_sha256'].item() == result['identity_sha256']
    pred[arm] = data['scores'].argmax(axis=1)
overall = comparison(truth, pred['raw120'], pred['clean120'])
per_fold = [{'fold': f, **comparison(truth[folds == f], pred['raw120'][folds == f], pred['clean120'][folds == f])} for f in range(5)]
per_class = []
for i, label in enumerate(labels):
    item = {'label': label, 'support': int((truth == i).sum())}
    for arm in ARMS:
        correct = int(((truth == i) & (pred[arm] == i)).sum())
        calls = int((pred[arm] == i).sum())
        item[arm] = {'correct': correct, 'recall': correct / item['support'], 'predicted': calls,
                     'false_positive': calls - correct, 'precision': correct / calls if calls else None}
    item['delta_correct'] = item['clean120']['correct'] - item['raw120']['correct']
    item['delta_recall'] = item['clean120']['recall'] - item['raw120']['recall']
    per_class.append(item)
groups = sorted(set(prompts))
den, diff = np.zeros((150, 18), dtype=np.int64), np.zeros((150, 18), dtype=np.int64)
for g, group in enumerate(groups):
    mask = prompts == group
    assert len(set(folds[mask])) == 1
    den[g] = np.bincount(truth[mask], minlength=18)
    diff[g] = np.bincount(truth[mask & (pred['clean120'] == truth)], minlength=18) - np.bincount(truth[mask & (pred['raw120'] == truth)], minlength=18)
values, invalid = [], 0
for sample in np.random.default_rng(42).integers(0, 150, size=(2000, 150)):
    count = np.bincount(sample, minlength=150)
    support = count @ den
    if np.any(support == 0):
        invalid += 1
    else:
        values.append(float(np.mean((count @ diff) / support)))
ci = None if invalid else np.quantile(values, [.025, .975], method='linear').tolist()

jobs = result['completed_jobs']
assert len(jobs) == 10 and {(j['fold'], j['arm']) for j in jobs} == {(f, a) for f in range(5) for a in ARMS}
sparse_parts = defaultdict(lambda: defaultdict(list))
job_runtime, events, compute_events = [], [], []
stage_seconds = defaultdict(float)
for job in jobs:
    fold, arm = job['fold'], job['arm']
    directory = B / job['path']
    assert directory.name == 'attempt-0001'
    assert sha(directory / 'completion.json') == job['completion_sha256']
    seal = read(directory / 'completion.json')
    assert seal['identity'] == identity and seal['status'] == 'COMPLETE' and seal['fold'] == fold and seal['arm'] == arm
    closed_inventory(directory, seal['files'], {'completion.json'})
    dense = load(directory / 'dense.npz')
    ix = np.flatnonzero(folds == fold)
    assert np.array_equal(dense['ids'], ids[ix]) and np.array_equal(dense['true'], truth[ix])
    assert np.array_equal(dense['scores'], oof[arm]['scores'][ix]) and np.array_equal(dense['plain_scores'], oof[arm]['plain_scores'][ix])
    jr, process = read(directory / 'result.json'), read(directory / 'process.json')
    assert jr['status'] == 'COMPLETE' and jr['trained_svms'] == 2
    job_runtime.append({'fold': fold, 'arm': arm, 'seconds': jr['seconds'], 'process': process})
    events.extend([(process['create_time'], 1), (seal['completed_at'], -1)])
    stages = read(directory / 'timings.json')['stages']
    compute_events.extend([(min(datetime.fromisoformat(s['started_at']).timestamp() for s in stages), 1),
                           (max(datetime.fromisoformat(s['started_at']).timestamp() + s['seconds'] for s in stages), -1)])
    for stage in stages:
        assert stage['status'] == 'COMPLETE'
        stage_seconds[stage['stage']] += stage['seconds']
    for kind in ('plain', 'centred'):
        fit = read(directory / 'fits' / f'f{fold}-{arm}-{kind}.json')
        assert fit['classes'] == list(range(18)) and fit['dtype'] == 'float64'
        assert not fit['warnings'], fit['warnings']
    for m in (2, 3, 6):
        for repeat in range(5):
            data = load(directory / 'sparse' / f'm{m}-r{repeat}.npz')
            expected = []
            for group in sorted(set(prompts[ix])):
                positions = [j for j in ix if prompts[j] == group]
                chosen = sorted(positions, key=lambda j: hashlib.sha256(f'42:{repeat}:{ids[j]}'.encode()).hexdigest())[:m]
                expected.extend(chosen)
            expected = np.array(sorted(expected))
            assert np.array_equal(data['ids'], ids[expected]) and np.array_equal(data['true'], truth[expected])
            assert data['labels'].tolist() == labels and data['m'].item() == m and data['repeat'].item() == repeat
            assert np.isfinite(data['scores']).all() and data['scores'].shape == (len(expected), 18)
            sparse_parts[(m, repeat)][arm].append((data['true'], data['scores'].argmax(1)))
active = maximum = 0
for _, delta_value in sorted(events, key=lambda item: (item[0], item[1])):
    active += delta_value
    maximum = max(maximum, active)
assert active == 0
compute_active = compute_maximum = 0
for _, delta_value in sorted(compute_events, key=lambda item: (item[0], item[1])):
    compute_active += delta_value
    compute_maximum = max(compute_maximum, compute_active)
assert compute_maximum == 2 and compute_active == 0
assert len({(j['process']['pid'], j['process']['create_time']) for j in job_runtime}) == 10
sparse = []
for (m, repeat), arms in sorted(sparse_parts.items()):
    joined = {a: (np.concatenate([x[0] for x in arms[a]]), np.concatenate([x[1] for x in arms[a]])) for a in ARMS}
    assert np.array_equal(joined['raw120'][0], joined['clean120'][0])
    sparse.append({'m': m, 'repeat': repeat, **comparison(joined['raw120'][0], joined['raw120'][1], joined['clean120'][1])})
sparse_summary = []
for m in (2, 3, 6):
    subset = [x for x in sparse if x['m'] == m]
    sparse_summary.append({'m': m, 'mean_raw120_ba': float(np.mean([x['raw120']['balanced_accuracy'] for x in subset])),
       'mean_clean120_ba': float(np.mean([x['clean120']['balanced_accuracy'] for x in subset])),
       'mean_delta_ba': float(np.mean([x['delta_ba'] for x in subset])),
       'positive': sum(x['delta_ba'] > 0 for x in subset), 'negative': sum(x['delta_ba'] < 0 for x in subset), 'zero': sum(x['delta_ba'] == 0 for x in subset)})
for published in result['summaries']:
    got = overall[published['arm']]
    assert abs(got['balanced_accuracy'] - published['balanced_accuracy']) < 1e-14 and got['correct'] == published['correct']
verifier = read(HERE / 'verified.json')
assert verifier['status'] == 'PASS_FROZEN_RAW_CLEAN_OOF' and verifier['identity'] == identity
for arm in ARMS:
    for key in ('balanced_accuracy', 'accuracy', 'correct', 'rows'):
        assert abs(verifier['primary'][arm][key] - overall[arm][key]) < 1e-14
assert verifier['paired_prompt_bootstrap']['ci95'] == ci
assert verifier['primary']['fixed_errors'] == overall['fixed_errors']
assert verifier['primary']['introduced_errors'] == overall['new_errors']
assert verifier['primary']['prediction_disagreements'] == overall['changed_predictions']
for published, got in zip(verifier['fold_comparisons'], per_fold):
    assert published['fold'] == got['fold']
    for arm in ARMS:
        assert published[arm] == got[arm]
for published, got in zip(verifier['sparse_scenarios'], sparse):
    assert (published['m'], published['repeat']) == (got['m'], got['repeat'])
    for arm in ARMS:
        assert published[arm] == got[arm]
report = {'status': 'PASS_INDEPENDENT_RESULT_REVIEW', 'run_id': identity['run_id'],
    'script_sha256': sha(Path(__file__)), 'archive_sha256': ZIP_SHA, 'archive_bytes': 8498985, 'extracted_files_verified': 348,
    'manifest_sha256': sha(EX / 'manifest.json'), 'config_sha256': identity['config_sha256'],
    'code_sha256': {x['path']: x['sha256'] for x in manifest['code']},
    'verifier_comparison': {'status': 'MATCH', 'verified_json_sha256': sha(HERE / 'verified.json'),
        'primary_and_five_folds_and_fifteen_sparse_scenarios': True, 'paired_bootstrap_ci_exact': True,
        'original_verifier_or_metric_functions_called': False},
    'result_sha256': {name: sha(B / name) for name in ['artifacts.json', 'result.json', 'run.json', 'guard.json', 'budget.json', 'state.json']},
    'overall': overall, 'per_fold': per_fold, 'per_class': per_class,
    'bootstrap': {'sampling_unit': 'official prompt_id', 'groups': 150, 'draws': 2000, 'seed': 42,
        'valid_draws': len(values), 'invalid_draws': invalid, 'ci95_delta_ba': ci,
        'ci95_percentage_points': None if ci is None else [100*x for x in ci],
        'scope': 'fixed paired trained predictions; no retraining uncertainty'},
    'sparse': sparse, 'sparse_summary': sparse_summary,
    'runtime': {'guard': guard, 'platform_elapsed_user_screenshot_seconds': 1720,
        'platform_scope': '28m40 platform total supplied by parent/user screenshot; not independently fetched',
        'scheduler': scheduler, 'compute_stage_interval_max_overlap': compute_maximum,
        'pid_create_to_post_exit_seal_envelope_max_overlap': maximum,
        'pid_envelope_note': 'Process create_time and post-exit completion receipt are not exact live intervals. Adjacent boundaries overlap by about0.02-0.06s; this conservative envelope is not evidence of three live workers. Actual compute-stage intervals and frozen scheduler both show max2.',
        'jobs': job_runtime, 'sum_job_seconds': sum(j['seconds'] for j in job_runtime),
        'sum_stage_seconds': dict(stage_seconds), 'cloud_selftest': selftest,
        'attempts': 1, 'completed_jobs': 10, 'trained_svms': 20, 'sparse_files': 150, 'warnings': 0, 'failure_files': 0},
    'limitations': ['This is a raw-old120 versus format-cleaned-same120 Stylegram comparison; neither arm is EXP042 aligned-new120.',
        'Official folds are inspected development OOF; no new holdout or leaderboard inference.',
        'Class-specific and sparse secondary outcomes cannot replace the frozen pooled18-class primary outcome after observation.',
        '28m40 platform total and1693.798696257s guard runtime measure different intervals.',
        'EXP042 used different arm data and recipes; its48m16 versus this28m14 is not a controlled pure parallel-speedup estimate.',
        'Completed run demonstrates two-worker scheduling and zero-error synthetic serial parity, not an executed cloud interruption/recovery trial.']}
Path(__file__).with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'overall', 'per_fold', 'per_class', 'bootstrap', 'sparse_summary']}, ensure_ascii=False, indent=2))
