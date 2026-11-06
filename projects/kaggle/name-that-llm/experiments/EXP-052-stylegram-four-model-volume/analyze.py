"""Analyze the fixed 3-arm EXP-052 Stylegram OOF. No training or network calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/EXP-042-luna-terra120-ablation'
EXP010 = ROOT / 'experiments/EXP-010-012-stylegram-share'
CLOUD = ROOT / 'cloud/luna-terra120-042'
HISTORY = CLOUD / 'outputs-v2/results'
ARMS = ('control', 'four120', 'four240')
PAIRS = (('four120', 'four240'), ('control', 'four240'), ('control', 'four120'))
SCENARIOS = [(m, repeat) for m in (2, 3, 6) for repeat in range(5)]
FROZEN_MANIFEST = '98b3469bcebc190db0bb6cf9f1a69ca4a9328f339e6fe6529de367251c29053a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_hash(path, expected):
    require(sha(path) == expected, f'Hash mismatch: {path}')
    return path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_metrics():
    path = verify_hash(OLD / 'verify_outputs.py', '525bb72d185a176773c5e4e30ec7771cc3bf451f8c85efda3f4c597599d0661e')
    spec = importlib.util.spec_from_file_location('exp042_verified_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = load_metrics()


def file_index(entries):
    index = {}
    for item in entries:
        path = item['path']
        parts = PurePosixPath(path)
        require(path and path == parts.as_posix() and not parts.is_absolute() and '..' not in parts.parts
                and ':' not in path and '\\' not in path and path not in index, 'Unsafe or duplicate inventory entry')
        index[path] = item
    return index


def anchored(folder, index, name):
    require(name in index, f'Missing inventory entry: {name}')
    path = (folder / name).resolve(strict=True)
    entry = index[name]
    require(path.is_relative_to(folder.resolve()) and path.is_file() and path.stat().st_size == entry['bytes'],
            f'Invalid inventory target: {name}')
    return verify_hash(path, entry['sha256'])


def history():
    controls = dict(
        train=verify_hash(CLOUD / 'payload/train.csv', '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'),
        folds=verify_hash(CLOUD / 'payload/folds.csv', 'e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551'),
        acceptance=verify_hash(OLD / 'metrics/final-acceptance.json', 'cb31c04481bedab1ad9af1751ee64f0758b95fb41b2f8656eb305d2cdbf4ed58'),
        verification=verify_hash(OLD / 'analysis-v1/verification.json', '8f26ded67e0d275151cbdb3dde20ddd12d6692ee2c1163b12d90c08c36817a2a'),
        inventory=verify_hash(HISTORY / 'artifacts.json', 'bd4d9c2913924659fa0b83af99e08e0aa65393ee8848af54b2889939086ec2bb'),
        exp010_acceptance=verify_hash(EXP010 / 'acceptance-20261101/acceptance-status.json', '6295fd0b9f93b67bd1c4ecf6102082ffb2f17f08b1d30b02b1ed946b21ec4023'),
        exp010_oof=verify_hash(EXP010 / 'download-v2/results/oof-share0.5.npy', 'd0215a7bd06d5f7dfbf2119c67db5892eef54e78dae4173225254da664e8baa7'),
        exp010_csv=verify_hash(EXP010 / 'download-v2/results/oof-share0.5.csv', '5ffdf9ed9085a4a1dd0d11f78113638ae609694e4b389d466d08e9400946aa0e'),
    )
    verification = read(controls['verification'])
    require(verification['status'] == 'PASS' and verification['historical_control']['matrix_byte_values_equal']
            and read(controls['exp010_acceptance'])['status'] == 'PASS'
            and read(controls['acceptance'])['source_artifacts_sha256'] == sha(controls['inventory']),
            'Historical acceptance is not bound to the fixed inventory')
    inv = read(controls['inventory'])
    identity = {key: inv[key] for key in ('run_id', 'config_sha256', 'runner_sha256')}
    index = file_index(inv['files'])
    controls['control_oof'] = anchored(HISTORY, index, 'oof/control.npz')
    verify_hash(controls['control_oof'], '633bb49ee1c25bbdf94921de6c9497be31423de1941a7a6e62208fa6d8b8f4d2')
    frame = pd.read_csv(controls['train'], dtype={'id': str, 'prompt_id': str})
    folds = pd.read_csv(controls['folds'], dtype={'id': str, 'prompt_id': str})
    require(len(frame) == len(folds) == 2699 and frame.id.is_unique and frame.prompt_id.nunique() == 150,
            'Wrong official row or prompt coverage')
    for col in ('id', 'prompt_id', 'model'):
        V.equal(frame[col], folds[col], 'official fold ' + col)
    frame['fold'] = folds['fold'].to_numpy()
    require(frame.groupby('prompt_id').fold.nunique().eq(1).all() and set(frame.fold) == set(range(5)),
            'Official prompt separation failed')
    labels = sorted(frame.model.unique())
    require(len(labels) == 18, 'Expected 18 classes')
    y = np.array([labels.index(v) for v in frame.model], dtype=np.int64)
    control = V.load_npz(controls['control_oof'], identity, labels)
    pred = checked_predictions(control, frame, y, 'control')
    V.equal(control['scores'], np.load(controls['exp010_oof'], allow_pickle=False), 'EXP010/042 complete score matrices')
    old_csv = pd.read_csv(controls['exp010_csv'], dtype={'id': str, 'prompt_id': str})
    for col in ('id', 'prompt_id', 'model', 'fold'):
        V.equal(frame[col], old_csv[col], 'EXP010 ' + col)
    V.equal(old_csv.predicted_model, np.asarray(labels)[pred], 'EXP010 argmax')
    metrics = V.metrics(y, pred)
    require(metrics['correct'] == 1264 and abs(metrics['balanced_accuracy'] - 0.4682426050211286) < 1e-14,
            'Historical control metrics differ')
    return frame, y, labels, control, index, identity, controls


def checked_predictions(data, frame, truth, arm):
    return V.check_prediction(data, frame.id.to_numpy(), truth, frame.fold.to_numpy(), arm)


def require_same_routing(control, other):
    for name in ('ids', 'true', 'labels', 'fold', 'centred_mask', 'route'):
        V.equal(control[name], other[name], 'paired ' + name)
    # Unsupervised group IDs have meaning within a fold, not across folds.
    for fold in np.unique(control['fold']):
        ix = control['fold'] == fold
        a, b = control['group_ids'][ix], other['group_ids'][ix]
        V.equal(a[:, None] == a[None, :], b[:, None] == b[None, :], 'paired group partition')


def sample_row_table(frame, y, labels, predictions):
    rows = frame[['id', 'prompt_id', 'model', 'fold']].copy()
    labels = np.asarray(labels)
    for arm, pred in predictions.items():
        rows[arm + '_prediction'] = labels[pred]
        rows[arm + '_correct'] = pred == y
    for left, right in PAIRS:
        a, b = predictions[left], predictions[right]
        rows[right + '_minus_' + left] = np.select(
            [(a != y) & (b == y), (a == y) & (b != y), a != b],
            ['fixed', 'new_error', 'both_wrong_changed'], default='unchanged')
    return rows


def dense_analysis(output, frame, y, labels, datasets):
    predictions = {arm: checked_predictions(data, frame, y, arm) for arm, data in datasets.items()}
    rows = sample_row_table(frame, y, labels, predictions)
    rows.to_csv(output / 'predictions.csv', index=False)
    class_rows, fold_rows, compare_rows = [], [], []
    summary = {}
    for arm, pred in predictions.items():
        metric = V.metrics(y, pred)
        require(not metric['missing_class_indices'], 'Full OOF missing class')
        summary[arm] = {k: metric[k] for k in ('rows', 'correct', 'accuracy', 'balanced_accuracy')}
        for i, label in enumerate(labels):
            tp = int(np.sum((y == i) & (pred == i)))
            fp = int(np.sum((y != i) & (pred == i)))
            support = metric['support'][i]
            class_rows.append(dict(arm=arm, label=label, support=support, correct=tp, false_positives=fp,
                                   recall=tp / support, precision=tp / (tp + fp) if tp + fp else None))
        for fold in range(5):
            ix = frame.fold.to_numpy() == fold
            fm = V.metrics(y[ix], pred[ix])
            require(not fm['missing_class_indices'], 'Fold missing class')
            fold_rows.append(dict(arm=arm, fold=fold, prompts=frame.prompt_id[ix].nunique(),
                                  **{k: fm[k] for k in ('rows', 'correct', 'accuracy', 'balanced_accuracy')}))
    draws, fold_comparisons, class_comparisons = {}, [], []
    for left, right in PAIRS:
        key = right + '_minus_' + left
        paired = V.paired_metrics(y, predictions[left], predictions[right], frame.prompt_id.to_numpy(),
                                  classes=18, draws=2000, seed=42)
        bs = paired['bootstrap']
        draws[key] = bs.pop('deltas')
        require(len(draws[key]) == 2000 and bs['valid_draws'] + bs['null_draws'] == 2000,
                'Incomplete bootstrap')
        counts = rows[key].value_counts()
        for name, field in [('fixed', 'fixed_errors'), ('new_error', 'introduced_errors')]:
            require(int(counts.get(name, 0)) == paired[field], 'Changed-row count does not reconcile')
        for fold in range(5):
            block = rows[rows.fold.eq(fold)]
            a, b = [next(v for v in fold_rows if v['arm'] == arm and v['fold'] == fold) for arm in (left, right)]
            fold_comparisons.append(dict(comparison=key, fold=fold, rows=len(block),
                delta_ba=b['balanced_accuracy'] - a['balanced_accuracy'], net_correct=b['correct'] - a['correct'],
                fixed=int(block[key].eq('fixed').sum()), new_errors=int(block[key].eq('new_error').sum()),
                changed=int(block[key].ne('unchanged').sum())))
        for label in labels:
            a, b = [next(v for v in class_rows if v['arm'] == arm and v['label'] == label) for arm in (left, right)]
            class_comparisons.append(dict(comparison=key, label=label, support=a['support'],
                delta_correct=b['correct'] - a['correct'], delta_recall=b['recall'] - a['recall'],
                delta_false_positives=b['false_positives'] - a['false_positives']))
        compare_rows.append(dict(comparison=key, primary=key == 'four240_minus_four120', delta_ba=paired['delta_ba'],
              fixed=paired['fixed_errors'], new_errors=paired['introduced_errors'],
              net_correct=summary[right]['correct'] - summary[left]['correct'],
              changed=paired['prediction_disagreements'], both_wrong_changed=int(counts.get('both_wrong_changed', 0)),
              bootstrap=bs))
    pd.DataFrame({'draw': range(1, 2001), **draws}).to_csv(output / 'bootstrap-draws.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output / 'per-class.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output / 'per-fold.csv', index=False)
    pd.DataFrame(fold_comparisons).to_csv(output / 'fold-comparisons.csv', index=False)
    pd.DataFrame(class_comparisons).to_csv(output / 'class-comparisons.csv', index=False)
    return summary, compare_rows


def new_results(results, result_digest, frame, y, labels, control):
    frozen = read(verify_hash(HERE / 'manifest.json', FROZEN_MANIFEST))
    require(frozen['freeze_status'] == 'FROZEN_BEFORE_TRAINING' and frozen['no_new_control_fits'] is True,
            'Training protocol was not frozen')
    frozen_files = file_index(frozen['files'])
    for name in frozen_files:
        anchored(ROOT, frozen_files, name)
    result_path = verify_hash(results / 'result.json', result_digest)
    result = read(result_path)
    identity = result['identity']
    config = read(HERE / 'config.json')
    require(config['experiment_id'] == 'EXP-052' and config['run_id'] == result['run_id'] == identity['run_id']
            and identity['arms'] == config['arms'] == list(ARMS[1:]) and identity['labels'] == config['labels'] == labels,
            'Wrong EXP052 protocol')
    require(result['status'] == 'COMPLETE' and result['trained_svms'] == 20 and result['control_fits'] == 0
            and result['gpu_used'] is False, 'Training receipt incomplete')
    for name in ('state.json', 'guard.json'):
        receipt = read(results / name)
        require(receipt['status'] == 'COMPLETE' and receipt['run_id'] == result['run_id'] and
                receipt['result_sha256'] == result_digest, f'{name} does not certify this result')
    require(read(results / 'identity.json') == identity, 'Result identity file mismatch')
    require(sha(HERE / 'config.json') == result['config_sha256'] == identity['config_sha256']
            and sha(HERE / 'train_volume.py') == result['runner_sha256'] == identity['runner_sha256']
            and identity['input_hashes'] == config['input_hashes'], 'Current frozen config/runner mismatch')
    source_paths = {'four120.csv': HERE / 'data/four120.csv', 'four240.csv': HERE / 'data/four240.csv',
                    'volume_gate.json': HERE / 'metrics/data-gate.json', 'train_volume.py': HERE / 'train_volume.py',
                    'legacy_runner.py': CLOUD / 'payload/runner.py',
                    'format_runner.py': ROOT / 'cloud/format120-044/payload/runner.py'}
    for key, digest in config['input_hashes'].items():
        require(PurePosixPath(key).name == key and '\\' not in key and ':' not in key, 'Invalid frozen input name')
        verify_hash(source_paths.get(key, CLOUD / 'payload' / key), digest)
    require(identity['format_runner_sha256'] == config['input_hashes']['format_runner.py'] and
            identity['legacy_runner_sha256'] == config['input_hashes']['legacy_runner.py'] and
            identity['core_sha256'] == config['input_hashes']['stylegram_core.py'], 'Worker helper bindings differ')
    identity_sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, allow_nan=False,
                                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    flat = {k: identity[k] for k in ('run_id', 'config_sha256', 'runner_sha256', 'legacy_runner_sha256', 'core_sha256')}
    flat['identity_sha256'] = identity_sha
    require(all(result[k] == v for k, v in flat.items()), 'Result identity differs')
    dense = {'control': control}
    require(set(result['oof']) == set(ARMS[1:]), 'Missing aggregate arm')
    for arm in ARMS[1:]:
        item = result['oof'][arm]
        require(item['path'] == f'oof/{arm}.npz', 'Wrong aggregate path')
        path = verify_hash(results / item['path'], item['sha256'])
        dense[arm] = V.load_npz(path, flat, labels)
        checked_predictions(dense[arm], frame, y, arm)
        require_same_routing(control, dense[arm])
    jobs = {}
    label_index = {label: idx for idx, label in enumerate(labels)}
    def training_labels(path):
        model = pd.read_csv(path, usecols=['model']).model
        require(model.isin(label_index).all(), 'Unknown training label')
        return model.map(label_index).to_numpy(dtype=int)
    public_y = training_labels(CLOUD / 'payload/extra.csv')
    added_y = {arm: training_labels(HERE / 'data' / (arm + '.csv')) for arm in ARMS[1:]}
    for item in result['completed_jobs']:
        arm, fold = item['arm'], item['fold']
        require(arm in ARMS[1:] and type(fold) is int and 0 <= fold < 5 and
                (arm, fold) not in jobs and item['path'] == f'jobs/f{fold}-{arm}', 'Invalid/duplicate job')
        folder = results / item['path']
        done = read(verify_hash(folder / 'completion.json', item['completion_sha256']))
        require(done['status'] == 'COMPLETE' and done['arm'] == arm and done['fold'] == fold and
                done['identity'] == identity, 'Fold completion identity changed')
        index = file_index(done['files'])
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} - {'completion.json'}
        require(set(index) == actual, 'Unclosed job inventory')
        for name in index:
            anchored(folder, index, name)
        job_result = read(folder / 'result.json')
        require(job_result['status'] == 'COMPLETE' and job_result['arm'] == arm and job_result['fold'] == fold
                and job_result['trained_svms'] == 2, 'Invalid fold result')
        part = V.load_npz(folder / 'dense.npz', flat, labels)
        ix = frame.fold.to_numpy() == fold
        weight = read(folder / 'weights.json')
        require(weight['arm'] == arm and weight['fold'] == fold, 'Weight receipt target differs')
        V.check_weights(weight, y[~ix], public_y, added_y[arm], labels)
        for kind in ('plain', 'centred'):
            fit = read(folder / f'fits/f{fold}-{arm}-{kind}.json')
            require(fit['C'] == .5 and fit['dual'] is True and fit['seed'] == 42 and fit['max_iter'] == 4000
                    and fit['dtype'] == 'float64' and fit['classes'] == list(range(18))
                    and not any(w['category'] == 'ConvergenceWarning' for w in fit['warnings']),
                    'Fit parameters or convergence differ')
        checked_predictions(part, frame[ix], y[ix], arm)
        for key in ('ids', 'true', 'fold', 'scores', 'plain_scores', 'group_ids', 'centred_mask', 'route'):
            V.equal(part[key], dense[arm][key][ix], 'fold-to-aggregate ' + key)
        jobs[arm, fold] = folder
    require(set(jobs) == {(arm, fold) for arm in ARMS[1:] for fold in range(5)}, 'Incomplete ten-job coverage')
    return dense, jobs, flat, result


def sparse_analysis(output, frame, y, labels, dense, jobs, old_index, old_identity, new_identity):
    summaries, samples = [], []
    for m, repeat in SCENARIOS:
        truth_parts, samples_by_arm = [], {arm: [] for arm in ARMS}
        for fold in range(5):
            valid = frame[frame.fold.eq(fold)].reset_index(drop=True)
            selected = V.sparse_selection(valid, m, repeat)
            ix = np.flatnonzero(frame.fold.to_numpy() == fold)[selected]
            included = frame.iloc[ix]
            truth_parts.append(y[ix])
            baseline = None
            for arm in ARMS:
                if arm == 'control':
                    path = anchored(HISTORY, old_index, f'sparse/f{fold}-control-m{m}-r{repeat}.npz')
                    identity = old_identity
                else:
                    path = jobs[arm, fold] / f'sparse/m{m}-r{repeat}.npz'
                    identity = new_identity
                part = V.load_npz(path, identity, labels)
                require(part['m'].item() == m and part['repeat'].item() == repeat, 'Sparse scenario differs')
                pred = checked_predictions(part, included, y[ix], arm)
                V.equal(part['plain_scores'], dense[arm]['plain_scores'][ix], 'sparse/dense plain margin')
                if baseline is None:
                    baseline = part
                else:
                    require_same_routing(baseline, part)
                samples_by_arm[arm].append(pred)
            sample = included[['id', 'prompt_id', 'model', 'fold']].copy()
            sample.insert(0, 'repeat', repeat)
            sample.insert(0, 'm', m)
            for arm in ARMS:
                sample[arm] = np.asarray(labels)[samples_by_arm[arm][-1]]
            samples.append(sample)
        pooled = np.concatenate(truth_parts)
        values = {arm: V.metrics(pooled, np.concatenate(parts)) for arm, parts in samples_by_arm.items()}
        require(all(v['balanced_accuracy'] is not None for v in values.values()), 'Pooled sparse class missing')
        row = {'m': m, 'repeat': repeat, 'rows': len(pooled)}
        for arm, metric in values.items():
            row[arm + '_ba'] = metric['balanced_accuracy']
        for left, right in PAIRS:
            row[right + '_minus_' + left] = row[right + '_ba'] - row[left + '_ba']
        summaries.append(row)
    pd.concat(samples, ignore_index=True).to_csv(output / 'sparse-predictions.csv', index=False)
    table = pd.DataFrame(summaries)
    table.to_csv(output / 'sparse-repeats.csv', index=False)
    return table.drop(columns=['repeat', 'rows']).groupby('m', as_index=False).mean().to_dict('records')


def analyze(results, result_digest, output):
    results, output = results.resolve(strict=True), output.resolve()
    require(not output.exists() and not output.is_relative_to(results) and not results.is_relative_to(output),
            'Output must be fresh and separate from training outputs')
    frame, y, labels, control, old_index, old_identity, control_paths = history()
    dense, jobs, identity, result = new_results(results, result_digest, frame, y, labels, control)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / 'analysis'
        staging.mkdir()
        metrics, comparisons = dense_analysis(staging, frame, y, labels, dense)
        sparse = sparse_analysis(staging, frame, y, labels, dense, jobs, old_index, old_identity, identity)
        summary = dict(status='PASS_FIXED_DEVELOPMENT_OOF', experiment_id='EXP-052',
            primary_comparison='four240_minus_four120', metrics=metrics, comparisons=comparisons,
            sparse_mean_by_m=sparse, official_rows=2699, official_prompts=150, classes=labels,
            folds=5, run_id=result['run_id'], result_sha256=result_digest, oof=result['oof'],
            control_reused=True, control_oof_equals_exp010_matrix=True,
            scope='Previously inspected official prompt-group development OOF of one Stylegram recipe. '
                  'Within-prompt paired intervals condition on the fitted models. Four120 is a fixed nested '
                  'selection, not independent randomized learning-curve replicates. No test-set or leaderboard inference.',
            sparse_scope='Each repeat pools all five folds before macro recall. The five repeats share fitted models.')
        write(staging / 'summary.json', summary)
        evidence = dict(analyzer_sha256=sha(Path(__file__)), result_sha256=result_digest, frozen_manifest_sha256=FROZEN_MANIFEST,
            config_sha256=result['config_sha256'], runner_sha256=result['runner_sha256'],
            historical_inputs={key: {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
                               for key, path in control_paths.items()},
            inventory=[{'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha(f)} for f in sorted(staging.iterdir())],
            generated_at=datetime.now(timezone.utc).isoformat(), numpy_version=np.__version__, pandas_version=pd.__version__)
        write(staging / 'analysis-manifest.json', evidence)
        require(not output.exists(), 'Analysis already exists')
        staging.rename(output)
    return summary


def verify_analysis(output):
    """Recompute metrics from the published row predictions, independent of margin analysis."""
    output = output.resolve(strict=True)
    receipt = read(output / 'analysis-manifest.json')
    require(receipt['analyzer_sha256'] == sha(Path(__file__)), 'Analyzer changed after result')
    index = file_index([dict(path=item['name'], bytes=item['bytes'], sha256=item['sha256'])
                        for item in receipt['inventory']])
    require(set(index) == {p.name for p in output.iterdir()} - {'analysis-manifest.json'}, 'Unclosed analysis output')
    for name in index:
        anchored(output, index, name)
    summary = read(output / 'summary.json')
    require(summary['status'] == 'PASS_FIXED_DEVELOPMENT_OOF' and summary['result_sha256'] == receipt['result_sha256'],
            'Unbound or incomplete analysis')
    rows = pd.read_csv(output / 'predictions.csv', dtype={'id': str, 'prompt_id': str})
    frame, y, labels, control, *_ = history()
    for name in ('id', 'prompt_id', 'model', 'fold'):
        V.equal(rows[name], frame[name], 'recomputed ' + name)
    V.equal(rows.control_prediction, np.array(labels)[control['scores'].argmax(1)], 'recomputed control')
    # Represent each prompt by a support and correct-count vector. Bootstrap weights
    # count repeated sampled prompts, unlike paired_metrics which indexes repeated arrays.
    group, inverse = np.unique(rows.prompt_id, return_inverse=True)
    require(len(group) == 150, 'Invalid recomputation prompt count')
    gsupport = pd.crosstab(rows.prompt_id, rows.model).reindex(index=group, columns=labels, fill_value=0).to_numpy()
    group_correct = {}
    for arm in ARMS:
        correct = rows[arm + '_prediction'].eq(rows.model)
        V.equal(rows[arm + '_correct'], correct, 'recomputed correct')
        group_correct[arm] = np.zeros_like(gsupport)
        np.add.at(group_correct[arm], (inverse[correct], y[correct]), 1)
        ba = (group_correct[arm].sum(0) / gsupport.sum(0)).mean()
        require(int(correct.sum()) == summary['metrics'][arm]['correct'] and
                abs(ba - summary['metrics'][arm]['balanced_accuracy']) < 1e-14, 'Recomputed arm metrics differ')
    rng = np.random.default_rng(42)
    mult = np.vstack([np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group)) for _ in range(2000)])
    denominator = mult @ gsupport
    valid = (denominator > 0).all(axis=1)
    draws = pd.read_csv(output / 'bootstrap-draws.csv')
    V.equal(draws.draw, np.arange(1, 2001), 'bootstrap draw order')
    for left, right in PAIRS:
        key = right + '_minus_' + left
        values = np.full(2000, np.nan)
        values[valid] = ((mult[valid] @ (group_correct[right] - group_correct[left])) / denominator[valid]).mean(1)
        require(np.allclose(draws[key], values, rtol=0, atol=1e-14, equal_nan=True), 'Recomputed draws differ')
        recorded = next(x for x in summary['comparisons'] if x['comparison'] == key)
        require(recorded['primary'] == (key == 'four240_minus_four120'), 'Primary contrast changed')
        require(np.allclose(np.quantile(values[valid], [.025, .975]), recorded['bootstrap']['ci95'], rtol=0, atol=1e-14),
                'Recomputed confidence interval differs')
    return {'status': 'PASS_INDEPENDENT_ROW_RECOMPUTATION', 'rows': len(rows), 'arms': len(ARMS),
            'comparisons': len(PAIRS), 'paired_draws_per_comparison': 2000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-control', action='store_true')
    parser.add_argument('--verify-analysis', type=Path, help='independently recompute from saved row-level CSV')
    parser.add_argument('--results', type=Path, help='extracted fixed-version results folder')
    parser.add_argument('--result-sha256', help='result.json digest from the download evidence')
    parser.add_argument('--output', type=Path, default=HERE / 'analysis')
    args = parser.parse_args()
    if args.verify_analysis is not None:
        print(json.dumps(verify_analysis(args.verify_analysis)))
        return
    if args.check_control:
        frame, y, labels, baseline, *_ = history()
        predicted = checked_predictions(baseline, frame, y, 'control')
        print(json.dumps({'status':'PASS_FIXED_CONTROL','rows':len(frame),'prompts':frame.prompt_id.nunique(),
                          'classes':len(labels),'correct':int((y == predicted).sum()),
                          'balanced_accuracy':V.metrics(y, predicted)['balanced_accuracy']}))
        return
    if args.results is None or not args.result_sha256:
        parser.error('--results and --result-sha256 are required for completed run analysis')
    summary = analyze(args.results, args.result_sha256, args.output)
    print(json.dumps({key: summary[key] for key in ('status', 'metrics', 'comparisons')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
