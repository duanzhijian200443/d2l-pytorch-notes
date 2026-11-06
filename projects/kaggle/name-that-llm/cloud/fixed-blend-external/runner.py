"""Fit the three frozen CPU SVMs and evaluate one accepted external panel.

Training recipes are EXP-010's Stylegram share=.5 (plain and centred) and
EXP-018's raw-public svc4 row weight=.35. Inference uses EXP-024's fixed
row-z-score blend alpha=.10. Evaluation labels/prompt IDs are never passed
to feature fitting, classifiers, or Stylegram inference grouping.

The original source modules and training CSV bytes are required unchanged.
No estimators/vocabularies are persisted; only scores and audit receipts are
saved. Importing this file does not load data or run training.
"""
from __future__ import annotations

import os
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '2'

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.sparse import vstack
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_info, threadpool_limits


HERE = Path(__file__).resolve().parent
ALPHA = 0.10
REQUIRED_VERSIONS = {'numpy': '2.1.3', 'pandas': '2.3.3', 'scipy': '1.16.3', 'sklearn': '1.6.1'}
LABELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol',
    'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max',
]
TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
FROZEN_INPUTS = {
    'style_extra.csv': '1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e',
    'public.csv': '0a1e31a0f8b0ece7ca44a961075252fe275107f3710804971838d714a4e77aaa',
    'stylegram_core.py': 'caa46bd121164aba891c3d78e662540db9294d653db09f53d02833c1b4ab77ef',
    'svc4.py': '80f66aedad27e8017defe31de7f0286c1b54efa5e96ba6e2cb42516a86e73c18',
    'margin_blend.py': '8c4288447de755d9bd160e5646794b2307f56ae00ae11104d062d9697ef5dc83',
}
DYNAMIC_NAMES = {'state.json', 'guard.json', 'training.log', '.run.lock', 'artifacts.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def numpy_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError('Unsupported JSON value: ' + type(value).__name__)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=numpy_json) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    for name, value in arrays.items():
        require(np.asarray(value).dtype.kind != 'O', 'Object array is forbidden: ' + name)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False, lineterminator='\n')
    temporary.replace(path)


def validate_protocol(protocol):
    # The separate data audit must explicitly accept this panel before fitting.
    require(protocol.get('evaluation_accepted') is True, 'External evaluation data gate is not accepted')
    require(protocol.get('labels') == LABELS, 'Changed 18-class training label order')
    require(protocol.get('required_versions') == REQUIRED_VERSIONS, 'Changed required dependencies')
    require(protocol.get('train_sha256') == TRAIN_SHA256, 'Changed official training input')
    for key, expected in (('alpha', ALPHA), ('style_share', .5), ('public_row_weight', .35)):
        require(protocol.get(key, expected) == expected, 'Changed frozen parameter: ' + key)
    covered = protocol.get('evaluation_labels')
    require(isinstance(covered, list) and len(covered) == len(set(covered)) == 4,
            'Expected four distinct external-panel labels')
    require(set(covered).issubset(LABELS), 'Unknown external-panel labels')
    for key in ('evaluation_rows', 'evaluation_groups'):
        require(type(protocol.get(key)) is int and protocol[key] > 0, 'Invalid ' + key)
    hashes = protocol.get('input_hashes')
    require(isinstance(hashes, dict) and set(FROZEN_INPUTS).union({'evaluation.csv', 'runner.py', 'guard.py'}).issubset(hashes),
            'Missing pinned payload inputs')
    for name, expected in FROZEN_INPUTS.items():
        require(hashes[name] == expected, 'Original training/core input changed: ' + name)
    for name, digest in hashes.items():
        require(isinstance(name, str) and Path(name).name == name and '\\' not in name and ':' not in name,
                'Unsafe payload input name')
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                'Invalid input SHA-256: ' + name)


def validate_frame(frame, rows, name, labels, require_id=True):
    required = {'prompt_id', 'model', 'text'} | ({'id'} if require_id else set())
    require(required.issubset(frame.columns), 'Missing columns: ' + name)
    require(len(frame) == rows, 'Wrong row count: ' + name)
    require(frame.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all(), 'Blank/non-string text: ' + name)
    require(frame.prompt_id.map(lambda x: isinstance(x, str) and bool(x)).all(), 'Blank prompt ID: ' + name)
    require(set(frame.model) == set(labels), 'Wrong label coverage: ' + name)
    identifier = 'id' if 'id' in frame.columns else 'row_id'
    require(identifier in frame and frame[identifier].is_unique and frame[identifier].ne('').all(),
            'Missing/duplicate sample IDs: ' + name)
    return frame[identifier].to_numpy(dtype=str)


def style_sample_weights(y, extra_y, n_classes=18):
    """EXP-010 runner.py:29-35, with its fixed share .5 and float64 weights."""
    train_counts = np.bincount(y, minlength=n_classes).astype(float)
    extra_counts = np.bincount(extra_y, minlength=n_classes).astype(float)
    require((train_counts > 0).all() and (extra_counts > 0).all(), 'Stylegram training lacks a class')
    weights = np.r_[np.full(len(y), .5), .5 * train_counts[extra_y] / extra_counts[extra_y]]
    mass = np.bincount(np.r_[y, extra_y], weights=weights, minlength=n_classes)
    require(np.allclose(mass, train_counts), 'Stylegram class weight mass changed')
    return weights


def make_style_classifier():
    return LinearSVC(C=.5, max_iter=4000, random_state=42, dual=True)


def matrix_receipt(matrix):
    digest = hashlib.sha256(str(matrix.shape).encode() + str(matrix.dtype).encode())
    for part in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(memoryview(np.ascontiguousarray(part)).cast('B'))
    return {'shape': list(matrix.shape), 'dtype': str(matrix.dtype), 'nnz': int(matrix.nnz),
            'csr_bytes': int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
            'sha256': digest.hexdigest()}


def vectorizer_receipts(vectorizers):
    receipts = []
    for name, vectorizer, scale in vectorizers:
        vocabulary = json.dumps(sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items()),
                                ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        receipts.append({'view': name, 'scale': scale, 'columns': len(vectorizer.vocabulary_),
                         'vocabulary_sha256': hashlib.sha256(vocabulary).hexdigest(),
                         'idf_sha256': hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest()})
    return receipts


def fit_checked(model, matrix, target, weights, member, labels, out):
    print('FIT START ' + member, flush=True)
    begun = time.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(matrix, target, sample_weight=weights)
    require(not any(issubclass(w.category, ConvergenceWarning) for w in captured), member + ' did not converge')
    require(model.classes_.tolist() == list(range(len(labels))), member + ' class order changed')
    require(int(model.n_iter_) < model.max_iter, member + ' reached the iteration cap')
    record = {'member': member, 'rows': len(target), 'features': matrix.shape[1],
              'classes': model.classes_.tolist(), 'labels': labels, 'n_iter': int(model.n_iter_),
              'convergence_warnings': 0, 'fit_seconds': time.monotonic() - begun,
              'parameters': model.get_params(), 'rss_bytes_after_fit': psutil.Process().memory_info().rss,
              'models_saved': False}
    save_json(out / ('fit-' + member + '.json'), record)
    print('FIT COMPLETE ' + json.dumps(record, default=numpy_json), flush=True)
    return record


def predict_stylegram(features, plain, centered, texts, core):
    """Original text-only inferred grouping and plain/centred row routing.

    The function deliberately accepts neither truth labels nor prompt IDs.
    """
    matrix = features.transform(texts)
    plain_scores = plain.decision_function(matrix)
    groups = core.normalised_content_groups(texts)
    mask = core.eligible(groups)
    scores = plain_scores.copy()
    if mask.any():
        centered_matrix = core.sparse_group_centre(matrix[mask], groups[mask])
        scores[mask] = centered.decision_function(centered_matrix)
    require(scores.shape == plain_scores.shape and np.isfinite(scores).all(), 'Invalid Stylegram inference scores')
    require(np.array_equal(scores[~mask], plain_scores[~mask]), 'Stylegram plain fallback changed')
    return scores, plain_scores, groups, mask


def covered_label_metrics(truth, prediction, labels):
    """Keep 18-way predictions; average recalls only over covered true labels."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    require(truth.ndim == 1 and truth.shape == prediction.shape and len(truth) > 0, 'Invalid metric rows')
    require(truth.dtype.kind in 'iu' and prediction.dtype.kind in 'iu', 'Metric labels must be integer indices')
    require(((truth >= 0) & (truth < len(labels))).all() and ((prediction >= 0) & (prediction < len(labels))).all(),
            'Metric label index out of bounds')
    covered = np.unique(truth)
    support = np.bincount(truth, minlength=len(labels))
    correct = np.bincount(truth[truth == prediction], minlength=len(labels))
    return {'metric': 'covered_label_macro_recall', 'score_columns': len(labels),
            'covered_labels': [labels[i] for i in covered], 'covered_label_count': len(covered),
            'macro_recall': float(np.mean(correct[covered] / support[covered])),
            'accuracy': float(np.mean(truth == prediction)), 'correct': int(np.sum(truth == prediction)),
            'rows': len(truth), 'per_class': [
                {'model': labels[i], 'support': int(support[i]), 'correct': int(correct[i]),
                 'recall': float(correct[i] / support[i])} for i in covered]}


def paired_interval(truth, reference, prediction, prompts, labels, draws=2000, seed=42):
    names = sorted(set(prompts))
    code = {name: i for i, name in enumerate(names)}
    groups = np.array([code[name] for name in prompts])
    covered = np.unique(truth)
    support = np.zeros((len(names), len(labels)), dtype=np.int64)
    difference = np.zeros_like(support)
    np.add.at(support, (groups, truth), 1)
    np.add.at(difference, (groups, truth), (prediction == truth).astype(int) - (reference == truth).astype(int))
    sampled = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    denominators = support[sampled].sum(axis=1)[:, covered]
    invalid = int((denominators == 0).any(axis=1).sum())
    interval = None
    if invalid == 0:
        deltas = (difference[sampled].sum(axis=1)[:, covered] / denominators).mean(axis=1)
        interval = np.quantile(deltas, [.025, .975], method='linear').tolist()
    return {'delta_definition': 'blend_minus_stylegram_covered_label_macro_recall',
            'unit': 'external_prompt_id', 'draws': draws, 'seed': seed,
            'valid_draws': draws - invalid, 'invalid_draws': invalid, 'ci95': interval,
            'missing_class_policy': 'null interval if any resample lacks a covered true class',
            'conditional_on_fitted_models': True, 'includes_retraining_uncertainty': False}


def weight_receipt(target, weights, official_rows, labels, class_weights):
    k = len(labels)
    official_mass = np.bincount(target[:official_rows], weights=weights[:official_rows], minlength=k)
    public_mass = np.bincount(target[official_rows:], weights=weights[official_rows:], minlength=k)
    return {'labels': labels, 'class_counts': np.bincount(target, minlength=k),
            'class_weights': class_weights, 'official_sample_mass': official_mass,
            'public_sample_mass': public_mass, 'effective_class_mass': (official_mass + public_mass) * class_weights,
            'sample_weight_dtype': str(weights.dtype)}


def run(here, out):
    import gc
    started = time.monotonic()
    protocol_path = here / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    versions = {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    require(versions == REQUIRED_VERSIONS, 'Runtime versions differ: ' + str(versions))
    input_hashes = {protocol_path: sha(protocol_path)}
    for name, digest in protocol['input_hashes'].items():
        path = here / name
        require(sha(path) == digest, 'Payload hash mismatch: ' + name)
        input_hashes[path] = digest
    candidates = [p for p in Path('/kaggle/input').rglob('train.csv')
                  if 'name-that-llm' in str(p) and sha(p) == TRAIN_SHA256]
    require(len(candidates) == 1, 'Expected one hash-pinned official train.csv')
    official_path = candidates[0]
    input_hashes[official_path] = TRAIN_SHA256
    official = pd.read_csv(official_path, dtype=str, keep_default_na=False)
    extra = pd.read_csv(here / 'style_extra.csv', dtype=str, keep_default_na=False)
    public = pd.read_csv(here / 'public.csv', dtype=str, keep_default_na=False)
    evaluation = pd.read_csv(here / 'evaluation.csv', dtype=str, keep_default_na=False)
    official_ids = validate_frame(official, 2699, 'official', LABELS)
    extra_ids = validate_frame(extra, 10073, 'style_extra', LABELS, require_id=False)
    public_ids = validate_frame(public, 5400, 'public', LABELS)
    evaluation_ids = validate_frame(evaluation, protocol['evaluation_rows'], 'evaluation', protocol['evaluation_labels'])
    require(official.prompt_id.nunique() == 150 and extra.prompt_id.nunique() == 694 and public.prompt_id.nunique() == 300,
            'Training prompt population changed')
    require(evaluation.prompt_id.nunique() == protocol['evaluation_groups'], 'External prompt population changed')
    require(evaluation.id.is_unique, 'Repeated external sample ID')
    sys.path.insert(0, str(here))
    core = importlib.import_module('stylegram_core')
    svc = importlib.import_module('svc4')
    blend = importlib.import_module('margin_blend')
    for name, module in (('stylegram_core', core), ('svc4', svc), ('margin_blend', blend)):
        require(Path(module.__file__).resolve() == (here / (name + '.py')).resolve(), 'Wrong imported core: ' + name)
    labels = list(LABELS)
    index = {name: i for i, name in enumerate(labels)}
    y = official.model.map(index).to_numpy(dtype=int)
    extra_y = extra.model.map(index).to_numpy(dtype=int)
    public_y = public.model.map(index).to_numpy(dtype=int)
    environment = {'observed_at': stamp(), 'versions': versions, 'python': platform.python_version(),
                   'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'threadpools': threadpool_info(),
                   'gpu_requested': False, 'protocol': protocol, 'models_saved': False,
                   'score_type': 'uncalibrated 18-class LinearSVC decision margins',
                   'scope': 'full-data refit followed by one external four-label panel evaluation'}
    save_json(out / 'environment.json', environment)
    save_npz(out / 'training-membership.npz', labels=np.array(labels), official_ids=official_ids,
             official_prompt_ids=official.prompt_id.to_numpy(dtype=str), official_true=y,
             style_public_ids=extra_ids, style_public_prompt_ids=extra.prompt_id.to_numpy(dtype=str), style_public_true=extra_y,
             svc_public_ids=public_ids, svc_public_prompt_ids=public.prompt_id.to_numpy(dtype=str), svc_public_true=public_y)
    save_json(out / 'input_preflight.json', {'status': 'PASS', 'observed_at': stamp(),
              'official_rows': 2699, 'style_public_rows': 10073, 'svc_public_rows': 5400,
              'evaluation_rows': len(evaluation), 'evaluation_groups': protocol['evaluation_groups'],
              'evaluation_labels': protocol['evaluation_labels'], 'training_labels': labels,
              'hashes_verified': True, 'evaluation_accepted': True,
              'evaluation_texts_used_for_classifier_feature_fit': False,
              'inference_grouping': 'unlabelled full external text batch only; no true prompt IDs or labels',
              'content_grouping_float64_square_bytes': 8 * len(evaluation) ** 2})
    completed = []
    timings = {'preflight_seconds': time.monotonic() - started}

    def phase(name, member=None):
        save_json(out / 'state.json', {'status': 'RUNNING', 'phase': name, 'member': member,
                                     'completed_fits': completed, 'observed_at': stamp()})

    print('PREFLIGHT PASS ' + json.dumps({'versions': versions, 'evaluation_rows': len(evaluation)}), flush=True)
    phase('fit', 'stylegram_features')
    begun = time.monotonic()
    texts = official.text.tolist() + extra.text.tolist()
    style_features = core.StyleGrams().fit(texts)
    matrix = style_features.transform(texts)
    require(matrix.dtype == np.float64, 'Stylegram feature dtype changed')
    centered_matrix = vstack([
        core.sparse_group_centre(matrix[:len(official)], official.prompt_id.to_numpy()),
        core.sparse_group_centre(matrix[len(official):], extra.prompt_id.to_numpy(), 2, 10**6),
    ], format='csr')
    target = np.r_[y, extra_y]
    weights = style_sample_weights(y, extra_y)
    timings['stylegram_feature_seconds'] = time.monotonic() - begun
    save_json(out / 'features-stylegram.json', {'plain': matrix_receipt(matrix), 'centred': matrix_receipt(centered_matrix),
              'views': vectorizer_receipts([(name, spec[0], spec[2]) for name, spec in
                       zip(('char', 'word', 'shape', 'function'), style_features.vecs, strict=True)]),
              'feature_seconds': timings['stylegram_feature_seconds'], 'evaluation_rows_in_fit': 0})
    save_npz(out / 'weights-stylegram.npz', labels=np.array(labels), target=target, weights=weights)
    save_json(out / 'weights-stylegram.json', {**weight_receipt(target, weights, len(official), labels, np.ones(18)),
              'share': .5, 'semantics': 'official .5 per row; public .5 * official class count / public class count; no class_weight'})
    phase('fit', 'stylegram_plain')
    plain = make_style_classifier()
    records = [fit_checked(plain, matrix, target, weights, 'stylegram_plain', labels, out)]
    completed.append('stylegram_plain')
    phase('fit', 'stylegram_centred')
    centered = make_style_classifier()
    records.append(fit_checked(centered, centered_matrix, target, weights, 'stylegram_centred', labels, out))
    completed.append('stylegram_centred')
    del matrix, centered_matrix, target, weights, texts
    gc.collect()
    phase('predict', 'stylegram')
    begun = time.monotonic()
    style_scores, plain_scores, groups, mask = predict_stylegram(
        style_features, plain, centered, evaluation.text.tolist(), core)
    require(style_scores.shape == (len(evaluation), 18), 'Stylegram output class width changed')
    timings['stylegram_prediction_seconds'] = time.monotonic() - begun
    save_npz(out / 'stylegram-scores.npz', ids=evaluation_ids, labels=np.array(labels), scores=style_scores,
             plain_scores=plain_scores, group_ids=groups, centred_mask=mask)
    del style_features, plain, centered
    gc.collect()

    phase('fit', 'svc4_features')
    begun = time.monotonic()
    texts = official.text.tolist() + public.text.tolist()
    matrix, svc_features = svc.fit_features(texts)
    require(matrix.dtype == np.float32, 'svc4 feature dtype changed')
    target = np.r_[y, public_y]
    weights = svc.public_sample_weights(len(official), len(public), .35)
    require(weights.dtype == np.float32, 'svc4 sample-weight dtype changed')
    timings['svc4_feature_seconds'] = time.monotonic() - begun
    save_json(out / 'features-svc4.json', {'plain': matrix_receipt(matrix),
              'views': vectorizer_receipts([(spec[0], vectorizer, spec[-1]) for spec, vectorizer in
                       zip(svc_features.VIEWS, svc_features.vectorizers_, strict=True)]),
              'feature_seconds': timings['svc4_feature_seconds'], 'evaluation_rows_in_fit': 0})
    class_weights = compute_class_weight('balanced', classes=np.arange(18), y=target)
    save_npz(out / 'weights-svc4.npz', labels=np.array(labels), target=target, weights=weights)
    save_json(out / 'weights-svc4.json', {**weight_receipt(target, weights, len(official), labels, class_weights),
              'public_row_weight': .35, 'semantics': 'sklearn 1.6.1 balanced from unweighted class counts, multiplied by float32 per-row weights'})
    phase('fit', 'svc4')
    classifier = svc.make_classifier()
    records.append(fit_checked(classifier, matrix, target, weights, 'svc4', labels, out))
    completed.append('svc4')
    del matrix, target, weights, texts
    gc.collect()
    phase('predict', 'svc4_and_fixed_blend')
    begun = time.monotonic()
    svc_scores = classifier.decision_function(svc_features.transform(evaluation.text.tolist()))
    require(svc_scores.shape == style_scores.shape and np.isfinite(svc_scores).all(), 'Invalid svc4 inference scores')
    blended_scores = blend.blend_margins(style_scores, svc_scores, ALPHA)
    require(np.isfinite(blended_scores).all(), 'Nonfinite fixed-blend scores')
    timings['svc4_and_blend_prediction_seconds'] = time.monotonic() - begun
    del classifier, svc_features
    gc.collect()
    save_npz(out / 'component-and-blend-scores.npz', ids=evaluation_ids, labels=np.array(labels),
             stylegram_scores=style_scores, svc4_scores=svc_scores, blend_scores=blended_scores,
             alpha=np.array(ALPHA), group_ids=groups, centred_mask=mask)
    phase('result')
    # The evaluation target enters the computation only from this metric stage.
    truth = evaluation.model.map(index).to_numpy(dtype=int)
    predictions = {'stylegram': style_scores.argmax(1), 'svc4': svc_scores.argmax(1), 'blend': blended_scores.argmax(1)}
    metrics = {name: covered_label_metrics(truth, pred, labels) for name, pred in predictions.items()}
    reference_good = predictions['stylegram'] == truth
    blend_good = predictions['blend'] == truth
    comparison = {'alpha': ALPHA, 'delta_macro_recall': metrics['blend']['macro_recall'] - metrics['stylegram']['macro_recall'],
                  'fixed_errors': int((~reference_good & blend_good).sum()), 'new_errors': int((reference_good & ~blend_good).sum()),
                  'prediction_disagreements': int((predictions['stylegram'] != predictions['blend']).sum()),
                  'paired_prompt_bootstrap': paired_interval(truth, predictions['stylegram'], predictions['blend'],
                                                            evaluation.prompt_id.to_numpy(), labels)}
    save_json(out / 'metrics.json', {'scope': 'external four-label panel with 18-way predictions; not 18-class balanced accuracy',
              'metrics': metrics, 'comparison': comparison})
    frame = evaluation[['id', 'prompt_id', 'model']].copy()
    for name, pred in predictions.items():
        frame[name + '_prediction'] = np.array(labels)[pred]
        frame[name + '_correct'] = pred == truth
    frame['stylegram_group_id'] = groups
    frame['stylegram_centred'] = mask
    save_csv(out / 'predictions.csv', frame)
    confusion = []
    for member, pred in predictions.items():
        for true_class in np.unique(truth):
            counts = np.bincount(pred[truth == true_class], minlength=18)
            for predicted_class in range(18):
                confusion.append({'member': member, 'true_model': labels[true_class],
                                  'predicted_model': labels[predicted_class], 'count': int(counts[predicted_class])})
    save_csv(out / 'confusion.csv', pd.DataFrame(confusion))
    require(len(records) == 3 and set(completed) == {'stylegram_plain', 'stylegram_centred', 'svc4'}, 'Incomplete three-model fit')
    for path, digest in input_hashes.items():
        require(sha(path) == digest, 'Input changed during execution: ' + path.name)
    save_json(out / 'result.json', {'status': 'COMPLETE', 'observed_at': stamp(),
              'scope': 'fixed alpha=.10 full-data refit evaluated once on an accepted external four-label panel',
              'trained_svms': 3, 'completed_fits': completed, 'convergence_warnings': 0,
              'evaluation_rows': len(evaluation), 'evaluation_groups': protocol['evaluation_groups'],
              'training_labels': labels, 'evaluation_labels': protocol['evaluation_labels'], 'score_columns': 18,
              'metric': 'covered_label_macro_recall', 'alpha': ALPHA, 'metrics': metrics, 'comparison': comparison,
              'fit_records': records, 'timings': timings, 'total_seconds': time.monotonic() - started,
              'protocol_sha256': input_hashes[protocol_path], 'gpu_used': False, 'models_saved': False,
              'calibrated_probabilities': False, 'sparse_scenarios_run': False,
              'limits': ['Four covered labels do not establish performance across all 18 classes.',
                         'Prompt bootstrap is conditional on these refitted models and this external source.',
                         'Original vocabularies/estimators were not persisted; further external inference needs another authorized refit.',
                         'Data independence and provenance remain bounded by the separately accepted data audit.']})
    paths = [p for p in sorted(out.rglob('*')) if p.is_file() and p.name not in DYNAMIC_NAMES and not p.name.endswith('.tmp')]
    save_json(out / 'artifacts.json', {'files': [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths]})
    save_json(out / 'state.json', {'status': 'COMPLETE', 'phase': 'result', 'completed_fits': completed, 'observed_at': stamp()})
    print('EXPERIMENT COMPLETE ' + json.dumps({'trained_svms': 3, 'evaluation_rows': len(evaluation),
                                              'metrics': {name: item['macro_recall'] for name, item in metrics.items()}}), flush=True)


def main():
    out = Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results'))
    out.mkdir(parents=True, exist_ok=True)
    # Refusing a duplicate must not overwrite a previous run's terminal state.
    require(not (out / 'result.json').exists(), 'Refusing to overwrite an existing result')
    try:
        with threadpool_limits(limits=2):
            run(HERE, out)
    except BaseException:
        if not (out / 'failure.json').exists():
            save_json(out / 'failure.json', {'observed_at': stamp(), 'traceback': traceback.format_exc()})
        save_json(out / 'state.json', {'status': 'ERROR', 'observed_at': stamp()})
        raise


if __name__ == '__main__':
    main()
