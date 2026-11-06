"""A1-only Stylegram replacement: official2699 + frozen public10073.

This module owns only its local extra data. It neither reads nor changes the
comp stack's global EX_T/EX_Y/EX_G used by family gates. Official train.csv byte
authentication is the wrapper's responsibility before passing TR/y/PROMPT.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import vstack
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

LABELS = ('claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max')
EXTRA_SHA256 = '1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e'
CORE_SHA256 = 'caa46bd121164aba891c3d78e662540db9294d653db09f53d02833c1b4ab77ef'
OFFICIAL_TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_sha(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()).hexdigest()


def load_core(path):
    path = Path(path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA256, 'Frozen Stylegram core hash mismatch')
    spec = importlib.util.spec_from_file_location('exp046_frozen_stylegram_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def _weights(y, extra_y):
    counts = np.bincount(y, minlength=18).astype(float)
    public_counts = np.bincount(extra_y, minlength=18).astype(float)
    require((counts > 0).all() and (public_counts > 0).all(), 'Stylegram training lacks an output class')
    weights = np.r_[np.full(len(y), .5), .5 * counts[extra_y] / public_counts[extra_y]]
    require(np.allclose(np.bincount(np.r_[y, extra_y], weights=weights, minlength=18), counts), 'Per-class training mass changed')
    return weights


def _classifier():
    return LinearSVC(C=.5, max_iter=4000, random_state=42, dual=True)


def _fit_checked(matrix, target, weights, name):
    require(matrix.dtype == np.float64, 'Stylegram feature dtype must remain float64')
    model = _classifier()
    began = time.monotonic()
    print('A1 FIT START ' + name, flush=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(matrix, target, sample_weight=weights)
    require(not any(issubclass(w.category, ConvergenceWarning) for w in captured), name + ' convergence warning')
    require(model.classes_.tolist() == list(range(18)), name + ' class order changed')
    require(int(model.n_iter_) < model.max_iter, name + ' reached iteration cap')
    receipt = {'member': name, 'rows': len(target), 'features': int(matrix.shape[1]),
               'classes': model.classes_.tolist(), 'n_iter': int(model.n_iter_),
               'parameters': model.get_params(), 'fit_seconds': time.monotonic()-began,
               'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in captured]}
    print('A1 FIT COMPLETE ' + json.dumps(receipt, sort_keys=True), flush=True)
    return model, receipt


def _fit_predict(training_texts, y, training_groups, test_texts, extra_texts, extra_y, extra_groups, core):
    """Numerical core, separately testable on small synthetic 18-class inputs."""
    texts = list(training_texts) + list(extra_texts)
    features = core.StyleGrams().fit(texts)  # TE never enters classifier TF-IDF fit.
    matrix = features.transform(texts)
    centered_matrix = vstack([
        core.sparse_group_centre(matrix[:len(training_texts)], training_groups),
        core.sparse_group_centre(matrix[len(training_texts):], extra_groups, 2, 10**6),
    ], format='csr')
    target = np.r_[y, extra_y]
    weights = _weights(y, extra_y)
    fit_receipts = []
    plain, receipt = _fit_checked(matrix, target, weights, 'stylegram_plain')
    fit_receipts.append(receipt)
    centred, receipt = _fit_checked(centered_matrix, target, weights, 'stylegram_centred')
    fit_receipts.append(receipt)
    feature_shape = list(matrix.shape)
    del matrix, centered_matrix, texts
    test_matrix = features.transform(test_texts)
    plain_scores = plain.decision_function(test_matrix)
    groups = core.normalised_content_groups(test_texts)  # Original unlabelled inference grouping.
    mask = core.eligible(groups)
    scores = plain_scores.copy()
    if mask.any():
        scores[mask] = centred.decision_function(core.sparse_group_centre(test_matrix[mask], groups[mask]))
    require(scores.shape == (len(test_texts), 18) and scores.dtype == np.float64 and np.isfinite(scores).all(), 'Invalid A1 routed raw margins')
    require(np.array_equal(scores[~mask], plain_scores[~mask]), 'Plain fallback changed')
    shifted = scores - scores.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    require(np.isfinite(probability).all() and np.allclose(probability.sum(1), 1), 'Invalid A1 probabilities')
    receipt = {'recipe': 'A1 official2699 + frozen extra10073; raw margins without normalization or blend',
               'official_rows': len(training_texts), 'public_rows': len(extra_texts), 'test_rows': len(test_texts),
               'feature_shape': feature_shape, 'feature_dtype': 'float64', 'share': .5,
               'evaluation_rows_in_classifier_tfidf_fit': 0, 'unlabelled_test_content_grouping': True,
               'centred_rows': int(mask.sum()), 'labels': list(LABELS), 'fits': fit_receipts,
               'weights_sha256': array_sha(weights), 'target_sha256': array_sha(target),
               'raw_margins_sha256': array_sha(scores), 'probability_sha256': array_sha(probability),
               'official_class_counts': np.bincount(y, minlength=18).tolist(),
               'public_class_counts': np.bincount(extra_y, minlength=18).tolist(),
               'public_per_class_row_weight': (.5*np.bincount(y, minlength=18)/np.bincount(extra_y, minlength=18)).tolist(),
               'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    return probability, {'groups': groups, 'centred_mask': mask, 'raw_margins': scores,
                         'plain_margins': plain_scores, 'receipt': receipt}


def predict(TR, y, PROMPT, TE, extra_path, *, extra_sha256=EXTRA_SHA256, core_path=None, labels=LABELS):
    """Return (P_style, metadata) in the fixed 18-label order.

    metadata.groups/centred_mask reproduce G_style/rows_style. Raw unnormalized
    margins remain available for proof; metadata.receipt is JSON compatible.
    The caller must authenticate official train.csv bytes against
    OFFICIAL_TRAIN_SHA256 before extracting TR/y/PROMPT. There is no bypass of the
    fixed extra CSV hash, cardinality, class order, or core source hash.
    """
    require(tuple(labels) == LABELS, 'Fixed 18-label order mismatch')
    require(extra_sha256 == EXTRA_SHA256, 'Caller changed fixed extra SHA256')
    TR, TE = list(TR), list(TE)
    y, PROMPT = np.asarray(y), np.asarray(PROMPT)
    require(len(TR) == 2699 and y.shape == PROMPT.shape == (2699,), 'Official training cardinality mismatch')
    require(y.dtype.kind in 'iu' and set(y.tolist()) == set(range(18)), 'Official class indices mismatch')
    require(all(isinstance(text, str) and bool(text) for text in TR), 'Invalid official training text')
    require(len(np.unique(PROMPT)) == 150, 'Official prompt group population changed')
    require(all(isinstance(value, str) and bool(value) for value in PROMPT), 'Invalid official prompt IDs')
    require(len(TE) > 0 and all(isinstance(text, str) for text in TE), 'Test texts must be a nonempty string sequence')
    path = Path(extra_path)
    # Parse precisely the bytes whose hash was checked, avoiding a hash/read race.
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == EXTRA_SHA256, 'Frozen extra CSV hash mismatch')
    import io
    extra = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    require({'row_id', 'prompt_id', 'model', 'text', 'canonical_prompt'}.issubset(extra.columns), 'Missing extra columns')
    require(len(extra) == 10073 and extra.row_id.is_unique and extra.prompt_id.nunique() == 694, 'Frozen extra cardinality mismatch')
    require(set(extra.model) == set(LABELS) and extra.text.ne('').all(), 'Frozen extra labels/text mismatch')
    require(extra.groupby('prompt_id').canonical_prompt.nunique().eq(1).all()
            and not extra.canonical_prompt.str.startswith('official:').any(), 'Frozen extra prompt grouping mismatch')
    extra_y = extra.model.map({label: index for index, label in enumerate(LABELS)}).to_numpy(dtype=int)
    core_path = Path(core_path) if core_path is not None else Path(__file__).resolve().with_name('stylegram_core.py')
    core = load_core(core_path)
    with threadpool_limits(limits=2):
        probability, metadata = _fit_predict(TR, y, PROMPT, TE, extra.text.tolist(), extra_y, extra.prompt_id.to_numpy(), core)
    metadata['receipt'].update(extra_sha256=EXTRA_SHA256, core_sha256=CORE_SHA256,
                               official_train_sha256_expected=OFFICIAL_TRAIN_SHA256,
                               official_file_hash_verified_by='calling wrapper', family_gate_global_extra_changed=False)
    return probability, metadata


def self_test(core_path=None):
    """Small real StyleGrams parity gate; never read/train official or extra CSV.

    The independent serial reference follows EXP027's frozen training/inference
    path. Local tests additionally execute that original source's AST slice.
    """
    path = Path(core_path) if core_path is not None else Path(__file__).resolve().with_name('stylegram_core.py')
    core = load_core(path)
    training = [f'Official topic {p}; style{i} gives helpful details and careful reasons, example {i}.'
                for p in range(3) for i in range(18)]
    extra = [f'Public topic {p}; style{i} supplies another answer with explicit details and example {i}.'
             for p in range(2) for i in range(18)]
    y, ey = np.tile(np.arange(18), 3), np.tile(np.arange(18), 2)
    prompts, eg = np.repeat(['o0', 'o1', 'o2'], 18), np.repeat(['e0', 'e1'], 18)
    testing = ['HIDDENSENTINEL garden soil watering flowers', 'HIDDENSENTINEL garden soil watering flowers',
               'HIDDENSENTINEL nebula star orbital telescope']
    with threadpool_limits(limits=2):
        # Kept separate from _weights/_classifier/_fit_predict so a future local
        # change cannot make both sides silently share a changed recipe.
        features = core.StyleGrams().fit(training + extra)
        matrix = features.transform(training + extra)
        centred_matrix = vstack([core.sparse_group_centre(matrix[:len(training)], prompts),
                                 core.sparse_group_centre(matrix[len(training):], eg, 2, 10**6)], format='csr')
        tr_count, ex_count = np.bincount(y, minlength=18).astype(float), np.bincount(ey, minlength=18).astype(float)
        weights = np.r_[np.full(len(y), .5), .5 * tr_count[ey] / ex_count[ey]]
        target = np.r_[y, ey]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            plain = LinearSVC(C=.5, max_iter=4000, random_state=42, dual=True).fit(matrix, target, sample_weight=weights)
            centered = LinearSVC(C=.5, max_iter=4000, random_state=42, dual=True).fit(centred_matrix, target, sample_weight=weights)
        require(not any(issubclass(w.category, ConvergenceWarning) for w in captured), 'Synthetic reference convergence warning')
        test_matrix = features.transform(testing)
        reference_plain = plain.decision_function(test_matrix)
        groups = core.normalised_content_groups(testing)
        mask = core.eligible(groups)
        reference_scores = reference_plain.copy()
        if mask.any():
            reference_scores[mask] = centered.decision_function(core.sparse_group_centre(test_matrix[mask], groups[mask]))
        reference_probability = np.exp(reference_scores-reference_scores.max(1, keepdims=True))
        reference_probability /= reference_probability.sum(1, keepdims=True)
        probability, metadata = _fit_predict(training, y, prompts, testing, extra, ey, eg, core)
    margin_difference = float(np.max(np.abs(metadata['raw_margins']-reference_scores)))
    probability_difference = float(np.max(np.abs(probability-reference_probability)))
    require(margin_difference <= 1e-12 and probability_difference <= 1e-12, 'A1 synthetic numerical parity failed')
    require(np.array_equal(metadata['groups'], groups) and np.array_equal(metadata['centred_mask'], mask), 'A1 synthetic routing changed')
    require(np.array_equal(probability.argmax(1), reference_probability.argmax(1)), 'A1 synthetic prediction changed')
    require(mask.any() and (~mask).any(), 'Synthetic check did not exercise both inference routes')
    return {'status': 'PASS', 'scope': 'small synthetic real-StyleGrams parity only; no official/public-data fit',
            'training_rows': 54, 'public_rows': 36, 'test_rows': 3, 'classes': 18, 'svm_fits': 4,
            'maximum_raw_margin_difference': margin_difference, 'maximum_probability_difference': probability_difference,
            'argmax_equal': True, 'both_inference_routes_exercised': True, 'core_sha256': CORE_SHA256,
            'versions': metadata['receipt']['versions']}
