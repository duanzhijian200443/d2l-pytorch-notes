"""Fixed four-arm complementary-stack arithmetic; no I/O, fitting or truth input.

The caller owns cache/source provenance and candidate row/column alignment.
This module stops at pre-Sinkhorn probabilities. Subsequent fixed .45 Sinkhorn
and typography 1.5 belong to the caller's already verified control function.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from src import full_stack_replay as original
from src.margin_blend import normalize_margins

LABELS = original.LABELS
ARMS = ('A0', 'A1', 'A2', 'A3')
MODES = ('strict', 'fixed_captured_mask')
STYLE_WEIGHT = 1.516
COMP_WEIGHT = .15
ATLAS_WEIGHT = .2
ATLAS_THRESHOLD = .3
ReplayError = original.ReplayError
need = original._require


def _comp_variant(events, n, probability):
    logit = np.zeros((n, 18))
    non_style = np.zeros_like(logit)
    order = []
    for _, member in events.by_name['member']:
        key, rows, weight = member['key'], member['rows'], member['weight']
        order.append(key)
        p = probability if key == 'stylegram' else member['probability']
        contribution = weight * np.log(np.clip(p, 1e-9, 1.))
        if rows is None:
            logit[:] += contribution
            if key != 'stylegram':
                non_style[:] += contribution
        else:
            logit[rows] += contribution
            if key != 'stylegram':
                non_style[rows] += contribution
    before = logit.copy()
    gates = events.one('comp_before_gates')['gates']
    need(set(gates) == set(original.GATE_ORDER) and all(gates[key] is not None for key in original.GATE_ORDER),
         'The fixed five-family-gate inventory is required')
    audit = []
    for key in original.GATE_ORDER:
        family = 'claude' if key == 'claude_lora' else key
        indices = np.array([i for i, label in enumerate(LABELS) if label.startswith(family)])
        score = np.asarray(gates[key], dtype=np.float64)
        if key != 'claude_lora':
            score = score - score.mean(1, keepdims=True)
            scale = np.sqrt(np.mean(score * score, axis=1, keepdims=True))
            score = score / np.maximum(scale, .25)
        gate_lp = original._logsoftmax64(score)
        base = logit[:, indices].copy()
        base_lp = original._logsoftmax64(base)
        ordered = np.sort(np.exp(base_lp), axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        confidence = np.clip((.45-margin)/.45, 0, 1)
        lam = 1.5 if key == 'claude' else 1. if key == 'claude_lora' else .25
        strength = min(.45, lam/(1+lam)) * confidence
        mixed_lp = original._logsoftmax64((1-strength[:, None])*base_lp + strength[:, None]*gate_lp)
        mass = logsumexp(base, axis=1)
        logit[:, indices] = mass[:, None] + mixed_lp
        after_mass = logsumexp(logit[:, indices], axis=1)
        need(np.allclose(mass, after_mass, rtol=0, atol=1e-10), 'Family gate changed family log-mass')
        audit.append({'key': key, 'indices': indices, 'strength': strength,
                      'family_log_mass_before': mass, 'family_log_mass_after': after_mass})
    return {'comp_before_gates': before, 'comp_logit': logit.astype(np.float32),
            'comp_logit_float64': logit, 'non_style_before_gates': non_style,
            'comp_member_order': order, 'family_gates': audit}


def _main_base(main, n, checks, expected_members):
    event = main.one('main_members')
    args = event['args']
    fixed = {'perpassage': 1, 'post': 0, 'svc_w': 1.5, 'tfidf_w': .25, 'distort_w': .10,
             'comp_w': .15, 'atlas_margin': .3, 'sk_div': 1, 'adapt_w': 1., 'adapt_thr': .15,
             'sinkhorn': .45, 'sinkhorn_target_mix': .35}
    for key, expected in fixed.items():
        need(args.get(key) == expected, f'Frozen main argument changed: {key}')
    need(not args.get('sim'), 'Simulation main cache is not supported')
    members, total = original._main_inventory(event, args, n, checks, expected_members)
    need('svc4' in members, 'Original main svc4 must remain present in all arms')
    divisor = sum(abs(value).max() > 0 for value in members.values())
    captured = main.one('main_probability')
    need(captured['divisor'] == divisor, 'Original main divisor changed')
    normalized = total / max(1, divisor)
    probability = original._softmax_original(normalized)
    checks.compare('A0.main.normalized_logits', captured['normalized_logits'], normalized, scores=True)
    checks.compare('A0.main.probability', captured['probability'], probability, scores=True)
    return probability, list(members), divisor, args


def _mix_comp(main_probability, comp_serialized):
    # Reproduce np.load(... )[:, permutation], including the F-layout float32
    # row-reduction boundary even though the permutation is the identity here.
    columns = comp_serialized[:, list(range(18))]
    comp_probability = original._softmax_original(columns)
    mixed = (1-COMP_WEIGHT)*main_probability + COMP_WEIGHT*comp_probability
    mixed /= mixed.sum(1, keepdims=True)
    return mixed, comp_probability


def _requested(probability):
    ordered = np.sort(probability, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    rows = np.where(margin < ATLAS_THRESHOLD)[0] if len(probability) >= 50 else np.arange(len(probability))
    return rows, margin


def _atlas_cache(main, n, ids, args, original_probability, checks):
    request = main.one('atlas_requested')
    rows, margin = _requested(original_probability)
    checks.compare('A0.atlas.requested_rows', request['rows'], rows, exact=True)
    checks.compare('A0.atlas.margin', request['margin'], margin)
    checks.compare('A0.atlas.probability', request['probability'], original_probability, scores=True)
    bundles = [value for value in args['atlas_dir'].split(',') if value]
    weights = [float(value) for value in str(args['atlas_w']).split(',')]
    need(len(bundles) == 1 and weights == [ATLAS_WEIGHT], 'Exactly one original ATLAS bundle at .2 is required')
    need(request['bundles'] == bundles and request['weights'] == weights, 'ATLAS request bundle/weight mismatch')
    returned = main.by_name.get('atlas_returned', [])
    need(len(returned) == 1, 'One ATLAS return is required; skipped ATLAS is not a baseline')
    result = returned[0][1]
    need(result['bundle'] == bundles[0] and result['weight'] == ATLAS_WEIGHT, 'ATLAS returned bundle/weight mismatch')
    requested = original._row_indices(result['requested_rows'], n, 'ATLAS returned rows')
    need(np.array_equal(requested, rows), 'ATLAS returned request ordering changed')
    mask = original._array(result['completed_mask'], (len(rows),), 'ATLAS completed mask', 'b')
    completed = original._row_indices(result['completed_rows'], n, 'ATLAS completed rows')
    need(np.array_equal(completed, rows[mask]), 'ATLAS completed rows/mask mismatch')
    probabilities = original._array(result['probabilities'], (len(rows), 18), 'ATLAS cached probabilities')
    need(probabilities.dtype == np.float32 and np.all(probabilities[~mask] == 0), 'ATLAS float32/unused placeholders changed')
    original._prob(probabilities[mask], int(mask.sum()), 'ATLAS completed scores')
    # Return ONLY real completed scores. Uncompleted zero placeholders never
    # enter an arm's probability lookup or its arithmetic.
    return {'bundle': bundles[0], 'original_requested_rows': rows.copy(), 'completed_rows': completed.copy(),
            'completed_probabilities': probabilities[mask].copy(),
            'original_missing_ids': [ids[i] for i in rows[~mask]]}


def _apply_atlas(pre_atlas, rows, cache):
    position = {int(row): i for i, row in enumerate(cache['completed_rows'])}
    need(all(int(row) in position for row in rows), 'ATLAS coverage insufficient; no scores may be imputed')
    local = np.array([position[int(row)] for row in rows], dtype=int)
    mixed = pre_atlas.copy()
    if len(rows):
        mixed[rows] += ATLAS_WEIGHT * (cache['completed_probabilities'][local] - pre_atlas[rows])
    mixed /= mixed.sum(1, keepdims=True)
    return mixed


def replay_four_arms(main_events, comp_events, style_margins, svc_margins, *, mode='strict',
                     expected_main_members=None, expected_comp_members=None):
    """Replay A0/A1/A2/A3, retaining the original main svc4 and every other member.

    Events must be decoded (numpy arrays), share IDs/text hashes/18-label order,
    and already be source/hash authenticated by the caller. Candidate margins
    must be raw float64 N-by-18 arrays in precisely that same order. No evaluation
    labels, texts, feature fitting, or tunable coefficients are accepted.

    Strict arms with missing ATLAS requests return ``pre_sinkhorn=None`` and
    ``accepted=False``. The explicitly selected fixed_captured_mask mode retains
    dynamic-route missing IDs but applies only the original request mask; it is a
    secondary mechanism diagnostic and never claims dynamic-policy acceptance.
    """
    need(mode in MODES, 'Unknown mode; only strict or explicit fixed_captured_mask is allowed')
    main, comp = original._Events(main_events, 'main'), original._Events(comp_events, 'comp')
    identity, comp_identity = original._identity(main), original._identity(comp)
    need(identity == comp_identity, 'Main/comp row, text hash, class order or scope mismatch')
    n = len(identity['ids'])
    for value, name in ((style_margins, 'style_margins'), (svc_margins, 'svc_margins')):
        original._array(value, (n, 18), name)
        need(value.dtype == np.float64, name + ' must retain original float64 margins')
    checks = original._Checks()
    serialized, members, gates = original._replay_comp(comp, n, expected_comp_members, checks)
    style = [payload for _, payload in comp.by_name['member'] if payload['key'] == 'stylegram']
    need(len(style) == 1 and style[0]['rows'] is None and style[0]['weight'] == STYLE_WEIGHT, 'Stylegram must contribute once over all rows at 1.516')
    main_probability, main_members, divisor, args = _main_base(main, n, checks, expected_main_members)
    baseline = _comp_variant(comp, n, style[0]['probability'])
    checks.compare('A0.comp_serialized', serialized, baseline['comp_logit'], scores=True, exact=True)
    a0_pre, a0_comp_probability = _mix_comp(main_probability, baseline['comp_logit'])
    loaded, status = main.one('comp_loaded'), main.one('comp_status')
    need(loaded['comp_meta']['ids'] == identity['ids'] and loaded['comp_meta']['labels'] == list(LABELS), 'Loaded comp identity mismatch')
    need(loaded['path'] == status['path'] == args['comp_path'] and status['requested_weight'] == COMP_WEIGHT, 'Loaded comp source/weight mismatch')
    checks.compare('A0.comp_loaded', loaded['logit'], serialized, scores=True, exact=True)
    checks.compare('A0.comp_probability', loaded['probability'], a0_comp_probability, scores=True)
    checks.compare('A0.comp_status_probability', status['probability'], a0_comp_probability, scores=True)
    checks.compare('A0.pre_atlas', main.one('pre_atlas')['probability'], a0_pre, scores=True)
    cache = _atlas_cache(main, n, identity['ids'], args, a0_pre, checks)
    captured_applied = _apply_atlas(a0_pre, cache['completed_rows'], cache)
    checks.compare('A0.pre_sinkhorn', main.one('pre_sinkhorn')['probability'], captured_applied, scores=True)
    if mode == 'fixed_captured_mask':
        need(not cache['original_missing_ids'], 'Fixed captured-mask diagnostic requires a fully completed original request mask')
    z_style, z_svc = normalize_margins(style_margins), normalize_margins(svc_margins)
    probabilities = {'A0': style[0]['probability'],
                     'A1': original._softmax_original(style_margins),
                     'A2': original._softmax_original(z_style),
                     'A3': original._softmax_original(.9*z_style + .1*z_svc)}
    arms = {}
    for arm in ARMS:
        result = baseline if arm == 'A0' else _comp_variant(comp, n, probabilities[arm])
        need(np.array_equal(result['non_style_before_gates'], baseline['non_style_before_gates']), 'A non-Stylegram comp member changed')
        pre_atlas, comp_probability = _mix_comp(main_probability, result['comp_logit'])
        requested, margin = _requested(pre_atlas)
        missing = np.setdiff1d(requested, cache['completed_rows'])
        complete = len(missing) == 0
        applied = requested if mode == 'strict' else cache['original_requested_rows']
        pre_sinkhorn = _apply_atlas(pre_atlas, applied, cache) if (complete or mode == 'fixed_captured_mask') else None
        status_name = ('COMPLETE_DYNAMIC_POLICY' if complete else 'BLOCKED_ATLAS_COVERAGE') if mode == 'strict' else 'DIAGNOSTIC_FIXED_CAPTURED_MASK'
        result.update({'style_probability': probabilities[arm].copy(), 'comp_probability': comp_probability,
            'pre_atlas': pre_atlas, 'atlas_margin': margin, 'requested_rows': requested.copy(),
            'requested_ids': [identity['ids'][i] for i in requested], 'missing_row_indices': missing,
            'missing_ids': [identity['ids'][i] for i in missing], 'dynamic_coverage_complete': complete,
            'applied_atlas_rows': applied.copy() if pre_sinkhorn is not None else None,
            'pre_sinkhorn': pre_sinkhorn, 'status': status_name, 'accepted': mode == 'strict' and complete,
            'coverage': {'dynamic_requested_count': len(requested), 'captured_completed_count': len(cache['completed_rows']),
                         'missing_count': len(missing), 'mode': mode}})
        arms[arm] = result
    return {'mode': mode, 'rows': n, 'ids': identity['ids'].copy(), 'labels': list(LABELS), 'arms': arms,
            'status': 'DIAGNOSTIC_ONLY' if mode != 'strict' else 'COMPLETE_DYNAMIC_POLICY' if all(a['accepted'] for a in arms.values()) else 'BLOCKED_ATLAS_COVERAGE',
            'accepted': mode == 'strict' and all(a['accepted'] for a in arms.values()),
            'baseline_checks': checks.stages, 'main_members_preserved': main_members, 'main_divisor': divisor,
            'comp_members_preserved_except_stylegram': members, 'family_gate_order': gates,
            'captured_atlas_rows': cache['completed_rows'].copy(), 'atlas_bundle': cache['bundle'],
            'parameters': {'style_weight': STYLE_WEIGHT, 'comp_weight': COMP_WEIGHT, 'atlas_weight': ATLAS_WEIGHT,
                           'atlas_threshold': ATLAS_THRESHOLD, 'A3_svc_fraction': .1, 'sinkhorn_next': .45, 'typography_next': 1.5},
            'limits': ['No evaluation truth, accuracy, fitting, cache file I/O, or additional model inference.',
                       'Source/cache authenticity and candidate alignment are caller responsibilities.',
                       'Fixed captured-mask results are diagnostic only and do not implement the candidate dynamic ATLAS policy.']}
