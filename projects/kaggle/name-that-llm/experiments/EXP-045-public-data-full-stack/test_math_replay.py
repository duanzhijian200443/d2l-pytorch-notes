"""Synthetic score arithmetic only; no data files, model fitting, or truth."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp, softmax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location('exp045_math', HERE / 'math_replay.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic_events():
    n = 60
    rng = np.random.default_rng(42)
    labels = list(m.LABELS)
    ids = [f'synthetic-{i}' for i in range(n)]
    common = {'ids': ids, 'labels': labels, 'text_sha256': ['0' * 64] * n, 'scope': 'synthetic_math'}
    groups = np.arange(n)
    groups[1], groups[3] = groups[0], groups[2]
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    mask = (counts[inverse] >= 2) & (counts[inverse] <= 18)
    raw = np.zeros((n, 18), dtype=np.float64)
    raw[:, 16] = 50.
    style_prob = softmax(raw, axis=1)
    comp = [('data', dict(common, component='comp'))]
    before = np.zeros((n, 18))
    for key, probability, weight, rows in (
        ('fixed_other', np.full((n, 18), 1/18), .7, None),
        ('stylegram', style_prob, 1.516, None),
        ('q3_8b_twin', softmax(rng.normal(size=(int(mask.sum()), 18)), axis=1), .2, np.flatnonzero(mask)),
    ):
        lp = np.log(np.clip(probability, 1e-9, 1))
        comp.append(('member', {'key': key, 'rows': rows, 'probability': probability, 'weight': weight, 'log_probability': lp}))
        if rows is None:
            before += weight * lp
        else:
            before[rows] += weight * lp
    gates = {}
    for gate in m.original.GATE_ORDER:
        family = 'claude' if gate == 'claude_lora' else gate
        columns = [i for i, label in enumerate(labels) if label.startswith(family)]
        gates[gate] = rng.normal(size=(n, len(columns))).astype(np.float32)
    comp.append(('comp_before_gates', {'groups': groups.copy(), 'style_groups': groups.copy(),
                                     'group_mask': mask.copy(), 'style_mask': mask.copy(),
                                     'logit': before.copy(), 'gates': gates}))
    # Independent construction of captured family stages; the tests below also
    # assert mass preservation and recomputed strength, rather than just hashes.
    logit = before.copy()
    for key in m.original.GATE_ORDER:
        family = 'claude' if key == 'claude_lora' else key
        indices = np.array([i for i, label in enumerate(labels) if label.startswith(family)])
        score = gates[key].astype(np.float64)
        if key != 'claude_lora':
            score -= score.mean(axis=1, keepdims=True)
            score /= np.maximum(np.sqrt(np.mean(score**2, axis=1, keepdims=True)), .25)
        gate_lp = score - logsumexp(score, axis=1, keepdims=True)
        base = logit[:, indices].copy()
        base_lp = base - logsumexp(base, axis=1, keepdims=True)
        ranked = np.sort(np.exp(base_lp), axis=1)
        confidence = np.clip((.45-(ranked[:, -1]-ranked[:, -2]))/.45, 0, 1)
        coefficient = .45 if key in ('claude', 'claude_lora') else .2
        strength = coefficient * confidence
        gate_mix = (1-strength[:, None])*base_lp + strength[:, None]*gate_lp
        gate_mix -= logsumexp(gate_mix, axis=1, keepdims=True)
        mass = logsumexp(base, axis=1)
        logit[:, indices] = mass[:, None] + gate_mix
        comp.append(('family_gate_applied', {'key': key, 'family': family, 'indices': indices,
            'raw_score': gates[key], 'gate_log_probability': gate_lp, 'base': base,
            'strength': strength, 'family_mass': mass, 'logit': logit.copy()}))
    serial = logit.astype(np.float32)
    comp.append(('component_end', {'logit': logit.copy(), 'serialized_logit': serial.copy()}))
    normalized = np.zeros((n, 18))
    normalized[:30, 16] = 2.7
    members = {key: normalized.copy() for key in ('tfidf', 'svc4', 'distort')}
    total = sum(members.values())
    mainprob = np.exp(total/3 - (total/3).max(axis=1, keepdims=True))
    mainprob /= mainprob.sum(axis=1, keepdims=True)
    columns = serial[:, list(range(18))]
    compprob = np.exp(columns - columns.max(axis=1, keepdims=True))
    compprob /= compprob.sum(axis=1, keepdims=True)
    pre = .85*mainprob + .15*compprob
    pre /= pre.sum(axis=1, keepdims=True)
    ranked = np.sort(pre, axis=1)
    margin = ranked[:, -1]-ranked[:, -2]
    rows = np.flatnonzero(margin < .3)
    atlas = np.full((len(rows), 18), 1/18, dtype=np.float32)
    post = pre.copy()
    post[rows] += .2*(atlas-pre[rows])
    post /= post.sum(axis=1, keepdims=True)
    args = {'perpassage': 1, 'post': 0, 'sim': '', 'svc_w': 1.5, 'tfidf_w': .25, 'distort_w': .10,
            'comp_w': .15, 'comp_path': 'captured/comp_logit.npy', 'atlas_margin': .3, 'sk_div': 1,
            'adapt_w': 1., 'adapt_thr': .15, 'sinkhorn': .45, 'sinkhorn_target_mix': .35,
            'atlas_dir': 'captured/atlas', 'atlas_w': '0.2'}
    main = [('data', dict(common, component='main')),
            ('main_members', {'args': args, 'members': members, 'specs': [], 'groups': groups.copy(), 'total': total}),
            ('comp_loaded', {'comp_meta': {'ids': ids, 'labels': labels}, 'path': args['comp_path'], 'logit': serial.copy(), 'probability': compprob.copy()}),
            ('comp_status', {'path': args['comp_path'], 'requested_weight': .15, 'probability': compprob.copy()}),
            ('main_probability', {'divisor': 3, 'normalized_logits': total/3, 'probability': mainprob.copy()}),
            ('pre_atlas', {'probability': pre.copy()}),
            ('atlas_requested', {'rows': rows.copy(), 'margin': margin, 'probability': pre.copy(), 'bundles': [args['atlas_dir']], 'weights': [.2]}),
            ('atlas_returned', {'bundle': args['atlas_dir'], 'weight': .2, 'requested_rows': rows.copy(),
                                'completed_mask': np.ones(len(rows), dtype=bool), 'completed_rows': rows.copy(), 'probabilities': atlas}),
            ('pre_sinkhorn', {'probability': post.copy()}), ('component_end', {})]
    return main, comp, raw


class FourArmTests(unittest.TestCase):
    def setUp(self):
        self.main, self.comp, self.raw = synthetic_events()
        self.options = {'expected_main_members': ['tfidf', 'svc4', 'distort'],
                        'expected_comp_members': ['fixed_other', 'stylegram', 'q3_8b_twin']}

    def replay(self, style=None, svc=None, **kwargs):
        return m.replay_four_arms(self.main, self.comp, self.raw if style is None else style,
                                 self.raw if svc is None else svc, **self.options, **kwargs)

    def test_a0_exact_original_boundaries_and_a1_identity_replacement(self):
        result = self.replay()
        a0, a1 = result['arms']['A0'], result['arms']['A1']
        np.testing.assert_array_equal(a0['comp_logit'], self.comp[-1][1]['serialized_logit'])
        np.testing.assert_array_equal(a0['pre_atlas'], dict(self.main)['pre_atlas']['probability'])
        np.testing.assert_array_equal(a0['pre_sinkhorn'], dict(self.main)['pre_sinkhorn']['probability'])
        for key in ('comp_logit', 'pre_atlas', 'pre_sinkhorn'):
            np.testing.assert_array_equal(a0[key], a1[key])
        self.assertIn('svc4', result['main_members_preserved'])
        self.assertEqual(result['main_divisor'], 3)
        self.assertEqual(a0['comp_member_order'], ['fixed_other', 'stylegram', 'q3_8b_twin'])

    def test_replacement_is_local_and_does_not_mutate_inputs(self):
        before = copy.deepcopy((self.main, self.comp, self.raw))
        candidate = np.roll(self.raw, 1, axis=1)
        result = self.replay(candidate)
        base = result['arms']['A0']
        for arm in ('A1', 'A2', 'A3'):
            value = result['arms'][arm]
            np.testing.assert_array_equal(value['non_style_before_gates'], base['non_style_before_gates'])
            expected = 1.516*(np.log(np.clip(value['style_probability'], 1e-9, 1)) - np.log(np.clip(base['style_probability'], 1e-9, 1)))
            np.testing.assert_allclose(value['comp_before_gates']-base['comp_before_gates'], expected, atol=1e-12, rtol=0)
        for key, value in dict(self.main)['main_members']['members'].items():
            np.testing.assert_array_equal(value, dict(before[0])['main_members']['members'][key])
        for (_, new), (_, old) in zip(self.comp, before[1]):
            for key, value in new.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, old[key])
        np.testing.assert_array_equal(self.raw, before[2])

    def test_gate_family_mass_preserved_and_strength_recomputed(self):
        candidate = np.random.default_rng(7).normal(size=self.raw.shape)
        result = self.replay(candidate)
        for arm in result['arms'].values():
            self.assertEqual([g['key'] for g in arm['family_gates']], list(m.original.GATE_ORDER))
            for gate in arm['family_gates']:
                np.testing.assert_allclose(gate['family_log_mass_before'], gate['family_log_mass_after'], atol=1e-12, rtol=0)
        self.assertTrue(any(not np.array_equal(a['strength'], b['strength'])
                            for a, b in zip(result['arms']['A0']['family_gates'], result['arms']['A1']['family_gates'])))

    def test_missing_candidate_atlas_coverage_blocks_strict_without_imputation(self):
        candidate = np.roll(self.raw, 1, axis=1)
        result = self.replay(candidate)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'BLOCKED_ATLAS_COVERAGE')
        self.assertTrue(result['arms']['A0']['accepted'])
        for arm in ('A1', 'A2', 'A3'):
            value = result['arms'][arm]
            self.assertIsNone(value['pre_sinkhorn'])
            self.assertIsNone(value['applied_atlas_rows'])
            self.assertGreater(len(value['missing_ids']), 0)
            self.assertEqual(len(value['missing_ids']), value['coverage']['missing_count'])

    def test_fixed_original_mask_is_explicit_diagnostic_not_dynamic_acceptance(self):
        result = self.replay(np.roll(self.raw, 1, axis=1), mode='fixed_captured_mask')
        self.assertEqual(result['status'], 'DIAGNOSTIC_ONLY')
        self.assertFalse(result['accepted'])
        original_rows = dict(self.main)['atlas_requested']['rows']
        for value in result['arms'].values():
            self.assertEqual(value['status'], 'DIAGNOSTIC_FIXED_CAPTURED_MASK')
            self.assertFalse(value['accepted'])
            np.testing.assert_array_equal(value['applied_atlas_rows'], original_rows)
            self.assertIsNotNone(value['pre_sinkhorn'])
        self.assertGreater(len(result['arms']['A1']['missing_ids']), 0)

    def test_invalid_inputs_and_parameter_drift_fail_closed(self):
        with self.assertRaises(ValueError):
            self.replay(self.raw.astype(np.float32))
        bad = self.raw.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.replay(bad)
        with self.assertRaises(ValueError):
            self.replay(mode='fill_missing')
        dict(self.main)['main_members']['args']['svc_w'] = 0
        with self.assertRaisesRegex(ValueError, 'svc_w'):
            self.replay()

    def test_a0_capture_mismatch_is_not_a_valid_ablation(self):
        dict(self.main)['pre_atlas']['probability'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'pre_atlas'):
            self.replay()


if __name__ == '__main__':
    unittest.main()
