"""Synthetic gates, weights and grouping only; never run the full training job."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp042_runner', HERE / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def config():
    return {'run_id': 'synthetic-only', 'labels': runner.LABELS.copy(),
            'new_labels': runner.NEW_LABELS.copy(), 'validation_scheme': 'prompt-group-v1',
            'C': .5, 'dual': True, 'max_iter': 4000, 'seed': 42,
            'sparse_per_prompt': [2, 3, 6], 'sparse_repeats': 5,
            'identity': {'source': 'synthetic'}, 'budget': {'hard_seconds': 6600},
            'holdout_prompt_group_hashes': [runner.text_hash('excluded prompt')]}


def new_rows():
    rows = []
    for i in range(60):
        for label in runner.NEW_LABELS:
            text = f'  Original\nraw text {i} {label}  '
            rows.append({'row_id': f'{i}:{label}', 'origin': 'synthetic', 'prompt_id': f'g{i}',
                         'model': label, 'text': text, 'base_prompt': f'Prompt {i}',
                         'text_sha256': runner.text_hash(text), 'job_key': f'{i}:{label}',
                         'split': 'train', 'source_protocol': 'synthetic'})
    return pd.DataFrame(rows)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.new = new_rows()
        self.train = pd.DataFrame({'text': ['unrelated official']})
        self.extra = pd.DataFrame({'text': ['unrelated extra']})
        self.holdout = config()['holdout_prompt_group_hashes']

    def gate(self, new=None, holdout=None):
        return runner.validate_new_rows(self.new if new is None else new, self.train, self.extra,
                                        self.holdout if holdout is None else holdout)

    def test_weights_keep_old_rows_and_add_exact_class_mass(self):
        official = np.array([0, 0, 0, 1, 1, 1, 1])
        public = np.array([0, 0, 1, 1, 1, 1, 1])
        added = np.array([0, 1, 1])
        control, add, audit = runner.fixed_weights(official, public, added, classes=2)
        np.testing.assert_array_equal(control, add[:len(control)])
        np.testing.assert_allclose(control[:len(official)], .5)
        np.testing.assert_allclose(add[-3:], [.75, .4, .4])
        np.testing.assert_allclose([x['control_mass'] for x in audit], [3., 4.])
        np.testing.assert_allclose([x['added_mass'] for x in audit], [.75, .8])
        np.testing.assert_allclose(np.bincount(np.r_[official, public, added], weights=add), [3.75, 4.8])
        with self.assertRaisesRegex(ValueError, 'Missing training class'):
            runner.fixed_weights(official, np.array([0]), added, classes=2)

    def test_raw_text_preserved(self):
        before = self.new.copy(deep=True)
        self.assertEqual(self.gate().nunique(), 60)
        pd.testing.assert_frame_equal(before, self.new)
        changed = self.new.copy()
        changed.loc[0, 'text'] = changed.loc[0, 'text'].strip()
        with self.assertRaisesRegex(ValueError, 'Raw text hash'):
            self.gate(changed)

    def test_row_count_split_and_labels_rejected(self):
        for column, value in [('split', 'holdout'), ('model', 'gpt-6-astra'), ('job_key', self.new.loc[1, 'job_key'])]:
            changed = self.new.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.gate(changed)
        with self.assertRaises(ValueError):
            self.gate(self.new.iloc[:-1])

    def test_exact_overlap_rejected_in_both_sources(self):
        for source in (self.train, self.extra):
            original = source.loc[0, 'text']
            source.loc[0, 'text'] = self.new.loc[0, 'text']
            with self.assertRaisesRegex(ValueError, 'overlaps baseline'):
                self.gate()
            source.loc[0, 'text'] = original

    def test_normalized_holdout_and_group_mapping_rejected(self):
        digest = runner.text_hash(runner.normalize('  ＰＲＯＭＰＴ\t0  '))
        with self.assertRaisesRegex(ValueError, 'Holdout'):
            self.gate(holdout=[digest])
        changed = self.new.copy()
        changed.loc[0, 'base_prompt'] = 'different prompt'
        with self.assertRaises(ValueError):
            self.gate(changed)
        changed = self.new.copy()
        changed.loc[0, 'prompt_id'] = 'g1'
        with self.assertRaises(ValueError):
            self.gate(changed)

    def test_config_freezes_labels_hyperparameters_and_budget(self):
        runner.validate_config(config())
        for key, value in [('labels', list(reversed(runner.LABELS))), ('C', 1), ('dual', False),
                           ('seed', 13), ('sparse_per_prompt', [2]), ('sparse_repeats', 4),
                           ('validation_scheme', 'random'), ('budget', {'hard_seconds': float('nan')}),
                           ('holdout_prompt_group_hashes', [])]:
            c = config()
            c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_config(c)

    def test_hash_gate_rejects_tampering_and_baseline_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory)
            hashes = {}
            for name in [*runner.FROZEN_HASHES, 'new120.csv']:
                (payload / name).write_text(name, encoding='utf-8')
                hashes[name] = runner.sha(payload / name)
            c = {'input_hashes': hashes}
            with self.assertRaisesRegex(ValueError, 'Frozen baseline changed'):
                runner.verify_hashes(payload, c)
            with mock.patch.object(runner, 'FROZEN_HASHES', {k: hashes[k] for k in runner.FROZEN_HASHES}):
                runner.verify_hashes(payload, c)
                (payload / 'new120.csv').write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
                    runner.verify_hashes(payload, c)

    def test_training_centre_keeps_public_and_codex_groups_separate(self):
        core_path = HERE.parent / 'EXP-010-012-stylegram-share/payload/stylegram_core.py'
        core_spec = importlib.util.spec_from_file_location('test_core', core_path)
        core = importlib.util.module_from_spec(core_spec)
        core_spec.loader.exec_module(core)
        x = csr_matrix(np.array([[1., 1.], [3., 3.], [10., 10.], [20., 20.], [100., 100.], [140., 140.]]))
        with mock.patch.object(core, 'sparse_group_centre', wraps=core.sparse_group_centre) as centre:
            result = runner.centre_training(core, x, ['same', 'same'], ['same', 'same'], ['same', 'same'])
            self.assertEqual(centre.call_count, 3)
            self.assertEqual(centre.call_args_list[2].args[1].tolist(), ['codex:same', 'codex:same'])
        np.testing.assert_allclose(result.toarray(), [[-1, -1], [1, 1], [-5, -5], [5, 5], [-20, -20], [20, 20]])

    def test_sparse_selection_does_not_use_labels_or_text(self):
        valid = pd.DataFrame({'id': [str(i) for i in range(18)], 'prompt_id': ['p0'] * 9 + ['p1'] * 9,
                              'model': ['irrelevant'] * 18, 'text': ['text'] * 18})
        for m in (2, 3, 6):
            for repeat in range(5):
                chosen = runner.sparse_indices(valid, m, repeat)
                self.assertEqual(len(chosen), 2 * m)
                changed = valid.assign(model='changed', text='changed')
                np.testing.assert_array_equal(chosen, runner.sparse_indices(changed, m, repeat))

    def test_receipts_manifest_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cfg = out / 'config-source.json'
            cfg.write_text(json.dumps(config()), encoding='utf-8')
            evidence = runner.Evidence(out, config(), cfg)
            evidence.save('result.json', {'status': 'COMPLETE'})
            (out / 'guard.json').write_text('{}')
            evidence.matrix('tiny.npz', scores=np.zeros((2, 18)), ids=np.array(['a', 'b']))
            evidence.manifest()
            manifest = json.loads((out / 'artifacts.json').read_text())
            self.assertNotIn('guard.json', [item['path'] for item in manifest['files']])
            self.assertEqual(manifest['run_id'], 'synthetic-only')
            for item in manifest['files']:
                self.assertEqual(item['sha256'], runner.sha(out / item['path']))
            with np.load(out / 'tiny.npz', allow_pickle=False) as data:
                self.assertEqual(data['run_id'].item(), 'synthetic-only')
                self.assertEqual(data['labels'].tolist(), runner.LABELS)
            evidence.started -= 6601
            with self.assertRaises(TimeoutError):
                evidence.check_budget()


if __name__ == '__main__':
    unittest.main()
