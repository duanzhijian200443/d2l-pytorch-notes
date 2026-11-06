"""Focused tests for audit false-clean failure modes; no training or network."""
import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from open_pool_audit import (
    add_hashes, conflict_mask, main_key, normalize, panel_checks,
    propagate_group_hits, reconstruct_comp, reconstruct_main, text_hash, verified_path,
    output_path, run,
)


class OpenPoolAuditTests(unittest.TestCase):
    def test_exact_and_normalized_keys_are_distinct_scopes(self):
        a, b = 'Ａ  Test\nTEXT', 'a test text'
        self.assertNotEqual(text_hash(a, False), text_hash(b, False))
        self.assertEqual(text_hash(a), text_hash(b))
        self.assertNotEqual(main_key(a), main_key(b))
        self.assertEqual(normalize(a), b)

    def test_duplicate_same_label_is_not_conflict(self):
        d = add_hashes(pd.DataFrame({'text': ['Same', 'same', 'other'], 'model': ['a', 'a', 'b']}))
        self.assertEqual(conflict_mask(d).tolist(), [False, False, False])
        d.loc[1, 'model'] = 'b'
        self.assertEqual(conflict_mask(d).tolist(), [True, True, False])

    def test_empty_canons_do_not_create_a_fake_shared_prompt(self):
        d = pd.DataFrame({'id': ['1', '2', '3', '4'], 'model': ['a', 'b', 'a', 'b'],
                          'prompt_canon': ['', '', 'p1', 'p1'],
                          'source_dataset': ['s'] * 4, 'panel_labels': [0, 0, 2, 2]})
        groups, bad = panel_checks(d)
        self.assertEqual(list(groups.index), ['p1'])
        self.assertFalse(bad.any())
        self.assertEqual(propagate_group_hits(d, pd.Series([True, False, False, False])).tolist(), [False] * 4)
        self.assertEqual(propagate_group_hits(d, pd.Series([False, False, True, False])).tolist(), [False, False, True, True])

    def test_panel_mismatch_is_reported_on_every_group_row(self):
        d = pd.DataFrame({'id': ['1', '2'], 'model': ['a', 'a'], 'prompt_canon': ['p1', 'p1'],
                          'source_dataset': ['x', 'y'], 'panel_labels': [2, 2]})
        _, bad = panel_checks(d)
        self.assertEqual(bad.tolist(), [True, True])

    def test_main_first_seen_is_text_only_and_case_sensitive(self):
        official = pd.DataFrame({'text': ['A text'], 'model': ['a']})
        matched = pd.DataFrame({'text': ['A  text', 'new text', ''], 'model': ['b', 'b', 'a']})
        pool = pd.DataFrame({'text': ['new text', 'a text', 'unknown'], 'model': ['a', 'a', 'x']})
        actual = reconstruct_main(official, matched, pool, ['a', 'b'])
        self.assertEqual(actual.text.tolist(), ['A text', 'new text', 'a text'])
        self.assertEqual(actual.model.tolist(), ['a', 'b', 'a'])

    def test_comp_requires_complete_enough_panel_after_finish_and_dedup(self):
        labels = ['m' + str(i) for i in range(12)]
        rows = [{'prompt_id': 'p', 'label': lab, 'text': (lab + ' ') * 35, 'finish_reason': 'stop'} for lab in labels]
        rows.append(dict(rows[0], text=('replacement ' * 35)))
        spec = {'name': 'x', 'label_column': 'label', 'finish_column': 'finish_reason'}
        result = reconstruct_comp([(spec, pd.DataFrame(rows))], labels)
        self.assertEqual(len(result), 12)
        self.assertEqual(result.iloc[0].text, rows[0]['text'].strip())
        rows[11]['finish_reason'] = 'length'
        self.assertTrue(reconstruct_comp([(spec, pd.DataFrame(rows))], labels).empty)

    def test_prvs_prefix_preserves_separate_bank(self):
        labels = ['m' + str(i) for i in range(12)]
        d = pd.DataFrame([{'prompt_id': 'p', 'model': lab, 'text': (lab + ' ') * 35} for lab in labels])
        spec = {'name': 'prvs', 'label_column': 'model', 'prompt_prefix': 'pv_'}
        self.assertEqual(reconstruct_comp([(spec, d)], labels).prompt_id.unique().tolist(), ['pv_p'])

    def test_expected_hash_is_required_even_when_size_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'input.csv'
            p.write_bytes(b'abc')
            with self.assertRaises(ValueError):
                verified_path(Path(tmp), {'path': 'input.csv', 'bytes': 3, 'sha256': '0' * 64})

    def test_output_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'experiment/config.json'
            with self.assertRaises(ValueError):
                output_path(root, config, {'output_dir': 'unrelated'})

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            out = exp / 'metrics'
            out.mkdir()
            sentinel = out / 'keep.txt'
            sentinel.write_text('unchanged', encoding='utf8')
            config = exp / 'config.json'
            config.write_text(json.dumps({'output_dir': str(out)}), encoding='utf8')
            with self.assertRaises(FileExistsError):
                run(config)
            self.assertEqual(sentinel.read_text(encoding='utf8'), 'unchanged')


if __name__ == '__main__':
    unittest.main()
