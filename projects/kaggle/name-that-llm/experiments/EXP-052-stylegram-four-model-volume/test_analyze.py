"""Contract and independent recomputation tests using already accepted historical margins."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import analyze as analysis


class FixedOOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.y, cls.labels, cls.control, *_ = analysis.history()

    def test_reject_wrong_fold_order_class_order_or_nonfinite_scores(self):
        for field in ['ids', 'fold', 'scores']:
            with self.subTest(field=field):
                broken = self.control.copy()
                arr = broken[field].copy()
                if field == 'ids':
                    arr[:2] = arr[:2][::-1]
                elif field == 'fold':
                    arr[0] = (arr[0] + 1) % 5
                else:
                    arr[0, 0] = float('nan')
                broken[field] = arr
                with self.assertRaises(ValueError):
                    analysis.checked_predictions(broken, self.frame, self.y, 'control')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wrong.npz'
            np.savez(path, labels=np.array(self.labels[::-1]))
            with self.assertRaises(ValueError):
                analysis.V.load_npz(path, {}, self.labels)

    def test_paired_direction_and_independent_recomputation(self):
        # One formerly correct sample switches to an error in B. One formerly
        # wrong sample is corrected in C. No model is executed by this fixture.
        base_pred = self.control['scores'].argmax(1)
        arms = {'control': self.control}
        for arm in ('four120', 'four240'):
            part = dict(self.control, arm=np.array(arm))
            part['scores'] = part['scores'].copy()
            if arm == 'four120':
                ix = np.flatnonzero((base_pred == self.y) & part['centred_mask'])[0]
                target = (self.y[ix] + 1) % 18
            else:
                ix = np.flatnonzero((base_pred != self.y) & part['centred_mask'])[0]
                target = self.y[ix]
            part['scores'][ix, target] = part['scores'][ix].max() + 1
            arms[arm] = part
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metrics, contrasts = analysis.dense_analysis(out, self.frame, self.y, self.labels, arms)
            primary = next(x for x in contrasts if x['primary'])
            self.assertEqual(primary['comparison'], 'four240_minus_four120')
            self.assertEqual((primary['fixed'], primary['new_errors'], primary['net_correct']), (2, 0, 2))
            self.assertEqual(len(pd.read_csv(out / 'per-class.csv')), 54)
            self.assertEqual(len(pd.read_csv(out / 'fold-comparisons.csv')), 15)
            self.assertEqual(len(pd.read_csv(out / 'class-comparisons.csv')), 54)
            analysis.write(out / 'summary.json', dict(status='PASS_FIXED_DEVELOPMENT_OOF',
                           result_sha256='fixture', metrics=metrics, comparisons=contrasts))
            entries = [{'name':p.name,'bytes':p.stat().st_size,'sha256':analysis.sha(p)} for p in out.iterdir()]
            analysis.write(out / 'analysis-manifest.json', dict(analyzer_sha256=analysis.sha(Path(analysis.__file__)),
                           result_sha256='fixture', inventory=entries))
            check = analysis.verify_analysis(out)
            self.assertEqual(check['status'], 'PASS_INDEPENDENT_ROW_RECOMPUTATION')
            with (out / 'predictions.csv').open('ab') as stream:
                stream.write(b'\n')
            with self.assertRaises(ValueError):
                analysis.verify_analysis(out)

    def test_incomplete_inventory_and_duplicate_fold_cannot_silently_merge(self):
        with self.assertRaises(ValueError):
            analysis.file_index([{'path':'dense.npz'}, {'path':'dense.npz'}])
        with self.assertRaises(ValueError):
            analysis.file_index([{'path':'../wrong'}])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'sample').write_bytes(b'corrupt')
            digest = hashlib.sha256(b'correct').hexdigest()
            with self.assertRaises(ValueError):
                analysis.anchored(folder, {'sample': {'bytes':7, 'sha256':digest}}, 'sample')


if __name__ == '__main__':
    unittest.main()
