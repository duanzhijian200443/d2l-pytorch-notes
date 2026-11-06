"""Offline checks for fixed-source postprocessing and strict cache coverage."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

import replay


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = replay.source_modules()
        cls.cache = replay.capture()

    def test_author_gate_reproduces_cached_bytes(self):
        replay.check_author_baseline(self.modules, self.cache)
        self.assertEqual(len(self.cache.atlas['completed_rows']), 206)
        self.assertEqual(self.cache.divisor, 16)

    def test_new_atlas_row_blocks_entire_arm_and_no_probabilities_imputed(self):
        n = 50
        prob = np.full((n, 18), .01)
        prob[:, 0] = .83
        prob[:2] = 1 / 18
        cache = SimpleNamespace(math=self.cache.math, panel=pd.DataFrame({'id':[str(i) for i in range(n)]}),
            atlas={'completed_rows':np.array([0]), 'completed_probabilities':np.full((1,18), 1/18,dtype=np.float32)})
        info, _, scores = replay.strict_route(cache, prob)
        self.assertEqual(info['requested_ids'], ['0','1'])
        self.assertEqual(info['missing_ids'], ['1'])
        self.assertIsNone(scores)
        cache.atlas = {'completed_rows':np.array([0,1]), 'completed_probabilities':np.full((2,18), 1/18,dtype=np.float32)}
        info, _, scores = replay.strict_route(cache, prob)
        self.assertTrue(info['all_requested_covered'])
        self.assertIsNotNone(scores)
        self.assertTrue(np.allclose(scores.sum(1), 1))

    def test_source_post_can_run_on_reference_input_and_preserves_family_mass(self):
        # This exercises the author module on the accepted older capture. It is not
        # an evaluation of any of the new training arms.
        p = self.cache.main.one('pre_sinkhorn')['probability']
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=2):
            diag = Path(tmp)/'source-v40a.json'
            sk, reference, final, status = replay.author_post(self.modules,p,self.cache.train,self.cache.panel,diag)
            self.assertEqual(status, 'EXECUTED')
            self.assertTrue(np.isfinite(final).all())
            self.assertEqual(final.shape, (729,18))
            self.assertEqual(set(replay.read(diag)), {'claude','gemini','gpt','glm'})
            for fam in ('claude', 'gemini', 'gpt', 'glm'):
                cols = [i for i,s in enumerate(replay.LABELS) if s.startswith(fam)]
                np.testing.assert_allclose(logsumexp(final[:,cols],axis=1),logsumexp(reference[:,cols],axis=1),atol=1e-8)
            self.assertFalse(np.array_equal(sk,reference))

    def test_require_authentic_v13_log_probability(self):
        n = len(self.cache.panel)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'main.npz'
            np.savez_compressed(path, ids=np.array(self.cache.id['ids']), labels=np.array(replay.LABELS),
                                weighted_logp=np.zeros((n,18)))
            replay.write(root/'main-result.json', dict(status='COMPLETE',source_sha256=replay.SOURCE_SHA,
                coefficient=12,pool_rows=190588,fit_log='svc4 fit rows=200478 synthetic',score_sha256=replay.sha(path)))
            with self.assertRaisesRegex(ValueError, 'log probability mismatch'):
                replay.read_main(root, replay.sha(root/'main-result.json'), self.cache)


if __name__ == '__main__':
    unittest.main()
