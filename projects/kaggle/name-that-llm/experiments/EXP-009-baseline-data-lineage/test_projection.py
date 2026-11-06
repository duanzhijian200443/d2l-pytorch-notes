import json
import unittest
import pandas as pd
import numpy as np
from project_rows import HERE, OUT, ROOT, main_key, comp_text, sha


class ProjectionChecks(unittest.TestCase):
    def test_baseline_dedup_keeps_case_and_normalizes_unicode(self):
        self.assertNotEqual(main_key("Model A"), main_key("model a"))
        self.assertEqual(main_key("Ａ\n  B"), main_key("A B"))
        self.assertEqual(comp_text("  **hello   world**  "), "hello world")

    def test_executed_source_and_original_function_equivalence(self):
        result=json.loads((OUT/'projection.json').read_text(encoding='utf-8'))
        self.assertEqual(result['audit_sha256'],sha(HERE/'project_rows.py'))
        self.assertTrue(all(result['checks'].values()))
        rec=json.loads((ROOT/'tracking/records/experiments/EXP-004.json').read_text(encoding='utf-8'))
        self.assertEqual(result['source_notebook_sha256'],sha(ROOT/rec['source']['local_path']))

    def test_filter_funnel_and_rows_reconcile(self):
        stages=pd.read_csv(OUT/'complement_filter_stages.csv')
        nums=stages.drop(columns='origin').to_numpy()
        self.assertTrue((np.diff(nums,axis=1)<=0).all())
        rows=pd.read_csv(OUT/'complement_extra_rows.csv')
        self.assertEqual(int(stages.after_min_12_labels.sum()),len(rows))
        self.assertFalse(rows.duplicated(['prompt_id','model']).any())
        self.assertTrue(rows.groupby('prompt_id').model.nunique().ge(12).all())

    def test_per_class_weight_mass_is_conserved(self):
        mass=pd.read_csv(OUT/'stylegram_class_weight_mass.csv',index_col=0)
        train=pd.read_csv(ROOT/'train.csv').model.value_counts().reindex(mass.index)
        self.assertTrue(np.allclose(mass.official,mass.public))
        self.assertTrue(np.allclose(mass.official+mass.public,train))

    def test_old_prvs_rows_are_not_called_new(self):
        rows=pd.read_csv(OUT/'candidate_overlap.csv')
        new=rows.loc[rows.table.eq('prvs-clean-v6') & rows.split.eq('new')]
        self.assertEqual(len(new),4158)
        self.assertTrue(new.already_in_projected_comp_stylegram.all())
        possible=pd.read_csv(OUT/'potential_increment_rows.csv')
        self.assertEqual(len(possible),1385)
        self.assertFalse(possible.already_in_any_audited_runtime_branch.any())
        self.assertEqual(set(possible.row_id),set(rows.loc[~rows.already_in_any_audited_runtime_branch,'row_id']))


if __name__=='__main__':
    unittest.main()
