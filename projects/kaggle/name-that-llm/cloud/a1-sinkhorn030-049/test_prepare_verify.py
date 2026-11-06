"""Fail-closed mutation checks; never edits the frozen baseline."""
import copy, unittest
from prepare_verify import source_gate,expected_notebook,expected_metadata,validate_candidate
class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.old,self.meta=source_gate();self.new=expected_notebook(self.old);self.newmeta=expected_metadata(self.meta)
    def test_exact_candidate(self):
        self.assertEqual(validate_candidate(self.old,self.meta,self.new,self.newmeta)['changed_cells'],[0,2,12])
    def test_reject_changed_model_cell(self):
        self.new['cells'][6]['source']='print("different model")'
        with self.assertRaises(AssertionError):validate_candidate(self.old,self.meta,self.new,self.newmeta)
    def test_reject_added_training_attachment(self):
        self.newmeta['dataset_sources'].append('unapproved/self-generated120/1')
        with self.assertRaises(AssertionError):validate_candidate(self.old,self.meta,self.new,self.newmeta)
    def test_reject_second_math_change(self):
        self.new['cells'][12]['source']=self.new['cells'][12]['source'].replace("'--atlas_w', '0.20'","'--atlas_w', '0.00'")
        self.assertNotEqual(self.new,expected_notebook(self.old))
        with self.assertRaises(AssertionError):validate_candidate(self.old,self.meta,self.new,self.newmeta)
if __name__=='__main__':unittest.main()
