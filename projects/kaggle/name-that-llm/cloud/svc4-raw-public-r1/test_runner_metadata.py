"""Regression for the observed NumPy vocabulary-index serialization failure."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent

def extract(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='feature_receipt')
    namespace={'hashlib':hashlib,'json':json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['feature_receipt']

class VocabularyReceiptTests(unittest.TestCase):
    def fixture(self,integer_type):
        vectorizer=SimpleNamespace(vocabulary_={'词':integer_type(1),'abc':integer_type(0)},idf_=np.array([1.0,2.0],dtype=np.float32))
        return SimpleNamespace(VIEWS=[('word','word',str,(1,3),180000,.7)],vectorizers_=[vectorizer])

    def test_observed_failure_is_reproduced_and_fixed(self):
        with self.assertRaisesRegex(TypeError,'not JSON serializable'):
            extract(HERE.parent/'svc4-raw-public/runner.py')(self.fixture(np.int64))
        corrected=extract(HERE/'runner.py')
        expected=corrected(self.fixture(int))
        self.assertEqual(corrected(self.fixture(np.int64)),expected)
        self.assertEqual(corrected(self.fixture(np.int32)),expected)
        json.dumps(expected,allow_nan=False)

    def test_only_metadata_normalization_changed_in_runner(self):
        original=(HERE.parent/'svc4-raw-public/runner.py').read_text(encoding='utf-8')
        current=(HERE/'runner.py').read_text(encoding='utf-8')
        self.assertEqual(current,original.replace('sorted(vectorizer.vocabulary_.items())',
            'sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items())'))

if __name__=='__main__':unittest.main()
