import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[2]
KERNEL=ROOT/'cloud/a1-public-stylegram-046/kernel'

class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate=json.loads((KERNEL/'a1.ipynb').read_text(encoding='utf-8'))
        cls.original=json.loads((ROOT/'reviews/own-scheme-readiness-20261030/baseline-v2.ipynb').read_text(encoding='utf-8'))

    def test_hidden_flag_forces_full_even_empty_or_zero(self):
        source=''.join(self.candidate['cells'][2]['source'])
        expression=next(x.split('=',1)[1].split('#',1)[0] for x in source.splitlines() if x.startswith('FAST_COMMIT ='))
        class Frame:
            def __init__(self,n):self.n=n
            def __len__(self):return self.n
        class PD:
            def __init__(self,n):self.n=n
            def read_csv(self,*args):return Frame(self.n)
        class OS:
            def __init__(self,e):self.environ=e
        for rows in [1,3,49,50,4000]:
            for flag in [None,'','0','1']:
                env={} if flag is None else {'KAGGLE_IS_COMPETITION_RERUN':flag}
                actual=eval(expression,{'os':OS(env),'_pd':PD(rows),'DATA':'dummy'})
                self.assertEqual(actual,flag is None and rows<50)

    def test_main_math_unchanged_and_other_source_cells_exact(self):
        for index in [4,6]:
            self.assertEqual(''.join(self.candidate['cells'][index]['source']),''.join(self.original['cells'][index]['source']))
        original=''.join(self.original['cells'][12]['source'])
        current=''.join(self.candidate['cells'][12]['source'])
        self.assertEqual(current.split("print('RUNNING MAIN INFERENCE'")[0],original.split("print('RUNNING MAIN INFERENCE'")[0])
        self.assertIn('verify_main_log',current)
        comp=''.join(self.candidate['cells'][10]['source'])
        self.assertIn('raise subprocess.CalledProcessError',comp)
        self.assertNotIn('grep -v',comp)

    def test_payload_allowlist_hashes_and_no_panel(self):
        source=''.join(self.candidate['cells'][2]['source'])
        tree=ast.parse(source)
        call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_blob' for t in n.targets))
        blob=base64.b64decode(ast.literal_eval(call.args[0]))
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            self.assertEqual(set(z.namelist()),{'a1_stylegram.py','stylegram_core.py','input_policy.py','input_preflight.py','runtime_preflight.py','run_receipts.py','extra-inputs.json','preflight_manifest.json','payload-hashes.json'})
            for name,digest in json.loads(z.read('payload-hashes.json')).items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)
        metadata=json.loads((KERNEL/'kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private'])
        self.assertFalse(metadata['enable_internet'])
        self.assertEqual(sum(len(metadata[k]) for k in ['competition_sources','dataset_sources','kernel_sources','model_sources']),32)
        self.assertIn('zhjianduan/ntl-stylegram-cv-inputs-v1/1',metadata['dataset_sources'])

if __name__=='__main__':unittest.main()
