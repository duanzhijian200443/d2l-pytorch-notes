"""Real captured logs and mutations; no model execution or network."""
import importlib.util
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('run_receipts', ROOT/'cloud/a1-public-stylegram-046/run_receipts.py')
checks=importlib.util.module_from_spec(spec); spec.loader.exec_module(checks)
LOGS=ROOT/'cloud/full-stack-capture-r1/outputs-v2/results/logs'


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comp=(LOGS/'comp.log').read_text(encoding='utf-8')
        cls.main=(LOGS/'main.log').read_text(encoding='utf-8')

    def test_real_exp031_complete_logs(self):
        comp=checks.verify_comp_log(self.comp,729)
        main=checks.verify_main_log(self.main,729)
        self.assertEqual(comp['member_count'],20)
        self.assertEqual(comp['eligible_comp_rows'],728)
        self.assertEqual(main['atlas_completed_rows'],206)

    def test_comp_missing_duplicate_member_weight_rows_gate_smoke_skip(self):
        mutations=[re.sub(r'^.*\+ stylegram .*\n','',self.comp,flags=re.M),
          self.comp+'\n  + stylegram weight 1.516 on 729 rows (1s)\n',
          self.comp.replace('weight 1.516','weight 1.515'),
          self.comp.replace('on 728 rows','on 727 rows',1),
          self.comp.replace('family gate applied glm:','family gate skipped glm:'),
          self.comp.replace('smoke False','smoke True'),
          self.comp+'\nSKIP tokattn_deb: unavailable\n']
        for text in mutations:
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError): checks.verify_comp_log(text,729)

    def test_comp_no_eligible_groups_conditional_members_absent(self):
        text=self.comp.replace('singletons 1  2-3: 8  4-18: 720  >18: 0','singletons 729  2-3: 0  4-18: 0  >18: 0')
        text='\n'.join(line for line in text.splitlines() if not any(re.match(r'\s*\+ '+re.escape(k)+r'\s',line) for k in checks.CONDITIONAL))
        self.assertEqual(checks.verify_comp_log(text,729)['member_count'],10)
        with self.assertRaises(ValueError): checks.verify_comp_log(text+'\n  + glm9b_twin weight 0.165 on 0 rows (1s)',729)

    def test_main_missing_member_adapter_gate_partial_or_skipped_atlas(self):
        mutations=[re.sub(r'^.*\] svc4 done\n','',self.main,flags=re.M),
          self.main.replace('adapter q8v5_f0 done','adapter unexpected done'),
          self.main.replace('atlas completed 206/206','atlas completed 205/206'),
          self.main.replace('mixed 206/206','mixed 205/206'),
          self.main.replace('rows at 0.2','rows at 0.1'),
          self.main.replace('comp stack loaded','COMP SKIPPED'),
          self.main+'\n[ 999s] ATLAS SKIPPED: problem\n',
          self.main+'\n[ 999s] atlas budget checkpoint: preserving 200/206 completed predictions\n',
          self.main.replace('gaussian head q14v5: 724','gaussian head q14v5: 723')]
        for text in mutations:
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError): checks.verify_main_log(text,729)

    def test_zero_atlas_request_is_valid(self):
        text=self.main.replace('atlas on 206/729','atlas on 0/729').replace('atlas completed 206/206','atlas completed 0/0').replace('mixed 206/206','mixed 0/0')
        self.assertEqual(checks.verify_main_log(text,729)['atlas_completed_rows'],0)


if __name__=='__main__':unittest.main()
