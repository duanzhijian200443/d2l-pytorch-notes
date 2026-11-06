"""Adversarial unit checks; no network or model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import workbench_core as core
import workbench_remote as remote
import experiment_workbench as wb

TARGET={'ref':'zhjianduan/example','submission_id':123,'script_version_id':456,'version':2}

class ContractTests(unittest.TestCase):
    def test_fixed_version_and_zero_score(self):
        row={'ref':123,'status':'COMPLETE','publicScore':'0','url':'/code/zhjianduan/example?scriptVersionId=456'}
        result=core.parse_submission([row],TARGET)
        self.assertEqual(result['score'],0);self.assertTrue(result['version_verified'])

    def test_missing_score_or_version_preserves_history(self):
        record={'submission':{'status':'COMPLETE','public_score':.663,'observed_at':'2026-11-01T00:00:00Z'}};before=copy.deepcopy(record)
        for row in [{'ref':123,'status':'COMPLETE','url':'/code/zhjianduan/example?scriptVersionId=456'},
                    {'ref':123,'status':'COMPLETE','publicScore':'.9','url':'/submissions/123/123.raw'}]:
            self.assertIsNone(core.parse_submission([row],TARGET)['score'])
            self.assertEqual(core.historical_score(record)['value'],.663)
        self.assertEqual(record,before)

    def test_wrong_version_ref_score_duplicates(self):
        good={'ref':123,'status':'COMPLETE','publicScore':'.663','url':'/code/zhjianduan/example?scriptVersionId=456'}
        for changes in [{'url':'/code/zhjianduan/example?scriptVersionId=457'}, {'url':'/code/other/example?scriptVersionId=456'},
                        {'publicScore':'NaN'},{'publicScore':'2'},{'status':'UNEXPECTED'}]:
            with self.subTest(changes=changes),self.assertRaises(core.GateError):core.parse_submission([dict(good,**changes)],TARGET)
        with self.assertRaises(core.GateError):core.parse_submission([good,good],TARGET)
        self.assertEqual(core.parse_submission([],TARGET)['status'],'UNKNOWN_NOT_RETURNED')

    def test_unsafe_paths(self):
        for name in ['../x','C:/Windows/test','/etc/passwd','foo/../../x']:
            with self.subTest(name=name),self.assertRaises(core.GateError):core.inside(name)

    def test_no_provider_secrets_in_failures(self):
        raw='https://storage.googleapis.com/x?Signature=SECRET Authorization: Bearer sk-secret password=secret'
        result=json.dumps(core.safe_failure(RuntimeError(raw)))
        for value in ['SECRET','sk-secret','Authorization','Signature','password']:self.assertNotIn(value,result)

    def test_unknown_experiment_is_rejected(self):
        with self.assertRaises(core.GateError):core.profile('../EXP-046')

    def test_download_hosts(self):
        remote.valid_url('https://storage.googleapis.com/a?Signature=opaque')
        for u in ['http://storage.googleapis.com/a','https://127.0.0.1/a','https://storage.googleapis.com.evil/a','https://user:pw@storage.googleapis.com/a','https://storage.googleapis.com:81/a']:
            with self.subTest(u=u),self.assertRaises(core.GateError):remote.valid_url(u)

    def test_stream_size_guards_and_no_partial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();self.assertTrue(root.is_absolute())
            with self.assertRaises(core.GateError):remote.stream_file([b'123456'],root/'over',5,5)
            self.assertFalse((root/'over').exists())
            with self.assertRaises(core.GateError):remote.stream_file([b'123'],root/'short',5,5)
            self.assertFalse((root/'short').exists())
            self.assertEqual(remote.stream_file([b'12',b'345'],root/'ok',5,5),5)
            self.assertEqual((root/'ok').read_bytes(),b'12345')

    def test_manifest_tamper_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'a').write_bytes(b'yes')
            entry={'name':'a','bytes':3,'sha256':hashlib.sha256(b'yes').hexdigest()}
            manifest={'status':'COLLECTED_COMMIT_OUTPUTS','files':[entry]};core.verify_files(manifest,folder)
            (folder/'a').write_bytes(b'bad')
            with self.assertRaises(core.GateError):core.verify_files(manifest,folder)
            (folder/'a').write_bytes(b'yes');manifest['files'].append(entry)
            with self.assertRaises(core.GateError):core.verify_files(manifest,folder)

    def test_confirmation_survives_subsequent_error_and_no_score(self):
        parent=core.WORK/'metrics';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='wb-test-',dir=parent) as tmp:
            temp=Path(tmp).resolve();self.assertTrue(temp.is_relative_to(core.ROOT))
            with patch.object(core,'WORK',temp),patch.object(wb,'WORK',temp):
                good={'status':'COMPLETE','score':.665,'query_ok':True,'version_verified':True,'submission_id':123,'script_version_id':456,'observed_at':'2026-11-02T00:00:00Z'}
                core.remember('EXP-048','status',good,temp/'success.json')
                core.remember('EXP-048','status',{'status':'ERROR','query_ok':False,'observed_at':'2026-11-02T01:00:00Z'},temp/'error.json')
                self.assertEqual(core.latest('EXP-048','confirmed-score')['score'],.665)
                p={'id':'EXP-048','remote':TARGET,'report':'README.md'};r={'title':'test','submission':None}
                report=wb.make_report(p,r).read_text(encoding='utf-8')
                self.assertIn('0.665',report);self.assertIn('历史确认分数保留',report)
                core.remember('EXP-048','status',dict(good,score=None),temp/'missing.json')
                self.assertEqual(core.latest('EXP-048','confirmed-score')['score'],.665)

    def test_unverified_or_tampered_summary_not_displayed(self):
        parent=core.WORK/'metrics';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='wb-test-',dir=parent) as tmp:
            temp=Path(tmp).resolve();self.assertTrue(temp.is_relative_to(core.ROOT))
            summary=temp/'summary.json';core.atomic_json(summary,{'status':'COMPLETE_LOCAL_DIAGNOSTIC','baseline_045':{'correct':488},'candidate_030':{'correct':999}})
            p={'id':'EXP-049','remote':None,'local_summary':summary.relative_to(core.ROOT).as_posix(),'report':'README.md'};r={'title':'test'}
            with patch.object(core,'WORK',temp),patch.object(wb,'WORK',temp):
                core.remember('EXP-049','verify',{'status':'ERROR','observed_at':core.now()},temp/'error.json')
                report=wb.make_report(p,r).read_text(encoding='utf-8');self.assertNotIn('999',report)
                core.remember('EXP-049','verify',{'status':'PASS_FIXED_LOCAL_JOB','verified_summary':{'path':p['local_summary'],'sha256':'0'*64}},temp/'wrong-hash.json')
                self.assertNotIn('999',wb.make_report(p,r).read_text(encoding='utf-8'))

    def test_local_command_allowlist_and_optimization(self):
        self.assertNotIn(('EXP-046','run-local'),wb.JOB_SPECS)
        with patch.dict('os.environ',{'PYTHONOPTIMIZE':'2'}):self.assertNotIn('PYTHONOPTIMIZE',wb.child_env({}))
        with self.assertRaises(core.GateError):wb.child_env({'proxy':'http://evil.example'})

if __name__=='__main__':unittest.main()
