"""Offline handoff tests. No credentials, network, or model executables."""
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=mod('handoff_under_test',CODE/'run_batch.py')
oldtests=mod('old_batch_fixtures',ROOT/'experiments/EXP-039-codex-aligned-300/code/test_run_batch.py')

class FakeRunner:
    def __init__(self):self.calls=[];self.fail=False;self.active=0;self.peak=0;self.lock=threading.Lock()
    def __call__(self,command,prompt,attempt,timeout,stop,cancel,cwd):
        with self.lock:self.calls.append(attempt.name);index=len(self.calls);self.active+=1;self.peak=max(self.peak,self.active)
        time.sleep(.005)
        text='Keep **raw** formatting exactly.\nThe same response remains preserved.'
        failed=self.fail and index==1
        events=[{'type':'thread.started','thread_id':attempt.name},{'type':'item.completed','item':{'type':'agent_message','text':text}},
                {'type':'turn.failed'} if failed else {'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':7}}]
        (attempt/'final.txt').write_bytes(text.encode());(attempt/'raw_stdout.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events),encoding='utf-8');(attempt/'stderr.txt').write_text('',encoding='utf-8')
        with self.lock:self.active-=1
        return {'returncode':1 if failed else 0,'interrupted':False}

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture=oldtests.BatchTests('test_dry_run_does_not_create_collection_or_run_model');self.fixture.setUp()
        self.root=self.fixture.root;self.old=self.fixture.exp;self.exe=self.fixture.exe;self.core=self.fixture.core
        base=b.module('original_batch',self.root/b.BASE);fmt=b.module('original_format',self.root/b.FORMAT);original=self.core.inspect_response
        self.core.inspect_response=lambda a,o,m:base.inspect_saved(self.core,original,fmt.format_flags,a,o,m)
        state=self.core.run_collection(self.old,self.exe,limit=2,workers=1,run=True,runner=FakeRunner())
        self.core.inspect_response=original
        self.jobs=self.fixture.jobs;self.saved=self.core.read_jsonl(self.old/'collection/records.jsonl');self.gate=self.jobs[2]['key']
        gate=self.old/'collection/attempts'/self.gate;gate.mkdir();self.write(gate/'handover-gate.json',{'job_key':self.gate,'purpose':'authorized handoff fixture'})
        state.update(status='LOCAL_ERROR',controller_pid=99999991,active=[],failures=[{'job_key':self.gate,'status':'UNCERTAIN','reasons':['CONTROLLER_FileExistsError']}],
                     local_errors=[{'job_key':self.gate,'stage':'attempt_artifacts','error_type':'FileExistsError'}])
        self.write(self.old/'collection/state.json',state)
        self.exp=self.root/'experiments/EXP-040-codex300-concurrency2'
        for folder in ['code','metrics']:(self.exp/folder).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(CODE/'run_batch.py',self.exp/'code/run_batch.py')
        for name in ['prompts.jsonl','request-plan.jsonl']:shutil.copyfile(self.old/name,self.exp/name)
        self.config={'workers':2,'source_experiment':self.old.relative_to(self.root).as_posix(),'authorization':{'generation':True},'max_seconds':21600}
        self.write(self.exp/'config.json',self.config)
        self.receipt={'prior_experiment':self.old.relative_to(self.root).as_posix(),'prior_records_path':(self.old/'collection/records.jsonl').relative_to(self.root).as_posix(),
          'prior_state_path':(self.old/'collection/state.json').relative_to(self.root).as_posix(),'saved_count':2,'saved_job_keys':[r['job_key'] for r in self.saved],
          'gate_job_key':self.gate,'gate_attempt_path':gate.relative_to(self.root).as_posix(),'prior_controller_pid':99999991,
          'original_wall_deadline_utc':(datetime.fromisoformat(state['started_at'])+timedelta(seconds=21600)).isoformat()}
        self.write(self.exp/'recovery-receipt.json',self.receipt)
        now=datetime.now(timezone.utc)
        self.write(self.exp/'metrics/execution-preflight.json',{'status':'PASS','observed_at':now.isoformat(),'expires_at':(now+timedelta(minutes=15)).isoformat(),
           'login_chatgpt':True,'codex_executable_path':str(self.exe),'codex_executable_sha256':b.sha(self.exe.read_bytes())})
        self.freeze();self.runner=FakeRunner()
    def tearDown(self):self.fixture.tearDown()
    def write(self,p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def freeze(self):
        entry=lambda p:{'path':p.relative_to(self.root).as_posix(),'sha256':b.sha(p.read_bytes())}
        paths=[self.root/p for p in b.DEPENDENCIES]+[self.old/p for p in ['config.json','prompts.jsonl','request-plan.jsonl','collection/state.json','collection/records.jsonl']]
        paths += [p for p in (self.old/'collection/attempts').rglob('*') if p.is_file()]
        paths += [self.exp/'recovery-receipt.json']
        self.write(self.exp/'manifest.json',{'generation_authorized':True,'config':entry(self.exp/'config.json'),
          'inputs':[entry(p) for p in dict.fromkeys(paths)],'code':[entry(self.exp/'code/run_batch.py')],
          'outputs':[entry(self.exp/'prompts.jsonl'),entry(self.exp/'request-plan.jsonl')]})
    def run_fake(self,**kwargs):return b.run_batch(self.exp,run=True,codex=self.exe,runner=self.runner,pid_checker=lambda pid:False,**kwargs)

    def test_dry_run_remaining298_no_calls(self):
        r=b.run_batch(self.exp,codex=self.exe,runner=self.runner);self.assertEqual(r['remaining_in_this_run'],298);self.assertEqual(r['already_saved'],2)
        self.assertFalse((self.exp/'collection').exists());self.assertEqual(self.runner.calls,[])
    def test_worker3_rejected(self):
        with self.assertRaisesRegex(ValueError,'exactly2'):self.run_fake(workers=3)
        self.assertEqual(self.runner.calls,[])
    def test_old_pid_alive_blocks_start(self):
        with self.assertRaisesRegex(ValueError,'Old controller'):
            b.run_batch(self.exp,run=True,codex=self.exe,runner=self.runner,pid_checker=lambda pid:True)
        self.assertEqual(self.runner.calls,[])
    def test_history_hash_change_rejected(self):
        p=self.old/'collection/records.jsonl';p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_fake()
        self.assertEqual(self.runner.calls,[])
    def test_gate_request_evidence_blocks_replay(self):
        p=self.root/self.receipt['gate_attempt_path'];self.write(p/'request.json',{'not_safe':'could be sent'});self.freeze()
        with self.assertRaisesRegex(ValueError,'may have dispatched'):self.run_fake()
        self.assertEqual(self.runner.calls,[])
    def test_full_continuation_preserves_successes_and_original_keys(self):
        r=self.run_fake();self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['saved_total'],300);self.assertEqual(r['saved_new_total'],298)
        self.assertEqual(self.runner.peak,2);self.assertEqual(len(self.runner.calls),298)
        self.assertFalse(set(self.runner.calls)&set(self.receipt['saved_job_keys']));self.assertIn(self.gate,self.runner.calls[:2])
        audit=b.load(self.exp/'metrics/collection-audit.json');self.assertEqual(audit['status'],'COLLECTED_UNVALIDATED');self.assertEqual(audit['by_model'],{m:75 for m in b.MODELS})
        self.assertEqual(audit['by_split'],{'train':240,'holdout':60});self.assertEqual(audit['gate_model_dispatches'],0)
        combined=self.core.read_jsonl(self.exp/'collection/combined-records.jsonl');self.assertEqual(len(combined),300);self.assertEqual(combined[0]['text'],self.saved[0]['text'])
        again=self.run_fake();self.assertEqual(again['status'],'COMPLETE');self.assertEqual(len(self.runner.calls),298)
    def test_new_failure_stops_without_replay(self):
        self.runner.fail=True;r=self.run_fake();self.assertEqual(r['status'],'STOPPED_FAILURE');self.assertLessEqual(len(self.runner.calls),2)
        self.assertEqual(r['already_saved'],2);self.assertEqual(r['already_saved_new'],0)
        before=len(self.runner.calls)
        with self.assertRaises(Exception):self.run_fake()
        self.assertEqual(len(self.runner.calls),before)
    def test_audit_failure_never_complete(self):
        self.runner.fail=True
        with patch.object(b,'audit_collection',side_effect=RuntimeError('synthetic audit')):r=self.run_fake()
        self.assertEqual(r['status'],'AUDIT_FAILED');self.assertEqual(b.load(self.exp/'collection/state.json')['status'],'AUDIT_FAILED')
    def test_stale_preflight_blocks(self):
        p=self.exp/'metrics/execution-preflight.json';v=b.load(p);v['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();self.write(p,v)
        with self.assertRaisesRegex(ValueError,'expired'):self.run_fake()
        self.assertEqual(self.runner.calls,[])
    def test_changed_original_prompt_plan_rejected(self):
        p=self.exp/'prompts.jsonl';p.write_bytes(p.read_bytes()+b'\n');self.freeze()
        with self.assertRaisesRegex(ValueError,'inputs differ'):self.run_fake()
        self.assertEqual(self.runner.calls,[])
    def test_old_lock_blocks_start(self):
        (self.old/'collection/batch.lock').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'lock remains'):self.run_fake()
        self.assertEqual(self.runner.calls,[])
    def test_original_deadline_cannot_be_extended(self):
        self.receipt['original_wall_deadline_utc']=(datetime.now(timezone.utc)+timedelta(hours=7)).isoformat()
        self.write(self.exp/'recovery-receipt.json',self.receipt);self.freeze()
        with self.assertRaisesRegex(ValueError,'deadline changed'):self.run_fake()
        self.assertEqual(self.runner.calls,[])

if __name__=='__main__':unittest.main()
