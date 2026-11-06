"""Offline fixtures only: no model executable, authentication, network or inference."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

CODE=Path(__file__).resolve().parent
REAL_ROOT=CODE.parents[2]
spec=importlib.util.spec_from_file_location('batch_under_test',CODE/'run_batch.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class FakeRunner:
    def __init__(self):self.calls=[];self.fail_at=None;self.mismatch=False;self.peak=0;self.active=0;self.lock=threading.Lock()
    def __call__(self,command,prompt,attempt,timeout,stop_event,cancel_event,cwd):
        with self.lock:
            self.active+=1;self.peak=max(self.peak,self.active);self.calls.append(prompt);index=len(self.calls)
        text='Keep **raw** formatting exactly.\nThis repeated response must remain unchanged.'
        failed=index==self.fail_at
        events=[{'type':'thread.started','thread_id':str(index)},
                {'type':'item.completed','item':{'type':'agent_message','text':text+' mismatch' if self.mismatch else text}},
                {'type':'turn.failed'} if failed else {'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}}]
        (attempt/'final.txt').write_bytes(text.encode())
        (attempt/'raw_stdout.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
        (attempt/'stderr.txt').write_text('',encoding='utf-8')
        with self.lock:self.active-=1
        return {'returncode':1 if failed else 0,'interrupted':False}

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.exp=self.root/'experiments/EXP-039-codex-aligned-300'
        for folder in ['code','metrics','workspace']:(self.exp/folder).mkdir(parents=True,exist_ok=True)
        for relative in [b.CORE,b.FORMAT,b.BANK,b.ALIGNED_CONFIG,b.PILOT_CONFIG,b.INSTRUCTIONS]:
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REAL_ROOT/relative,target)
        shutil.copyfile(CODE/'run_batch.py',self.exp/'code/run_batch.py')
        self.core=b.module('fixture_core',self.root/b.CORE)
        allrows=self.core.read_jsonl(self.root/b.BANK);paired=set(b.load(self.root/b.ALIGNED_CONFIG)['prepared_paired_panel']['prompt_ids'])
        train=[p for p in allrows if p['split']=='train' and p['prompt_id'] not in paired][:60]
        hold=[p for p in allrows if p['split']=='holdout'][:15]
        self.rows=train+hold
        (self.exp/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in self.rows),encoding='utf-8')
        self.config=b.load(self.root/b.PILOT_CONFIG)
        self.config.update(experiment_id='EXP-039',authorization={'generation':True},
                           instructions_file='../EXP-021-codex-subscription-pilot/model-instructions.txt',workspace='workspace',prompts_file='prompts.jsonl',
                           models=[{'slug':m,'reasoning_effort':'low'} for m in sorted(b.MODELS)])
        self.config['budget']['max_concurrent_requests']=1
        self.write(self.exp/'config.json',self.config)
        self.exe=self.root/'fake-codex.exe';self.exe.write_bytes(b'fixture only;never execute')
        now=datetime.now(timezone.utc)
        self.write(self.exp/'metrics/execution-preflight.json',{'status':'PASS','observed_at':now.isoformat(),
            'expires_at':(now+timedelta(minutes=15)).isoformat(),'login_chatgpt':True,
            'codex_executable_path':str(self.exe),'codex_executable_sha256':b.sha(self.exe.read_bytes())})
        self.jobs=self.core.prepare(self.exp,self.exe)[-1]
        (self.exp/'request-plan.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in self.jobs),encoding='utf-8')
        self.freeze();self.runner=FakeRunner()

    def write(self,p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def tearDown(self):self.tmp.cleanup()
    def freeze(self):
        def entry(p):return {'path':p.relative_to(self.root).as_posix(),'sha256':b.sha(p.read_bytes())}
        self.write(self.exp/'manifest.json',{'generation_authorized':True,'config':entry(self.exp/'config.json'),
          'inputs':[entry(self.root/p) for p in [b.CORE,b.FORMAT,b.BANK,b.ALIGNED_CONFIG,b.PILOT_CONFIG,b.INSTRUCTIONS]],
          'code':[entry(self.exp/'code/run_batch.py')],
          'outputs':[entry(self.exp/'prompts.jsonl'),entry(self.exp/'request-plan.jsonl')]})
    def refresh_jobs(self):
        self.jobs=self.core.prepare(self.exp,self.exe)[-1]
        (self.exp/'request-plan.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in self.jobs),encoding='utf-8');self.freeze()
    def run_fake(self,**kw):return b.run_batch(self.exp,run=True,codex=self.exe,runner=self.runner,**kw)

    def test_dry_run_does_not_create_collection_or_run_model(self):
        result=b.run_batch(self.exp,codex=self.exe,runner=self.runner)
        self.assertEqual(result['status'],'DRY_RUN');self.assertEqual(result['target_total'],300)
        self.assertFalse((self.exp/'collection').exists());self.assertEqual(self.runner.calls,[])

    def test_worker_two_rejected_before_any_call(self):
        with self.assertRaisesRegex(ValueError,'exactly1'):self.run_fake(workers=2)
        self.assertEqual(self.runner.calls,[])

    def test_changed_frozen_input_rejected(self):
        with (self.exp/'prompts.jsonl').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_fake()
        self.assertEqual(self.runner.calls,[])

    def test_expired_auth_preflight_rejected(self):
        p=self.exp/'metrics/execution-preflight.json';pre=b.load(p);pre['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();self.write(p,pre)
        with self.assertRaisesRegex(ValueError,'expired'):self.run_fake()
        self.assertEqual(self.runner.calls,[])

    def test_generation_authorization_required(self):
        self.config['authorization']['generation']=False;self.write(self.exp/'config.json',self.config);self.refresh_jobs()
        with self.assertRaisesRegex(ValueError,'authorization'):self.run_fake()
        self.assertEqual(self.runner.calls,[])

    def test_paired_panel_overlap_rejected(self):
        bank=self.core.read_jsonl(self.root/b.BANK);self.rows[0]=bank[0]
        (self.exp/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in self.rows),encoding='utf-8');self.refresh_jobs()
        with self.assertRaisesRegex(ValueError,'Paired panel'):self.run_fake()
        self.assertEqual(self.runner.calls,[])

    def test_full300_preserves_raw_duplicates_and_format_flags(self):
        state=self.run_fake();self.assertEqual(state['status'],'COMPLETE');self.assertEqual(len(self.runner.calls),300);self.assertEqual(self.runner.peak,1)
        audit=b.load(self.exp/'metrics/collection-audit.json');self.assertEqual(audit['status'],'COLLECTED_UNVALIDATED');self.assertEqual(audit['by_model'],{m:75 for m in b.MODELS});self.assertEqual(audit['by_split'],{'train':240,'holdout':60})
        self.assertEqual(audit['duplicate_text_rows'],299);self.assertEqual(audit['flags']['line_break'],300);self.assertEqual(audit['flags']['bold'],300)
        records=self.core.read_jsonl(self.exp/'collection/records.jsonl');self.assertEqual(records[0]['text'],'Keep **raw** formatting exactly.\nThis repeated response must remain unchanged.')
        again=self.run_fake();self.assertEqual(again['status'],'COMPLETE');self.assertEqual(len(self.runner.calls),300)

    def test_failure_stops_and_uncertain_attempt_cannot_replay(self):
        self.runner.fail_at=2;state=self.run_fake();self.assertEqual(state['status'],'STOPPED_FAILURE');self.assertEqual(len(self.runner.calls),2)
        self.assertEqual(b.load(self.exp/'metrics/collection-audit.json')['status'],'INCOMPLETE')
        with self.assertRaises(Exception):self.run_fake()
        self.assertEqual(len(self.runner.calls),2)

    def test_event_final_mismatch_stops_first_request(self):
        self.runner.mismatch=True;state=self.run_fake();self.assertEqual(state['status'],'STOPPED_FAILURE');self.assertEqual(state['saved_total'],0);self.assertEqual(len(self.runner.calls),1)

    def test_audit_cannot_publish_complete_until_passes(self):
        observed=[]
        def invalid_audit(exp,core,inspect,checker,state,jobs,binding):
            observed.append(b.load(exp/'collection/state.json')['status'])
            return {'status':'INCOMPLETE','run_id':state['run_id']}
        with patch.object(b,'audit_collection',side_effect=invalid_audit):state=self.run_fake()
        self.assertEqual(observed,['FINALIZING']);self.assertEqual(state['status'],'AUDIT_FAILED')
        for name in ['state.json','summary.json','run-'+state['run_id']+'.json']:self.assertEqual(b.load(self.exp/'collection'/name)['status'],'AUDIT_FAILED')

    def test_audit_exception_reports_failure(self):
        self.runner.fail_at=1
        with patch.object(b,'audit_collection',side_effect=RuntimeError('synthetic audit failure')):state=self.run_fake()
        self.assertEqual(state['status'],'AUDIT_FAILED');self.assertEqual(state['audit_status'],'ERROR')

if __name__=='__main__':unittest.main()
