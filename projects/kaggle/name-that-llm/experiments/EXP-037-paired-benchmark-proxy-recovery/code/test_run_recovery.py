"""Offline recovery checks: fake transports, temporary artifacts, no credentials/network."""
from collections import Counter
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
spec=importlib.util.spec_from_file_location('recovery_under_test',CODE/'run_recovery.py')
trial=importlib.util.module_from_spec(spec);spec.loader.exec_module(trial)

class FakeTransports:
    def __init__(self):
        self.api_calls=[]
        self.codex_calls=[]
        self.api_transform=None
        self.api_exception=None
        self.nano_cost=1000000
        self.live_ready=True
        self.make_calls=0
        self.preflight_calls=0
        self.delay=0

    def make(self):
        self.make_calls+=1
        return object()

    def preflight(self,context):
        self.preflight_calls+=1
        return {'ready':self.live_ready,'listed_models':['openai/gpt-5.6-luna','openai/gpt-5.6-terra'],
                'quota':{'daily':{'remaining_usd':'9.0'},'monthly':{'remaining_usd':'99.0'}}}

    def generate(self,context,payload,*,attempt_dir,job):
        assert attempt_dir.is_dir() and job['api_payload']==payload
        assert (attempt_dir/'request.json').is_file()
        self.api_calls.append(payload)
        if self.api_exception:
            raise self.api_exception
        raw={'id':f'api-{len(self.api_calls)}','model':payload['model'],
             'choices':[{'finish_reason':'stop','message':{'content':'Identical **text**.\nSecond line.','refusal':None}}],
             'usage':{'cost':{'input_tokens_cost_nanodollars':0,'output_tokens_cost_nanodollars':self.nano_cost}}}
        return self.api_transform(raw) if self.api_transform else raw

    def codex(self,job,attempt):
        self.codex_calls.append(job)
        if self.delay:
            time.sleep(self.delay)
        text='Identical **text**.\nSecond line.'
        (attempt/'final.txt').write_bytes(text.encode('utf-8'))
        events=[{'type':'thread.started','thread_id':'codex-'+job['job_id']},
                {'type':'item.completed','item':{'type':'agent_message','text':text}},
                {'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}}]
        (attempt/'raw_stdout.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
        return {'success':True,'reasons':[],'flags':[],'text':text,'text_sha256':trial.sha_bytes(text.encode('utf-8')),
                'model_observed':[],'model_observation_scope':'requested_only','cost_usd':None,'cost_nanodollars':None,
                'usage':[{'input_tokens':10,'output_tokens':5}],'finish_reason':'turn.completed',
                'provider_response_id':'codex-'+job['job_id']}



class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.exp=self.root/'experiments/EXP-037-paired-benchmark-proxy-recovery'
        (self.exp/'code').mkdir(parents=True);(self.exp/'metrics').mkdir()
        template=ROOT/'experiments/EXP-036-aligned-paired-connection-recovery'
        for name in ['config.json','request-plan.jsonl']:
            shutil.copyfile(template/name,self.exp/name)
        self.config=trial.load(self.exp/'config.json');self.jobs=trial.read_jsonl(self.exp/'request-plan.jsonl')
        self.receipt=trial.load(template/'recovery-receipt.json')
        for rel in trial.RUNTIME_FILES+[trial.BASE_REL]:
            dest=self.root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        oldrel='experiments/EXP-035-aligned-paired-repeat-pilot/collection'
        old=self.exp/'recovery/history/collection';old.mkdir(parents=True)
        for name in ['state.json','records.jsonl']:shutil.copyfile(ROOT/oldrel/name,old/name)
        shutil.copytree(ROOT/oldrel/'attempts',old/'attempts')
        for attempt in (template/'collection/attempts').iterdir():
            shutil.copytree(attempt,old/'attempts'/attempt.name)
        original_collection=self.root/oldrel
        shutil.copytree(ROOT/oldrel,original_collection)
        state=trial.load(old/'state.json');state.update(status='HISTORICAL_SNAPSHOT',
            requests_sent_total=3,requests_by_route={'codex_subscription':1,'kaggle_model_proxy':2})
        trial.write_json(old/'state.json',state)
        self.receipt.update(prior_state_path=(old/'state.json').relative_to(self.root).as_posix(),
            prior_records_path=(old/'records.jsonl').relative_to(self.root).as_posix(),
            historical_counts=state['requests_by_route'],
            historical_uncertain_job_ids=[p.name for p in (old/'attempts').iterdir()
                if trial.load(p/'result.json')['status']=='UNCERTAIN'])
        self.receipt['historical_attempt_origins']={ident:{
            'experiment':'EXP-035-aligned-paired-repeat-pilot',
            'attempt_path':oldrel+'/attempts/'+ident}
            for ident in self.receipt['historical_success_job_ids']}
        trial.write_json(self.exp/'recovery-receipt.json',self.receipt)
        shutil.copyfile(CODE/'run_recovery.py',self.exp/'code/run_recovery.py')
        (self.exp/'code/kaggle_transport.py').write_text('# fake transport injected',encoding='utf-8')
        self.codex=self.root/'fake-codex.exe';self.codex.write_bytes(b'never executable')
        now=datetime.now(timezone.utc)
        trial.write_json(self.exp/'metrics/execution-preflight.json',{'status':'PASS','observed_at':now.isoformat(),
            'expires_at':(now+timedelta(minutes=15)).isoformat(),'free_quota_only':True,'free_quota_remaining_usd':'9',
            'models_available':['openai/'+m for m in trial.MODELS],'requested_reasoning_effort':'low',
            'codex_executable_path':str(self.codex),'codex_executable_sha256':trial.sha_bytes(self.codex.read_bytes())})
        self.fake=FakeTransports();self.freeze()

    def tearDown(self):self.temp.cleanup()

    def freeze(self):
        entry=lambda p:{'path':p.relative_to(self.root).as_posix(),'sha256':trial.sha_bytes(p.read_bytes())}
        files=[self.root/r for r in trial.RUNTIME_FILES+[trial.BASE_REL]]
        files += [self.exp/'recovery-receipt.json',self.exp/'request-plan.jsonl']
        files += list((self.exp/'recovery/history/collection').rglob('*'))
        files += list((self.root/'experiments/EXP-035-aligned-paired-repeat-pilot/collection').rglob('*'))
        files=[p for p in files if p.is_file()]
        trial.write_json(self.exp/'manifest.json',{'generation_authorized':True,'config':entry(self.exp/'config.json'),
            'inputs':[entry(p) for p in files],
            'code':[entry(self.exp/'code/run_recovery.py'),entry(self.exp/'code/kaggle_transport.py')]})

    def run_fake(self,**kwargs):
        return trial.run_recovery(self.exp,run=True,codex_path=self.codex,kaggle_generate=self.fake.generate,
            kaggle_make_context=self.fake.make,kaggle_preflight=self.fake.preflight,codex_call=self.fake.codex,
            format_checker=lambda text:['line_break','bold'],**kwargs)

    def test_dry_run_is_offline_and_skips_three_old_dispatches(self):
        r=trial.run_recovery(self.exp)
        self.assertEqual(r['remaining'],77);self.assertEqual(r['already_saved'],1)
        self.assertFalse(r['historical_cost_known']);self.assertFalse((self.exp/'collection').exists())

    def test_full_counts_no_repeated_old_success_or_unknown(self):
        r=self.run_fake();self.assertEqual(r['status'],'DISPATCH_COMPLETE_PARTIAL')
        self.assertEqual(r['saved_total'],78);self.assertEqual(r['requests_sent_total'],80)
        self.assertEqual(r['requests_by_route'],{'codex_subscription':40,'kaggle_model_proxy':40})
        self.assertEqual(len(self.fake.codex_calls),39);self.assertEqual(len(self.fake.api_calls),38)
        new=trial.read_jsonl(self.exp/'collection/records.jsonl');combined=trial.read_jsonl(self.exp/'collection/combined-records.jsonl')
        oldids=set(self.receipt['historical_success_job_ids']+self.receipt['historical_uncertain_job_ids'])
        self.assertFalse({x['job_id'] for x in new}&oldids);self.assertEqual(len(combined),78)
        inherited=next(x for x in combined if x['inherited_history_snapshot'])
        self.assertEqual(inherited['origin_experiment'],'EXP-035-aligned-paired-repeat-pilot')
        self.assertEqual(inherited['attempt_original_path_project'],
                         self.receipt['historical_attempt_origins'][inherited['job_id']]['attempt_path'])
        self.assertFalse(r['cost_known']);self.assertTrue(r['new_cost_known'])
        self.assertTrue(all(x['text']=='Identical **text**.\nSecond line.' for x in new))
        self.assertTrue(all('line_break' in x['flags'] for x in new))
        self.assertEqual(r['audit_status'],'DISPATCH_COMPLETE_PARTIAL_AUDITED')
        audit=trial.load(self.exp/'metrics/collection-audit.json')
        self.assertEqual(audit['complete_paired_cells'],19)
        self.assertEqual(audit['counts']['kaggle_model_proxy|gpt-5.6-terra'],18)

    def test_partial_resume_does_not_repeat_success(self):
        first=self.run_fake(max_jobs=2);self.assertEqual(first['requests_sent_total'],5)
        self.assertEqual(first['saved_total'],3);before=len(self.fake.api_calls)+len(self.fake.codex_calls)
        second=self.run_fake(max_jobs=1);self.assertEqual(second['requests_sent_total'],6)
        self.assertEqual(len(self.fake.api_calls)+len(self.fake.codex_calls),before+1)

    def test_new_unknown_cost_stops_and_blocks_replay(self):
        self.fake.api_transform=lambda raw:{**raw,'usage':{}}
        r=self.run_fake();self.assertEqual(r['requests_sent_total'],5)
        self.assertFalse(r['new_cost_known']);self.assertEqual(r['status'],'STOPPED_FAILURE')
        with self.assertRaises(trial.InspectionRequired):self.run_fake()

    def test_known_budget_crossing_stops_next_dispatch(self):
        self.fake.nano_cost=110000000
        r=self.run_fake();self.assertEqual(r['status'],'STOPPED_BUDGET')
        self.assertEqual(r['requests_sent_total'],5);self.assertEqual(r['confirmed_cost_usd'],'0.11')

    def test_new_transport_error_retains_safe_chain(self):
        class TransportError(Exception):
            cause_chain=[{'type':'ConnectError','errno':10054,'message':'must not leak'}]
            transport_trace=['connection.connect_tcp.failed']
        self.fake.api_exception=TransportError('secret request header')
        r=self.run_fake();failure=r['failures'][0]
        self.assertEqual(failure['cause_chain'],[{'type':'ConnectError','errno':10054}])
        self.assertEqual(failure['transport_trace'],['connection.connect_tcp.failed'])
        self.assertNotIn('secret',(self.exp/'collection/state.json').read_text())

    def test_audit_failure_never_publishes_success(self):
        with patch.object(trial,'audit_collection',side_effect=RuntimeError('audit injected')):
            r=self.run_fake(max_jobs=1)
        self.assertEqual(r['status'],'AUDIT_FAILED')
        self.assertEqual(trial.load(self.exp/'collection/summary.json')['status'],'AUDIT_FAILED')

    def test_stale_preflight_blocks_transports(self):
        p=self.exp/'metrics/execution-preflight.json';v=trial.load(p);v['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();trial.write_json(p,v)
        with self.assertRaisesRegex(ValueError,'expired'):self.run_fake()
        self.assertEqual(self.fake.make_calls,0)

    def test_changed_history_fails_before_calls(self):
        p=self.root/self.receipt['prior_records_path'];p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_fake()
        self.assertEqual(self.fake.make_calls,0)

    def test_historical_uncertainty_requires_separate_explicit_authorization(self):
        self.receipt['authorization_to_continue_with_preserved_historical_uncertainty']=False
        trial.write_json(self.exp/'recovery-receipt.json',self.receipt);self.freeze()
        with self.assertRaisesRegex(ValueError,'authorization'):self.run_fake()
        self.assertEqual(self.fake.make_calls,0)

    def test_live_preflight_failure_dispatches_nothing(self):
        self.fake.live_ready=False;r=self.run_fake()
        self.assertEqual(r['requests_sent_total'],3);self.assertEqual(r['launched_this_run'],0)
        self.assertEqual(r['status'],'STOPPED_PREFLIGHT_OR_CONTROLLER')

if __name__=='__main__':unittest.main()
