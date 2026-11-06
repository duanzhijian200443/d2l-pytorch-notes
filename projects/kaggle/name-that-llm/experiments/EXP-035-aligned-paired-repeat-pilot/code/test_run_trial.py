"""Offline checks: no credentials, SDK imports, network or model processes."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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
REAL_ROOT=CODE.parents[2]
spec=importlib.util.spec_from_file_location('trial_under_test',CODE/'run_trial.py')
trial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


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

    def generate(self,context,payload):
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


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.exp=self.root/'experiments/EXP-035-aligned-paired-repeat-pilot'
        (self.exp/'code').mkdir(parents=True)
        (self.exp/'metrics').mkdir()
        self.config=trial.load(CODE.parent/'config.json')
        self.config['authorization']['generation']=True
        self.jobs=trial.read_jsonl(CODE.parent/'request-plan.jsonl')
        for job in self.jobs:
            job['execution_authorized']=True
        trial.write_json(self.exp/'config.json',self.config)
        (self.exp/'request-plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.jobs),encoding='utf-8')
        shutil.copyfile(CODE/'run_trial.py',self.exp/'code/run_trial.py')
        (self.exp/'code/kaggle_transport.py').write_text('# Fake injection is used; this module must not be imported.\n',encoding='utf-8')
        for relative in trial.RUNTIME_FILES:
            target=self.root/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REAL_ROOT/relative,target)
        self.codex=self.root/'fake-codex.exe'
        self.codex.write_bytes(b'not executable: fake transports only')
        now=datetime.now(timezone.utc)
        pre={'status':'PASS','observed_at':now.isoformat(),'expires_at':(now+timedelta(minutes=15)).isoformat(),
             'free_quota_only':True,'free_quota_remaining_usd':'9.0',
             'models_available':['openai/gpt-5.6-luna','openai/gpt-5.6-terra'],
             'requested_reasoning_effort':'low','codex_executable_path':str(self.codex),
             'codex_executable_sha256':trial.sha_bytes(self.codex.read_bytes())}
        trial.write_json(self.exp/'metrics/execution-preflight.json',pre)
        self.freeze()
        self.fake=FakeTransports()

    def tearDown(self):
        self.tmp.cleanup()

    def freeze(self):
        entry=lambda p:{'path':p.relative_to(self.root).as_posix(),'sha256':trial.sha_bytes(p.read_bytes())}
        manifest={'generation_authorized':True,'config':entry(self.exp/'config.json'),
                  'inputs':[entry(self.root/p) for p in trial.RUNTIME_FILES],
                  'code':[entry(self.exp/'code/run_trial.py'),entry(self.exp/'code/kaggle_transport.py')],
                  'outputs':[entry(self.exp/'request-plan.jsonl')]}
        trial.write_json(self.exp/'manifest.json',manifest)

    def run_fake(self,**kwargs):
        return trial.run_trial(self.exp,run=True,codex_path=self.codex,
            kaggle_generate=self.fake.generate,kaggle_make_context=self.fake.make,
            kaggle_preflight=self.fake.preflight,codex_call=self.fake.codex,
            format_checker=lambda text:['line_break','bold'] if '\n' in text else [],**kwargs)

    def test_default_dry_run_has_no_effect_or_transport(self):
        result=trial.run_trial(self.exp)
        self.assertEqual(result['status'],'DRY_RUN')
        self.assertFalse((self.exp/'collection').exists())
        self.assertEqual(self.fake.make_calls,0)

    def test_authorization_required_before_any_request(self):
        self.config['authorization']['generation']=False
        trial.write_json(self.exp/'config.json',self.config);self.freeze()
        with self.assertRaisesRegex(ValueError,'authorization'):
            self.run_fake()
        self.assertEqual(self.fake.make_calls,0)

    def test_changed_frozen_plan_refuses_before_any_request(self):
        with (self.exp/'request-plan.jsonl').open('a',encoding='utf-8') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            self.run_fake()
        self.assertEqual(self.fake.make_calls,0)

    def test_expired_preflight_refuses_before_any_request(self):
        pre=trial.load(self.exp/'metrics/execution-preflight.json')
        pre['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        trial.write_json(self.exp/'metrics/execution-preflight.json',pre)
        with self.assertRaisesRegex(ValueError,'expired'):
            self.run_fake()
        self.assertEqual(self.fake.make_calls,0)

    def test_live_preflight_before_first_codex_and_not_ready_sends_nothing(self):
        self.fake.live_ready=False
        result=self.run_fake()
        self.assertEqual(result['status'],'STOPPED_PREFLIGHT_OR_CONTROLLER')
        self.assertEqual(result['requests_sent_total'],0)
        self.assertEqual(len(self.fake.codex_calls),0)
        self.assertEqual(len(self.fake.api_calls),0)
        self.assertEqual(self.fake.preflight_calls,1)

    def test_full_success_preserves_identical_repeats_and_format_failures(self):
        result=self.run_fake()
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['saved_total'],80)
        self.assertEqual(len(self.fake.api_calls),40)
        self.assertEqual(len(self.fake.codex_calls),40)
        self.assertEqual(self.fake.make_calls,1)
        self.assertEqual(self.fake.preflight_calls,1)
        records=trial.read_jsonl(self.exp/'collection/records.jsonl')
        self.assertEqual(len({r['text_sha256'] for r in records}),1)
        self.assertTrue(all('line_break' in r['flags'] for r in records))
        self.assertEqual(Counter(r['model_observation_scope'] for r in records),{'requested_only':40,'returned_alias':40})
        audit=trial.load(self.exp/'metrics/collection-audit.json')
        self.assertEqual(audit['status'],'COLLECTED_UNVALIDATED')
        self.assertEqual(audit['identical_text_extra_rows'],79)
        self.assertEqual(Decimal(result['confirmed_cost_usd']),Decimal('0.04'))
        self.assertFalse((self.exp/'collection/run.lock').exists())

    def test_bounded_pause_is_not_complete_and_resume_skips_success(self):
        first=self.run_fake(max_jobs=4)
        self.assertEqual(first['status'],'STOPPED_REQUEST_LIMIT')
        self.assertEqual(first['target_total'],80)
        self.assertEqual(first['saved_total'],4)
        first_ids=[r['job_id'] for r in trial.read_jsonl(self.exp/'collection/records.jsonl')]
        second=self.run_fake(max_jobs=2)
        self.assertEqual(second['saved_total'],6)
        self.assertEqual(second['saved_this_run'],2)
        all_ids=[r['job_id'] for r in trial.read_jsonl(self.exp/'collection/records.jsonl')]
        self.assertEqual(all_ids[:4],first_ids)
        self.assertEqual(len(set(all_ids)),6)

    def assert_all_terminal_files_failed_audit(self,result):
        self.assertEqual(result['status'],'AUDIT_FAILED')
        self.assertEqual(result['saved_total'],80)
        for name in ['state.json','summary.json','run-'+result['run_id']+'.json']:
            saved=trial.load(self.exp/'collection'/name)
            self.assertEqual(saved['status'],'AUDIT_FAILED')
            self.assertEqual(saved['run_id'],result['run_id'])
        self.assertEqual(len(self.fake.api_calls),40)
        self.assertEqual(len(self.fake.codex_calls),40)

    def test_incomplete_audit_never_publishes_complete_terminal(self):
        real_audit=trial.audit_collection
        def incomplete(*args):
            audit=real_audit(*args)
            audit['status']='INCOMPLETE'
            audit['integrity_errors']=['injected audit mismatch']
            trial.write_json(self.exp/'metrics/collection-audit.json',audit)
            return audit
        with patch.object(trial,'audit_collection',side_effect=incomplete):
            result=self.run_fake()
        self.assert_all_terminal_files_failed_audit(result)
        self.assertEqual(result['audit_status'],'INCOMPLETE')

    def test_audit_exception_never_publishes_complete_terminal(self):
        with patch.object(trial,'audit_collection',side_effect=OSError('audit write failed')):
            result=self.run_fake()
        self.assert_all_terminal_files_failed_audit(result)
        self.assertEqual(result['audit_status'],'ERROR')
        self.assertEqual(result['audit_error_type'],'OSError')

    def test_soft_threshold_stops_all_routes_before_next_dispatch(self):
        self.fake.nano_cost=100000000
        result=self.run_fake()
        self.assertEqual(result['status'],'STOPPED_BUDGET')
        self.assertEqual(result['saved_total'],2)
        self.assertEqual(len(self.fake.api_calls),1)
        self.assertEqual(len(self.fake.codex_calls),1)
        self.assertEqual(Decimal(result['confirmed_cost_usd']),Decimal('0.1'))

    def test_missing_cost_stops_without_retry_and_blocks_resume(self):
        self.fake.api_transform=lambda raw:{**raw,'usage':{}}
        result=self.run_fake()
        self.assertEqual(result['status'],'STOPPED_FAILURE')
        self.assertFalse(result['cost_known'])
        self.assertEqual(result['saved_total'],1)
        self.assertEqual(len(self.fake.api_calls),1)
        with self.assertRaises(InspectionRequiredAlias):
            self.run_fake()
        self.assertEqual(len(self.fake.api_calls),1)

    def test_negative_boolean_and_unknown_extra_cost_rejected(self):
        for cost in [
            {'input_tokens_cost_nanodollars':True,'output_tokens_cost_nanodollars':1},
            {'input_tokens_cost_nanodollars':-1,'output_tokens_cost_nanodollars':1},
            {'input_tokens_cost_nanodollars':0,'output_tokens_cost_nanodollars':1,'other_tokens_cost_nanodollars':7},
            {'input_tokens_cost_nanodollars':0,'output_tokens_cost_nanodollars':1,'total_cost_nanodollars':2},
        ]:
            with self.subTest(cost=cost),self.assertRaises(ValueError):
                trial.api_cost({'usage':{'cost':cost}})
        nanos,cost=trial.api_cost({'usage':{'cost':{'input_tokens_cost_nanodollars':2,'output_tokens_cost_nanodollars':3,'total_cost_nanodollars':5,'unused_component':0}}})
        self.assertEqual(nanos,5)
        self.assertEqual(cost,Decimal('0.000000005'))

    def test_model_mismatch_and_truncation_are_failed_but_raw_preserved(self):
        self.fake.api_transform=lambda raw:{**raw,'model':'wrong-model','choices':[{'finish_reason':'length','message':{'content':'partial'}}]}
        result=self.run_fake()
        self.assertEqual(result['status'],'STOPPED_FAILURE')
        failure=result['failures'][0]
        self.assertIn('MODEL_IDENTIFIER_MISMATCH',failure['reasons'])
        self.assertIn('TRUNCATED',failure['reasons'])
        attempt=self.exp/'collection/attempts'/failure['job_id']
        self.assertTrue((attempt/'raw_response.json').exists())
        self.assertEqual((attempt/'final.txt').read_text(encoding='utf-8'),'partial')

    def test_refusal_is_kept_as_observed_result_not_retried(self):
        self.fake.api_transform=lambda raw:{**raw,'choices':[{'finish_reason':'stop','message':{'content':None,'refusal':'I cannot fulfill this request.'}}]}
        result=self.run_fake(max_jobs=2)
        self.assertEqual(result['saved_total'],2)
        row=trial.read_jsonl(self.exp/'collection/records.jsonl')[-1]
        self.assertIn('refusal',row['flags'])
        self.assertEqual(row['text'],'I cannot fulfill this request.')

    def test_transport_error_stores_only_sanitized_fields_and_unknown_cost(self):
        class SafeTransportError(Exception):
            stage='generate'
            error_type='PermissionDeniedError'
            http_status=403
        self.fake.api_exception=SafeTransportError('SECRET_MUST_NOT_BE_STORED')
        result=self.run_fake()
        self.assertFalse(result['cost_known'])
        self.assertEqual(result['failures'][0]['http_status'],403)
        self.assertEqual(result['failures'][0]['stage'],'generate')
        for path in (self.exp/'collection').rglob('*.json'):
            self.assertNotIn('SECRET_MUST_NOT_BE_STORED',path.read_text(encoding='utf-8'))

    def test_existing_lock_is_not_stolen(self):
        collection=self.exp/'collection';collection.mkdir()
        trial.write_json(collection/'run.lock',{'controller_pid':123,'token':'other'},True)
        with self.assertRaises(InspectionRequiredAlias):
            self.run_fake()
        self.assertEqual(trial.load(collection/'run.lock')['token'],'other')
        self.assertEqual(self.fake.make_calls,0)

    def test_heartbeat_updates_while_request_is_inflight(self):
        self.fake.delay=1.4
        observed=[]
        real_write=trial.write_json
        def spy(path,value,exclusive=False):
            if path.name=='state.json' and value.get('active'):
                observed.append(value['observed_at'])
            return real_write(path,value,exclusive)
        with patch.object(trial,'write_json',side_effect=spy):
            result=self.run_fake(max_jobs=1)
        self.assertEqual(result['status'],'STOPPED_REQUEST_LIMIT')
        self.assertGreaterEqual(len(set(observed)),2)

    def test_append_failure_preserves_original_and_prevents_next_request(self):
        with patch.object(trial,'append_jsonl',side_effect=OSError('local disk failure')):
            result=self.run_fake()
        self.assertEqual(result['status'],'LOCAL_ERROR')
        self.assertEqual(len(self.fake.codex_calls),1)
        self.assertEqual(len(self.fake.api_calls),0)
        path=self.exp/'collection/attempts'/self.jobs[0]['job_id']
        self.assertTrue((path/'record.json').exists())
        self.assertTrue((path/'result.json').exists())
        with self.assertRaises(InspectionRequiredAlias):
            self.run_fake()


InspectionRequiredAlias=trial.InspectionRequired


if __name__=='__main__':
    unittest.main(verbosity=2)
