"""Offline fake-service and generated-source tests. No auth or network."""
from __future__ import annotations
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
import zipfile

CODE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud_transport_under_test',CODE/'kaggle_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


class Missing(Exception):
    response=NS(status_code=404)


def enum(name):return NS(name=name)


class FakeApi:
    def __init__(self,payload,job):
        self.payload,self.job=payload,job
        self.creates=[];self.gets=0;self.create_error=None;self.exists=False
        self.bad_identity=False;self.extra_runs=False;self.output_failure=False;self.run_version=1
        self.raw={'id':'raw-unique','model':payload['model'],'choices':[{'finish_reason':'stop','message':{'content':'Original **text**.\nSecond line.'}}],
                  'usage':{'cost':{'input_tokens_cost_nanodollars':123,'output_tokens_cost_nanodollars':456}},'unknown_extra':'preserve'}
        self.benchmarks=NS(benchmark_tasks_api_client=self,benchmarks_api_client=self)
        self.slug=None

    def build_kaggle_client(self):return nullcontext(self)
    def http_client(self):return object()
    def _make_task_slug(self,slug):return NS(task_slug=slug,version_number=0,owner_slug='fixture')
    def _convert_py_to_notebook(self,source):return source
    def task(self):
        return NS(slug=NS(task_slug=self.slug,version_number=1),creation_state=enum('BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED'),
                  is_public=False,is_backing_notebook_published=False,source_kernel_id=123,error=None)
    def get_benchmark_task(self,request):
        self.gets+=1
        if not self.creates and not self.exists:raise Missing('no such private task')
        self.slug=request.slug.task_slug;return self.task()
    def create_benchmark_task(self,request):
        self.creates.append(request);self.slug=request.slug
        if self.create_error:raise self.create_error
        return self.task()
    def _fetch_task_runs(self,client,slug):
        r=NS(id=999,task_slug=NS(task_slug=slug,version_number=self.run_version),state=enum('BENCHMARK_TASK_RUN_STATE_COMPLETED'),model_version_slug='gpt-5.6-luna')
        return [r,r] if self.extra_runs else [r]
    def download_benchmark_task_run_output(self,request):
        assert request.run_id==999 and request.include_source is False
        return object()
    def download_file(self,response,path,client,**kwargs):
        assert kwargs['quiet'] is True and kwargs['max_retries']==0
        raw=(json.dumps(self.raw,indent=2)+'\n').encode()
        result={'job_id':self.job['job_id'] if not self.bad_identity else 'f'*64,
                'payload_sha256':t.sha(t.payload_bytes(self.payload)),'generation_requests':1,
                'status':'FAILED_OR_UNKNOWN' if self.output_failure else 'RECEIVED',
                'model_requested':self.payload['model'],'sdk_max_retries':0,'response_cache':False,
                'raw_response_sha256':t.sha(raw),'error_type':'NotFoundError','http_status':404}
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('cloud-result.json',json.dumps(result))
            if not self.output_failure:z.writestr('raw-response.json',raw)
            z.writestr('../../must-not-extract.txt','NEVER EXTRACT')
    def benchmarks_quota(self):
        return NS(quota_balances=[NS(refill_period=enum(name),quota_used=0,total_quota_allowed=10) for name in ['DAILY','MONTHLY']])
    def list_benchmark_models(self,request):
        return NS(benchmark_models=[NS(version=NS(slug=x.removeprefix('openai/'),model_proxy_slug=x,allow_model_proxy=True)) for x in t.EXPECTED_MODELS],next_page_token='')


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.payload={'model':'openai/gpt-5.6-luna','messages':[{'role':'system','content':t.BASE_INSTRUCTION},{'role':'user','content':'Keep quotes " and Unicode 雨 and exact newlines.\n'}],
                      'reasoning_effort':'low','max_completion_tokens':1024,'stream':False,'n':1}
        self.job={'job_id':'a'*64,'route':'kaggle_model_proxy','api_payload':self.payload}
        self.api=FakeApi(self.payload,self.job)
        types=NS(Create=NS,Get=NS,Download=NS,Validation=NS,Stage=NS,Models=NS,
                 StageType=NS(BENCHMARK_TASK_VALIDATION_STAGE_DEFAULT_CANDIDATE='DEFAULT_CANDIDATE'))
        self.ctx=t.CloudContext(self.api,types)
        self.ctx._ready=True;self.ctx._allowed_models=frozenset(t.EXPECTED_MODELS)

    def tearDown(self):self.tmp.cleanup()
    def call(self):return t.generate(self.ctx,self.payload,attempt_dir=self.path,job=self.job)

    def test_source_compiles_preserves_payload_and_has_one_generation_site(self):
        source=t.cloud_source(self.payload,self.job['job_id'],'private-test')
        tree=ast.parse(source);compile(source,'cloud-source','exec')
        literal=next(n.value.args[0].value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='PAYLOAD')
        self.assertEqual(json.loads(literal),self.payload)
        self.assertEqual(source.count('client.chat.completions.create('),1)
        self.assertIn("marker.open('x')",source)
        self.assertIn("os.environ['ENABLE_LOCAL_CACHING'] = 'false'",source)
        self.assertNotIn('judge',source)

    def test_one_create_one_raw_response_no_arbitrary_zip_extraction(self):
        raw=self.call();self.assertEqual(raw,self.api.raw)
        self.assertEqual(len(self.api.creates),1)
        req=self.api.creates[0]
        self.assertEqual(len(req.validation_config.validation_stages),1)
        self.assertEqual(req.validation_config.validation_stages[0].default_model_version_slug,'gpt-5.6-luna')
        self.assertEqual(self.ctx._requests_sent,1)
        self.assertTrue((self.path/'cloud-result.json').exists())
        self.assertFalse((self.path/'must-not-extract.txt').exists())

    def test_existing_task_never_repushed(self):
        self.api.exists=True
        with self.assertRaisesRegex(t.KaggleTransportError,'TaskAlreadyExists'):self.call()
        self.assertEqual(self.api.creates,[])

    def test_failed_create_no_retry_and_safe_chain(self):
        error=RuntimeError('SECRET bearer never stored');error.__cause__=OSError(10054,'SECRET')
        self.api.create_error=error
        with self.assertRaises(t.KaggleTransportError) as caught:self.call()
        self.assertEqual(len(self.api.creates),1)
        self.assertFalse(self.ctx._ready)
        safe=json.dumps(caught.exception.sanitized());self.assertNotIn('SECRET',safe)
        self.assertIn('10054',safe)

    def test_repeated_local_attempt_fails_before_second_create(self):
        self.call()
        with self.assertRaises(FileExistsError):self.call()
        self.assertEqual(len(self.api.creates),1)

    def test_count_cap_prevents_remote_call(self):
        self.ctx._requests_sent=38
        with self.assertRaisesRegex(t.KaggleTransportError,'RequestLimit'):self.call()
        self.assertEqual(self.api.creates,[])

    def test_wrong_result_identity_rejected(self):
        self.api.bad_identity=True
        with self.assertRaisesRegex(t.KaggleTransportError,'IdentityMismatch'):self.call()

    def test_multiple_server_runs_stop(self):
        self.api.extra_runs=True
        with self.assertRaisesRegex(t.KaggleTransportError,'ExactlyOne'):self.call()

    def test_omitted_run_version_uses_unique_task_binding(self):
        self.api.run_version=0
        self.assertEqual(self.call(),self.api.raw)
        meta=json.loads((self.path/'cloud-run.json').read_text())
        self.assertEqual(meta['run_task_version_number'],0)
        self.assertEqual(meta['version'],1)

    def test_nonzero_mismatched_run_version_rejected(self):
        self.api.run_version=2
        with self.assertRaisesRegex(t.KaggleTransportError,'RunIdentity'):self.call()

    def test_cloud_failure_retained_without_new_attempt(self):
        self.api.output_failure=True
        with self.assertRaisesRegex(t.KaggleTransportError,'NotFoundError'):self.call()
        result=json.loads((self.path/'cloud-result.json').read_text())
        self.assertEqual(result['status'],'FAILED_OR_UNKNOWN');self.assertEqual(len(self.api.creates),1)

    def test_preflight_does_not_create_or_infer(self):
        r=t.preflight(self.ctx)
        self.assertTrue(r['ready']);self.assertEqual(r['model_requests_sent'],0)
        self.assertEqual(self.api.creates,[])

    def test_tampered_protocol_rejected_before_create(self):
        self.payload['reasoning_effort']='high'
        with self.assertRaisesRegex(t.KaggleTransportError,'Contract'):self.call()
        self.assertEqual(self.api.creates,[])


if __name__=='__main__':unittest.main()
