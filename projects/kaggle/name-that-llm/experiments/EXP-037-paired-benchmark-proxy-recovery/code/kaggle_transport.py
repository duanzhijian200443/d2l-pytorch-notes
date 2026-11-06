"""One private Benchmark creation run per approved Kaggle request.

No model inference occurs on import, client creation, or preflight. A generation
uses the task-creation validation run exactly once; it NEVER schedules a separate
task run. Credentials and signed download URLs remain inside official clients.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from io import StringIO
import json
import logging
from pathlib import Path
import re
import time
from types import SimpleNamespace
import zipfile

EXPECTED_MODELS = ('openai/gpt-5.6-luna', 'openai/gpt-5.6-terra')
BASE_INSTRUCTION = "Answer the user's request."
MAX_REQUESTS = 38
POLL_SECONDS = 5
TASK_TIMEOUT_SECONDS = 600


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def payload_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    # All names are local constants and each artifact is written exactly once.
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
        stream.flush()
        import os
        os.fsync(stream.fileno())


class KaggleTransportError(RuntimeError):
    def __init__(self, stage, error_type, http_status=None, cause_chain=None):
        self.stage, self.error_type, self.http_status = stage, error_type, http_status
        self.cause_chain = cause_chain or []
        self.transport_trace = []
        super().__init__(f'{stage}: {error_type}')

    def sanitized(self):
        return {'stage':self.stage, 'error_type':self.error_type,
                'http_status':self.http_status, 'cause_chain':self.cause_chain}


def failure(stage, exc):
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    chain, seen, current = [], set(), exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current)); row = {'type':type(current).__name__}
        for field in ('errno', 'verify_code'):
            value = getattr(current, field, None)
            if type(value) is int: row[field] = value
        chain.append(row); current = current.__cause__ or current.__context__
    return KaggleTransportError(stage, type(exc).__name__, status if type(status) is int else None, chain)


class CloudContext:
    def __init__(self, api, types):
        self._api, self._types = api, types
        self._ready, self._requests_sent = False, 0
        self._allowed_models = frozenset()

    def __repr__(self): return '<PrivateBenchmarkContext credentials=official-client-only>'
    def __getstate__(self): raise TypeError('Do not serialize authenticated context')
    def close(self): self._ready = False


def make_client():
    for name in ('kaggle','kagglesdk','requests','urllib3','openai','httpx','httpcore'):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.benchmarks.types.benchmark_tasks_api_service import (
                ApiCreateBenchmarkTaskRequest, ApiGetBenchmarkTaskRequest,
                ApiDownloadBenchmarkTaskRunOutputRequest, ApiTaskCreationValidationConfig, ApiValidationStage)
            from kagglesdk.benchmarks.types.benchmark_types import BenchmarkTaskValidationStage
            from kagglesdk.benchmarks.types.benchmarks_api_service import ApiListBenchmarkModelsRequest
            api = KaggleApi(); api.authenticate()
        # Disable the CLI convenience wrapper, including its mutation retry path.
        api.with_retry = lambda method: method
        types = SimpleNamespace(Create=ApiCreateBenchmarkTaskRequest, Get=ApiGetBenchmarkTaskRequest,
            Download=ApiDownloadBenchmarkTaskRunOutputRequest, Validation=ApiTaskCreationValidationConfig,
            Stage=ApiValidationStage, StageType=BenchmarkTaskValidationStage, Models=ApiListBenchmarkModelsRequest)
        return CloudContext(api, types)
    except (Exception, SystemExit) as exc:
        raise failure('make_client', exc) from None


def preflight(context, minimum_free_usd='0.25'):
    context._ready = False
    result = {'observed_at':utc_now(), 'ready':False, 'model_requests_sent':context._requests_sent,
              'transport_route':'private_benchmark_creation_validation_run', 'max_retries':0,
              'generation_probe_sent':False, 'expected_models':list(EXPECTED_MODELS)}
    try:
        required = Decimal(str(minimum_free_usd))
        if not required.is_finite() or required < 0: raise ValueError('InvalidRequiredQuota')
        quota = context._api.benchmarks_quota(); balances = {}
        for row in quota.quota_balances or []:
            name = getattr(row.refill_period, 'name', '').lower()
            if name not in ('daily','monthly'): continue
            used, total = Decimal(str(row.quota_used)), Decimal(str(row.total_quota_allowed))
            if not all(x.is_finite() and x >= 0 for x in (used,total)): raise ValueError('InvalidQuota')
            balances[name] = {'used_usd':str(used), 'total_usd':str(total), 'remaining_usd':str(max(Decimal(0),total-used))}
        result['quota'] = balances
        if set(balances) != {'daily','monthly'} or any(Decimal(v['remaining_usd']) < required for v in balances.values()):
            result['stop_reason'] = 'INSUFFICIENT_VERIFIED_FREE_QUOTA'; return result
        matched, slugs, token, pages, tokens = {}, set(), '', 0, set()
        with context._api.build_kaggle_client() as kaggle:
            while True:
                req = context._types.Models()
                if token: req.page_token = token
                response = kaggle.benchmarks.benchmarks_api_client.list_benchmark_models(req)
                pages += 1
                for model in response.benchmark_models or []:
                    version = model.version
                    slugs.add(version.slug)
                    proxy_slug = getattr(version, 'model_proxy_slug', '')
                    if proxy_slug in EXPECTED_MODELS and getattr(version, 'allow_model_proxy', False) is True:
                        matched[proxy_slug] = {'catalog_slug':version.slug, 'model_proxy_slug':proxy_slug,
                                              'allow_model_proxy':True}
                token = response.next_page_token
                if not token: break
                if pages >= 10 or token in tokens: raise ValueError('CatalogPaginationIncomplete')
                tokens.add(token)
        result.update(listed_models=sorted(slugs), matched_requested_models=sorted(matched), model_listing_matches=matched)
        if set(matched) != set(EXPECTED_MODELS):
            result['stop_reason']='TARGET_MODEL_NOT_LISTED'; return result
        context._ready = True; context._allowed_models = frozenset(matched)
        result.update(ready=True, inference_access='unknown_until_authorized_cloud_request',
                      low_reasoning_validation='fixed request low; actual acceptance checked in generated response')
        return result
    except (Exception, SystemExit) as exc:
        result['error'] = failure('preflight',exc).sanitized()
        result['stop_reason']='PREFLIGHT_NOT_ESTABLISHED'
        return result


def validate_payload(context, payload, job):
    keys = {'model','messages','reasoning_effort','max_completion_tokens','stream','n'}
    messages = payload.get('messages')
    if (set(payload) != keys or payload.get('model') not in context._allowed_models
        or payload.get('reasoning_effort') != 'low' or payload.get('max_completion_tokens') != 1024
        or payload.get('stream') is not False or payload.get('n') != 1
        or not isinstance(messages,list) or len(messages) != 2
        or messages[0] != {'role':'system','content':BASE_INSTRUCTION}
        or set(messages[1]) != {'role','content'} or messages[1]['role'] != 'user'
        or not isinstance(messages[1]['content'],str) or not messages[1]['content'].strip()
        or job.get('api_payload') != payload or job.get('route') != 'kaggle_model_proxy'
        or not re.fullmatch('[a-f0-9]{64}',job.get('job_id',''))):
        raise KaggleTransportError('generate','PayloadDiffersFromApprovedContract')


def cloud_source(payload, job_id, task_slug):
    encoded = payload_bytes(payload).decode('utf-8')
    return '''# %%
import os
os.environ['ENABLE_LOCAL_CACHING'] = 'false'
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import kaggle_benchmarks as kbench
from openai import OpenAI, DefaultHttpxClient
PAYLOAD = json.loads(%r)
JOB_ID = %r
PAYLOAD_SHA256 = %r

# %%
@kbench.task(name=%r)
def collect_one(llm):
    root = Path('/kaggle/working')
    marker = root / ('dispatch-' + JOB_ID + '.started')
    try:
        with marker.open('x') as f:
            f.write(JOB_ID)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        return None
    result = {'job_id':JOB_ID, 'payload_sha256':PAYLOAD_SHA256,
              'generation_requests':0, 'started_at':datetime.now(timezone.utc).isoformat(),
              'status':'STARTED', 'model_requested':PAYLOAD['model'],
              'sdk_max_retries':0, 'response_cache':False}
    client = None
    try:
        actor = kbench.llms[PAYLOAD['model']]
        client = OpenAI(api_key=actor.client.api_key, base_url=actor.client.base_url,
                        organization='', project='', max_retries=0, timeout=180,
                        http_client=DefaultHttpxClient(timeout=180, follow_redirects=False))
        # Keep unrelated OpenAI account routing headers absent, including empty
        # values; constructor blanks prevent ambient env values being inherited.
        client.organization = None
        client.project = None
        result['generation_requests'] = 1
        (root/'cloud-dispatch.json').write_text(json.dumps(result,indent=2))
        response = client.chat.completions.create(**PAYLOAD)
        raw = response.model_dump(mode='json')
        data = (json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False)+'\\n').encode('utf-8')
        with (root/'raw-response.json').open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        result.update(status='RECEIVED', raw_response_sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        result.update(status='FAILED_OR_UNKNOWN', error_type=type(exc).__name__)
        status = getattr(exc,'status_code',None)
        if type(status) is int: result['http_status'] = status
        chain, seen, current = [], set(), exc
        while current is not None and id(current) not in seen and len(chain)<8:
            seen.add(id(current))
            item = {'type':type(current).__name__}
            for name in ('errno','verify_code'):
                value = getattr(current,name,None)
                if type(value) is int: item[name] = value
            chain.append(item)
            current = current.__cause__ or current.__context__
        result['cause_chain'] = chain
    finally:
        result['ended_at'] = datetime.now(timezone.utc).isoformat()
        with (root/'cloud-result.json').open('x') as f:
            json.dump(result,f,indent=2)
            f.flush()
            os.fsync(f.fileno())
        if client is not None: client.close()
    return None

# %%
collect_one.run(kbench.llm)
''' % (encoded, job_id, sha(payload_bytes(payload)), task_slug)


def _enum_name(value):
    return getattr(value,'name',str(value))


def _assert_task_identity(task, slug, version=1):
    if (task.slug.task_slug != slug or task.slug.version_number != version
        or task.is_public is not False or task.is_backing_notebook_published is True):
        raise KaggleTransportError('task','UnexpectedTaskIdentityOrVisibility')


def read_output_zip(path, job_id, payload_sha):
    with zipfile.ZipFile(path) as archive:
        # Do not extract arbitrary notebook artifacts, paths, or possible secrets.
        names = archive.namelist()
        selected = {}
        for wanted in ('cloud-result.json','raw-response.json'):
            matches = [n for n in names if n == wanted or n.endswith('/'+wanted)]
            if len(matches) != 1:
                if wanted == 'raw-response.json' and not matches: continue
                raise KaggleTransportError('download','OutputFileMissingOrDuplicated')
            if archive.getinfo(matches[0]).file_size > 5_000_000:
                raise KaggleTransportError('download','OutputTooLarge')
            selected[wanted] = archive.read(matches[0])
    result = json.loads(selected['cloud-result.json'])
    if result.get('job_id') != job_id or result.get('payload_sha256') != payload_sha:
        raise KaggleTransportError('download','CloudResultIdentityMismatch')
    if result.get('generation_requests') not in (0,1):
        raise KaggleTransportError('download','CloudDispatchCountMismatch')
    raw = None
    if 'raw-response.json' in selected:
        if sha(selected['raw-response.json']) != result.get('raw_response_sha256'):
            raise KaggleTransportError('download','RawResponseHashMismatch')
        raw = json.loads(selected['raw-response.json'])
    return result, raw


def generate(context, payload, *, attempt_dir, job):
    """One task-creation POST; every other remote operation is read-only."""
    if not context._ready: raise KaggleTransportError('generate','PreflightNotReady')
    if context._requests_sent >= MAX_REQUESTS: raise KaggleTransportError('generate','KaggleRequestLimitReached')
    validate_payload(context,payload,job)
    attempt = Path(attempt_dir).resolve()
    if not attempt.is_dir(): raise KaggleTransportError('generate','MissingDurableAttemptDirectory')
    ident = job['job_id']; task_slug = 'ntl-exp037-'+ident[:32]
    payload_sha = sha(payload_bytes(payload))
    source = cloud_source(payload,ident,task_slug)
    with (attempt/'cloud-source.py').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(source)
    write_json(attempt/'cloud-submit-intent.json',{'job_id':ident,'task_slug':task_slug,
        'payload_sha256':payload_sha,'source_sha256':sha(source.encode()),'observed_at':utc_now(),
        'private_only':True,'mutation_retries':0,'separate_task_run':False})
    api, types = context._api, context._types
    started = time.monotonic()
    stage = 'task_preexistence'
    try:
        with api.build_kaggle_client() as kaggle:
            service = kaggle.benchmarks.benchmark_tasks_api_client
            check = types.Get(); check.slug = api._make_task_slug(task_slug)
            try:
                existing = service.get_benchmark_task(check)
            except Exception as exc:
                status = getattr(getattr(exc,'response',None),'status_code',None)
                if status != 404: raise
            else:
                raise KaggleTransportError('task','TaskAlreadyExistsDoNotRepush')
            req = types.Create(); req.slug = task_slug; req.text = api._convert_py_to_notebook(source)
            validation = types.Validation(); one = types.Stage()
            one.type = types.StageType.BENCHMARK_TASK_VALIDATION_STAGE_DEFAULT_CANDIDATE
            one.default_model_version_slug = payload['model'].removeprefix('openai/')
            validation.validation_stages = [one]; req.validation_config = validation
            stage = 'create_task'; context._requests_sent += 1
            response = service.create_benchmark_task(req)  # NEVER use with_retry here.
            if getattr(response,'error',None): raise KaggleTransportError(stage,'TaskCreationRejected')
            _assert_task_identity(response,task_slug)
            write_json(attempt/'cloud-task.json',{'task_slug':task_slug,'version':1,
                'source_kernel_id':response.source_kernel_id,'is_public':response.is_public,
                'is_backing_notebook_published':response.is_backing_notebook_published,'created_at':utc_now()})
            stage = 'poll_creation'; task = response
            while True:
                _assert_task_identity(task,task_slug)
                state = _enum_name(task.creation_state)
                if state.endswith(('_COMPLETED','_ERRORED','_VALIDATION_FAILED')): break
                if time.monotonic()-started >= TASK_TIMEOUT_SECONDS:
                    raise KaggleTransportError(stage,'CloudTaskTimeoutOutcomeUnknown')
                time.sleep(POLL_SECONDS)
                query = types.Get(); query.slug = api._make_task_slug(task_slug); query.slug.version_number = 1
                task = service.get_benchmark_task(query)
            stage = 'list_creation_runs'
            runs = api._fetch_task_runs(kaggle,task_slug)
            if len(runs) != 1:
                raise KaggleTransportError(stage,'ExpectedExactlyOneCreationRun')
            run = runs[0]
            # The list-runs API omits version_number (SDK reads it as 0).
            # A missing version is bound by this never-before-existing slug,
            # the verified task version 1, and its sole creation run. A returned
            # nonzero different version is still rejected.
            if run.task_slug.task_slug != task_slug or run.task_slug.version_number not in (0,1):
                raise KaggleTransportError(stage,'UnexpectedCreationRunIdentity')
            run_state = _enum_name(run.state)
            if not run_state.endswith(('_COMPLETED','_ERRORED')):
                raise KaggleTransportError(stage,'CreationRunNotTerminal')
            write_json(attempt/'cloud-run.json',{'task_slug':task_slug,'version':1,'run_id':run.id,
                'creation_state':state,'run_state':run_state,'candidate_model':run.model_version_slug,
                'run_task_version_number':run.task_slug.version_number,
                'version_binding':'fresh_unique_task_version1_and_sole_creation_run',
                'actual_requested_model':payload['model'],'separate_task_run':False,'observed_at':utc_now()})
            stage = 'download_output'; request = types.Download(); request.run_id = run.id; request.include_source = False
            download = service.download_benchmark_task_run_output(request)
            archive = attempt/'cloud-output.zip'
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                api.download_file(download,str(archive),kaggle.http_client(),quiet=True,max_retries=0,timeout=60)
            result, raw = read_output_zip(archive,ident,payload_sha)
            write_json(attempt/'cloud-result.json',result)
            if result.get('status') != 'RECEIVED' or result.get('generation_requests') != 1 or raw is None:
                raise KaggleTransportError('cloud_generate',result.get('error_type','CloudResultNotReceived'),
                    result.get('http_status'),result.get('cause_chain'))
            if result.get('model_requested') != payload['model'] or result.get('sdk_max_retries') != 0 or result.get('response_cache') is not False:
                raise KaggleTransportError('cloud_generate','CloudProtocolMismatch')
            return raw
    except KaggleTransportError:
        context._ready = False
        raise
    except (Exception,SystemExit) as exc:
        context._ready = False
        raise failure(stage,exc) from None
