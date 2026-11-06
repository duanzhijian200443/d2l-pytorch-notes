"""One-shot local continuation: await the existing CPU V1, then start one GPU run.

Run in the existing official Kaggle CLI Python environment. Never retries a
save/run request, never submits predictions, and never modifies another job.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CPU='zhjianduan/name-that-llm-cache-preflight-029/1'
GPU='zhjianduan/name-that-llm-cache-capture-029'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def update(**changes):
    path=HERE/'execution-state.json';state=json.loads(path.read_text());state.update(changes,observed_at_utc=now());save(path,state)
    print(json.dumps(changes,ensure_ascii=False),flush=True)
def enum_name(value):return str(getattr(value,'name',value)).split('.')[-1].upper()
def seconds(value):
    return value.total_seconds() if hasattr(value,'total_seconds') else float(str(value).rstrip('s'))


def verify_saved(api,version,expected_dir,out):
    out.mkdir(exist_ok=False)
    api.kernels_pull(GPU+'/'+str(version),str(out),metadata=True,quiet=True)
    expected=json.loads((expected_dir/'kernel-metadata.json').read_text())
    actual=json.loads((out/'kernel-metadata.json').read_text())
    if actual['id']!=GPU:raise ValueError('Wrong saved notebook')
    for key in ('is_private','enable_gpu','enable_tpu','enable_internet','docker_image','machine_shape'):
        if actual.get(key)!=expected.get(key):raise ValueError('Saved setting differs: '+key)
    for key in ('dataset_sources','kernel_sources'):
        if sorted(actual[key])!=sorted(x.rsplit('/',1)[0] for x in expected[key]):raise ValueError('Saved attachment differs: '+key)
    for key in ('model_sources','competition_sources'):
        if sorted(actual[key])!=sorted(expected[key]):raise ValueError('Saved attachment differs: '+key)
    if (out/actual['code_file']).read_text(encoding='utf-8')!=(expected_dir/expected['code_file']).read_text(encoding='utf-8'):
        raise ValueError('Saved source bytes differ after newline normalization')
    save(out/'verified.json',{'status':'PASS','version':version,'verified_at':now(),
        'source_sha256':sha(expected_dir/expected['code_file']),'settings':actual,
        'version_binding_note':'Saved API lists dataset/notebook refs without version; original exact version requests plus CPU byte manifest are retained separately.'})


def require_saved(response):
    if response.error:raise ValueError('Kaggle rejected save: '+response.error)
    for key in ('invalid_dataset_sources','invalid_kernel_sources','invalid_model_sources','invalid_competition_sources'):
        if getattr(response,key,None):raise ValueError('Invalid attachment in save response: '+key)
    if response.ref!=GPU or response.version_number<1:raise ValueError('Wrong save response identity')
    return response.version_number


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-python',required=True)
    args=parser.parse_args()
    with (HERE/'continuation.lock').open('x',encoding='utf-8') as stream:stream.write(now())
    bindings=json.loads((HERE/'continuation-bindings.json').read_text())
    def verify_local():
        for name,digest in bindings.items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen continuation input changed: '+name)
    try:
        verify_local()
        api=KaggleApi();api.authenticate()
        update(status='CPU_PREFLIGHT_RUNNING',continuation_pid=os.getpid(),automatic_gpu_continuation=True,gpu_started=False)
        deadline=time.monotonic()+5400
        while True:
            result=api.kernels_status(CPU);status=enum_name(result.status)
            update(cpu_status=status)
            if status=='COMPLETE':break
            if status not in ('RUNNING','QUEUED'):
                output=HERE/'preflight-output-v1';output.mkdir(exist_ok=True)
                api.kernels_output(CPU,str(output),quiet=True)
                raise RuntimeError('CPU preflight did not complete: '+status)
            if time.monotonic()>deadline:raise TimeoutError('CPU observation limit exceeded; GPU not started')
            time.sleep(45)
        output=HERE/'preflight-output-v1';output.mkdir(exist_ok=True)
        api.kernels_output(CPU,str(output),quiet=True)
        candidates=list(output.rglob('preflight_manifest.json'))
        if len(candidates)!=1:raise ValueError('Expected one actual CPU manifest')
        manifest=candidates[0]
        verify_local()
        quota=api.quota_view().gpu_quota
        remaining=seconds(quota.total_time_allowed)-seconds(quota.time_used)-seconds(quota.time_reserved)
        if remaining<28800:raise ValueError('Less than 8h unreserved Kaggle GPU quota remains')
        update(status='CPU_PREFLIGHT_COMPLETE',cpu_manifest=str(manifest.relative_to(ROOT)),
               cpu_manifest_sha256=sha(manifest),gpu_quota_remaining_seconds=remaining)
        destination=HERE/'launch-v1'
        process=subprocess.run([args.build_python,'-B','-X','utf8',str(HERE/'prepare_capture.py'),
                 '--preflight-manifest',str(manifest),'--destination',str(destination)],cwd=ROOT,
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=120)
        (HERE/'prepare-capture.log').write_text(process.stdout,encoding='utf-8')
        if process.returncode:raise RuntimeError('Capture packaging failed; see prepare-capture.log')
        verify_local()
        kernel=destination/'kernel'
        package=json.loads((destination/'launch-package.json').read_text())
        def verify_generated():
            if sha(kernel/'capture.py')!=package['bootstrap_sha256'] or sha(kernel/'kernel-metadata.json')!=package['metadata_sha256']:
                raise ValueError('Prepared GPU upload files changed')
        verify_generated()
        existing=api.kernels_list(user='zhjianduan',mine=True,page_size=100)
        if existing is None or len(existing)>=100:
            raise ValueError('Cannot establish unused GPU notebook destination from bounded owned list')
        if any(item is not None and item.ref==GPU for item in existing):
            raise ValueError('GPU notebook destination already exists; refusing to overwrite or rerun')
        update(status='GPU_SAVE_IN_FLIGHT',gpu_started=False)
        saved=api.kernels_push(str(kernel),timeout=28800,acc='NvidiaTeslaT4',no_run=True)
        version=require_saved(saved)
        update(status='GPU_SAVED_NOT_RUN',gpu_saved_version=version)
        verify_saved(api,version,kernel,HERE/'gpu-saved-readback')
        verify_local()
        verify_generated()
        # Persist before the only run request. An ambiguous network error is not retried.
        update(status='GPU_RUN_REQUEST_IN_FLIGHT',gpu_run_requests=1,gpu_started=None,
               gpu_may_be_active=True,gpu_start_outcome='UNKNOWN_UNTIL_RESPONSE')
        launched=api.kernels_push(str(kernel),timeout=28800,acc='NvidiaTeslaT4',no_run=False)
        version=require_saved(launched)
        update(status='GPU_SUBMITTED',gpu_started=True,gpu_version=version,gpu_submitted_at_utc=now(),gpu_start_outcome='ACKNOWLEDGED')
        verify_saved(api,version,kernel,HERE/'gpu-run-readback')
        result=api.kernels_status(GPU+'/'+str(version))
        update(status='GPU_'+enum_name(result.status),gpu_status=enum_name(result.status),gpu_readback_verified=True,
               automatic_gpu_continuation=False,continuation_finished=True)
    except BaseException as error:
        # Exception strings may contain request URLs; remove all query parameters.
        message=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[redacted]',str(error))
        state=json.loads((HERE/'execution-state.json').read_text())
        save(HERE/'continuation-failure.json',{'at':now(),'error_type':type(error).__name__,'message':message,
             'last_state':state.get('status'),'gpu_started':state.get('gpu_started',False),'run_request_count':state.get('gpu_run_requests',0)})
        update(status='CONTINUATION_REQUIRES_REVIEW',automatic_gpu_continuation=False,continuation_finished=True)
        raise SystemExit(1)


if __name__=='__main__':main()
