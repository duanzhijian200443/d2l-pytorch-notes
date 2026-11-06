"""Read-only official Kaggle backend. No push, create, cancel or submit APIs."""
import argparse
from pathlib import Path
import re
from urllib.parse import urlparse,urljoin
from workbench_core import *

MAX_FILE=16*1024**2
MAX_TOTAL=64*1024**2
MAX_LOG=2*1024**2
EXACT_FILES={'submission.parquet','submission_receipt.json','input_preflight.json','availability.json',
             'a1_synthetic_parity.json','reproduction_environment.json','add120_preflight.json',
             'comp_execution_receipt.json','main_execution_receipt.json','a1_training_receipt.json','comp_meta.json'}

def valid_url(url):
    p=urlparse(url)
    require(p.scheme=='https' and not p.username and not p.password and p.port in (None,443),'UNSAFE_DOWNLOAD_URL')
    h=p.hostname or ''
    require(h in {'storage.googleapis.com','www.kaggle.com'} or h.endswith('.kaggleusercontent.com') or h.endswith('.googleusercontent.com'),'UNEXPECTED_DOWNLOAD_HOST')

def http(url,method='GET'):
    import requests
    for _ in range(5):
        valid_url(url)
        response=requests.request(method,url,stream=True,timeout=(20,90),allow_redirects=False)
        if response.status_code in (301,302,303,307,308):
            location=response.headers.get('Location');response.close();require(bool(location),'EMPTY_REDIRECT')
            url=urljoin(url,location);continue
        response.raise_for_status();return response
    raise GateError('TOO_MANY_REDIRECTS')

def stream_file(chunks,dest,expected,limit):
    require(0<expected<=limit,'ARTIFACT_SIZE_EXCEEDS_LIMIT')
    part=Path(str(dest)+'.part');total=0
    with part.open('xb') as f:
        for block in chunks:
            if not block:continue
            total+=len(block);require(total<=expected and total<=limit,'STREAM_EXCEEDS_DECLARED_LIMIT')
            f.write(block)
        f.flush();os.fsync(f.fileno())
    require(total==expected,'TRUNCATED_DOWNLOAD')
    require(not dest.exists(),'OUTPUT_ALREADY_EXISTS');part.replace(dest)
    return total

def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a=KaggleApi();a.authenticate();return a

def status(api,p):
    remote=p['remote'];rows=[];found=False
    for page in range(1,6):
        chunk=api.competition_submissions('name-that-llm',page_number=page,page_size=100)
        require(chunk is not None,'SUBMISSION_LIST_UNAVAILABLE')
        for item in chunk:
            if item is None:continue
            d=item.to_dict();d['status']=str(item.status).split('.')[-1];rows.append(d)
            if d.get('ref')==remote['submission_id']:found=True
        if found or len(chunk)<100:break
    result=parse_submission(rows,remote)
    # Notebook commit completion is separate from competition completion.
    k=api.kernels_status(remote['ref']+'/'+str(remote['version']))
    result.update(commit_status=str(k.status).split('.')[-1],observed_at=now(),scope='formal_submission_status_and_separate_commit_status')
    return result

def frozen_source(api,p):
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
    remote=p['remote'];owner,slug=remote['ref'].split('/')
    q=ApiGetKernelRequest();q.user_name=owner;q.kernel_slug=slug;q.version_label='v'+str(remote['version'])
    with api.build_kaggle_client() as client:r=client.kernels.kernels_api_client.get_kernel(q)
    require(r.metadata.ref==remote['ref'] and r.metadata.current_version_number==remote['version'],'REMOTE_NOTEBOOK_IDENTITY_MISMATCH')
    require(r.metadata.is_private and not r.metadata.enable_internet,'REMOTE_PRIVACY_OR_NETWORK_CHANGED')
    expected=code_hash(inside(remote['source_path']).read_text(encoding='utf-8'))
    require(code_hash(r.blob.source)==expected,'REMOTE_NOTEBOOK_CODE_CHANGED')
    return {'code_sha256':expected,'private':True,'internet':False,'version':remote['version']}

def collect(api,p,dest):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    remote=p['remote'];identity=frozen_source(api,p)
    committed=api.kernels_status(remote['ref']+'/'+str(remote['version']))
    require(str(committed.status).split('.')[-1]=='COMPLETE','COMMIT_NOT_COMPLETE')
    owner,slug=remote['ref'].split('/');token=None;tokens=set();files=[];seen=set();log=''
    for page in range(10):
        q=ApiListKernelSessionOutputRequest();q.user_name=owner;q.kernel_slug=slug;q.version_label='v'+str(remote['version']);q.page_size=100
        if token:q.page_token=token
        with api.build_kaggle_client() as client:r=client.kernels.kernels_api_client.list_kernel_session_output(q)
        if page==0:log=r.log or ''
        require(len(log.encode())<=MAX_LOG,'COMMIT_LOG_TOO_LARGE')
        for f in r.files or []:
            name=f.file_name
            require(name not in seen,'DUPLICATE_OUTPUT_LISTING');seen.add(name)
            if name not in EXACT_FILES:continue
            valid_url(f.url)
            match=re.search(r'/kf/(\d+)/',f.url)
            require(match is not None and int(match.group(1))==remote['script_version_id'],'OUTPUT_SCRIPT_VERSION_MISMATCH')
            files.append((name,f.url))
        token=r.next_page_token
        if not token:break
        require(token not in tokens,'REPEATED_OUTPUT_PAGE');tokens.add(token)
    require(not token,'OUTPUT_PAGINATION_INCOMPLETE')
    require({'submission.parquet','submission_receipt.json','input_preflight.json','reproduction_environment.json','a1_synthetic_parity.json'}.issubset({n for n,u in files}),'REQUIRED_COMMIT_OUTPUTS_MISSING')
    # Validate every permitted file size before fetching any body.
    sized=[];total=0
    for name,url in files:
        # Signed GET URLs may reject HEAD. Streaming GET gives headers without
        # materializing the body; close it before the bounded body download.
        response=http(url)
        try:size=int(response.headers.get('Content-Length','-1'))
        finally:response.close()
        require(0<size<=MAX_FILE,'ARTIFACT_SIZE_EXCEEDS_LIMIT');total+=size
        require(total<=MAX_TOTAL,'COLLECTION_SIZE_EXCEEDS_LIMIT');sized.append((name,url,size))
    folder=dest/'files';folder.mkdir(exist_ok=False);entries=[]
    for name,url,size in sized:
        response=http(url)
        try:stream_file(response.iter_content(chunk_size=1024*256),folder/name,size,MAX_FILE)
        finally:response.close()
        entries.append({'name':name,'bytes':size,'sha256':sha(folder/name)})
    # Raw logs may contain secrets. Retain only known safe event summaries.
    log_events=[]
    for phrase in ['FAST_COMMIT True','INPUT_PREFLIGHT PASS','EXP046 INPUT/ENVIRONMENT/SYNTHETIC PARITY PASS','Traceback (most recent call last)']:
        if phrase in log:log_events.append(phrase)
    result={'status':'COLLECTED_COMMIT_OUTPUTS','scope':'saved_notebook_outputs_not_hidden_rerun_artifacts',
            'observed_at':now(),'query_ok':True,'ref':remote['ref'],'version':remote['version'],'script_version_id':remote['script_version_id'],
            'code_identity':identity,'files':entries,'files_path':folder.relative_to(ROOT).as_posix(),'bytes':total,
            'safe_log_events':log_events,'raw_logs_saved':False,'signed_urls_saved':False}
    verify_files(result,folder);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['status','collect']);parser.add_argument('experiment');parser.add_argument('destination');args=parser.parse_args()
    dest=inside(args.destination);require(dest.is_relative_to(WORK/'observations'),'INVALID_OBSERVATION_DESTINATION')
    try:
        _,p,_=profile(args.experiment);require(bool(p.get('remote')),'EXPERIMENT_HAS_NO_SAVED_REMOTE_VERSION')
        api=api_client();result=status(api,p) if args.operation=='status' else collect(api,p,dest)
    except Exception as error:result=safe_failure(error)
    atomic_json(dest/'backend.json',result,exclusive=True)
    print(result.get('status','UNKNOWN'))
    return 1 if result['status']=='ERROR' else 0

if __name__=='__main__':raise SystemExit(main())
