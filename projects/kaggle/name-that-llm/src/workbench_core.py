"""Small, standard-library contracts shared by the experiment workbench."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import uuid
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/EXP-051-experiment-workbench'
CONFIG=ROOT/'configs/experiment-workbench.json'

class GateError(ValueError):
    pass

def require(ok,code):
    if not ok:raise GateError(code)

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def inside(value,root=ROOT):
    require(isinstance(value,str) and value and not PureWindowsPath(value).anchor,'PATH_MUST_BE_PROJECT_RELATIVE')
    p=PurePosixPath(value.replace('\\','/'))
    require(not p.is_absolute() and '..' not in p.parts and ':' not in value,'UNSAFE_PROJECT_PATH')
    result=(root/p).resolve()
    require(result.is_relative_to(root.resolve()),'PATH_ESCAPES_PROJECT')
    return result

def atomic_json(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text);f.flush();os.fsync(f.fileno())
    else:
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('x',encoding='utf-8',newline='\n') as f:f.write(text);f.flush();os.fsync(f.fileno())
        temporary.replace(path)

def new_observation(ident,operation):
    require(re.fullmatch(r'EXP-\d{3}',ident) is not None,'INVALID_EXPERIMENT_ID')
    require(operation in {'status','collect','verify','prepare','run-local','doctor','check'},'INVALID_OPERATION')
    p=WORK/'observations'/ident/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+operation+'-'+uuid.uuid4().hex[:6])
    p.mkdir(parents=True,exist_ok=False)
    return p

@contextmanager
def local_lock(ident):
    require(re.fullmatch(r'EXP-\d{3}',ident) is not None,'INVALID_EXPERIMENT_ID')
    p=WORK/'locks'/(ident+'.lock');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as stream:
        if stream.seek(0,2)==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:raise GateError('LOCAL_OPERATION_ALREADY_ACTIVE') from None
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def profile(key,config_path=CONFIG):
    cfg=read(config_path);matches=[p for p in cfg['experiments'] if key==p['id'] or key in p.get('aliases',[])]
    require(len(matches)==1,'UNKNOWN_EXPERIMENT')
    p=matches[0];require(re.fullmatch(r'EXP-\d{3}',p['id']) is not None,'INVALID_REGISTRY_ID')
    r=read(inside('tracking/records/experiments/'+p['id']+'.json'))
    require(r['id']==p['id'],'REGISTRY_RECORD_MISMATCH')
    source=p.get('remote')
    if source:
        require(source['ref'].startswith('zhjianduan/') and len(source['ref'].split('/'))==2,'UNEXPECTED_REMOTE_OWNER')
        require(type(source['version']) is int and source['version']>0,'MISSING_FIXED_VERSION')
        require(type(source['script_version_id']) is int and type(source['submission_id']) is int,'MISSING_FIXED_SUBMISSION_BINDING')
        require(r['source']['notebook_ref']==source['ref'] and r['source']['notebook_version']==source['version']
                and r['source']['script_version_id']==source['script_version_id'],'SOURCE_BINDING_CHANGED')
        require(r['submission']['id']==source['submission_id'] and r['submission']['script_version_id']==source['script_version_id'],'SUBMISSION_BINDING_CHANGED')
        require(sha(inside(source['source_path']))==source['source_sha256'],'FROZEN_SOURCE_CHANGED')
    return cfg,p,r

def code_hash(source):
    n=json.loads(source)
    require(n.get('nbformat')==4,'INVALID_NOTEBOOK')
    code='\n\n'.join((''.join(c['source']) if isinstance(c['source'],list) else c['source']).replace('\r\n','\n') for c in n['cells'] if c['cell_type']=='code')
    return hashlib.sha256(code.encode()).hexdigest()

def parse_submission(rows,remote):
    hits=[r for r in rows if r.get('ref')==remote['submission_id']]
    if not hits:return {'query_ok':True,'status':'UNKNOWN_NOT_RETURNED','score':None,'version_verified':False}
    require(len(hits)==1,'DUPLICATE_SUBMISSION_ROWS')
    row=hits[0];match=re.search(r'(?:[?&])scriptVersionId=(\d+)(?:&|$)',row.get('url',''))
    if not match:return {'query_ok':True,'status':'UNKNOWN_VERSION_NOT_RETURNED','score':None,'version_verified':False}
    require(int(match.group(1))==remote['script_version_id'],'REMOTE_SUBMISSION_VERSION_MISMATCH')
    expected='/code/'+remote['ref']
    require(row['url'].split('?')[0] in (expected,'https://www.kaggle.com'+expected),'REMOTE_SUBMISSION_REF_MISMATCH')
    status=row.get('status','UNKNOWN_STATUS_NOT_RETURNED')
    require(status in {'PENDING','RUNNING','COMPLETE','ERROR','CANCELLED','FAILED','UNKNOWN_STATUS_NOT_RETURNED'},'UNRECOGNIZED_SUBMISSION_STATUS')
    score=None
    if status=='COMPLETE' and row.get('publicScore') not in (None,''):
        score=float(row['publicScore']);require(math.isfinite(score) and 0<=score<=1,'INVALID_PUBLIC_SCORE')
    return {'query_ok':True,'status':status,'score':score,'version_verified':True,
            'submission_id':remote['submission_id'],'script_version_id':remote['script_version_id'],
            'notebook_version':remote['version'],'submitted_at':row.get('date')}

def historical_score(record):
    s=record.get('submission') or {}
    if s.get('status')=='COMPLETE' and s.get('public_score') is not None:
        v=float(s['public_score']);require(math.isfinite(v) and 0<=v<=1,'INVALID_HISTORICAL_SCORE')
        return {'value':v,'observed_at':s.get('observed_at'),'submission_id':s.get('id'),'source':'canonical experiment record'}
    return None

def verify_files(manifest,folder):
    require(manifest.get('status')=='COLLECTED_COMMIT_OUTPUTS','NOT_A_COMPLETED_COLLECTION')
    names=[]
    for e in manifest['files']:
        name=e['name'];require(PurePosixPath(name).name==name and '\\' not in name and ':' not in name,'UNSAFE_ARTIFACT_NAME')
        require(name not in names,'DUPLICATE_ARTIFACT');names.append(name)
        p=inside(name,Path(folder));require(p.is_file(),'MISSING_ARTIFACT')
        require(p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'ARTIFACT_HASH_MISMATCH')
    require(bool(names),'EMPTY_COLLECTION')

def latest(key,operation):
    p=WORK/'latest'/key/(operation+'.json')
    if not p.exists():return None
    pointer=read(p);f=inside(pointer['path'])
    require(sha(f)==pointer['sha256'],'OBSERVATION_POINTER_CHANGED')
    result=read(f);require(result['experiment_id']==key,'OBSERVATION_ID_MISMATCH')
    return result

def remember(key,operation,result,path):
    result.update(experiment_id=key,operation=operation)
    atomic_json(path,result,exclusive=True)
    atomic_json(WORK/'latest'/key/(operation+'.json'),{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
    if operation=='status' and result.get('query_ok') and result.get('version_verified') and result.get('status')=='COMPLETE' and result.get('score') is not None:
        value=result['score'];require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'INVALID_CONFIRMED_SCORE')
        atomic_json(WORK/'latest'/key/'confirmed-score.json',{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})

def safe_failure(error):
    # Provider exception messages/URLs and stderr are intentionally never saved.
    return {'status':'ERROR','error_code':str(error) if isinstance(error,GateError) and re.fullmatch('[A-Z0-9_]+',str(error)) else type(error).__name__,
            'query_ok':False,'observed_at':now(),'note':'Operation failed; previous confirmed results were not changed.'}
