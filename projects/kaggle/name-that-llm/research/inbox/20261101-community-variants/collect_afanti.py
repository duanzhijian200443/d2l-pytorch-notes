from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,hashlib,difflib
from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
refs=['afantihust/ntl-fork-664','afantihust/ntl-fork664-m25','afantihust/ntl-fork664-s50','afantihust/ntl-fork664-s70']
def code(n):return '\n\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in n['cells'] if c['cell_type']=='code')
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
olds=[]
for f in (ROOT/'tracking/records/sources').glob('SRC-*.json'):
 r=json.loads(f.read_text(encoding='utf-8-sig'));ip=[p for p in r['evidence'] if p.endswith('.ipynb')]
 if ip:olds.append((r['id'],code(json.loads((ROOT/ip[0]).read_text(encoding='utf-8-sig')))))
def fetch(ref):
 a=KaggleApi();a.authenticate();q=ApiGetKernelRequest();q.user_name,q.kernel_slug=ref.split('/')
 with a.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(q)
 m=r.metadata.to_dict();assert not m['isPrivate'];v=m['currentVersionNumber'];raw=r.blob.source;n=json.loads(raw);s=code(n)
 q.version_label='v'+str(v)
 with a.build_kaggle_client() as c: fixed=c.kernels.kernels_api_client.get_kernel(q)
 assert sha(fixed.blob.source)==sha(raw)
 folder=BASE/'solutions'/(ref.replace('/','--')+f'-v{v}');folder.mkdir(parents=True,exist_ok=False)
 for name,data in [('source.ipynb',raw),('code.txt',s),('description.md','\n\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in n['cells'] if c['cell_type']=='markdown')),('metadata.json',json.dumps(m,ensure_ascii=False,indent=2))]:(folder/name).write_text(data,encoding='utf-8')
 candidates=sorted([(difflib.SequenceMatcher(None,s.splitlines(),old.splitlines(),autojunk=False).ratio(),ident,old) for ident,old in olds],reverse=True)
 same=[ident for ident,old in olds if sha(old)==sha(s)]
 for ratio,ident,old in candidates[:1]:(folder/f'vs-{ident}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=ident,tofile=ref)),encoding='utf-8')
 row={'ref':ref,'title':m['title'],'author':m['author'],'category':'solutions','notebook_version':v,'script_version_id':None,'observed_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(raw),'code_sha256':sha(s),'public_score':None,'source_executed':False,'fixed_version_refetch_hash_match':True,'artifact_dir':folder.relative_to(ROOT).as_posix(),'same_code_sources':same,'related_sources':[r[1] for r in candidates[:2]],'nearest_sources':[{'source':ident,'line_similarity':ratio} for ratio,ident,old in candidates[:3]],'action':'NEW_FIXED_VERSION'}
 (folder/'receipt.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
 return row
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,refs))
p=BASE/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'));m['entries']+=rows;p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows:print(json.dumps(r,ensure_ascii=False))
