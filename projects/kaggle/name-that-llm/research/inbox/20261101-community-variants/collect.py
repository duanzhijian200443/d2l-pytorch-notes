"""Read public Kaggle notebooks; never execute fetched cells. Temporary intake helper."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import difflib
import hashlib
import json
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
REFS=[
 'lightningv08/0-662-qwen3-8-27b-lora-weighted-ensemble',
 'junaid512/ntl-v1-baseline',
 'sujanmajhisuzan/ntl-apex-sota-atlas-hungarian',
 'dazhijiang/name-that-llm-ntl-0662',
 'gloriachen0816/ntl-0-662-all-public-a-fork-of-alexchilton',
]
def code(nb):
 return '\n\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='code')
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
records=[json.loads(p.read_text(encoding='utf-8-sig')) for p in (ROOT/'tracking/records/sources').glob('SRC-*.json')]
def fetch(ref):
 a=KaggleApi(); a.authenticate()
 q=ApiGetKernelRequest(); q.user_name,q.kernel_slug=ref.split('/')
 with a.build_kaggle_client() as c: r=c.kernels.kernels_api_client.get_kernel(q)
 m=r.metadata.to_dict(); assert not m.get('isPrivate'), 'private'
 v=m['currentVersionNumber']; observed=datetime.now(timezone.utc).isoformat()
 raw=r.blob.source; nb=json.loads(raw); src=code(nb)
 matches=[s for s in records if s['url'].split('?')[0]=='https://www.kaggle.com/code/'+ref and s['source_version'].get('notebook_version')==v]
 item={'ref':ref,'title':m['title'],'author':m['author'],'category':'solutions','notebook_version':v,'script_version_id':None,'observed_at':observed,'source_sha256':sha(raw),'code_sha256':sha(src),'public_score':{'value':0.662,'bound_to_source':False,'evidence_type':'parent_observed_platform_page','script_version_id':None},'related_sources':[s['id'] for s in records if s['url'].split('?')[0]=='https://www.kaggle.com/code/'+ref],'same_code_sources':[s['id'] for s in records if s['source_version'].get('code_sha256')==sha(src)],'source_executed':False}
 if matches and matches[0]['source_version'].get('code_sha256')==sha(src):
  item.update(action='REUSE',existing_src=matches[0]['id'],existing_evidence=matches[0]['evidence']); return item
 folder=BASE/'solutions'/(ref.replace('/','--')+f'-v{v}'); folder.mkdir(parents=True,exist_ok=False)
 (folder/'source.ipynb').write_text(raw,encoding='utf-8')
 (folder/'code.txt').write_text(src,encoding='utf-8')
 md='\n\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='markdown')
 (folder/'description.md').write_text(md,encoding='utf-8')
 (folder/'metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
 item.update(action='NEW_FIXED_VERSION',artifact_dir=folder.relative_to(ROOT).as_posix())
 candidates=[]
 for s in records:
  paths=[p for p in s['evidence'] if p.endswith('.ipynb')]
  if not paths: continue
  old=code(json.loads((ROOT/paths[0]).read_text(encoding='utf-8-sig')))
  ratio=difflib.SequenceMatcher(None,old.splitlines(),src.splitlines(),autojunk=False).ratio()
  candidates.append((ratio,s['id'],old))
 candidates.sort(reverse=True)
 item['nearest_sources']=[{'source':i,'line_similarity':r} for r,i,c in candidates[:4]]
 for ratio,ident,old in candidates[:1]:
  (folder/f'vs-{ident}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),src.splitlines(True),fromfile=ident,tofile=ref)),encoding='utf-8')
 (folder/'receipt.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8')
 return item
with ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(fetch,REFS))
(BASE/'manifest.json').write_text(json.dumps({'status':'PENDING_IMPORT','role':'temporary_intake_not_authoritative_index','entries':rows},ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows: print(json.dumps(r,ensure_ascii=False))
