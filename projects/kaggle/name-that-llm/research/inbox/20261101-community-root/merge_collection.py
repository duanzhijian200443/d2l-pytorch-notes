"""Merge this read-only community collection; never execute notebook sources."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, shutil

ROOT = Path(__file__).resolve().parents[3]
assert (ROOT / 'AGENTS.md').exists(), ROOT
SNAP = ROOT / 'tracking/snapshots/20261101T073557Z-community-refresh'
SNAP.mkdir(exist_ok=False)
NOW = datetime.now(timezone.utc).isoformat()
PAGE_TIME = '2026-11-01T07:35:57Z'
REG = ROOT / 'tracking/records/sources'
existing = {p.stem: json.loads(p.read_text(encoding='utf-8-sig')) for p in REG.glob('SRC-*.json')}
assert max(int(k.split('-')[1]) for k in existing) == 35, 'recheck concurrent allocations'
items, updates, movements, imports = [], [], {}, []

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rel(p): return p.relative_to(ROOT).as_posix()
def save(p, value, exclusive=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inside(p):
    assert p.resolve().is_relative_to(ROOT.resolve()), p
def move(src, dst):
    inside(src); inside(dst); assert src.exists() and not dst.exists(), (src,dst)
    dst.parent.mkdir(parents=True,exist_ok=True); src.rename(dst)
    movements[rel(src)] = rel(dst)
def clean_times(text):
    # Root's hand-transcribed per-page minutes were estimates, not clock readings.
    text = re.sub(r'2026-11-01T07:(?:36:00|39:00|43(?::00)?|49(?::00)?)(?:\+00:00|Z)', PAGE_TIME, text)
    text = re.sub(r'07:(?:36|39|43|49) UTC', '07:35:57 UTC（采集窗口终点，非逐页精确时间）', text)
    return text
def sections(path):
    text=path.read_text(encoding='utf8'); out={}
    for heading, body in re.findall(r'^## (.+)\n(.*?)(?=^## |\Z)', text, re.M|re.S):
        out[heading] = re.sub(r'\s+', ' ', body.strip())
    return out

# Shared snapshots are retrieval evidence, not a second latest-status registry.
root_stage=ROOT/'research/inbox/20261101-community-root'
for name in ['kernels-score-list.json','search-0664.json','score-list-dom-observation.json','search-0664-dom-observation.json','one-swap-versions-dom-observation.json']:
    p=root_stage/name
    if 'dom-observation' in name:
        d=read(p); d['observed_at']=PAGE_TIME; d.pop('observation_time_precision',None)
        d['observation_time_precision']='collection-end; exact per-page timestamp not separately recorded'
        d['observation_window_utc']='2026-11-01T07:30:00Z/2026-11-01T07:35:57Z'
        save(p,d)
    move(p,SNAP/name)

stage_names=['20261101-community-root','20261101-community-newmodels','20261101-community-variants','20261101-community-discussions']
manifests={name:read(ROOT/'research/inbox'/name/'manifest.json') for name in stage_names}
assert len(manifests[stage_names[1]]['entries'])==6
assert len(manifests[stage_names[2]]['entries'])==9
for stage_name in stage_names:
    stage=ROOT/'research/inbox'/stage_name; m=manifests[stage_name]
    if stage_name.endswith('discussions'):
        for e in m['records']:
            d=stage/Path(e['snapshot_path']).parent.parent
            prop=e['source_record_proposal']
            item={**e,**prop,'_stage':stage_name,'_dir':d,'_slug':d.name}
            if e.get('existing_src_candidate'): item['_existing']=e['existing_src_candidate']; updates.append(item)
            else: items.append(item)
    else:
        for e in m['entries']:
            p=Path(e['artifact_dir']); d=ROOT/p if str(p).replace('\\','/').startswith('research/') else stage/p
            item={**e,'_stage':stage_name,'_dir':d,'_slug':e.get('slug') or d.name.replace('--','-')}
            if 'source_version' not in item:
                item['source_version']={k:e[k] for k in ['notebook_version','script_version_id','source_sha256','code_sha256'] if k in e}
            items.append(item)
        for u in m.get('updates',[]): updates.append({**u,'_stage':stage_name,'_existing':u['existing_src'],'_notes_only':True})

assert len(items)==27, len(items)
new_records=[]
for n,item in enumerate(items,36):
    sid=f'SRC-{n:03d}'; src=item['_dir']; cat=item['category']
    short=item['_slug'][:88].rstrip('-'); dst=ROOT/'research'/cat/f'{sid}-{short}'
    move(src,dst)
    notes=dst/'notes.md'; assert notes.exists(), notes
    sec=sections(notes)
    r={k:item[k] for k in ['kind','category','title','author','url','observed_at','source_version','public_score','summary','author_claim','verified','use_for_us','test_plan','limitations','tags','related_sources'] if k in item}
    r.update(schema_version=1,id=sid,artifact_dir=rel(dst),notes_path=rel(notes))
    r.setdefault('kind',{'solutions':'notebook','datasets':'dataset','discussions':'discussion','official':'official'}[cat])
    r.setdefault('url','https://www.kaggle.com/code/'+item.get('ref',''))
    r.setdefault('related_sources',item.get('related_source_candidates',[]))
    r.setdefault('tags',[])
    for field,heading in [('author_claim','作者声称与实际设置'),('verified','我们核查了什么'),('use_for_us','对当前方案的用途与限制'),('test_plan','待验证想法'),('limitations','对当前方案的用途与限制')]:
        r.setdefault(field,sec.get(heading,'详见独立笔记；本次只读收集，未复现。')[:420])
    if datetime.fromisoformat(r['observed_at'].replace('Z','+00:00')) > datetime.now(timezone.utc):
        r['observed_at']=PAGE_TIME
    r['collection_observation_note']='源码/API时间见原始回执；网页汇总按07:30—07:35:57 UTC窗口记录，不虚构逐页秒级时间。'
    r['supersedes']=None
    meta=dst/'metadata.json'
    if meta.exists():
        md=read(meta);r['source_version']['notebook_id']=md.get('id')
    if r['source_version'].get('code_sha256'):
        r['family_id']='code-'+r['source_version']['code_sha256'][:12]
    r['evidence']=[rel(notes)]+[rel(p) for p in sorted(dst.rglob('*')) if p.is_file() and p!=notes]
    if cat=='solutions':
        r['evidence'] += [rel(SNAP/'score-list-dom-observation.json'),rel(SNAP/'search-0664-dom-observation.json')]
    for f in item.get('shared_evidence',[]): r['evidence'].append(rel(SNAP/f))
    new_records.append(r);imports.append({'source_id':sid,'action':'NEW_FIXED_SOURCE','category':cat,'artifact_dir':r['artifact_dir'],'url':r['url'],'version':r['source_version']})

# Existing records preserve their source version, old evidence and score binding.
updated=[]
for u in updates:
    sid=u['_existing']; old=existing[sid];save(SNAP/'before-records'/f'{sid}.json',old)
    r=json.loads(json.dumps(old))
    if u.get('_notes_only'):
        prop=u['proposed_update'];dst=ROOT/u['intended_notes_path'];src=ROOT/u['staged_notes_path']
        move(src,dst);r.update(prop);r['observed_at']=PAGE_TIME
        r['evidence']=list(dict.fromkeys(old['evidence']+[rel(dst),rel(SNAP/'score-list-dom-observation.json')]))
    else:
        dst=ROOT/'research/discussions'/f'{sid}-{u["_slug"]}'
        move(u['_dir'],dst)
        r.update({k:u[k] for k in ['title','author','category','tags','summary','author_claim','verified','use_for_us','test_plan','limitations','observed_at','source_version']})
        r.update(artifact_dir=rel(dst),notes_path=rel(dst/'notes.md'))
        r['related_sources']=list(dict.fromkeys(old.get('related_sources',[])+u.get('related_sources',[])))
        r['evidence']=list(dict.fromkeys(old['evidence']+[rel(dst/'notes.md')]+[rel(p) for p in dst.rglob('*') if p.is_file()]))
        snap=next(dst.glob('snapshots/*.json'))
        body=f'# {sid} · {r["title"]}\n\n状态：已读原帖及新快照，复用既有SRC。\n\n## 值得保留的结论\n\n{old["summary"]}\n\n## 来源与原件\n\n[原帖]({r["url"]}) · [本次快照](snapshots/{snap.name})。旧证据继续由正式台账引用，不复制。\n\n'
        for heading,key in [('作者声称与实际设置','author_claim'),('我们核查了什么','verified'),('对当前方案的用途与限制','use_for_us'),('适用条件与限制','limitations'),('待验证想法','test_plan')]: body+=f'## {heading}\n\n{r[key]}\n\n'
        body+='## 更新记录\n\n- 2026-11-01：复核评论并保留原快照，未新增实验；原编号继续有效。\n'
        (dst/'notes.md').write_text(body,encoding='utf8')
    updated.append(r);imports.append({'source_id':sid,'action':'UPDATE_EXISTING','category':r['category'],'artifact_dir':r['artifact_dir'],'url':r['url']})

records={r['id']:r for r in new_records+updated}
def addrel(a,b):
    r=records[a]; r['related_sources']=list(dict.fromkeys(r.get('related_sources',[])+[b]))
def by_ref(ref): return [r['id'] for r in new_records if r['url'].split('?')[0]=='https://www.kaggle.com/code/'+ref]
# Cross-source relationships; no copied analysis bodies.
for a,b in [('SRC-036','SRC-037'),('SRC-036','SRC-038'),('SRC-037','SRC-038'),('SRC-038','SRC-027')]: addrel(a,b);addrel(b,a)
for r in new_records:
    h=r['source_version'].get('code_sha256')
    if h:
        same=[s['id'] for s in list(existing.values())+new_records if s['id']!=r['id'] and s.get('source_version',{}).get('code_sha256')==h]
        r['same_code_sources']=same
        r['related_sources']=list(dict.fromkeys(r['related_sources']+same))
        if same:r['tags']=list(dict.fromkeys(r['tags']+['same-code']))
announcement=next(r for r in new_records if r['source_version'].get('topic_id')==753974)
enai=next(r for r in new_records if r['source_version'].get('dataset_id')==12956120)
addrel(announcement['id'],enai['id']);addrel(enai['id'],announcement['id'])

# Convert staging notes to durable links; source documents and raw snapshots stay intact.
for r in records.values():
    notes=ROOT/r['notes_path'];text=clean_times(notes.read_text(encoding='utf8'))
    text=re.sub(r'^# .*$',f'# {r["id"]} · {r["title"]}',text,count=1,flags=re.M)
    text=text.replace('__SRC_ID__',r['id'])
    text=re.sub(r'^原始记录：.*\n','',text,flags=re.M)
    text=text.replace('待合并窗口登记/更新正式 SRC','已归档，正式记录见下方链接').replace('待主窗口分配正式 SRC','正式编号已登记')
    text=text.replace('正式SRC待合并窗口分配','正式编号已登记').replace('正式编号由合并窗口分配','正式编号已登记')
    text=text.replace('待主窗口分配正式SRC、合并页面快照并校验','正式记录与页面快照已合并').replace('待正式导入','已正式归档')
    text=text.replace('（合并时替换占位编号）','')
    link=os.path.relpath(REG/f'{r["id"]}.json',notes.parent).replace('\\','/')
    lines=text.splitlines();lines.insert(2,f'原始记录：[正式台账]({link})。')
    text='\n'.join(lines)+'\n'
    notes.write_text(text,encoding='utf8')
    # Normalise only processed browser-observation files, never raw API/source bytes.
    for p in notes.parent.rglob('*'):
        if p.is_file() and p.suffix=='.json' and any(t in p.name for t in ['browser-observation','page-observation']):
            d=json.loads(clean_times(p.read_text(encoding='utf8')))
            d['observed_at']=PAGE_TIME;d['observation_time_precision']='collection-end; exact per-page timestamp not separately recorded'
            d['observation_window_utc']='2026-11-01T07:30:00Z/2026-11-01T07:35:57Z';save(p,d)
    r['verified']=clean_times(r['verified'])
    r['evidence_sha256']={p:sha(ROOT/p) for p in r['evidence'] if (ROOT/p).exists()}
    save(REG/f'{r["id"]}.json',r,exclusive=r in new_records)

# Refresh-only receipt for the already archived V6; discard our duplicate temporary copy.
dup=next((root_stage/'solutions').glob('judith*--v6'))
receipt=read(dup/'receipt.json');save(SNAP/'SRC-022-refetch-receipt.json',receipt)
r=records['SRC-022'];r['evidence'].append(rel(SNAP/'SRC-022-refetch-receipt.json'));r['evidence_sha256'][r['evidence'][-1]]=sha(SNAP/'SRC-022-refetch-receipt.json');save(REG/'SRC-022.json',r)
inside(dup);assert dup.resolve().is_relative_to(root_stage.resolve());shutil.rmtree(dup)

for name,m in manifests.items():
    m['status']='IMPORTED';m['imported_at']=NOW;m['role']='historical_import_receipt_not_authoritative_registry'
    m['canonical_records']=[i for i in imports if any(r['id']==i['source_id'] for r in records.values())]
    m['path_migrations']={a:b for a,b in movements.items() if '/'+name+'/' in a}
    save(ROOT/'research/inbox'/name/'manifest.json',m)

index={'role':'dated_collection_inventory_not_source_registry','observed_at':NOW,'scope':{'competition':'name-that-llm','api_candidates_by_score':60,'browser_score_rows':20,'browser_keyword_matches':13,'keyword':'0.664','forum_window':'Recent discussions from Oct 29 to Nov 1 plus existing SRC-026/027 recheck','not_exhaustive':True},'entries':imports,'new_sources':[r['id'] for r in new_records],'updated_sources':[r['id'] for r in updated],'excluded':[{'ref':'leoprovorov/he-who-must-not-be-named','reason':'display score1645.5 outside competition range'},{'ref':'ozertuu/fresh-top-30','reason':'display score1019.8 outside competition range'},{'ref':'cosmorin/why-1-is-1-gap-analysis-across-6-competitions','reason':'older generic multi-competition comparison; outside recent solution scope'},{'ref':'johnnykhan1/name-that-llm-family-first-then-the-model','reason':'older Oct24 lead; no new result verified in this refresh'}],'known_gaps':['Unbound page scores explicitly retained as unbound','Full weights/input versions and hidden runtimes not reproduced','Two dataset entries are cards/file lists, not row audits','No new official ruling verified; no official entry fabricated'],'no_training_or_submission':True,'path_migrations':movements}
save(SNAP/'collection-index.json',index)
print(json.dumps({'new':len(new_records),'updated':len(updated),'new_ids':index['new_sources'],'updated_ids':index['updated_sources'],'snapshot':rel(SNAP)},ensure_ascii=False))
