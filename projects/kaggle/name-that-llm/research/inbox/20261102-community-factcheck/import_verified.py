"""Import this bounded research check. Does not execute or modify cloud jobs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[3]
INBOX=Path(__file__).resolve().parent
SOURCES=ROOT/'tracking/records/sources'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,value,exclusive=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def allocate(item):
    records=[read(p) for p in SOURCES.glob('SRC-*.json')]
    for r in records:
        if r['url'].split('?')[0].rstrip('/').replace('/versions/'+str(item['version']),'')==item['base_url'] and any(r.get('source_version',{}).get(k)==item['version'] for k in ('notebook_version','dataset_version_number')):
            raise ValueError('Already registered: '+r['id'])
    ident='SRC-'+str(max(int(r['id'][4:]) for r in records)+1).zfill(3)
    dest=ROOT/'research'/item['category']/(ident+'-'+item['slug'])
    assert not dest.exists() and dest.resolve().is_relative_to(ROOT.resolve())
    shutil.copytree(INBOX/item['inbox'],dest)
    # Preserve handoff manifest/factcheck as observed evidence; canonical index below owns current identity.
    note=dest/'notes.md';original=note.read_text(encoding='utf-8')
    original=original.replace('待导入核查','核查').replace('核查（待合并）','核查')
    note.write_text(f'# {ident} · {item["title"]}\n\n已归档；正式索引：[来源记录](../../../tracking/records/sources/{ident}.json)。下方保留采集时的范围、证据及交接说明；其中PENDING_IMPORT描述的是原始交接阶段。\n\n'+original,encoding='utf-8')
    record=read(ROOT/'tracking/templates/source.json')
    record.update(schema_version=1,id=ident,kind='notebook' if item['category']=='solutions' else 'dataset',category=item['category'],
        artifact_dir=rel(dest),notes_path=rel(note),author=item['author'],title=item['title'],url=item['url'],observed_at=item['observed'],
        source_version=item['source_version'],public_score=None,summary=item['summary'],author_claim=item['author_claim'],verified=item['verified'],
        use_for_us=item['use_for_us'],test_plan=item['test_plan'],limitations=item['limitations'],tags=item['tags'],related_sources=item['related'])
    files=[p for p in sorted(dest.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    record['evidence']=[rel(p) for p in files]
    record['evidence_sha256']={rel(p):sha(p) for p in files}
    record['supersedes']=None
    write(SOURCES/(ident+'.json'),record,exclusive=True)
    write(INBOX/item['inbox']/'import-result.json',{'status':'IMPORTED','canonical_source':ident,'canonical_directory':rel(dest),'observed_at':datetime.now(timezone.utc).isoformat()},exclusive=True)
    return ident,rel(dest)

def refresh_existing():
    updated=[]
    for d in sorted((INBOX/'solutions').iterdir()):
        p=d/'receipt.json'
        if not p.exists():continue
        receipt=read(p)
        if 'existing_source' not in receipt:continue
        assert receipt['code_equal'] and receipt['current_version']==receipt['old_version']
        rp=SOURCES/(receipt['existing_source']+'.json');r=read(rp)
        old_source=ROOT/receipt['old_source_path'];artifact=ROOT/r.get('artifact_dir',old_source.parent.relative_to(ROOT).as_posix())
        out=artifact/'snapshots/20261102-community-factcheck';out.mkdir(parents=True,exist_ok=False)
        for name in ['receipt.json','metadata.json']:shutil.copyfile(d/name,out/name)
        notes=ROOT/r.get('notes_path',(artifact/'notes.md').relative_to(ROOT).as_posix())
        if not notes.exists():notes.write_text(f'# {r["id"]} · {r["title"]}\n\n来源：{r["url"]}\n\n{r["summary"]}\n\n限制：{r["limitations"]}\n',encoding='utf-8')
        with notes.open('a',encoding='utf-8') as f:f.write(f'\n## 2026-11-02 固定版本复核\n\n官方API当前仍为V{receipt["current_version"]}；以同一规范提取全部代码单元，与已归档源码完全一致。代码SHA256 `{receipt["code_sha256"]}`。仅更新版本/代码观察，不把标题或历史最佳分绑定到当前版本。详见`snapshots/20261102-community-factcheck/receipt.json`。\n')
        r.update(observed_at=receipt['observed_at'],category='solutions',artifact_dir=rel(artifact),notes_path=rel(notes))
        r['verified']=r.get('verified','')+f' 2026-11-02再次读取当前V{receipt["current_version"]}，全部代码单元与历史固定版本相同。'
        r['evidence']=list(dict.fromkeys(r['evidence']+[rel(notes),rel(out/'receipt.json'),rel(out/'metadata.json')]))
        r.setdefault('evidence_sha256',{}).update({rel(p):sha(p) for p in [notes,out/'receipt.json',out/'metadata.json']})
        write(rp,r)
        write(d/'import-result.json',{'status':'DUPLICATE_FIXED_VERSION_REUSED','canonical_source':r['id'],'new_observation':rel(out/'receipt.json')},exclusive=True)
        updated.append(r['id'])
    return updated

def main():
    mercury=read(INBOX/'solutions/mercurydust-v11/factcheck.json')
    pool=read(INBOX/'datasets/open-pool-v13/factcheck.json')
    lucas=read(INBOX/'datasets/lucas-free-proxy-v12/manifest.json')
    items=[dict(category='solutions',inbox='solutions/mercurydust-v11',slug='mercurydust-v11-pairwise-index-audit',
        author='mercurydust',title='Mercurydust V11 / V34.1：配对修正与批次索引缺陷',base_url='https://www.kaggle.com/code/mercurydust/name-that-llm-submission',url='https://www.kaggle.com/code/mercurydust/name-that-llm-submission',version=11,observed=mercury['observed_at'],
        source_version={'notebook_version':11,'script_version_id':None,'source_sha256':mercury['source_sha256'],'code_sha256':mercury['code_sha256'],'body_version':'V34.1'},
        summary='配对判别第二批起混用首批4项特征；AST微测试复现。0.10为门控log修正，非10%概率融合；附件公开可查，未绑定新成绩。',
        author_claim='正文以旧V31为基础，新增A/B弱属性和同模型配对判别；0.10/0.35为未验证初值。',verified='固定V11等于当前latest。执行提取的原推理循环：3/128行无差异，129行起四列特征错位；仅小型stub测试。Revision03附件Public搜索和metadata可读。',
        use_for_us='作为独立配对残差候选的代码警示，不能当成熟高分方案直接复现。',test_plan='修索引后先验证批次边界一致；固定外部prompt验证，单独测试门控修正；不同时改变基线权重。',
        limitations='未跑完整模型或比赛；V11发布时间、scriptVersionId及新LB未知。附件许可Unknown，正文/全部依赖未完整审计。',tags=['pairwise','bug','static-audit','version-binding','unscored'],related=['SRC-017']),
        dict(category='datasets',inbox='datasets/open-pool-v13',slug='open-pool-kaggle-v13',author='crtproduction',title='Open Pool平台V13 / 作者v11：路线隔离、分组和匹配权重',base_url='https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool',url=pool['source_url'],version=13,observed=pool['verified_at_utc'],
        source_version={'dataset_id':pool['dataset_id'],'dataset_version_number':13,'author_release_label':'v11 fix','version_created_at':pool['checks']['v13_created_utc']},summary='主parquet元数据确认190588行/23列；Gemini3.7隔离10593。新分组/权重有用，但上游已接触全train，单按折重算权重不足以恢复独立OOF。',
        author_claim='新增10294、隔离10593、移除7620；V13只改5501 flag且正文不变。两seed小模型单项消融，不含新行和全组合。',verified='固定V12/V13时间和文件清单；尾部元数据确认主池190588、隔离10593；CSV汇总/分组缺口实际计数。新日志确认1024条按train逐类标点删行、264条按train相似删行。',
        use_for_us='优先审计来源可信度、prompt祖先与净新增；match_weight另作采样对照，不直接替换当前实验。',test_plan='先按现有训练/验证数据做文本与题组交集；按折重做依赖train的筛选/匹配，或采用未参与的外部验证。固定分类器与权重后分开换池、分组、采样。',
        limitations='未下载完整主parquet逐行diff；5501flag正文不变、迁移ID、真实served model、592完整panel仍为作者主张。新组合没有独立提分证据。',tags=['data','provenance','grouping','sampling','oof-leakage','version-binding'],related=['SRC-038','SRC-027','SRC-033']),
        dict(category='datasets',inbox='datasets/lucas-free-proxy-v12',slug='lucas-free-proxy-v12',author='lucasboesen',title='Lucas GPT/Gemini平台V12：7906行与新增192实测',base_url='https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations',url=lucas['source_url'],version=12,observed=lucas['observed_at'],
        source_version={'dataset_version_number':12,'author_release_label':'v2b21','version_created_at':'2026-11-01T14:39:54Z'},summary='固定CSV实测7906行/1872题/8类；较V11+192、较V7+3651。只有363题齐8类，合并OpenPool需去重。',
        author_claim='新增192条，来自prompt-bank-v2/v2b21，Kaggle Proxy单段协议。',verified='V11/V12全字段多重集与V7原件比对；+192/-0、48新增题、四类各48；行/正文/prompt-model无精确重复。OpenPool V11 source清单已列该来源3511行。',
        use_for_us='可作为原始公共数据增量审计候选；标签覆盖与来源重叠需单独处理。',test_plan='先与当前10073公共池/OpenPool做全文及prompt祖先去重，再做固定协议的加入/不加入对照。',
        limitations='无finish_reason列，未完成截断/拒答/语义去重或实际served-model认证。未逐行联结OpenPool，不能断言新192全部净新增。',tags=['data','incremental','dedup','provenance','version-binding'],related=['SRC-034','SRC-038'])]
    created=[allocate(item) for item in items]
    updated=refresh_existing()
    index={'status':'IMPORTED_VERIFIED_ITEMS','created':created,'updated':updated,'supplementary_screening':'Three score-ranked candidates retain individual PENDING_REVIEW inbox entries; no score/version certification.',
           'no_training':True,'no_submission':True,'no_remote_mutation':True,'observed_at':datetime.now(timezone.utc).isoformat()}
    write(ROOT/'tracking/snapshots/20261102-community-factcheck/import-index.json',index,exclusive=True)
    print(json.dumps(index,ensure_ascii=False))

if __name__=='__main__':main()
