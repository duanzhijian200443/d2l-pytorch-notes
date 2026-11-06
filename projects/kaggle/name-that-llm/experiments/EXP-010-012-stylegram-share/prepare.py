# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas==3.0.6", "pyarrow==23.0.1"]
# ///
import ast, hashlib, json, pathlib, sys, zipfile
import pandas as pd

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'experiments/EXP-009-baseline-data-lineage'
sys.path.insert(0,str(PRIOR))
from project_rows import SOURCES, RAW, comp_text, audit_hash
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    payload=HERE/'payload';payload.mkdir(exist_ok=True)
    upload=HERE/'dataset';upload.mkdir(exist_ok=True)
    kernel=HERE/'kernel';kernel.mkdir(exist_ok=True)
    baseline=json.loads((ROOT/'tracking/records/experiments/EXP-004.json').read_text(encoding='utf-8'))
    assert sha(ROOT/baseline['source']['local_path'])==baseline['source']['sha256']
    source=PRIOR/'comp_stack.py'
    nb=json.loads((ROOT/baseline['source']['local_path']).read_text(encoding='utf-8'))
    assert source.read_text(encoding='utf-8')==''.join(nb['cells'][8]['source']).split('\n',1)[1]
    ledger=pd.read_csv(PRIOR/'outputs/complement_extra_rows.csv')
    flags=pd.read_parquet(ROOT/'experiments/EXP-008-public-pool-audit/outputs/row_flags.parquet')
    checked=ledger.merge(flags,on='row_id',suffixes=('','_audit'),validate='one_to_one')
    assert len(checked)==10325 and checked.linked_official_prompt.sum()==252
    kept=checked.loc[~checked.linked_official_prompt].copy()
    assert len(kept)==10073 and kept.prompt_id.nunique()==694
    risk=['official_exact_match','official_high_similarity','official_shingle_review','label_conflict','identity_mismatch','provenance_issue']
    assert not kept[risk].any().any()
    assert kept.canonical_prompt.ne('').all()
    assert kept.groupby('prompt_id').canonical_prompt.nunique().eq(1).all()
    assert kept.groupby('canonical_prompt').prompt_id.nunique().eq(1).all()
    paths={table:RAW/path for table,path,_,_ in SOURCES}
    downloads=json.loads((ROOT/'experiments/EXP-008-public-pool-audit/download-manifest.json').read_text(encoding='utf-8'))
    hashes={f['local_path']:f['sha256'] for ds in downloads['datasets'] for f in ds['files']}
    restored={}
    for table,path in paths.items():
        assert sha(path)==hashes[path.relative_to(ROOT).as_posix()]
        df=pd.read_csv(path);label='model' if 'model' in df else 'label'
        for row in kept.loc[kept.origin.eq(table)].itertuples():
            original=df.iloc[int(row.row_id.rsplit(':',1)[1])]
            text=comp_text(original['text'])
            assert str(original[label])==row.model and audit_hash(text)==row.audit_text_hash
            expected=('pv_' if table=='prvs-clean-v6' else '')+str(original['prompt_id'])
            assert expected==row.prompt_id
            restored[row.row_id]=text
    extra=kept[['row_id','origin','prompt_id','model','audit_text_hash','canonical_prompt']].copy()
    extra['text']=extra.row_id.map(restored)
    assert extra.text.notna().all() and extra.row_id.is_unique
    extra.to_csv(payload/'extra.csv',index=False,lineterminator='\n')
    (payload/'folds.csv').write_bytes((ROOT/'validation/prompt-group-v1/folds.csv').read_bytes())
    split=json.loads((ROOT/'validation/prompt-group-v1/manifest.json').read_text(encoding='utf-8'))
    allowed={'shape_view','function_view','StyleGrams','sparse_group_centre','content_cosine_distance','average_linkage_groups','normalised_content_groups','eligible'}
    tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in allowed]
    nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'FUNCTION_WORDS','TOKEN_RE'} for t in n.targets)]
    nodes.sort(key=lambda n:n.lineno)
    imports='import re\nimport numpy as np\nfrom scipy.sparse import csr_matrix, diags, hstack\nfrom scipy.cluster.hierarchy import fcluster, linkage\nfrom scipy.spatial.distance import squareform\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nGROUP_MIN, GROUP_MAX, STYLE_GROUP_THRESH = 2, 18, 0.90\n'
    (payload/'stylegram_core.py').write_text(imports+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n',encoding='utf-8')
    (payload/'runner.py').write_bytes((HERE/'runner.py').read_bytes())
    protocol={'split_id':'prompt-group-v1','train_sha256':split['train_sha256'],'labels':split['labels'],'shares':[0.5,0.25,0.75],'public_rows':10073,'removed_known_regen':252,'public_groups':694,'folds':5,'seed':42,'C':0.5,'max_iter':4000,'dual':True,'sparse_per_prompt':[2,3,6],'sparse_repeats':5,'dense_primary':True,'source_notebook_sha256':baseline['source']['sha256'],'core_source_sha256':sha(source),'input_hashes':{p.name:sha(p) for p in payload.iterdir() if p.name!='protocol.json' and p.is_file()},'scope':'isolated stylegram, not full ensemble; known prompt links isolated; semantic duplicate audit incomplete; validation grouping uses only held-out text','grouping_note':'Unsupervised validation batch clustering is part of inference; it sees no true labels or prompt IDs','no_paid_api':True,'no_gpu':True}
    (payload/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(upload/'experiment_payload.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.iterdir()):
            if p.is_file():z.write(p,p.name)
    metadata={'id':'zhjianduan/ntl-stylegram-cv-inputs-v1','title':'NTL Stylegram CV Inputs v1','licenses':[{'name':'other'}],'description':'Private experiment inputs. Public generated passages retain upstream licenses and attribution. No redistribution authorization implied. Known official prompt reconstructions excluded. Sources: seyitkaangunes/ntl-extra-api-passages V2; lucasboesen/ntl-free-proxy-generations V7; dophuchung/ntl-proxy-generations V1; prvsiyan/prvsiyans-extended-name-that-llm-dataset V6. Official training text is loaded separately from the competition. The fixed-fold table is private experiment metadata.'}
    (upload/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    bootstrap='''import hashlib, pathlib, json, runpy, sys, os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
paths=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()==PROTOCOL_HASH]
assert len(paths)==1, 'Expected one hash-pinned protocol'
target=paths[0].parent
protocol=json.loads(paths[0].read_text())
for name,digest in protocol['input_hashes'].items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest, name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'runner.py'),run_name='__main__')
'''.replace('PROTOCOL_HASH',repr(sha(payload/'protocol.json')))
    (kernel/'bootstrap.py').write_text(bootstrap,encoding='utf-8')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':'zhjianduan/name-that-llm-stylegram-share-cv','title':'Name That LLM Stylegram Share CV','code_file':'bootstrap.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['zhjianduan/ntl-stylegram-cv-inputs-v1/1'],'competition_sources':['name-that-llm'],'kernel_sources':[],'model_sources':[]},indent=2),encoding='utf-8')
    (HERE/'preflight.json').write_text(json.dumps({'status':'PASS','protocol':protocol,'payload_sha256':sha(upload/'experiment_payload.zip'),'payload_bytes':(upload/'experiment_payload.zip').stat().st_size},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared 10073 public rows, 694 groups, fixed folds; no training yet')

if __name__=='__main__':main()
