"""Exercise the two actually generated candidate programs without running models or Kaggle."""
import ast
import fnmatch
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, vstack

from audit import ARMS, EXP53, HERE, ROOT, ZERO, ZERO_SHA, RAW_HASH, audit_data, cells, digest, need, read, save, weights


def inspect_candidate(size):
    folder = ROOT/'cloud'/ARMS[size]/'kernel'
    path = folder/'apex.ipynb'
    z, c = read(ZERO), read(path)
    zs, cs = cells(z),cells(c)
    need(len(cs)==len(zs)==17 and [i for i,(a,b) in enumerate(zip(zs,cs)) if a!=b] == [12,14],
         'Unexpected algorithm or orchestration changes')
    need({k:v for k,v in z.items() if k!='cells'} == {k:v for k,v in c.items() if k!='cells'} and
         all({k:v for k,v in a.items() if k!='source'} == {k:v for k,v in b.items() if k!='source'}
             for a,b in zip(z['cells'],c['cells'])), 'Notebook metadata differs')
    zm,cm=read(ZERO.parent/'kernel-metadata.json'),read(folder/'kernel-metadata.json')
    need(set(zm)==set(cm) and {k for k in cm if cm[k]!=zm[k]} == {'id','title','code_file'},'Launch setting changed')
    need(cm['code_file']=='apex.ipynb' and cm['id']==f'zhjianduan/ntl-apex-codex{size}-'+('056' if size==120 else '057')
         and cm['is_private'] and cm['enable_gpu'] and not cm['enable_internet']
         and sum(len(cm[k]) for k in ('competition_sources','kernel_sources','dataset_sources','model_sources'))==32,
         'Unexpected candidate source identity/attachment inventory')
    oc, nc=zs[12].partition('\n')[2],cs[12].partition('\n')[2]
    start = '# Added raw responses are private to this Stylegram member.'
    need(nc[:nc.index(start)] == oc[:oc.index('grams = StyleGrams().fit(TR + EX_T)')], 'Shared loader/weight function changed')
    gate = '# ---- gates: style SVCs'
    need(nc[nc.index(gate):] == oc[oc.index(gate):], 'Gate or later member code changed')
    route = oc[oc.index('G_style = normalised_content_groups(TE)'):oc.index("add_member('stylegram', P_style)")+len("add_member('stylegram', P_style)")]
    need(route in nc and nc.count("add_member('stylegram', P_style)")==1, 'Routed contribution changed')
    need(cs[2]==zs[2] and cs[16]==zs[16] and cs[14].index('_c0 = _t.time()')<cs[14].index('if not FAST_COMMIT:')
         <cs[14].index('COMP_SEC = _t.time() - _c0'), 'Quick/hidden timing scope differs')
    return path, c, nc, nc[nc.index(start):nc.index(gate)], route


def run_numeric(size, body, part, train, shared, rows):
    labels,y,ey,ny,old,added=weights(train,shared,rows)
    texts,extra,add=[[x['text'] for x in t] for t in (train,shared,rows)]
    groups=[np.asarray([x['prompt_id'] for x in t]) for t in (train,shared,rows)]
    te=['sentinel-a','sentinel-b','sentinel-c']
    lookup = {text:i+1 for i,text in enumerate(dict.fromkeys(texts+extra+add+te))}
    def values(tt):return csr_matrix([[float(lookup[t])] for t in tt])
    observed=[]; fit_text=[]; centres=[]; members=[]
    class Grams:
        def fit(self,tt):fit_text.extend(tt);return self
        def transform(self,tt):return values(tt)
    class SVC:
        def __init__(self,**kw):self.kw=kw; self.n_iter_=1; self.classes_=np.arange(18)
        def fit(self,X,target,sample_weight):
            observed.append((X.copy(),target.copy(),sample_weight.copy(),self.kw))
            self.mark=len(observed);return self
    def centre(x,g,lo=2,hi=18):
        centres.append((x.copy(),list(g),lo,hi))
        return x.multiply(0.5).tocsr()
    def probs(svc,xt):
        result=np.zeros((xt.shape[0],18));result[:,svc.mark]=1.;return result
    env=dict(np=np,pd=pd,json=json,os=os,pathlib=__import__('pathlib'),vstack=vstack,
             StyleGrams=Grams,LinearSVC=SVC,GROUP_MIN=2,GROUP_MAX=18,EXTRA_SHARE=.5,EXTRA=pd.DataFrame(shared),
             LABELS=labels,LAB2I={v:i for i,v in enumerate(labels)},K=18,TR=texts,EX_T=extra,TE=te,
             EX_Y=ey.copy(),y=y.copy(),EX_G=groups[1].copy(),PROMPT=groups[0].copy(),TINY=False,
             sparse_group_centre=centre,normalised_content_groups=lambda t: np.array([1,1,2]),
             eligible=lambda g:np.array([True,True,False]),svc_probs=probs,describe_groups=lambda *a:None,
             W={'stylegram':1.516},add_member=lambda key,p: members.append((key,p.copy())))
    tree=ast.parse(body)
    func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='extra_sample_weights')
    exec(compile(ast.Module(body=[func],type_ignores=[]),'<candidate frozen weight function>','exec'),env)
    with tempfile.TemporaryDirectory(dir=HERE) as tmp, patch.dict(os.environ,{'NTL_VOLUME_RUN_NONCE':'audit-nonce'}):
        receipt_file=Path(tmp)/'stylegram_volume_receipt.json'
        adjusted=part.replace('/kaggle/working/stylegram_volume_receipt.json',receipt_file.as_posix())
        exec(compile(adjusted,'<candidate Stylegram block>','exec'),env)
        receipt=read(receipt_file)
    need(env['_volume_sha']==RAW_HASH[size] and digest(env['_volume_data'].to_csv(index=False,lineterminator='\n').encode())==RAW_HASH[size],
         'Embedded CSV row order differs')
    need(fit_text==texts+extra+add and env['EX_T']==extra and np.array_equal(env['EX_Y'],ey)
         and np.array_equal(env['EX_G'],groups[1]) and env['EXTRA'].to_dict('records')==shared,
         'Training text or shared global changed')
    need(len(observed)==2 and len(centres)==4 and len(members)==1 and members[0][0]=='stylegram', 'Wrong fit/centre count')
    golden=np.r_[y,ey,ny]; expected=np.r_[old,added]
    allX=values(texts+extra+add)
    for idx,(x,yy,w,kw) in enumerate(observed):
        need(np.array_equal(yy,golden) and np.array_equal(w,expected) and kw=={'C':.5,'random_state':42,'max_iter':4000},
             'Candidate passed wrong row/weight/constructor to SVM')
        need((x != allX.multiply(1 if idx==0 else .5)).nnz==0, 'Feature row order differs between arms')
    for i,expected_groups in enumerate(groups):
        need(centres[i][1]==expected_groups.tolist(), 'Centred wrong prompt groups')
    need(centres[1][2:] == centres[2][2:] == (2,10**6) and centres[3][1]==[1,1],
         'Eligible centre group bounds or held test routing changed')
    need(members[0][1].argmax(1).tolist()==[2,2,1], 'Plain/centred route differs')
    need(receipt['arm']==f'four{size}' and receipt['fit_count']==2 and receipt['run_nonce']=='audit-nonce'
         and receipt['full_training_rows']==len(expected) and receipt['source_csv_sha256']==RAW_HASH[size],
         'Member receipt failed on actual candidate block')
    return {'fit_calls':2,'training_rows_each_svm':len(expected),'old_weight_rows_bitwise_equal':len(old),
            'new_weight_rows_bitwise_equal':len(added),'old_shared_globals_unchanged':True,'centred_subset_routed':True,
            'training_numeric_scope':'executed candidate block and original weight function; instrumented vectorizer and estimators record row order/weight without fitting models',
            'new_per_class':receipt['per_added_class']}


def simulate_guards(size, notebook, body):
    """Use original loader on staged physical sources and inject a controlled child result."""
    registry=read(EXP53/'inputs/source-snapshot/loader-file-receipt.json')
    def files(pattern, recursive=True):
        return [str(ROOT/x['path']) for x in registry if fnmatch.fnmatchcase(x['name'],pattern.rsplit('/',1)[-1])]
    guard=cells(notebook)[14]
    outcomes={}
    for case in ('fast','hidden_ok','hidden_child_failed','hidden_wrong_nonce','hidden_missing_seconds',
                 'hidden_nan_seconds','hidden_wrong_ids','hidden_nan_logits','hidden_stale'):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            wd=Path(tmp); (wd/'comp_stack.py').write_text(body,encoding='utf-8',newline='\n')
            test=pd.DataFrame({'id':['q1','q2','q3'],'text':['a','b','c']});test.to_csv(wd/'test.csv',index=False)
            original=guard.replace('/kaggle/working',wd.as_posix());env={'FAST_COMMIT':case=='fast','DATA':tmp}
            if case=='hidden_stale': (wd/'stylegram_volume_receipt.json').write_text('{}')
            calls=[]
            def fake_child(cmd, cwd, env, check):
                calls.append(cmd)
                need(check and Path(cwd)==wd,'Hidden child not fail-closed')
                if case=='hidden_child_failed':raise subprocess.CalledProcessError(1,cmd)
                pre=read(wd/'volume_preflight.json'); labels=sorted({v['model'] for v in pd.read_csv(ROOT/'train.csv',usecols=['model']).to_dict('records')})
                nonce='wrong' if case=='hidden_wrong_nonce' else env['NTL_VOLUME_RUN_NONCE']
                r={**pre,'status':'MEMBER_ADDED','fit_count':2,'run_nonce':nonce,'test_rows':3,'output_classes':18,
                   'member_weight':1.516,'official_rows':2699,'full_training_rows':2699+10743+size}
                (wd/'stylegram_volume_receipt.json').write_text(json.dumps(r))
                meta=dict(labels=labels,ids=test.id.tolist(),family_gates_applied=True,seconds=1.0)
                if case=='hidden_missing_seconds': del meta['seconds']
                if case=='hidden_nan_seconds': meta['seconds']=float('nan')
                if case=='hidden_wrong_ids': meta['ids']=list(reversed(meta['ids']))
                (wd/'comp_meta.json').write_text(json.dumps(meta))
                logits=np.zeros((3,18),dtype=np.float32)
                if case=='hidden_nan_logits': logits[0,0]=float('nan')
                np.save(wd/'comp_logit.npy',logits)
            caught=None
            try:
                with patch.object(glob,'glob',files),patch.object(subprocess,'run',fake_child),patch.dict(os.environ):
                    if case=='fast': os.environ.pop('KAGGLE_IS_COMPETITION_RERUN',None)
                    else: os.environ['KAGGLE_IS_COMPETITION_RERUN']='1'
                    exec(compile(original,'<candidate orchestration>','exec'),env)
            except (AssertionError,KeyError,subprocess.CalledProcessError) as err:caught=err
            success=case in ('fast','hidden_ok')
            need((caught is None)==success,'Guard returned unexpected status: '+case)
            pre=read(wd/'volume_preflight.json')
            need(pre['fast_commit']==(case=='fast') and pre['competition_rerun_flag_present']==(case!='fast') and pre['fit_count']==0 and pre['rows']==size,'Fast evidence scope invalid')
            need(len(calls)==int(case not in ('fast','hidden_stale')) and (wd/'volume_comp_execution.json').is_file()==(case=='hidden_ok'),
                 'Guard accepted missing/stale child artifacts')
            outcomes[case]='PASS_EXPECTED_BEHAVIOR'
    return outcomes


def main():
    train,shared,rows,data=audit_data()
    checked={}
    for n in ARMS:
        path,nb,body,part,_=inspect_candidate(n)
        numbers=run_numeric(n,body,part,train,shared,rows[n])
        guards=simulate_guards(n,nb,body)
        checked[n]=dict(status='PASS',source_sha256=digest(path.read_bytes()),metadata_sha256=digest((path.parent/'kernel-metadata.json').read_bytes()),
            comp_sha256=digest(body.encode()),changed_cells=[12,14],data=data[n],training=numbers,guards=guards)
    report={'status':'PASS_LOCAL_CANDIDATE_PAIR','base_sha256':ZERO_SHA,'arms':checked,
            'limits':['Local guard cases use simulated child outputs and real staged source files. Remote FAST, hidden scoring, model convergence and ATLAS coverage require separate evidence.',
                      'Original infer.py retains its documented fallback on runtime errors. Completion of the input preflight alone does not attest hidden inference.']}
    save('candidate-verification.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
