"""Kaggle CPU only: one fixed five-fold, three-share stylegram ablation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import gc, hashlib, json, platform, time, traceback, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from stylegram_core import StyleGrams, sparse_group_centre, normalised_content_groups, eligible

HERE=Path(__file__).resolve().parent
OUT=Path('/kaggle/working/results');OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=lambda:datetime.now(timezone.utc).isoformat()
P=json.loads((HERE/'protocol.json').read_text())

def save(name,data):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def weights(y,ey,share):
    tr=np.bincount(y,minlength=18).astype(float);ex=np.bincount(ey,minlength=18).astype(float)
    assert (tr>0).all() and (ex>0).all()
    w=np.r_[np.full(len(y),1-share),share*tr[ey]/ex[ey]]
    mass=np.bincount(np.r_[y,ey],weights=w,minlength=18)
    assert np.allclose(mass,tr)
    return w

def sparse_indices(valid,m,repeat):
    keys=valid.id.map(lambda x:hashlib.sha256(f"42:{repeat}:{x}".encode()).hexdigest())
    d=pd.DataFrame({'prompt':valid.prompt_id.to_numpy(),'key':keys.to_numpy(),'index':np.arange(len(valid))})
    return np.sort(d.sort_values(['prompt','key']).groupby('prompt',sort=True).head(m)['index'].to_numpy())

def grouping(texts,X):
    groups=normalised_content_groups(texts)
    mask=eligible(groups)
    centered=sparse_group_centre(X[mask],groups[mask]) if mask.any() else None
    return groups,mask,centered

def fit_model(X,y,w):
    model=LinearSVC(C=P['C'],max_iter=P['max_iter'],random_state=P['seed'],dual=P['dual'])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(X,y,sample_weight=w)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in ws),'SVM convergence warning'
    assert model.classes_.tolist()==list(range(18))
    return model

def run():
    start=time.monotonic()
    for name,digest in P['input_hashes'].items():assert sha(HERE/name)==digest,name
    competition=next(p for p in Path('/kaggle/input').rglob('train.csv') if 'name-that-llm' in str(p))
    assert sha(competition)==P['train_sha256']
    train=pd.read_csv(competition,dtype={'id':str,'prompt_id':str});folds=pd.read_csv(HERE/'folds.csv',dtype={'id':str,'prompt_id':str})
    extra=pd.read_csv(HERE/'extra.csv',dtype={'row_id':str,'prompt_id':str})
    assert train.id.tolist()==folds.id.tolist() and train.model.tolist()==folds.model.tolist()
    assert train.prompt_id.tolist()==folds.prompt_id.tolist()
    assert train.text.map(lambda s:hashlib.sha256(s.encode()).hexdigest()).tolist()==folds.text_sha256.tolist()
    assert folds.groupby('prompt_id').fold.nunique().eq(1).all()
    assert len(extra)==10073 and extra.prompt_id.nunique()==694 and extra.row_id.is_unique
    assert extra.groupby('prompt_id').canonical_prompt.nunique().eq(1).all()
    assert not extra.canonical_prompt.str.startswith('official:').any()
    labels=P['labels'];lab={v:i for i,v in enumerate(labels)}
    y=train.model.map(lab).to_numpy();ey=extra.model.map(lab).to_numpy();ext=extra.text.tolist()
    assert len(labels)==18 and set(extra.model)==set(labels)
    norms=lambda t:' '.join(__import__('unicodedata').normalize('NFKC',t).casefold().split())
    assert set(map(norms,train.text)).isdisjoint(map(norms,extra.text))
    environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'cpu_count':os.cpu_count(),'gpu_requested':False,'protocol':P,'started_at':stamp()}
    save('environment.json',environment)
    save('state.json',{'status':'RUNNING','observed_at':stamp(),'completed_folds':[]})
    oof={str(a):np.full((len(train),18),np.nan) for a in P['shares']}
    dense_records=[];sparse_records=[];timings=[]
    for fold in range(5):
        begun=time.monotonic();ti=np.flatnonzero(folds.fold.to_numpy()!=fold);vi=np.flatnonzero(folds.fold.to_numpy()==fold)
        valid=train.iloc[vi].reset_index(drop=True);texts=train.text.iloc[ti].tolist();vtexts=valid.text.tolist()
        assert set(train.prompt_id.iloc[ti]).isdisjoint(valid.prompt_id)
        print(f'FOLD {fold}: fitting four-view TF-IDF on {len(ti)} official + {len(extra)} public',flush=True)
        grams=StyleGrams().fit(texts+ext)
        x=grams.transform(texts+ext);xv=grams.transform(vtexts)
        xcenter=vstack([sparse_group_centre(x[:len(ti)],train.prompt_id.iloc[ti].to_numpy()),sparse_group_centre(x[len(ti):],extra.prompt_id.to_numpy(),2,10**6)],format='csr')
        groups,mask,xvc=grouping(vtexts,xv)
        feature_seconds=time.monotonic()-begun
        sparse_scenarios=[]
        for m in P['sparse_per_prompt']:
            for repeat in range(P['sparse_repeats']):
                selected=sparse_indices(valid,m,repeat)
                # Only unlabelled selected texts define inference clusters.
                g=normalised_content_groups(valid.text.iloc[selected].tolist());mk=eligible(g)
                sparse_scenarios.append((m,repeat,selected,g,mk))
        del grams;gc.collect()
        target=np.r_[y[ti],ey]
        for share in P['shares']:
            started=time.monotonic();w=weights(y[ti],ey,share)
            plain=fit_model(x,target,w);plain_secs=time.monotonic()-started
            started=time.monotonic();center=fit_model(xcenter,target,w);center_secs=time.monotonic()-started
            plain_scores=plain.decision_function(xv);scores=plain_scores.copy()
            if mask.any():scores[mask]=center.decision_function(xvc)
            assert np.isfinite(scores).all();oof[str(share)][vi]=scores
            pred=scores.argmax(1)
            record={'fold':fold,'share':share,'rows':len(vi),'ba':float(balanced_accuracy_score(y[vi],pred)),'plain_ba':float(balanced_accuracy_score(y[vi],plain_scores.argmax(1))),'centred_rows':int(mask.sum()),'features':int(x.shape[1]),'plain_n_iter':int(plain.n_iter_),'centred_n_iter':int(center.n_iter_),'feature_seconds':feature_seconds,'plain_fit_seconds':plain_secs,'centred_fit_seconds':center_secs}
            dense_records.append(record)
            np.savez_compressed(OUT/f'fold{fold}-share{share}.npz',ids=valid.id.to_numpy(dtype=str),labels=np.array(labels),plain_scores=plain_scores,scores=scores,true=y[vi],group_ids=groups,centred_mask=mask)
            for m,repeat,selected,g,mk in sparse_scenarios:
                sp=plain_scores[selected].copy()
                if mk.any():
                    xc=sparse_group_centre(xv[selected[mk]],g[mk]);sp[mk]=center.decision_function(xc);del xc
                assert np.isfinite(sp).all()
                sparse_records.append({'fold':fold,'share':share,'m':m,'repeat':repeat,'rows':len(selected),'centred_rows':int(mk.sum()),'ba':float(balanced_accuracy_score(y[vi[selected]],sp.argmax(1)))})
                np.savez_compressed(OUT/f'sparse-f{fold}-a{share}-m{m}-r{repeat}.npz',ids=valid.id.iloc[selected].to_numpy(dtype=str),scores=sp,true=y[vi[selected]],group_ids=g,centred_mask=mk)
            print('FIT COMPLETE '+json.dumps(record),flush=True)
            del plain,center;gc.collect()
            pd.DataFrame(dense_records).to_csv(OUT/'fold_metrics.csv',index=False)
            pd.DataFrame(sparse_records).to_csv(OUT/'sparse_fold_metrics.csv',index=False)
            save('state.json',{'status':'RUNNING','observed_at':stamp(),'completed_fold_shares':[[r['fold'],r['share']] for r in dense_records]})
        del x,xcenter,xv,xvc,sparse_scenarios;gc.collect()
        timings.append({'fold':fold,'total_seconds':time.monotonic()-begun})
    summary=[]
    for share in P['shares']:
        scores=oof[str(share)];assert scores.shape==(2699,18) and np.isfinite(scores).all()
        pred=scores.argmax(1);f=train[['id','prompt_id','model']].copy();f['fold']=folds.fold;f['predicted_model']=np.array(labels)[pred];f['correct']=pred==y
        f.to_csv(OUT/f'oof-share{share}.csv',index=False);np.save(OUT/f'oof-share{share}.npy',scores)
        summary.append({'share':share,'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'accuracy':float((pred==y).mean()),'correct':int((pred==y).sum())})
    save('result.json',{'status':'COMPLETE','observed_at':stamp(),'summaries':summary,'rows':2699,'public_rows':len(extra),'folds':5,'trained_svms':30,'timings':timings,'total_seconds':time.monotonic()-start,'scope':'stylegram-only grouped OOF; not leaderboard or full ATLAS20','convergence_warnings':0,'gpu_used':False})
    save('state.json',{'status':'COMPLETE','observed_at':stamp()})
    print('EXPERIMENT COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):run()
    except Exception:
        save('failure.json',{'observed_at':stamp(),'traceback':traceback.format_exc()})
        save('state.json',{'status':'ERROR','observed_at':stamp()})
        raise
