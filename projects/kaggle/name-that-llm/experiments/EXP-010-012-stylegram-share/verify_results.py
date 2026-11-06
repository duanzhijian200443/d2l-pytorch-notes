"""Verify downloaded Kaggle results, then make paired prompt-level comparisons.

No training, cloud mutation or competition submission. Call with --input DIR
pointing to the downloaded directory containing result.json.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def matrix_counts(truth,pred,groups,labels):
    correct=np.zeros((len(groups),len(labels)));support=np.zeros_like(correct)
    lab={v:i for i,v in enumerate(labels)}
    for gi,group in enumerate(groups):
        part=truth.prompt_id.eq(group).to_numpy()
        yi=truth.model.loc[part].map(lab).to_numpy()
        support[gi]=np.bincount(yi,minlength=len(labels))
        correct[gi]=np.bincount(yi,weights=(truth.model.loc[part].to_numpy()==pred[part]),minlength=len(labels))
    return correct,support

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args();src=args.input
    summary=json.loads((src/'result.json').read_text(encoding='utf-8'))
    assert summary['status']=='COMPLETE' and summary['trained_svms']==30 and summary['convergence_warnings']==0
    env=json.loads((src/'environment.json').read_text(encoding='utf-8'))
    local=json.loads((HERE/'payload/protocol.json').read_text(encoding='utf-8'))
    assert env['protocol']==local
    train=pd.read_csv(ROOT/'train.csv',dtype={'id':str,'prompt_id':str});folds=pd.read_csv(ROOT/'validation/prompt-group-v1/folds.csv',dtype={'id':str,'prompt_id':str})
    assert sha(ROOT/'train.csv')==local['train_sha256']
    labels=local['labels'];y=train.model.to_numpy();by_id=train.set_index('id');fold_of=dict(zip(folds.id,folds.fold))
    predictions={};dense=[];per_class=[];recomputed_folds=[]
    for share in local['shares']:
        oof=pd.read_csv(src/f'oof-share{share}.csv',dtype={'id':str,'prompt_id':str});scores=np.load(src/f'oof-share{share}.npy',allow_pickle=False)
        assert oof.id.tolist()==train.id.tolist() and oof.model.tolist()==train.model.tolist() and oof.prompt_id.tolist()==train.prompt_id.tolist()
        assert oof.fold.tolist()==folds.fold.tolist() and scores.shape==(2699,18) and np.isfinite(scores).all()
        pred=np.array(labels)[scores.argmax(1)];assert pred.tolist()==oof.predicted_model.tolist()
        assert oof.correct.tolist()==(pred==y).tolist()
        predictions[share]=pred
        ba=float(balanced_accuracy_score(y,pred));reported=next(x for x in summary['summaries'] if x['share']==share)
        assert abs(ba-reported['balanced_accuracy'])<1e-12
        dense.append({'share':share,'balanced_accuracy':ba,'accuracy':float((pred==y).mean()),'correct':int((pred==y).sum())})
        cm=confusion_matrix(y,pred,labels=labels)
        for label,support,correct in zip(labels,cm.sum(1),np.diag(cm)):
            per_class.append({'share':share,'model':label,'support':int(support),'correct':int(correct),'recall':float(correct/support)})
        for fold in range(5):
            z=np.load(src/f'fold{fold}-share{share}.npz',allow_pickle=False);ids=train.id[folds.fold.eq(fold)].tolist()
            assert z['ids'].tolist()==ids and z['labels'].tolist()==labels
            assert z['scores'].shape==z['plain_scores'].shape==(len(ids),18)
            assert np.array(labels)[z['true']].tolist()==by_id.loc[ids,'model'].tolist()
            assert np.array_equal(z['scores'],scores[folds.fold.eq(fold)])
            recomputed_folds.append({'fold':fold,'share':share,'ba':float(balanced_accuracy_score(by_id.loc[ids,'model'],np.array(labels)[z['scores'].argmax(1)]))})
    # Each repetition is a separate pooled five-fold evaluation. Do not average
    # per-fold BAs: some sparse validation folds lack one or two true classes.
    sparse=[]
    for share in local['shares']:
        for m in local['sparse_per_prompt']:
            for repeat in range(local['sparse_repeats']):
                ids=[];score_parts=[]
                for fold in range(5):
                    z=np.load(src/f'sparse-f{fold}-a{share}-m{m}-r{repeat}.npz',allow_pickle=False)
                    current=z['ids'].tolist();assert all(fold_of[x]==fold for x in current)
                    valid=train.loc[folds.fold.eq(fold)].copy();valid['_k']=valid.id.map(lambda x:hashlib.sha256(f'42:{repeat}:{x}'.encode()).hexdigest())
                    expected=set(valid.sort_values(['prompt_id','_k']).groupby('prompt_id').head(m).id)
                    assert set(current)==expected and len(current)==len(expected)
                    assert z['scores'].shape==(len(current),18) and np.isfinite(z['scores']).all()
                    assert z['true'].shape==(len(current),)
                    assert np.array(labels)[z['true']].tolist()==by_id.loc[current,'model'].tolist()
                    ids.extend(current);score_parts.append(z['scores'])
                assert len(ids)==len(set(ids))
                truth=by_id.loc[ids,'model'];pred=np.array(labels)[np.vstack(score_parts).argmax(1)]
                assert set(truth)==set(labels)
                sparse.append({'share':share,'m':m,'repeat':repeat,'rows':len(ids),'balanced_accuracy':float(balanced_accuracy_score(truth,pred))})
    output=HERE/'analysis';output.mkdir(exist_ok=True)
    pd.DataFrame(dense).to_csv(output/'dense_metrics.csv',index=False)
    pd.DataFrame(per_class).to_csv(output/'per_class.csv',index=False)
    pd.DataFrame(sparse).to_csv(output/'sparse_pooled_metrics.csv',index=False)
    sm=pd.DataFrame(sparse).groupby(['share','m']).balanced_accuracy.agg(['mean','std','min','max'])
    sm.to_csv(output/'sparse_summary.csv')
    # Cluster bootstrap of paired predictions, conditional on already-fitted
    # models and these development folds; not a fresh-training uncertainty claim.
    groups=sorted(train.prompt_id.unique());base,support=matrix_counts(train,predictions[.5],groups,labels)
    rng=np.random.default_rng(42);draws=rng.integers(0,len(groups),size=(2000,len(groups)))
    denominator=support[draws].sum(1);paired=[]
    folds_metrics=pd.read_csv(src/'fold_metrics.csv')
    assert len(folds_metrics)==15 and not folds_metrics.duplicated(['fold','share']).any()
    expected=pd.DataFrame(recomputed_folds).set_index(['fold','share']).sort_index()
    reported=folds_metrics.set_index(['fold','share']).sort_index()
    assert expected.index.equals(reported.index)
    assert np.allclose(expected.ba,reported.ba,rtol=0,atol=1e-12)
    folds_metrics=expected.reset_index()
    for share in [.25,.75]:
        candidate,_=matrix_counts(train,predictions[share],groups,labels)
        delta=((candidate[draws].sum(1)-base[draws].sum(1))/denominator).mean(1)
        b=predictions[.5]==y;c=predictions[share]==y
        fold_d=folds_metrics[folds_metrics.share.eq(share)].set_index('fold').ba-folds_metrics[folds_metrics.share.eq(.5)].set_index('fold').ba
        paired.append({'share':share,'baseline':.5,'delta_ba':float(balanced_accuracy_score(y,predictions[share])-balanced_accuracy_score(y,predictions[.5])),
                       'bootstrap_low':float(np.quantile(delta,.025)),'bootstrap_high':float(np.quantile(delta,.975)),
                       'fixed_errors':int((~b&c).sum()),'introduced_errors':int((b&~c).sum()),'prediction_disagreements':int((predictions[share]!=predictions[.5]).sum()),
                       'positive_folds':int(fold_d.gt(0).sum()),'negative_folds':int(fold_d.lt(0).sum()),'fold_deltas':{str(k):float(v) for k,v in fold_d.items()}})
    verified={'status':'PASS','scope':'isolated stylegram OOF, no leaderboard claim','protocol_verified':True,'folds':5,'shares':3,'rows':2699,
              'score_type':'uncalibrated LinearSVC decision_function; selected plain/centred rows; argmax matches original stylegram routing, not calibrated probabilities',
              'dense':dense,'paired':paired,'sparse_scenarios':len(sparse),'bootstrap_note':'2000 paired prompt resamples conditional on fitted models/development folds; exploratory, not corrected for selecting a winner',
              'source_result_sha256':sha(src/'result.json'),'runtime_seconds':summary['total_seconds'],'gpu_used':summary['gpu_used'],'environment':env}
    (output/'verified.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','dense':dense,'paired':paired,'sparse_summary':sm.reset_index().to_dict('records')},ensure_ascii=False))

if __name__=='__main__':main()
