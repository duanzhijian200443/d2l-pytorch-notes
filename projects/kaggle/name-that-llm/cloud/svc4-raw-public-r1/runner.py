"""One offline CPU sweep: four svc4 configurations, fixed prompt-group folds."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
import sklearn
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits, threadpool_info
from svc4 import fit_features, make_classifier, public_sample_weights

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results'))
OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
stamp = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def save_array(path, **arrays):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        if path.suffix=='.npy':
            np.save(stream, arrays['scores'], allow_pickle=False)
        else:
            np.savez_compressed(stream, **arrays)
    temp.replace(path)


def save_csv(path, frame):
    temp = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp,index=False)
    temp.replace(path)


def fingerprint_csr(matrix):
    digest = hashlib.sha256(str(matrix.shape).encode()+str(matrix.dtype).encode())
    for part in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(memoryview(np.ascontiguousarray(part)).cast('B'))
    return digest.hexdigest()


def feature_receipt(features):
    result = []
    for (view, _, _, _, _, scale), vectorizer in zip(features.VIEWS, features.vectorizers_):
        vocabulary = json.dumps(sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items()), ensure_ascii=False, separators=(',', ':')).encode()
        result.append({'view':view, 'scale':scale, 'columns':len(vectorizer.vocabulary_),
                       'vocabulary_sha256':hashlib.sha256(vocabulary).hexdigest(),
                       'idf_sha256':hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest()})
    return result


def run():
    start = time.monotonic()
    for name, digest in P['input_hashes'].items():
        assert sha(HERE/name)==digest, 'Input hash: '+name
    versions = {'numpy':np.__version__, 'pandas':pd.__version__, 'scipy':scipy.__version__, 'sklearn':sklearn.__version__}
    # sklearn >=1.7 changed balanced class weights when sample_weight is given.
    # This sweep fixes 1.6.1 semantics instead of silently changing the contrast.
    assert versions==P['required_versions'], {'actual':versions, 'required':P['required_versions']}
    candidates = [p for p in Path('/kaggle/input').rglob('train.csv') if 'name-that-llm' in str(p)]
    candidates = [p for p in candidates if sha(p)==P['train_sha256']]
    assert len(candidates)==1, 'Expected one pinned official train.csv'
    train = pd.read_csv(candidates[0], dtype={'id':str, 'prompt_id':str})
    folds = pd.read_csv(HERE/'folds.csv', dtype={'id':str, 'prompt_id':str})
    public = pd.read_csv(HERE/'public.csv', dtype={'row_id':str,'id':str,'prompt_id':str,'call_id':str})
    assert len(train)==2699 and len(public)==5400
    for column in ('id','prompt_id','model'):
        assert train[column].tolist()==folds[column].tolist(), column
    assert train.text.map(lambda s:hashlib.sha256(s.encode()).hexdigest()).tolist()==folds.text_sha256.tolist()
    assert train.id.is_unique and folds.groupby('prompt_id').fold.nunique().eq(1).all()
    assert public.id.is_unique and public.call_id.is_unique and public.row_id.is_unique
    assert public.groupby('prompt_id').size().eq(18).all() and public.prompt_id.nunique()==300
    assert public.model.value_counts().eq(300).all()
    assert public.text.map(lambda s:hashlib.sha256(s.encode()).hexdigest()).tolist()==public.text_sha256.tolist()
    labels = P['labels']; lab = {s:i for i,s in enumerate(labels)}
    assert len(labels)==18 and set(public.model)==set(train.model)==set(labels)
    y = train.model.map(lab).to_numpy(); py = public.model.map(lab).to_numpy()
    public_texts = public.text.tolist()
    variants = P['variants']
    assert len(variants)==4 and len({v['id'] for v in variants})==4
    assert {v['public_row_weight'] for v in variants}=={None,.175,.35,.7}
    environment = {'observed_at':stamp(), 'versions':versions, 'python':platform.python_version(),
                   'platform':platform.platform(), 'cpu_count':os.cpu_count(),
                   'memory_total_bytes':psutil.virtual_memory().total, 'gpu_requested':False,
                   'threadpools':threadpool_info(),
                   'protocol':P, 'score_type':'uncalibrated LinearSVC decision_function',
                   'class_weight_semantics':'sklearn 1.6.1 balanced weights from unweighted training class counts',
                   'fitted_weights_saved':False}
    save(OUT/'environment.json', environment)
    save(OUT/'input_preflight.json', {'status':'PASS','official_rows':2699,'public_rows':5400,
         'public_groups':300,'folds':5,'hashes_verified':True,'checked_at':stamp()})
    print('PREFLIGHT PASS '+json.dumps(versions), flush=True)
    oof = {v['id']:np.full((len(train),18),np.nan) for v in variants}
    records = []; completed = []; timings = []
    for fold in range(5):
        fold_start = time.monotonic()
        fit_rows = np.flatnonzero(folds.fold.to_numpy()!=fold)
        valid_rows = np.flatnonzero(folds.fold.to_numpy()==fold)
        assert set(train.prompt_id.iloc[fit_rows]).isdisjoint(train.prompt_id.iloc[valid_rows])
        official_texts = train.text.iloc[fit_rows].tolist()
        valid_texts = train.text.iloc[valid_rows].tolist()
        for augmented in (False, True):
            subset = [v for v in variants if (v['public_row_weight'] is not None)==augmented]
            mode = 'official-plus-raw' if augmented else 'official-only'
            save(OUT/'state.json', {'status':'RUNNING','observed_at':stamp(),'fold':fold,
                 'phase':'features','mode':mode,'completed':completed})
            print(f'FOLD {fold} {mode}: fitting four-view features', flush=True)
            feature_start = time.monotonic()
            texts = official_texts+public_texts if augmented else official_texts
            target = np.r_[y[fit_rows],py] if augmented else y[fit_rows]
            X, features = fit_features(texts)
            XV = features.transform(valid_texts)
            assert X.dtype==XV.dtype==np.dtype('float32')
            assert X.shape[0]==len(target) and XV.shape==(len(valid_rows),X.shape[1])
            feature_seconds = time.monotonic()-feature_start
            train_matrix_hash, valid_matrix_hash = fingerprint_csr(X), fingerprint_csr(XV)
            feature_info = {'fold':fold,'mode':mode,'training_rows':len(target),'validation_rows':len(valid_rows),
                'public_rows_in_feature_fit':5400 if augmented else 0,
                'training_official_ids':train.id.iloc[fit_rows].tolist(),
                'validation_ids':train.id.iloc[valid_rows].tolist(),
                'dtype':str(X.dtype),'features':X.shape[1], 'nnz':X.nnz,
                'csr_bytes':X.data.nbytes+X.indices.nbytes+X.indptr.nbytes,
                'validation_csr_bytes':XV.data.nbytes+XV.indices.nbytes+XV.indptr.nbytes,
                'liblinear_float64_csr_estimate':X.nnz*12+(X.shape[0]+1)*8,
                'train_matrix_sha256':train_matrix_hash,'valid_matrix_sha256':valid_matrix_hash,
                'feature_seconds':feature_seconds,'views':feature_receipt(features)}
            save(OUT/f'feature-fold{fold}-{mode}.json',feature_info)
            del features, texts; gc.collect()
            class_counts = np.bincount(target,minlength=18)
            cw = compute_class_weight('balanced', classes=np.arange(18), y=target)
            for variant in subset:
                exp = variant['id']; weight = variant['public_row_weight']
                weights = public_sample_weights(len(fit_rows), len(public) if augmented else 0, weight or 0.)
                official_mass = np.bincount(target[:len(fit_rows)],weights=weights[:len(fit_rows)],minlength=18)
                public_mass = np.bincount(target[len(fit_rows):],weights=weights[len(fit_rows):],minlength=18)
                save(OUT/'state.json', {'status':'RUNNING','observed_at':stamp(),'fold':fold,
                     'phase':'fit','experiment':exp,'mode':mode,'completed':completed})
                print(f'FIT START {exp} fold={fold} public_weight={weight}',flush=True)
                model = make_classifier()
                fit_start = time.monotonic()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(X,target,sample_weight=weights)
                assert not any(issubclass(w.category,ConvergenceWarning) for w in captured), 'SVM did not converge'
                fit_seconds = time.monotonic()-fit_start
                assert model.classes_.tolist()==list(range(18))
                pred_start = time.monotonic(); scores = model.decision_function(XV)
                prediction_seconds = time.monotonic()-pred_start
                assert scores.shape==(len(valid_rows),18) and np.isfinite(scores).all()
                oof[exp][valid_rows]=scores
                dest = OUT/exp;dest.mkdir(exist_ok=True)
                save_array(dest/f'fold{fold}.npz',ids=train.id.iloc[valid_rows].to_numpy(dtype=str),
                    labels=np.array(labels),true=y[valid_rows],scores=scores,
                    train_matrix_sha256=np.array(train_matrix_hash),valid_matrix_sha256=np.array(valid_matrix_hash))
                weights_info = {'fold':fold,'public_row_weight':weight,'labels':labels,'class_counts':class_counts.tolist(),
                    'class_weights':cw.tolist(),'official_sample_mass':official_mass.tolist(),
                    'public_sample_mass':public_mass.tolist(),'effective_class_mass':((official_mass+public_mass)*cw).tolist(),
                    'sample_weight_dtype':str(weights.dtype),'semantics':environment['class_weight_semantics']}
                save(dest/f'weights-fold{fold}.json',weights_info)
                rec={'experiment':exp,'fold':fold,'public_row_weight':weight,
                    'ba':float(balanced_accuracy_score(y[valid_rows],scores.argmax(1))),
                    'rows':len(valid_rows),'training_rows':len(target),'features':X.shape[1],
                    'n_iter':int(model.n_iter_),'feature_seconds':feature_seconds,
                    'fit_seconds':fit_seconds,'prediction_seconds':prediction_seconds,
                    'rss_bytes':psutil.Process().memory_info().rss,'train_matrix_sha256':train_matrix_hash,
                    'valid_matrix_sha256':valid_matrix_hash,'native_estimator_parameters':model.get_params()}
                records.append(rec);completed.append([exp,fold]);save(dest/f'metrics-fold{fold}.json',rec)
                save_csv(OUT/'fold_metrics.csv',pd.DataFrame([{k:v for k,v in row.items() if k!='native_estimator_parameters'} for row in records]))
                print('FIT COMPLETE '+json.dumps({k:rec[k] for k in ('experiment','fold','ba','fit_seconds','n_iter','rss_bytes')}),flush=True)
                del model,weights,scores;gc.collect()
            assert fingerprint_csr(X)==train_matrix_hash and fingerprint_csr(XV)==valid_matrix_hash,'Feature matrix mutated'
            del X,XV,target;gc.collect()
        timings.append({'fold':fold,'wall_seconds':time.monotonic()-fold_start})
    expected_completed={(v['id'],fold) for v in variants for fold in range(5)}
    assert len(records)==len(completed)==20 and {tuple(x) for x in completed}==expected_completed
    summaries=[]
    for variant in variants:
        exp=variant['id'];scores=oof[exp];assert np.isfinite(scores).all()
        pred=np.array(labels)[scores.argmax(1)]
        frame=folds[['id','prompt_id','model','fold']].copy();frame['predicted_model']=pred;frame['correct']=pred==train.model.to_numpy()
        save_csv(OUT/exp/'oof.csv',frame);save_array(OUT/exp/'oof_decision.npy',scores=scores)
        summaries.append({'experiment':exp,'public_row_weight':variant['public_row_weight'],
             'balanced_accuracy':float(balanced_accuracy_score(train.model,pred)),'correct':int(frame.correct.sum())})
    save(OUT/'result.json',{'status':'COMPLETE','observed_at':stamp(),'scope':P['scope'],
         'trained_svms':len(records),'convergence_warnings':0,'gpu_used':False,'summaries':summaries,
         'fold_timings':timings,'total_seconds':time.monotonic()-start,'fitted_weights_saved':False})
    save(OUT/'artifacts.json',{'files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
         for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in
         {'state.json','guard.json','guard.json.tmp','training.log','.run.lock','artifacts.json'} and not p.name.endswith('.tmp')]})
    save(OUT/'state.json',{'status':'COMPLETE','observed_at':stamp(),'completed':completed})
    print('EXPERIMENT COMPLETE '+json.dumps(summaries),flush=True)


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):
            run()
    except BaseException:
        save(OUT/'failure.json',{'observed_at':stamp(),'traceback':traceback.format_exc()})
        save(OUT/'state.json',{'status':'ERROR','observed_at':stamp()})
        raise
