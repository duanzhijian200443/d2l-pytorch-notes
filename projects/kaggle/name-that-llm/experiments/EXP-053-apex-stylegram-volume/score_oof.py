"""Verify official five-fold predictions and compare the prespecified three arms."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

EX = Path(__file__).resolve().parent
ROOT = EX.parents[1]
JOB = ROOT/"cloud/apex-stylegram-053/style"
ARMS = ("zero","four120","four240")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def analyze():
    hist=ROOT/"experiments/EXP-042-luna-terra120-ablation/verify_outputs.py"
    require(digest(hist)=="525bb72d185a176773c5e4e30ec7771cc3bf451f8c85efda3f4c597599d0661e","Metric code differs")
    spec=importlib.util.spec_from_file_location("fixed_official_metrics",hist)
    calc=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calc)
    results=JOB/"outputs/results"
    summary=read(results/"summary.json")
    remote=read(JOB/"outputs/download-receipt.json")
    require(remote["status"]=="COMPLETE" and remote["archive_sha256"]==digest(JOB/"outputs/exp053-results.zip"),"Archive identity")
    require(summary["status"]=="COMPLETE" and summary["trained_svms"]==36
            and summary["config_sha256"]==digest(JOB/"experiment-config.json"),"Fit identity")
    official=pd.read_csv(ROOT/"train.csv",dtype=str)
    folds=pd.read_csv(ROOT/"validation/prompt-group-v1/folds.csv")
    require(official.id.tolist()==folds.id.tolist() and folds.groupby("prompt_id").fold.nunique().eq(1).all(),"Fold IDs")
    labels=sorted(official.model.unique())
    y=official.model.map({label:i for i,label in enumerate(labels)}).to_numpy(dtype=int)
    groups=official.prompt_id.to_numpy()
    require(len(official)==2699 and len(set(groups))==150 and len(labels)==18,"Official grid")
    predictions={}
    ratios={}
    class_rows=[]
    row_table=official[["id","prompt_id","model"]].copy()
    row_table["fold"]=folds.fold
    for arm in ARMS:
        path=results/"oof"/(arm+".npz")
        bound=summary["outputs"]["oof"][arm]
        require(digest(path)==bound["sha256"],"Unbound OOF "+arm)
        with np.load(path,allow_pickle=False) as mat:
            require(mat["ids"].tolist()==official.id.tolist() and mat["labels"].tolist()==labels
                    and np.array_equal(mat["fold"],folds.fold) and mat["arm"].item()==arm
                    and mat["config_sha256"].item()==summary["config_sha256"],"OOF row/class identity")
            scores=mat["scores"];prob=mat["probabilities"]
            require(scores.shape==prob.shape==(2699,18) and np.isfinite(scores).all() and np.isfinite(prob).all()
                    and np.allclose(prob.sum(axis=1),1,atol=1e-6) and np.array_equal(scores.argmax(1),prob.argmax(1)),"OOF numerical contract")
            predictions[arm]=prob.argmax(1)
        pred=predictions[arm]
        row_table[arm]=np.asarray(labels)[pred]
        ratios[arm]=calc.metrics(y,pred)
        for index,label in enumerate(labels):
            hit=(y==index)&(pred==index)
            class_rows.append({"arm":arm,"label":label,"support":int((y==index).sum()),"correct":int(hit.sum()),
                               "recall":float(hit.sum()/(y==index).sum()),"false_positives":int(((pred==index)&(y!=index)).sum())})
    comparisons={}
    fold_rows=[]
    for a,b in [("four120","four240"),("zero","four240"),("zero","four120")]:
        report=calc.paired_metrics(y,predictions[a],predictions[b],groups)
        report["bootstrap"].pop("deltas")
        name=b+"_minus_"+a
        comparisons[name]=report
        for fold in range(5):
            mask=folds.fold.to_numpy()==fold
            ma=calc.metrics(y[mask],predictions[a][mask]);mb=calc.metrics(y[mask],predictions[b][mask])
            fold_rows.append({"comparison":name,"fold":fold,"delta_ba":mb["balanced_accuracy"]-ma["balanced_accuracy"],
                              "net_correct":mb["correct"]-ma["correct"]})
    out=EX/"metrics/oof"
    out.mkdir(parents=True,exist_ok=False)
    row_table.to_csv(out/"predictions.csv",index=False)
    pd.DataFrame(class_rows).to_csv(out/"per-class.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"fold-deltas.csv",index=False)
    result={"status":"PASS_OFFICIAL_FIVE_FOLD_STYLEGRAM","score_scope":"Stylegram on official development folds, not full Kaggle ensemble",
            "metrics":ratios,"comparisons":comparisons,"labels":labels,"source_sha256":summary["source_sha256"],
            "fit_summary_sha256":digest(results/"summary.json"),"fit_archive_sha256":remote["archive_sha256"],
            "outputs":{p.name:digest(p) for p in out.glob("*.csv")}}
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print({k:round(v["balanced_accuracy"],7) for k,v in ratios.items()})


if __name__=="__main__":
    analyze()
