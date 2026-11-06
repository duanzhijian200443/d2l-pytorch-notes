"""Run the common Apex four-view main SVC once, using its published code."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE_SHA = "ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(inputs, output):
    source = inputs / "source.ipynb"
    if digest(source) != SOURCE_SHA:
        raise ValueError("Unexpected author source")
    cells = json.loads(source.read_text())["cells"]
    setup, infer = ["".join(cells[i]["source"]) for i in (4, 10)]
    pool_raw = pd.read_parquet(inputs / "ntl_open_pool.parquet")
    train = pd.read_csv(inputs / "train.csv")
    panel = pd.read_csv(inputs / "panel.csv", dtype=str, keep_default_na=False)
    labels = sorted(train.model.unique().tolist())
    if list(panel.columns) != ["id", "text"] or len(panel) != 729 or not panel.id.is_unique:
        raise ValueError("Expected the blinded cache panel")
    safe = pool_raw[pool_raw.model.isin(labels)].copy()
    for field in ("flag_train_prompt_regen", "flag_gemini37_routing_risk"):
        safe = safe[~safe[field].astype(bool)]
    if len(safe) != 190588 or len(cells) < 17 or "flag_train_prompt_regen" not in setup:
        raise ValueError("Unexpected parent pool")
    safe = safe[["text","model"]].copy()
    safe["text"] = safe.text.fillna("").astype(str)
    safe_path = inputs / "pool_open_safe.parquet"
    safe.to_parquet(safe_path, index=False)
    del safe, pool_raw
    gc.collect()
    emitted = []
    def log(*args):
        message = " ".join(map(str, args))
        emitted.append(message)
        print(message, flush=True)
    namespace = dict(np=np, pd=pd, gc=gc, re=re, os=__import__("os"), LABELS=labels,
                     L2I={label: i for i, label in enumerate(labels)}, train=train, test=panel, LP={},
                     a=SimpleNamespace(svc_w=12., pool=str(safe_path), sim=None,
                                       extra=str(inputs/"nlm_matched_clean.parquet"), extra_w=.5),
                     TfidfVectorizer=TfidfVectorizer, log=log)
    matched = infer[infer.index("EXTRA_DF = pd.DataFrame"):infer.index("FIT_T = np.concatenate")]
    main = infer[infer.index("if a.svc_w > 0:"):infer.index("if a.distort_w > 0:")]
    exec(compile(matched, "author-matched-source", "exec"), namespace)
    exec(compile(main, "author-main-svc4", "exec"), namespace)
    scores = namespace["LP"]["svc4"]
    if scores.shape != (729, 18) or not np.isfinite(scores).all():
        raise ValueError("Incomplete main SVC prediction")
    fit_line = next(x for x in emitted if x.startswith("svc4 fit rows="))
    if not fit_line.startswith("svc4 fit rows=200478 "):
        raise ValueError("Unexpected V13 training-row count: "+fit_line)
    output.mkdir(exist_ok=True)
    target = output/"main.npz"
    np.savez_compressed(target, ids=panel.id.to_numpy(dtype=str),labels=np.array(labels), weighted_logp=scores)
    receipt = {"status":"COMPLETE","source_sha256":SOURCE_SHA,"pool_rows":190588,"fit_log":fit_line,
               "coefficient":12.0,"format":"weighted log probability, source dtype",
               "score_sha256":digest(target),"role":"common main branch on exposed diagnostic panel"}
    (output/"main-result.json").write_text(json.dumps(receipt,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    compute(args.input,args.output)
