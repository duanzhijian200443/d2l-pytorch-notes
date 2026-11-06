"""Local intent and binding receipts for the two specifically authorized submissions.

The Kaggle write is made separately. An existing intent without a response
requires reconciliation against Kaggle before any further write.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from workbench_core import atomic_json, read, sha, require

TARGETS = {
    54: ("apex-zero-054", "zhjianduan/ntl-apex-zero-054",
         "ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd",31,
         "EXP054 Apex V1 original / zero Codex rows"),
    55: ("a1-sinkhorn030-055", "zhjianduan/name-that-llm-a1-sinkhorn030-049",
         "35ddcc24900d4bace4b3c4cc167475ad2c119af83a9eb687ef717f7bea1200b5",32,
         "EXP055 fixed A1 + Sinkhorn 0.30 / zero Codex rows"),
}


def row_bound(row, ref, sid):
    url=urlsplit(row.get("url") or "")
    return url.path=="/code/"+ref and parse_qs(url.query).get("scriptVersionId")==[str(sid)]


def prepare(number, rows):
    name,ref,expected,n_inputs,title=TARGETS[number]
    home=ROOT/"cloud"/name
    path=home/"ready-for-submission.json"
    ready=read(path)
    require(ready["status"]=="READY_FOR_SUBMISSION" and ready["ref"]==ref,"READY_IDENTITY")
    require(ready["source_sha256"]==expected and ready["private"] is True and ready["internet"] is False
            and ready["asset_count"]==n_inputs,"RECIPE_OR_SETTINGS")
    ver=ready["version"];sid=ready["script_version_id"]
    require(type(ver) is int and ver>0 and type(sid) is int and sid>0,"MISSING_FIXED_VERSION")
    require(any(p.is_file() and sha(p)==ready["submission_sha256"] for p in home.rglob("submission.parquet")),
            "NO_VERIFIED_SAVE_OUTPUT")
    require(isinstance(rows,list) and any(r.get("ref")==57879463 for r in rows),"UNKNOWN_ACCOUNT_INVENTORY")
    hits=[row for row in rows if row_bound(row,ref,sid)]
    if hits:
        print(json.dumps({"already_submitted":[r["ref"] for r in hits],"script_version_id":sid}));return
    owner,slug=ref.split("/")
    request={"competitionName":"name-that-llm","kernelOwner":owner,"kernelSlug":slug,
             "hasKernelVersion":True,"kernelVersion":ver,"hasFileName":True,"fileName":"submission.parquet",
             "hasSubmissionDescription":True,"submissionDescription":title+f"; notebook v{ver}, script {sid}"}
    receipt={"status":"INTENT","created_at":datetime.now(timezone.utc).isoformat(),"experiment":f"EXP-{number:03d}",
             "request":request,"ready_sha256":sha(path),"script_version_id":sid,
             "submission_ids_observed_before":[r["ref"] for r in rows],"authorization":"Submit first two choices, hold slots 3 and 4"}
    atomic_json(home/"formal-submit-intent.json",receipt,exclusive=True)
    print(json.dumps(request))


def bind(number, payload):
    name,ref,expected,_,_=TARGETS[number]
    home=ROOT/"cloud"/name
    ready=read(home/"ready-for-submission.json")
    require(ready["ref"]==ref and ready["source_sha256"]==expected,"TARGET_DRIFT")
    sid=ready["script_version_id"]
    intent=read(home/"formal-submit-intent.json") if (home/"formal-submit-intent.json").is_file() else None
    if intent:require(intent["ready_sha256"]==sha(home/"ready-for-submission.json") and intent["script_version_id"]==sid,"INTENT_DRIFT")
    matches=[r for r in payload["rows"] if row_bound(r,ref,sid) and r["ref"]==payload["submission_id"]]
    require(len(matches)==1,"MISSING_SUBMISSION_BINDING")
    row=matches[0]
    status=row.get("status") or "ACCEPTED_STATUS_NOT_RETURNED"
    value=row.get("public_score",row.get("publicScore"))
    receipt={"id":row["ref"],"script_version_id":sid,"notebook_version":ready["version"],
             "observed_at":datetime.now(timezone.utc).isoformat(),"submitted_at":row["date"],"status":status,
             "public_score":float(value) if status=="COMPLETE" and value is not None else None,
             "url":"https://www.kaggle.com/code/"+ref+"?scriptVersionId="+str(sid),
             "description":row.get("description"),"source":"Kaggle official submitted version and submission listing"}
    atomic_json(home/"formal-submit-binding.json",receipt,exclusive=True)
    print(json.dumps(receipt))


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("action",choices=["prepare","bind"]);arg.add_argument("exp",type=int,choices=list(TARGETS))
    args=arg.parse_args()
    data=json.load(sys.stdin)
    (prepare if args.action=="prepare" else bind)(args.exp,data)
