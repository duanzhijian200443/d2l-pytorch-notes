"""Persist a one-shot submission intent after quick-run verification; does not submit."""
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import control

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    with control.command_lock():
        state=control.state()
        assert state['run_version']==2 and state['run_request_count']==1
        assert state['last_status']['version']==2 and state['last_status']['status']=='COMPLETE'
        verification=read(HERE/'quick-commit-verification.json')
        assert verification['status']=='PASS_QUICK_COMMIT_ONLY' and verification['version']==2
        for name,digest in verification['outputs'].items():
            assert sha(HERE/'outputs-v2'/name)==digest,name
        rb=state['readback_receipts']['2']; p=HERE/rb['path']
        assert sha(p)==rb['sha256']
        assert read(p)['status']=='PASS' and read(p)['local_bindings']==control.local_package()[2]
        review=read(ROOT/'experiments/EXP-046-atlas20-a1-submission/metrics/final-review.json')
        assert review['status'].startswith('PASS'), 'Final independent review is not PASS'
        assert not (HERE/'submission-intent.json').exists() and not (HERE/'submission-ack.json').exists()
        request={'competitionName':'name-that-llm','kernelOwner':'zhjianduan',
            'kernelSlug':'name-that-llm-a1-public-stylegram-046','kernelVersion':2,'hasKernelVersion':True,
            'fileName':'submission.parquet','hasFileName':True,
            'submissionDescription':'EXP046-A1-v2 public10073 Stylegram only; original ATLAS20 .20 comp .15 Sinkhorn .45',
            'hasSubmissionDescription':True}
        intent={'status':'REQUEST_INTENT_PERSISTED','request_count':1,'observed_at':datetime.now(timezone.utc).isoformat(),
            'request':request,'quick_verification_sha256':sha(HERE/'quick-commit-verification.json'),
            'local_bindings':control.local_package()[2],
            'authority':'Explicit user request in this chat to submit EXP045 A1 to Kaggle.',
            'unknown_response_policy':'Do not resend. Reconcile submission ref or unique description with exact saved version.'}
        control.save(HERE/'submission-intent.json',intent,exclusive=True)
        control.update(status='COMPETITION_SUBMISSION_IN_FLIGHT',submission_request_count=1,submission_outcome='UNKNOWN_UNTIL_RESPONSE')
        print(json.dumps(request,ensure_ascii=False))

if __name__=='__main__':main()
