"""Follow the three already-submitted jobs until terminal; no remote writes or new submissions."""
import os,sys,time,subprocess
from datetime import datetime,timezone
from formal import ROOT,OUT,props,read,atomic_json,now,require
from records import refresh

LIMIT=12*3600
INTERVAL=900

def main():
 ids=[read(props(n)[2]/'formal-submit-binding.json')['id'] for n in (56,57)]
 state=OUT/'score-watch.json'
 started=time.monotonic()
 from workbench_core import local_lock
 with local_lock('EXP-056'):
  last=None
  while True:
   if time.monotonic()-started>=LIMIT:
    atomic_json(state,{'status':'WATCH_TIMED_OUT','observed_at':now(),'ids':ids,'last':last});return
   try:
    refresh();last=read(OUT/'latest-comparison.json')
    require([last['arms'][s]['id'] for s in ('120','240')]==ids,'UNEXPECTED_ID')
    statuses=[a['status'] for a in last['arms'].values()]
    if all(s in ('COMPLETE','ERROR','FAILED','CANCELLED','CANCELED') for s in statuses):
     subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'tracking/ledger.py'),'render'],check=True,cwd=ROOT)
     atomic_json(state,{'status':'TERMINAL','ids':ids,'observed_at':now(),'scores':last});return
    atomic_json(state,{'status':'WAITING','observed_at':now(),'ids':ids,'jobs':statuses,'last_successful_query':last['observed_at'],'pid':os.getpid()})
   except Exception as e:
    atomic_json(state,{'status':'QUERY_FAILED_WILL_RETRY','observed_at':now(),'ids':ids,'error_type':type(e).__name__,'pid':os.getpid(),'prior_scores_retained':True})
   time.sleep(min(INTERVAL,max(1,LIMIT-(time.monotonic()-started))))
if __name__=='__main__':main()
