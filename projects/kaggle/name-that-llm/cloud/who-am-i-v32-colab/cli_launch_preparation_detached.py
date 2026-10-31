"""Launch the authorized CPU preparation in an independent WSL session."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
wrapper = root.parent/'colab-interface-validation/colab_with_proxy.py'
session = sys.argv[1]
log = root/'cli-preparation-recover.log'
with log.open('ab', buffering=0) as stream:
    process = subprocess.Popen(
        [sys.executable,'-u',str(wrapper),'--auth=oauth2','exec','-s',session,
         '--timeout','10800','-f',str(root/'cli_execute_preparation.py')],
        stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,
    )
print(json.dumps({'controller_linux_pid':process.pid,'session':session,'log':str(log)}),flush=True)
