"""Validate newly allocated hardware and cap setup time before the main watchdog."""
import json
import subprocess
import sys
from pathlib import Path

gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                    '--format=csv,noheader'],capture_output=True,text=True,check=True)
if 'A100' not in gpu.stdout:
    from google.colab import runtime
    runtime.unassign()
    raise RuntimeError('Unexpected accelerator; runtime released')
code='''import json,os,time,urllib.request
from pathlib import Path
deadline=time.monotonic()+1200
while time.monotonic()<deadline:
    p=Path('/content/v32_gpu_run_started.json')
    if p.is_file():
        try:
            os.kill(json.loads(p.read_text())['watchdog_pid'],0)
            raise SystemExit(0)
        except (ProcessLookupError,KeyError,ValueError):
            pass
    time.sleep(10)
address=os.environ['TBE_RUNTIME_ADDR']
for attempt in range(4):
    try:
        urllib.request.urlopen(urllib.request.Request('http://'+address+'/unassign',method='POST'),timeout=20).close()
        break
    except Exception:
        time.sleep(5)
'''
path=Path('/content/v32_setup_guard.py')
path.write_text(code)
proc=subprocess.Popen([sys.executable,str(path)],stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'gpu':gpu.stdout.strip(),'setup_guard_pid':proc.pid,'setup_timeout_seconds':1200}),flush=True)
