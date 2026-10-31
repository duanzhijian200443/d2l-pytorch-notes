"""Keep the official CLI controller alive across desktop tool reconnections."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
wrapper = root.parent / 'colab-interface-validation/colab_with_proxy.py'
session = sys.argv[1]
with (root / 'cli-gpu-controller.log').open('ab', buffering=0) as stream:
    process = subprocess.Popen(
        [sys.executable, '-u', str(wrapper), '--auth=oauth2', 'exec', '-s', session,
         '--timeout', '14400', '-f', str(root / 'cli_execute_gpu.py')],
        stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True,
    )
print(json.dumps({'controller_linux_pid': process.pid, 'session': session}), flush=True)
