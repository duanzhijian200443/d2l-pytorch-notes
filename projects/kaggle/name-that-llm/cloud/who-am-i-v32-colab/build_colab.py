"""Build the self-contained CPU-preparation / A100-execution notebook."""
import ast
import json
import base64
import zipfile
import io
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parent
lock=json.loads((ROOT.parent/'who-am-i-v30/input-lock.json').read_text(encoding='utf-8'))
lock.update(upstream_version=32,upstream_script_version_id=363222852)
(ROOT/'input-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['colab_runner.py','execute_v32.py','watchdog.py','upstream-v32.ipynb','input-lock.json','reference-submission.parquet']:
        path=ROOT/name
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
        z.write(path,name)
payload=base64.b64encode(buffer.getvalue()).decode()
code='''# CPU prepares inputs; an A100 runtime automatically runs inference and releases itself.
import shutil, subprocess
_gpu_probe = (subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            capture_output=True, text=True)
              if shutil.which('nvidia-smi') else None)
GPU_MODE = bool(_gpu_probe and _gpu_probe.returncode == 0 and _gpu_probe.stdout.strip())

import base64, io, zipfile
from pathlib import Path
_payload = PAYLOAD_LITERAL
_root = Path('/content/v32_payload')
_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload))) as _archive:
    assert all(Path(n).name == n for n in _archive.namelist())
    _archive.extractall(_root)
exec(compile((_root/'colab_runner.py').read_text(), 'colab_runner.py', 'exec'), globals())
'''.replace('PAYLOAD_LITERAL',repr(payload))
cells=[nbformat.v4.new_markdown_cell('''# Who am I V32 — Colab A100

Frozen source: [Roman Tamrazov V32](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i?scriptVersionId=363222852).

1. Run first with **CPU** to prepare the 31 version-pinned inputs. Weights move directly between cloud services; they are not downloaded to your PC.
2. Select **A100** and run the cell again. Hardware is detected automatically; an unexpected GPU is released immediately. The original algorithm and ensemble weights are retained; paths and execution plumbing are adapted for Colab.
3. Results and logs are saved under **MyDrive/Kaggle/name-that-llm/v32-colab/outputs/**. The GPU runtime is released in `finally` on success or error, with an independent four-hour watchdog as a fallback.

This executes the visible examples. It does **not** submit to the competition or establish the 0.660 leaderboard score. A100/Python/container differences are recorded. Existing Kaggle credentials are read only from Colab Secrets (`KAGGLE_API_TOKEN`); no credential is embedded here.

Source Python files and the frozen upstream notebook are preserved with each run for inspection.
'''),nbformat.v4.new_code_cell(code)]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},'colab':{'name':'Who_am_I_V32_A100.ipynb'}})
nbformat.validate(nb)
nbformat.write(nb,ROOT/'Who_am_I_V32_A100.ipynb')
print(json.dumps({'notebook':str(ROOT/'Who_am_I_V32_A100.ipynb'),'bytes':(ROOT/'Who_am_I_V32_A100.ipynb').stat().st_size,'payload_bytes':len(buffer.getvalue())}))
