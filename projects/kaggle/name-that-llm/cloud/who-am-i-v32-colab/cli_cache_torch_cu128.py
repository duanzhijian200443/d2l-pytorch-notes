"""Cache the same Torch release's CUDA 12.8 build before allocating a paid GPU."""
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

root = Path('/content/drive/MyDrive/Kaggle/name-that-llm/v32-colab')
destination = root / 'cli-wheels-cu128'
destination.mkdir(exist_ok=True)
report_path = root / 'cli_torch_cu128_result.json'
started = time.time()
report = {'status': 'RUNNING', 'started_at': started, 'torch': '2.11.0+cu128'}
report_path.write_text(json.dumps(report, indent=2))
try:
    with (root / 'cli-torch-cu128.log').open('w') as stream:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
             '--dest', str(destination), '--find-links', str(root / 'cli-wheels'),
             '--index-url', 'https://download.pytorch.org/whl/cu128',
             'torch==2.11.0+cu128'],
            stdout=stream, stderr=subprocess.STDOUT, timeout=900,
        )
    assert result.returncode == 0, f'Wheel download failed: {result.returncode}'
    wheels = list(destination.glob('torch-2.11.0+cu128-*.whl'))
    assert len(wheels) == 1, 'Expected one CUDA 12.8 Torch wheel'
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode()
    report.update(status='READY', wheel=wheels[0].name,
                  cuda_dependencies=[line for line in metadata.splitlines() if line.startswith('Requires-Dist: nvidia')])
except BaseException as exc:
    report.update(status='ERROR', error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    report['elapsed_seconds'] = time.time() - started
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
