"""Run frozen Who am I V32 on Colab; save evidence before releasing the runtime."""
import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
import traceback
from pathlib import Path

PAYLOAD = Path('/content/v32_payload')
INPUT = Path('/content/ntl_v32/input')
WORK = Path('/kaggle/working')
DRIVE = Path('/content/drive/MyDrive/Kaggle/name-that-llm/v32-colab')
GPU_MODE = globals().get('GPU_MODE', False)
PREPARED_BY_CLI = globals().get('PREPARED_BY_CLI', False)
GPU_LIMIT_SECONDS = 4 * 3600
started = time.time()
watchdog = None
output = None


def release_runtime():
    from google.colab import runtime
    for attempt in range(3):
        try:
            runtime.unassign()
            return
        except Exception as exc:
            print('Release retry', attempt + 1, type(exc).__name__, flush=True)
            time.sleep(3)


def backup_small_files():
    if output is None:
        return
    for source in WORK.glob('*'):
        if source.is_file() and source.suffix in {'.json', '.log', '.parquet', '.csv', '.py'}:
            shutil.copy2(source, output / source.name)


def run_logged(command, name, **kwargs):
    logfile = WORK / name
    with logfile.open('w', buffering=1) as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        while proc.poll() is None:
            time.sleep(15)
            if output:
                shutil.copy2(logfile, output / name)
            tail = logfile.read_text(errors='replace').splitlines()
            print(name, 'running', int(time.time() - started), 'seconds',
                  '\n'.join(tail[-2:])[:600], flush=True)
        if proc.returncode:
            raise RuntimeError(f'{name} failed with exit code {proc.returncode}; see saved log')


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if target.resolve() == Path(source).resolve():
            return
        raise RuntimeError(f'Refusing to overwrite input directory: {target}')
    target.symlink_to(Path(source).resolve(), target_is_directory=True)


def prepare_inputs():
    import kagglehub
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    lock = json.loads((PAYLOAD / 'input-lock.json').read_text())
    os.environ['KAGGLEHUB_CACHE'] = str(DRIVE / 'cache')

    def download(kind, handle):
        functions = {'dataset':kagglehub.dataset_download, 'notebook':kagglehub.notebook_output_download,
                     'model':kagglehub.model_download, 'competition':kagglehub.competition_download}
        fallback = DRIVE / 'api-cache' / kind / handle.replace('/versions/', '/')
        marker = fallback / '.v32-download-complete.json'
        if marker.is_file():
            return fallback
        try:
            return Path(functions[kind](handle.lower() if kind == 'model' else handle))
        except Exception as exc:
            if '404' not in str(exc):
                raise
            print('KaggleHub returned 404; using official Kaggle CLI SDK:', kind, handle, flush=True)
            fallback.mkdir(parents=True, exist_ok=True)
            if kind == 'dataset':
                api.dataset_download_files(handle.replace('/versions/', '/'), path=str(fallback), unzip=True, quiet=True)
            elif kind == 'notebook':
                api.kernels_output(handle.replace('/versions/', '/'), path=str(fallback), quiet=True)
            elif kind == 'model':
                api.model_instance_version_download(handle, path=str(fallback), untar=True, quiet=True)
            else:
                api.competition_download_files(handle, path=str(fallback), quiet=True)
                import zipfile
                with zipfile.ZipFile(fallback / (handle + '.zip')) as z:
                    z.extractall(fallback)
                (fallback / (handle + '.zip')).unlink()
            marker.write_text(json.dumps({'handle':handle,'kind':kind,'status':'COMPLETE'}))
            return fallback
    inventory = []
    for item in lock['datasets']:
        ref = item['ref']
        slug = ref.split('/')[-1]
        if slug == 'llama31-8b-base-verified':
            source = Path('/content/drive/MyDrive/Kaggle/name-that-llm/models/llama-3.1-8b-base-1f47e50')
            assert (source / 'transfer_manifest.json').is_file(), 'Verified Llama copy missing from Drive'
            if GPU_MODE:
                local = Path('/content/v32_llama')
                if not local.exists():
                    assert shutil.disk_usage('/content').free > 25 * 1024**3, 'Need 25 GiB free for local Llama cache'
                    print('Caching verified Llama weights on Colab disk', flush=True)
                    shutil.copytree(source, local)
                source = local
        else:
            handle = f"{ref}/versions/{item['pinned_version']}"
            print('Dataset', handle, flush=True)
            source = download('dataset', handle)
        link(source, INPUT / 'datasets' / ref)
        inventory.append({'kind': 'dataset', 'ref': ref, 'version': item['pinned_version'], 'path': str(source)})
    for item in lock['kernels']:
        handle = f"{item['ref']}/versions/{item['version']}"
        print('Notebook output', handle, flush=True)
        source = download('notebook', handle)
        link(source, INPUT / 'notebooks' / item['ref'].split('/')[-1])
        inventory.append({'kind': 'notebook', 'ref': handle, 'path': str(source)})
    for handle in lock['models']:
        print('Base model', handle, flush=True)
        source = download('model', handle)
        link(source, INPUT / 'models' / handle.lower())
        inventory.append({'kind': 'model', 'ref': handle, 'path': str(source)})
    source = download('competition', 'name-that-llm')
    link(source, INPUT / 'competitions' / 'name-that-llm')
    assert len(inventory) == 30
    (WORK / 'input_inventory.json').write_text(json.dumps(inventory, indent=2))
    checked = []
    import glob
    for relative, expected in lock['pinned_csv_sha256'].items():
        paths = glob.glob(str(INPUT / '**' / Path(relative).name), recursive=True)
        matches = [p for p in paths if hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected]
        assert len(matches) == 1, ('Historical input mismatch', relative)
        checked.append({'input': relative, 'path': matches[0], 'sha256': expected})
    (WORK / 'pinned_inputs_verified.json').write_text(json.dumps({'status': 'PASS', 'files': checked}, indent=2))
    print('All 31 inputs resolved; historical hashes PASS', flush=True)


def link_cli_prepared_inputs():
    preparation = json.loads((DRIVE / 'cli_preparation_result.json').read_text())
    assert preparation['status'] == 'READY', 'CPU input preparation is not complete'
    inventory = json.loads((DRIVE / 'cli_input_inventory.json').read_text())
    assert len(inventory) == 31
    for item in inventory:
        source = Path(item['path'])
        assert source.is_dir(), f'Missing prepared input: {item["ref"]}'
        if item['ref'] == 'zhjianduan/llama31-8b-base-verified':
            assert (source / 'transfer_manifest.json').is_file()
            local = Path('/content/v32_llama')
            assert not local.exists(), 'Unexpected pre-existing Llama local cache'
            assert shutil.disk_usage('/content').free > 25 * 1024**3
            print('Caching verified Llama weights on Colab disk', flush=True)
            shutil.copytree(source, local)
            source = local
        else:
            marker = json.loads((source / '.v32-download-complete.json').read_text())
            assert (marker['kind'],marker['ref'],marker['version']) == (item['kind'],item['ref'],item['version'])
        if item['kind'] == 'dataset':
            target = INPUT / 'datasets' / item['ref']
        elif item['kind'] == 'notebook':
            target = INPUT / 'notebooks' / item['ref'].split('/')[-1]
        elif item['kind'] == 'model':
            target = INPUT / 'models' / item['ref'].lower()
        else:
            target = INPUT / 'competitions' / item['ref']
        link(source, target)
    (WORK / 'input_inventory.json').write_text(json.dumps(inventory,indent=2))
    verified = json.loads((DRIVE / 'cli_pinned_inputs_verified.json').read_text())
    assert verified['status'] == 'PASS'
    for item in verified['files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    (WORK / 'pinned_inputs_verified.json').write_text(json.dumps(verified,indent=2))
    print('All 31 CLI-prepared inputs linked; historical hashes PASS', flush=True)


try:
    WORK.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    if GPU_MODE:
        # Separate process survives a kernel crash and releases this runtime at the hard limit.
        watchdog = subprocess.Popen([sys.executable, str(PAYLOAD / 'watchdog.py'), str(os.getpid()), str(GPU_LIMIT_SECONDS)],
                                    stdout=(WORK / 'watchdog.log').open('w'), stderr=subprocess.STDOUT,
                                    start_new_session=True)
        Path('/content/v32_gpu_run_started.json').write_text(json.dumps({'watchdog_pid':watchdog.pid,'started_at':started}))
        gpu_check = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   check=True, capture_output=True, text=True)
        assert 'A100' in gpu_check.stdout, 'An A100 is required; releasing the unexpected GPU'
    if PREPARED_BY_CLI:
        assert (DRIVE / 'cli_preparation_result.json').is_file(), 'Mount Drive with the official CLI first'
    else:
        from google.colab import drive, userdata
        drive.mount('/content/drive')
    DRIVE.mkdir(parents=True, exist_ok=True)
    output = DRIVE / 'outputs' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    output.mkdir(parents=True, exist_ok=False)
    if PREPARED_BY_CLI:
        (DRIVE / 'cli_gpu_run_state.json').write_text(json.dumps({'status':'RUNNING','output_directory':str(output),'started_at':started,'watchdog_seconds':GPU_LIMIT_SECONDS},indent=2))
    for path in PAYLOAD.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    if PREPARED_BY_CLI:
        link_cli_prepared_inputs()
    else:
        os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
        run_logged([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub==1.0.3', 'kaggle==2.2.5'], 'setup-data.log')
        prepare_inputs()
    if not GPU_MODE:
        (WORK / 'preparation_result.json').write_text(json.dumps({'status':'READY', 'inputs':31, 'output_directory':str(output)},indent=2))
        print('CPU PREPARATION READY. Switch to A100, then Run all; GPU detection is automatic.', flush=True)
    else:
        # Run inference in a fresh process after installing to avoid stale in-kernel imports.
        packages = ['torch==2.11.0', 'transformers==5.16.1', 'peft==0.20.0',
                    'numpy==2.1.3', 'pandas==2.3.3', 'scipy==1.16.3',
                    'scikit-learn==1.6.1', 'lightgbm==4.6.0', 'safetensors==0.8.0',
                    'pyarrow==23.0.1', 'sentencepiece', 'protobuf', 'accelerate']
        install_command = [sys.executable, '-m', 'pip', 'install', '-q']
        if PREPARED_BY_CLI:
            wheels = json.loads((DRIVE / 'cli_wheels_result.json').read_text())
            assert wheels['status'] == 'READY', 'CPU wheel preparation must complete before A100 starts'
            install_command += ['--no-index', '--find-links', str(DRIVE / 'cli-wheels')]
            driver = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip().splitlines()[0]
            driver_major = int(driver.split('.')[0])
            # The PyPI 2.11.0 wheel uses CUDA 13; older Colab drivers need the
            # same Torch release's official CUDA 12.8 build used on Kaggle.
            if driver_major < 580:
                compatibility = json.loads((DRIVE / 'cli_torch_cu128_result.json').read_text())
                assert compatibility['status'] == 'READY', 'CUDA 12.8 fallback cache is incomplete'
                packages[0] = 'torch==2.11.0+cu128'
                install_command += ['--find-links', str(DRIVE / 'cli-wheels-cu128')]
            (WORK / 'cuda_build_selection.json').write_text(json.dumps(
                {'driver': driver, 'torch_requirement': packages[0]}, indent=2))
        run_logged(install_command + packages, 'setup-inference.log')
        # Confirm an actual CUDA operation in the freshly installed environment.
        run_logged([sys.executable, '-c',
                    "import json,torch; assert torch.cuda.is_available(); "
                    "x=torch.ones(4,device='cuda'); assert (x+x).sum().item()==8; "
                    "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                    "'gpu':torch.cuda.get_device_name(0),'operation':'PASS'}))"],
                   'cuda-preflight.log')
        env = os.environ.copy()
        env.update({'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
                    'V32_INPUT_ROOT':str(INPUT), 'V32_OUTPUT_DIR':str(output)})
        env.pop('KAGGLE_API_TOKEN', None)
        run_logged([sys.executable, '-u', str(PAYLOAD / 'execute_v32.py')], 'v32-run.log', cwd=WORK, env=env)
        print('V32 RUN FINISHED; evidence saved to', output, flush=True)
except BaseException as exc:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'failure.json').write_text(json.dumps({'status':'ERROR','exception_type':type(exc).__name__,
        'message':str(exc), 'elapsed_seconds':time.time()-started},indent=2))
    traceback.print_exc()
    raise
finally:
    os.environ.pop('KAGGLE_API_TOKEN', None)
    try:
        backup_small_files()
        if PREPARED_BY_CLI and output:
            result_path = WORK / 'reproduction_result.json'
            failure_path = WORK / 'failure.json'
            result = json.loads(result_path.read_text()) if result_path.is_file() else (json.loads(failure_path.read_text()) if failure_path.is_file() else {'status':'INCOMPLETE'})
            (DRIVE / 'cli_gpu_run_state.json').write_text(json.dumps({'status':result['status'],'output_directory':str(output),'elapsed_seconds':time.time()-started,'release_requested':GPU_MODE,'result':result},indent=2))
        os.sync()
        if output:
            print('Saved output directory:', output, flush=True)
    finally:
        if GPU_MODE:
            print('Releasing GPU runtime now', flush=True)
            release_runtime()
