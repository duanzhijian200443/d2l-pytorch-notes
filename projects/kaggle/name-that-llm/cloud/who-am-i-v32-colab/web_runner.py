"""Browser entry point: reuse prepared Drive inputs, persist evidence, release A100."""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parent
DRIVE = Path('/content/drive/MyDrive/Kaggle/name-that-llm/v32-colab')
WORK = Path('/content/v32-web-work')
INPUT = Path('/content/ntl_v32/input')
LIMIT_SECONDS = 14400


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def copy_atomic(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise RuntimeError(f'Unexpected non-symlink input directory: {target}')
    target.symlink_to(source, target_is_directory=True)


def prepare_cached_inputs():
    ready = json.loads((DRIVE / 'cli_preparation_result.json').read_text())
    assert ready['status'] == 'READY' and ready['completed_inputs'] == 31 and ready['historical_hashes'] == 'PASS'
    inventory = json.loads((DRIVE / 'cli_input_inventory.json').read_text())
    assert len(inventory) == 31
    lock = json.loads((PAYLOAD / 'input-lock.json').read_text())
    declared = {('dataset', x['ref'], str(x['pinned_version'])) for x in lock['datasets']}
    declared |= {('notebook', x['ref'], str(x['version'])) for x in lock['kernels']}
    declared |= {('model', x.lower(), x.rsplit('/', 1)[-1]) for x in lock['models']}
    declared.add(('competition', 'name-that-llm', 'None'))
    actual = {(x['kind'], x['ref'].lower() if x['kind'] == 'model' else x['ref'], str(x['version'])) for x in inventory}
    assert len(actual) == 31 and actual == declared, 'Prepared cache does not match the frozen 31-input lock'
    for item in inventory:
        source = Path(item['path'])
        assert source.is_dir(), f'Cached input missing: {item["ref"]}; stop instead of downloading again'
        if item['ref'] == 'zhjianduan/llama31-8b-base-verified':
            manifest = json.loads((source / 'transfer_manifest.json').read_text())
            assert manifest['status'] == 'VERIFIED'
            assert manifest['source']['revision'] == '1f47e50cdbe801ad8a5174156ec3a0655108fb9f'
            local = Path('/content/v32_llama')
            marker = local / '.web-copy-complete.json'
            digest = sha256(source / 'transfer_manifest.json')
            valid = marker.is_file() and json.loads(marker.read_text()).get('manifest_sha256') == digest
            if valid:
                valid = all((local / f['file']).is_file() and (local / f['file']).stat().st_size == f['bytes'] for f in manifest['files'])
            if not valid:
                assert shutil.disk_usage('/content').free > 20 * 1024**3, 'Need 20 GiB free for the local Llama copy'
                local.mkdir(exist_ok=True)
                print('从已有 Drive 缓存复制 Llama 到本次运行时磁盘（不重新下载）。', flush=True)
                for f in source.iterdir():
                    if f.is_file():
                        copy_atomic(f, local / f.name)
                for f in manifest['files']:
                    assert (local / f['file']).stat().st_size == f['bytes']
                    if f.get('sha256'):
                        assert sha256(local / f['file']) == f['sha256']
                write_json(marker, {'manifest_sha256': digest})
            source = local
        else:
            marker = json.loads((source / '.v32-download-complete.json').read_text())
            assert (marker['kind'], marker['ref'], marker['version']) == (item['kind'], item['ref'], item['version'])
        kind = item['kind']
        if kind == 'dataset':
            target = INPUT / 'datasets' / item['ref']
        elif kind == 'notebook':
            target = INPUT / 'notebooks' / item['ref'].split('/')[-1]
        elif kind == 'model':
            target = INPUT / 'models' / item['ref'].lower()
        elif kind == 'competition' and item['ref'] == 'name-that-llm':
            target = INPUT / 'competitions' / item['ref']
        else:
            raise ValueError('Unexpected prepared input kind')
        link(source, target)
    verified = json.loads((DRIVE / 'cli_pinned_inputs_verified.json').read_text())
    assert verified['status'] == 'PASS'
    assert {x['input']: x['sha256'] for x in verified['files']} == lock['pinned_csv_sha256']
    for item in verified['files']:
        assert sha256(item['path']) == item['sha256'], f'Historical CSV changed: {item["input"]}'
    write_json(WORK / 'input_inventory.json', inventory)
    write_json(WORK / 'pinned_inputs_verified.json', verified)
    print('31 个缓存输入就绪；历史 CSV 哈希核验通过。', flush=True)


def release():
    from google.colab import runtime
    for attempt in range(3):
        try:
            runtime.unassign()
            return
        except Exception as exc:
            print('释放运行时重试', attempt + 1, type(exc).__name__, flush=True)
            time.sleep(3)
    address = os.environ.get('TBE_RUNTIME_ADDR')
    if address:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request('http://' + address + '/unassign', method='POST'), timeout=20).close()


def main():
    global WORK
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('请先在 Colab 选择 A100，然后重新运行此单元格。')
    lock_handle = Path('/content/v32-web-run.lock').open('w')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise RuntimeError('本运行时已有 V32 任务；请勿同时启动两份。')
    started = time.time()
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    WORK = WORK / run_id
    output = None
    child = None
    watcher_config = Path('/content/v32-web-watchdog-' + run_id + '.json')
    state = {'status': 'RUNNING', 'run_id': run_id, 'started_at': started,
             'deadline_at': started + LIMIT_SECONDS,
             'public_score_verified': False, 'phase': 'mount_drive'}

    def publish():
        if output is None:
            return
        state['updated_at'] = time.time()
        state['elapsed_seconds'] = time.time() - started
        try:
            import psutil
            memory = psutil.virtual_memory()
            state['system_ram_available_gib'] = round(memory.available / 1024**3, 2)
            state['system_ram_total_gib'] = round(memory.total / 1024**3, 2)
            if child and child.poll() is None:
                process = psutil.Process(child.pid)
                state['child_tree_rss_gib'] = round(sum(p.memory_info().rss for p in [process] + process.children(recursive=True)) / 1024**3, 2)
        except Exception:
            pass
        write_json(output / 'web_run_state.json', state)
        write_json(DRIVE / 'web_run_state.json', state)
        for p in WORK.iterdir():
            if p.is_file() and p.suffix in {'.json', '.log', '.parquet', '.py'} and p.stat().st_size < 20 * 1024**2:
                copy_atomic(p, output / p.name)

    def run(command, logfile, phase, env=None):
        nonlocal child
        state['phase'] = phase
        publish()
        with (WORK / logfile).open('w', buffering=1) as stream:
            child = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            while child.poll() is None:
                if time.time() - started >= LIMIT_SECONDS - 45:
                    raise TimeoutError('本次运行达到四小时上限；已完成阶段保留，准备释放 GPU。')
                time.sleep(15)
                publish()
                lines = (WORK / logfile).read_text(errors='replace').splitlines()
                print(f'[{phase}] 本次 {int((time.time()-started)/60)} 分钟；' + '\n'.join(lines[-3:])[-900:], flush=True)
            if child.returncode:
                raise RuntimeError(f'{phase} exited {child.returncode}; see {logfile}')
        child = None
        publish()

    try:
        WORK.mkdir(parents=True, exist_ok=False)
        write_json(watcher_config, {})
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
        assert 'A100' in gpu and len(gpu.splitlines()) == 1, '需要单张 A100；将释放当前不匹配的 GPU。'
        state['gpu'] = gpu
        watcher = subprocess.Popen([sys.executable, str(PAYLOAD / 'web_watchdog.py'), str(os.getpid()),
                                    str(LIMIT_SECONDS), str(watcher_config)], start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        state['watchdog_pid'] = watcher.pid
        from google.colab import drive
        drive.mount('/content/drive')
        assert DRIVE.is_dir(), 'Prepared Drive cache not found'
        output = DRIVE / 'outputs' / ('web-' + run_id)
        output.mkdir(parents=True, exist_ok=False)
        state['output_directory'] = str(output)
        write_json(watcher_config, {'output': str(output)})
        for p in PAYLOAD.iterdir():
            if p.is_file():
                copy_atomic(p, output / p.name)
        state['phase'] = 'link_prepared_cache'
        publish()
        prepare_cached_inputs()
        publish()
        wheels = json.loads((DRIVE / 'cli_wheels_result.json').read_text())
        assert wheels['status'] == 'READY'
        packages = ['torch==2.11.0', 'transformers==5.16.1', 'peft==0.20.0', 'numpy==2.1.3',
                    'pandas==2.3.3', 'scipy==1.16.3', 'scikit-learn==1.6.1', 'lightgbm==4.6.0',
                    'safetensors==0.8.0', 'pyarrow==23.0.1', 'sentencepiece', 'protobuf', 'accelerate']
        command = [sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', str(DRIVE / 'cli-wheels')]
        driver_major = int(gpu.split(',')[-1].strip().split('.')[0])
        assert driver_major >= 525, 'NVIDIA driver is too old'
        if driver_major < 580:
            assert json.loads((DRIVE / 'cli_torch_cu128_result.json').read_text())['status'] == 'READY'
            packages[0] = 'torch==2.11.0+cu128'
            command += ['--find-links', str(DRIVE / 'cli-wheels-cu128')]
        run(command + packages, 'setup-inference.log', 'install_cached_dependencies')
        run([sys.executable, '-c', "import json,torch; assert torch.cuda.is_available(); x=torch.ones(4,device='cuda'); assert (x+x).sum().item()==8; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'operation':'PASS'}))"],
            'cuda-preflight.log', 'cuda_preflight')
        env = os.environ.copy()
        env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                    'PYTHONPATH': str(PAYLOAD) + os.pathsep + env.get('PYTHONPATH', ''),
                    'V32_INPUT_ROOT': str(INPUT), 'V32_WORK_DIR': str(WORK),
                    'V32_CHECKPOINT_ROOT': str(DRIVE / 'web-checkpoints')})
        env.pop('KAGGLE_API_TOKEN', None)
        run([sys.executable, '-u', str(PAYLOAD / 'web_execute.py')], 'v32-run.log', 'v32_inference', env)
        state['result'] = json.loads((WORK / 'reproduction_result.json').read_text())
        state['status'] = 'COMPLETE'
        print('测试完成，样例一致：', state['result']['sample_predictions_match'], '；结果保存在', output, flush=True)
    except BaseException as exc:
        state.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else 'ERROR',
                     error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        if WORK.is_dir():
            write_json(WORK / 'failure.json', state)
        raise
    finally:
        try:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            state['release_requested'] = True
            publish()
            os.sync()
        except Exception as exc:
            print('最后一次保存未完成：', type(exc).__name__, str(exc), flush=True)
        finally:
            print('正在断开并释放本次 GPU 运行时。', flush=True)
            try:
                release()
            finally:
                lock_handle.close()


if __name__ == '__main__':
    main()
