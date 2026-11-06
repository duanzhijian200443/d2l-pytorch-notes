"""Save, verify, then dispatch exactly one private CPU run. No retry."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'cloud/format120-044'
EX = ROOT / 'experiments/EXP-044-format120-parallel'
REF = 'zhjianduan/name-that-llm-format120-044'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


def ack(r):
    return {'raw_ref': r.ref, 'ref': r.ref.removeprefix('/code/').strip('/'),
            'version': r.version_number, 'url': r.url, 'error': r.error,
            'invalid_datasets': r.invalid_dataset_sources or [],
            'invalid_kernels': r.invalid_kernel_sources or [], 'invalid_models': r.invalid_model_sources or []}


def normalized(nb):
    return [(c['cell_type'], ''.join(c['source']) if isinstance(c['source'], list) else c['source']) for c in nb['cells']]


def readback(api, version):
    destination = HERE / f'readback-v{version}'
    api.kernels_pull(REF + f'/{version}', str(destination), metadata=True, quiet=True)
    meta = read(destination / 'kernel-metadata.json')
    wanted = read(HERE / 'kernel/kernel-metadata.json')
    assert normalized(read(destination / meta['code_file'])) == normalized(read(HERE / 'kernel/experiment.ipynb')), 'Source differs'
    assert meta['is_private'] and not meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['dataset_sources'] == ['zhjianduan/ntl-stylegram-cv-inputs-v1']
    assert meta['competition_sources'] == ['name-that-llm']
    assert not meta['kernel_sources'] and not meta['model_sources']
    assert meta['docker_image'] == wanted['docker_image'] and meta['machine_shape'] in ('None', '', None)
    value = {'status': 'PASS', 'observed_at': now(), 'ref': REF, 'version': version,
             'source_equal': True, 'private': True, 'gpu': False, 'internet': False,
             'input_count': 2, 'container': meta['docker_image'],
             'source_sha256': sha(destination / meta['code_file']), 'metadata_sha256': sha(destination / 'kernel-metadata.json')}
    save(destination / 'verified.json', value)
    return value


def main():
    sp = HERE / 'execution-state.json'
    state = read(sp)
    assert state['status'] == 'PREPARED' and state['cloud_run_requests'] == 0, 'Already dispatched or uncertain; inspect, do not repeat'
    frozen = read(EX / 'manifest.json')
    for item in [frozen['config'], frozen['protocol']] + frozen['inputs'] + frozen['code']:
        assert sha(ROOT / item['path']) == item['sha256'], 'Frozen bytes changed: ' + item['path']
    review = read(EX / 'metrics/code-review.json')
    assert review['status'] == 'PASS'
    assert review['manifest_sha256'] == sha(EX / 'manifest.json'), 'Review manifest stale'
    for path, expected in review['code_sha256'].items():
        assert sha(ROOT / path) == expected, 'Review code stale: ' + path
    checks = read(EX / 'metrics/local-validation.json')
    assert checks['status'] == 'PASS' and checks['config_sha256'] == sha(EX / 'config.json')
    assert checks['runner_sha256'] == sha(EX / 'runner.py') and checks['guard_sha256'] == sha(EX / 'guard.py')
    api = KaggleApi()
    api.authenticate()
    old = api.kernels_status('zhjianduan/name-that-llm-luna-terra120-042/2')
    assert str(old.status).split('.')[-1] == 'COMPLETE', 'Earlier comparison not confirmed complete'
    refs = set()
    for page in range(1, 101):
        existing = api.kernels_list(mine=True, page=page, page_size=100)
        assert isinstance(existing, list) and all(k is not None for k in existing), 'Own notebook inventory unavailable'
        current_refs = {str(k.ref).removeprefix('/code/').strip('/') for k in existing}
        assert current_refs.isdisjoint(refs), 'Inventory pagination repeated; inspect before writing'
        assert REF not in current_refs, 'Remote target already exists; inspect before any write'
        refs.update(current_refs)
        if len(existing) < 100:
            break
    else:
        raise RuntimeError('Own inventory pagination limit; no write attempted')
    save(HERE / 'prelaunch-remote-check.json', {'observed_at': now(), 'previous_comparison_status': 'COMPLETE',
         'scope': 'Authenticated own notebook inventory, all returned pages', 'count': len(refs), 'target_absent': True})
    state.update(status='SAVE_REQUEST_IN_FLIGHT', observed_at=now())
    save(sp, state)
    saved = ack(api.kernels_push(str(HERE / 'kernel'), timeout=7200, no_run=True))
    save(HERE / 'save-ack.json', saved)
    assert saved['ref'] == REF and saved['version'] == 1 and not saved['error']
    assert not any(saved[k] for k in ('invalid_datasets', 'invalid_kernels', 'invalid_models'))
    readback(api, 1)
    state.update(status='SAVED_VERIFIED', version=1, observed_at=now())
    save(sp, state)
    state.update(status='RUN_REQUEST_IN_FLIGHT', cloud_run_requests=1, observed_at=now())
    save(sp, state)
    running = ack(api.kernels_push(str(HERE / 'kernel'), timeout=7200, no_run=False))
    save(HERE / 'run-ack.json', running)
    assert running['ref'] == REF and running['version'] == 2 and not running['error']
    assert not any(running[k] for k in ('invalid_datasets', 'invalid_kernels', 'invalid_models'))
    readback(api, 2)
    current = api.kernels_status(REF + '/2')
    state.update(status=str(current.status).split('.')[-1], version=2, observed_at=now(), readback_verified=True)
    save(sp, state)
    path = ROOT / 'tracking/records/experiments/EXP-044.json'
    record = read(path)
    record['source'].update(notebook_ref=REF, notebook_version=2, local_path=(HERE / 'kernel/experiment.ipynb').relative_to(ROOT).as_posix(), sha256=sha(HERE / 'kernel/experiment.ipynb'))
    record['observed_at'] = state['observed_at']
    record['execution'].update(status=state['status'], observed_at=state['observed_at'], run_id=state['run_id'], notebook_version=2, cloud_run_requests=1)
    record['conclusion'] = '原文120与去格式120对照已发起一次私有CPU运行；本地并行与续跑测试通过，等待云端严格环境/数值自检及实际训练结果。'
    record['next_action'] = '读取固定V2状态与产物，验收10个job、20个SVM、完整/稀疏OOF和数值一致性；不自动重跑或提交。'
    record['evidence'] = list(dict.fromkeys(record['evidence'] + [(HERE / 'run-ack.json').relative_to(ROOT).as_posix(), (HERE / 'readback-v2/verified.json').relative_to(ROOT).as_posix(), sp.relative_to(ROOT).as_posix()]))
    save(path, record)
    print(json.dumps({'status': state['status'], 'ref': REF, 'version': 2, 'gpu': False, 'private': True, 'internet': False}))


if __name__ == '__main__':
    main()
