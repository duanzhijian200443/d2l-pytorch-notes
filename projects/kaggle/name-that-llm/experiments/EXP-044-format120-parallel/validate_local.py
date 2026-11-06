"""Bounded local behavior/packaging checks; no full data fitting or network."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import base64
import hashlib
import io
import json
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAYLOAD = ROOT / 'cloud/format120-044/payload'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = HERE / 'metrics' / ('local-check-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'RUNNING', 'observed_at': stamp, 'full_training_started': False,
              'config_sha256': sha(HERE / 'config.json'), 'runner_sha256': sha(HERE / 'runner.py'),
              'guard_sha256': sha(HERE / 'guard.py'), 'steps': []}
    (HERE / 'metrics/local-validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    commands = [
        ('unit', [sys.executable, '-B', '-X', 'utf8', str(HERE / 'test_runner.py'), '-v']),
        ('preflight', [sys.executable, '-B', '-X', 'utf8', str(PAYLOAD / 'runner.py'), '--payload', str(PAYLOAD), '--output', str(output / 'preflight'), '--preflight-only']),
        ('parity', [sys.executable, '-B', '-X', 'utf8', str(PAYLOAD / 'runner.py'), '--payload', str(PAYLOAD), '--output', str(output / 'selftest'), '--self-test']),
    ]
    for name, command in commands:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', timeout=180)
        log = output / (name + '.log')
        log.write_text(result.stdout, encoding='utf-8')
        record['steps'].append({'name': name, 'exit_code': result.returncode, 'log': log.relative_to(ROOT).as_posix(), 'sha256': sha(log)})
        print(name, result.returncode, flush=True)
        if result.returncode:
            print(result.stdout[-6000:])
            raise RuntimeError(name + ' failed; full training not started')
    nb = read(ROOT / 'cloud/format120-044/kernel/experiment.ipynb')
    code = ''.join(nb['cells'][1]['source'])
    tree = ast.parse(code)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode']
    assert len(calls) == 1
    bundle = base64.b64decode(ast.literal_eval(calls[0].args[0]))
    allowed = {'stylegram_core.py', 'legacy_runner.py', 'runner.py', 'guard.py', 'test_runner.py',
               'raw120.csv', 'clean120.csv', 'data-gate.json', 'config.json', 'protocol.json'}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert len(archive.namelist()) == len(allowed) and set(archive.namelist()) == allowed
        body = []
        for name in archive.namelist():
            raw = archive.read(name)
            assert raw == (PAYLOAD / name).read_bytes(), 'Embedded payload differs: ' + name
            body.append(raw.decode('utf-8'))
        joined = '\n'.join(body)
    protected = []
    for relative in ('experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl',
                     'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl',
                     'experiments/EXP-043-codex-format-view/derived/holdout.jsonl'):
        for line in (ROOT / relative).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('split') == 'holdout' or relative.endswith('/holdout.jsonl'):
                protected.append(row['text'])
    assert all(text not in joined for text in protected if len(text) > 20), 'Protected answer found in uploaded package'
    manifest = read(HERE / 'manifest.json')
    for item in [manifest['config'], manifest['protocol']] + manifest['inputs'] + manifest['code']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    parity = read(output / 'selftest/selftest.json')
    assert parity['status'] == 'PASS' and parity['parallel_workers_observed'] == 2
    assert parity['max_abs_score_difference'] <= 1e-12 and parity['completed_resume_reused'] and parity['corruption_rejected']
    record.update(status='PASS', parity=parity, check_directory=output.relative_to(ROOT).as_posix(),
                  payload_allowlist=sorted(allowed), protected_answer_count=len(protected), protected_answer_matches=0,
                  environment_scope='Local behavior and small numerical test only; cloud strict versions still required',
                  manifest_sha256=sha(HERE / 'manifest.json'))
    (HERE / 'metrics/local-validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'max_abs_difference': parity['max_abs_score_difference'], 'full_training_started': False}))


if __name__ == '__main__':
    main()
