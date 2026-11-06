"""One bounded, private T4x2 capture after a separately pinned CPU file audit.

The supervisor owns a new worker process group. Setup, original component
processes, replay and archive preparation are bounded; no retry, submission,
network installation, Colab release or model change is performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import psutil


HERE = Path(__file__).resolve().parent
TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
PANEL_SHA256 = '8eeec7f64e924f0fd79dca57d324d42d269f400bdfbfff76d60b4c168d9ae6f2'
BASELINE_SHA256 = '6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8'
SMALL_FILE_BYTES = 16 * 1024**2
LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol',
          'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
MAIN_MEMBERS = ['tfidf', 'svc4', 'distort', 'q14v5_gauss', 'q14v5', 'q14v5_twin', 'q8v5_gauss',
                'q8v5', 'q8v5_twin', 'll8v5', 'll8v5_twin', 'n14b', 'n14b_twin', 'q25_14', 'q25_14_twin', 'deb']
COMP_MEMBERS = ['stylegram', 'stylometry_lgb', 'glm9b_twin', 'gauss_glm9b_L8', 'q3_8b_twin',
                'gauss_q3_8b_L22', 'q25_7b_twin', 'q3_4b_twin', 'gauss_q3_4b_L4', 'debv3l',
                'debv3l_twin', 'gauss_debv3l_L24', 'electl', 'desk', 'lora24_public', 'lora24_public_twin',
                'lora24_seed7', 'lora24_claude', 'lora24_claude_twin', 'lora20_q25_instyle', 'tokattn_deb', 'ultrabert']
FAMILY_GATES = ['claude', 'claude_lora', 'gpt', 'gemini', 'glm']
RUNTIME_FILES = ['setup.py', 'model_args.py', 'infer.py', 'comp_stack.py', 'gauss.py',
                 'full_stack_cache.py', 'full_stack_observer.py', 'full_stack_replay.py', 'runtime-build.json']
PACKAGES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn',
            'torch': 'torch', 'transformers': 'transformers', 'peft': 'peft', 'accelerate': 'accelerate',
            'safetensors': 'safetensors', 'lightgbm': 'lightgbm', 'pyarrow': 'pyarrow',
            'sentencepiece': 'sentencepiece', 'psutil': 'psutil'}


class CaptureIssue(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def require(value, message):
    if not value:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def bounded_sha(path, deadline, interrupted):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            require(not interrupted and time.monotonic() < deadline, 'Finalization time/signal boundary reached')
            block = stream.read(4 * 1024**2)
            if not block:
                break
            digest.update(block)
    require(not interrupted and time.monotonic() < deadline, 'Finalization time/signal boundary reached')
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_plan(plan):
    require(plan.get('schema_version') == 1 and plan.get('experiment_id') == 'EXP-029', 'Wrong capture plan')
    require(plan.get('scope') == 'exploratory_external', 'Capture is exploratory external only')
    require(isinstance(plan.get('run_id'), str) and bool(plan['run_id'].strip()), 'Missing unique run ID')
    require(plan.get('train_sha256') == TRAIN_SHA256 and plan.get('panel_sha256') == PANEL_SHA256,
            'Changed official or blinded evaluation input')
    require(plan.get('panel_rows') == 729, 'Changed blinded panel size')
    for key, expected in [('work_budget_seconds', 28200), ('platform_wall_seconds', 28800), ('shutdown_grace_seconds', 10)]:
        require(type(plan.get(key)) is int and plan[key] == expected, 'Changed fixed budget: ' + key)
    for key, expected in [('expected_main_members', MAIN_MEMBERS), ('expected_comp_members', COMP_MEMBERS),
                          ('expected_family_gates', FAMILY_GATES)]:
        values = plan.get(key)
        require(isinstance(values, list) and len(values) == len(set(values)) and set(values) == set(expected),
                'Changed predeclared inventory: ' + key)
    require(plan.get('allowed_omissions') == ['tokattn_q35'], 'Only the original unattached Q35 branch may be omitted')
    bindings = plan.get('attachment_bindings')
    require(isinstance(bindings, list) and len(bindings) == 31, 'Expected exactly 31 original attachments')
    require(all(isinstance(item, dict) for item in bindings), 'Invalid attachment binding')
    require(len({json.dumps(item, sort_keys=True) for item in bindings}) == 31, 'Duplicate attachment binding')
    require(digest_string(plan.get('preflight_manifest_sha256')), 'Missing CPU manifest hash')
    hashes = plan.get('input_hashes')
    require(isinstance(hashes, dict) and set(RUNTIME_FILES + ['runner.py', 'panel/test.csv', 'preflight_manifest.json']).issubset(hashes),
            'Missing pinned runtime inputs')
    for name, digest in hashes.items():
        relative = PurePosixPath(name)
        require(isinstance(name, str) and not relative.is_absolute() and '..' not in relative.parts
                and relative.as_posix() == name and '\\' not in name and ':' not in name,
                'Unsafe payload input path')
        require(name != 'plan.json' and 'label' not in relative.name.lower(), 'Disallowed plan self-hash or label sidecar')
        require(digest_string(digest), 'Invalid runtime hash: ' + name)
    require(hashes['preflight_manifest.json'] == plan['preflight_manifest_sha256'], 'CPU manifest binding differs')
    require(hashes['panel/test.csv'] == PANEL_SHA256, 'Panel hash binding differs')


def verify_payload(here, plan):
    for name, expected in plan['input_hashes'].items():
        require(sha(here / name) == expected, 'Payload changed: ' + name)
    actual = {path.relative_to(here).as_posix() for path in here.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts}
    require(actual == set(plan['input_hashes']) | {'plan.json'}, 'Undeclared/missing payload file')


def verify_cpu_manifest(manifest, plan):
    require(manifest.get('schema_version') == 1 and manifest.get('status') == 'PASS', 'CPU file audit did not pass')
    require(manifest.get('attachment_bindings') == plan['attachment_bindings'], 'GPU/CPU immutable attachment bindings differ')
    files = manifest.get('files')
    require(isinstance(files, list) and bool(files), 'CPU audit has no files')
    seen = set()
    counts = {'files': 0, 'small_files_rehashed': 0, 'large_files_version_bound': 0, 'large_bytes_not_rehashed': 0}
    for entry in files:
        name = entry['path']
        pure = PurePosixPath(name)
        require(pure.is_absolute() and pure.parts[:3] == ('/', 'kaggle', 'input') and '..' not in pure.parts
                and '\\' not in name and name not in seen, 'Unsafe/duplicate CPU input path')
        require(type(entry['bytes']) is int and entry['bytes'] >= 0 and digest_string(entry['sha256']), 'Invalid CPU file receipt')
        seen.add(name)
        path = Path(name)
        require(path.is_file() and path.stat().st_size == entry['bytes'], 'Mounted file absent or resized: ' + name)
        if entry['bytes'] <= SMALL_FILE_BYTES:
            require(sha(path) == entry['sha256'], 'Mounted small-file hash differs: ' + name)
            counts['small_files_rehashed'] += 1
        else:
            counts['large_files_version_bound'] += 1
            counts['large_bytes_not_rehashed'] += entry['bytes']
        counts['files'] += 1
    counts['large_file_policy'] = 'CPU full hashes plus identical pre-bound attachment versions and GPU path/size checks; not GPU byte rehash'
    return counts


def package_versions():
    versions = {name: importlib.metadata.version(distribution) for name, distribution in PACKAGES.items()}
    try:
        versions['bitsandbytes'] = importlib.metadata.version('bitsandbytes')
    except importlib.metadata.PackageNotFoundError:
        versions['bitsandbytes'] = None
    return versions


def gpu_preflight(plan, manifest):
    versions = package_versions()
    wanted = plan.get('required_versions', manifest.get('environment', {}).get('versions', {}))
    require(isinstance(wanted, dict) and bool(wanted), 'Missing CPU/runtime version receipt')
    for key, value in wanted.items():
        normalized = 'sklearn' if key == 'scikit-learn' else key
        if value is not None and normalized in versions:
            require(versions[normalized] == value, 'CPU/GPU package version differs: ' + key)
    # Actual no-model imports catch binary/API failures before expensive fits.
    for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'pyarrow', 'sentencepiece',
                 'safetensors.torch', 'transformers', 'accelerate'):
        importlib.import_module(name)
    original_find_spec = importlib.util.find_spec
    try:
        # Same unused-torchao compatibility rule as original comp_stack.py.
        importlib.util.find_spec = lambda name, *a, **kw: None if name.split('.')[0] == 'torchao' else original_find_spec(name, *a, **kw)
        importlib.import_module('peft')
    finally:
        importlib.util.find_spec = original_find_spec
    import pandas as pd
    buffer = io.BytesIO()
    probe = pd.DataFrame({'id': ['dependency-probe'], 'value': [1]})
    probe.to_parquet(buffer, index=False)
    buffer.seek(0)
    require(pd.read_parquet(buffer).equals(probe), 'Parquet dependency roundtrip failed')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected exactly two CUDA GPUs')
    devices = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        require(re.search(r'\bT4\b', properties.name.upper()) is not None, 'Expected T4 on GPU ' + str(index))
        values = (torch.ones(4, device=f'cuda:{index}') + 1).cpu().tolist()
        require(values == [2.0] * 4, 'CUDA operation failed')
        devices.append({'index': index, 'name': properties.name, 'total_memory_bytes': int(properties.total_memory),
                        'compute_capability': [properties.major, properties.minor]})
    torch.cuda.empty_cache()
    driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    return {'versions': versions, 'devices': devices, 'cuda': torch.version.cuda,
            'nvidia_smi': driver,
            'python': sys.version, 'cpu_count': os.cpu_count(), 'host_ram_bytes': psutil.virtual_memory().total,
            'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'requested_container': plan.get('container'), 'observed_at': stamp()}


def required_optional_inputs():
    patterns = {
        'lora24_public': '**/lora24/final/COMPLETE.json',
        'lora24_seed7': '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json',
        'lora24_claude': '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json',
        'lora20_q25_instyle': '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json',
        'tokattn_deb': '**/tokattn_debl23_final.pt',
        'ultrabert': '**/ultrabert-380m-ntl/fold*/config.json',
        'claude_gate': '**/ntl_extra_passages_claude.csv',
        'deberta_backbone': '**/deberta-v3-large/config.json',
    }
    resolved = {}
    for name, pattern in patterns.items():
        hits = sorted(glob.glob('/kaggle/input/' + pattern, recursive=True))
        require(bool(hits), 'Required original optional input is absent: ' + name)
        resolved[name] = hits
    for config in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            model_type = read_json(config).get('model_type')
        except (OSError, ValueError, AttributeError):
            continue
        require(model_type != 'qwen3_5', 'Undeclared Q35 backbone would change the fixed 22-member inventory')
    return resolved


def phase(out, name, **details):
    record = {'status': 'RUNNING', 'phase': name, 'observed_at': stamp(), **details}
    save(out / 'state.json', record)
    print('CAPTURE_STAGE ' + json.dumps(record), flush=True)


def command(command, label, out, env):
    log = out / 'logs' / (label + '.log')
    log.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    print('CHILD_START ' + json.dumps({'stage': label, 'command': command}), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', start_new_session=False)
        for line in child.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = child.wait()
    elapsed = time.monotonic() - started
    require(code == 0, f'{label} child failed with exit {code}')
    return {'seconds': elapsed, 'exit_code': code, 'log': str(log)}


def component_receipt(out, component, run_id):
    log = out / 'logs' / (component + '.log')
    lines = [line[len('CACHE_COMPONENT_COMMITTED '):] for line in log.read_text(encoding='utf-8').splitlines()
             if line.startswith('CACHE_COMPONENT_COMMITTED ')]
    require(len(lines) == 1, 'Missing/duplicate committed component receipt: ' + component)
    receipt = json.loads(lines[0])
    require(receipt['component'] == component and receipt['run_id'] == run_id, 'Wrong committed component identity')
    require(digest_string(receipt.get('manifest_sha256')), 'Invalid committed manifest hash')
    require(sha(out / 'cache' / component / 'manifest.json') == receipt['manifest_sha256'], 'Committed manifest changed')
    return receipt


def cap_atlas_budget(original_budget, remaining_work_seconds):
    require(math.isfinite(original_budget) and math.isfinite(remaining_work_seconds), 'Nonfinite time budget')
    applied = min(original_budget, max(0, remaining_work_seconds - 120))
    require(applied >= 180, 'BUDGET_INSUFFICIENT before main')
    return applied


def worker(here, out):
    started = time.monotonic()
    plan = read_json(here / 'plan.json')
    validate_plan(plan)
    initial_plan_hash = sha(here / 'plan.json')
    phase(out, 'payload_preflight')
    verify_payload(here, plan)
    build = read_json(here / 'runtime-build.json')
    require(build.get('baseline_sha256') == BASELINE_SHA256 and build.get('status') == 'BUILT_NOT_EXECUTED',
            'Runtime build is not bound to the fixed baseline')
    require(all(plan['input_hashes'].get(name) == digest for name, digest in build['files'].items()),
            'Runtime build/file receipts differ')
    (out / 'inputs').mkdir(exist_ok=True)
    (out / 'code').mkdir(exist_ok=True)
    for name in ('plan.json', 'preflight_manifest.json', 'runtime-build.json'):
        shutil.copyfile(here / name, out / 'inputs' / name)
    shutil.copyfile(here / 'panel/test.csv', out / 'inputs/test.csv')
    for name in RUNTIME_FILES + ['runner.py']:
        if name.endswith('.py'):
            shutil.copyfile(here / name, out / 'code' / name)
    manifest = read_json(here / 'preflight_manifest.json')
    mounted = verify_cpu_manifest(manifest, plan)
    save(out / 'mounted-input-check.json', mounted)
    working = Path('/kaggle/working')
    panel = working / 'panel'
    require(not panel.exists(), 'Refusing an existing panel staging directory')
    for name in ('comp_logit.npy', 'comp_meta.json', 'submission.parquet'):
        require(not (working / name).exists(), 'Refusing stale original output: ' + name)
    candidates = [entry['path'] for entry in manifest['files']
                  if entry['sha256'] == TRAIN_SHA256 and PurePosixPath(entry['path']).name == 'train.csv']
    official = plan.get('official_train_path')
    if official is None:
        require(len(candidates) == 1, 'Official train path must be unambiguous')
        official = candidates[0]
    require(official in candidates, 'Official path not bound by CPU audit')
    panel.mkdir()
    shutil.copyfile(official, panel / 'train.csv')
    shutil.copyfile(here / 'panel/test.csv', panel / 'test.csv')
    require(sha(panel / 'train.csv') == TRAIN_SHA256 and sha(panel / 'test.csv') == PANEL_SHA256, 'Staged data changed')
    import pandas as pd
    test = pd.read_csv(panel / 'test.csv', dtype=str, keep_default_na=False)
    require(list(test.columns) == ['id', 'text'] and len(test) == 729 and test.id.is_unique
            and test.text.str.strip().ne('').all(), 'Invalid blinded test schema')
    save(out / 'required-input-paths.json', required_optional_inputs())
    env = os.environ.copy()
    for key in ('NTL_SMOKE', 'NTL_TINY', 'NTL_FAKE_TEST', 'NTL_POOL', 'NTL_DATA'):
        env.pop(key, None)
    env.update({'NTL_DROP_Q35': '0', 'NTL_CAPTURE_DATA': str(panel), 'NTL_SETUP_RESULT': str(out / 'setup-result.json'),
                'NTL_MODEL_ARGS_RESULT': str(out / 'model-args.json'), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1'})
    phase(out, 'gpu_preflight')
    # Probe in a separate short-lived child; the supervisor/worker must not
    # retain CUDA contexts that steal memory from the original model process.
    command([sys.executable, '-u', str(here / 'runner.py'), '--gpu-preflight', str(out / 'environment-before-setup.json')],
            'gpu_preflight', out, env)
    context = {'schema_version': 1, 'run_id': plan['run_id'], 'scope': 'exploratory_external', 'launch_ready': True,
               'expected_main_members': plan['expected_main_members'], 'expected_comp_members': plan['expected_comp_members'],
               'expected_family_gates': plan['expected_family_gates'], 'allowed_omissions': plan['allowed_omissions'],
               'plan_sha256': initial_plan_hash, 'preflight_manifest_sha256': plan['preflight_manifest_sha256'],
               'attachment_bindings': plan['attachment_bindings'], 'panel_sha256': PANEL_SHA256,
               'required_events': {'main': ['data', 'main_svc4_margin', 'main_members', 'comp_loaded', 'comp_status',
                    'main_probability', 'pre_atlas', 'atlas_requested', 'atlas_returned', 'pre_sinkhorn', 'sinkhorn_applied',
                    'postprocessing_base', 'typography', 'component_end'],
                    'comp': ['data', 'member', 'style_margin_context', 'style_margin', 'style_routed',
                    'comp_before_gates', 'family_gate_applied', 'component_end']}}
    save(out / 'capture-context.json', context)
    env.update(NTL_CACHE_EXPORT=str(out / 'cache'), NTL_CACHE_CONTEXT=str(out / 'capture-context.json'))
    timings = {}
    phase(out, 'setup')
    timings['setup'] = command([sys.executable, '-u', str(here / 'setup.py')], 'setup', out, env)
    command([sys.executable, '-u', str(here / 'runner.py'), '--gpu-preflight', str(out / 'environment.json')],
            'gpu_after_setup', out, env)
    phase(out, 'comp')
    timings['comp'] = command([sys.executable, '-u', str(here / 'comp_stack.py')], 'comp', out, env)
    receipts = {'comp': component_receipt(out, 'comp', plan['run_id'])}
    save(out / 'component-receipts.json', receipts)
    for name in ('comp_logit.npy', 'comp_meta.json'):
        shutil.copyfile(working / name, out / name)
    env['NTL_COMP_SECONDS'] = str(timings['comp']['seconds'])
    phase(out, 'model_args')
    command([sys.executable, '-u', str(here / 'model_args.py')], 'model_args', out, env)
    model_args = read_json(out / 'model-args.json')
    original_command = model_args['command']
    main_command = list(original_command)
    remaining = float(os.environ['NTL_WORK_DEADLINE_MONOTONIC']) - time.monotonic()
    original_budget = float(main_command[main_command.index('--atlas_budget') + 1])
    actual_budget = cap_atlas_budget(original_budget, remaining)
    main_command[0] = sys.executable
    main_command[1] = str(here / 'infer.py')
    main_command[main_command.index('--atlas_budget') + 1] = str(actual_budget)
    save(out / 'main-command.json', {'original_command': original_command, 'actual_command': main_command,
         'original_atlas_budget': original_budget, 'actual_atlas_budget': actual_budget,
         'global_work_deadline_monotonic': float(os.environ['NTL_WORK_DEADLINE_MONOTONIC']),
         'main_launch_monotonic': time.monotonic(), 'main_T0_scope': 'original infer.py time.time at initialization; not reset or shifted',
         'only_command_changes': ['absolute script path/executable', 'ATLAS budget capped to remaining global work minus120s']})
    phase(out, 'main')
    timings['main'] = command(main_command, 'main', out, env)
    receipts['main'] = component_receipt(out, 'main', plan['run_id'])
    save(out / 'component-receipts.json', receipts)
    for name in ('submission.parquet', 'v31_diagnostics.json'):
        shutil.copyfile(working / name, out / name)
    phase(out, 'replay')
    sys.path.insert(0, str(here))
    modules = {name: importlib.import_module(name) for name in ('full_stack_cache', 'full_stack_observer', 'full_stack_replay')}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == (here / (name + '.py')).resolve(), 'Wrong imported replay helper')
    snapshots = {name: modules['full_stack_cache'].read_cache(out / 'cache', name, test.id.tolist(), test.text.tolist(), LABELS,
                    required_notes=['event-index'], manifest_sha256=receipts[name]['manifest_sha256']) for name in ('main', 'comp')}
    context_hash = sha(out / 'capture-context.json')
    require(all(item.metadata['context_sha256'] == context_hash and item.metadata['context'] == context
                for item in snapshots.values()), 'Captured context differs')
    decoded = {name: modules['full_stack_observer'].decode_events(snapshot) for name, snapshot in snapshots.items()}
    before_gates = next(payload for event, payload in decoded['comp'] if event == 'comp_before_gates')
    actual_inventory = {'main': list(next(payload for event, payload in decoded['main'] if event == 'main_members')['members']),
        'comp': [payload['key'] for event, payload in decoded['comp'] if event == 'member'],
        'family_gates': [payload['key'] for event, payload in decoded['comp'] if event == 'family_gate_applied'],
        'comp_eligible_rows': int(before_gates['group_mask'].sum())}
    save(out / 'captured-inventory.json', actual_inventory)
    if not before_gates['group_mask'].any():
        raise CaptureIssue('CONDITIONAL_COVERAGE_UNSATISFIED', 'No eligible comp groups; the predeclared22-member target was not met')
    if (set(actual_inventory['main']) != set(plan['expected_main_members'])
            or set(actual_inventory['comp']) != set(plan['expected_comp_members'])
            or actual_inventory['family_gates'] != FAMILY_GATES):
        raise CaptureIssue('CAPTURE_INVENTORY_MISMATCH', 'Observed members/gates differ from the predeclared inventory')
    replay = modules['full_stack_replay'].verify_replay(decoded['main'], decoded['comp'],
                    plan['expected_main_members'], plan['expected_comp_members'])
    save(out / 'replay.json', replay)
    if replay['accepted'] is not True:
        raise CaptureIssue(replay['status'], 'Cache replay not accepted: ' + replay['status'])
    require(replay['family_gates'] == FAMILY_GATES, 'Required family gate omitted')
    final = next(payload for event, payload in decoded['main'] if event == 'typography')['logits_after']
    submission = pd.read_parquet(out / 'submission.parquet')
    require(submission.id.astype(str).tolist() == test.id.tolist()
            and submission.model.tolist() == [LABELS[int(i)] for i in final.argmax(axis=1)], 'Final parquet differs from cached typography scores')
    verify_payload(here, plan)
    require(sha(here / 'plan.json') == initial_plan_hash, 'Plan changed during capture')
    save(out / 'worker-result.json', {'status': 'COMPLETE', 'scope': 'exploratory_external', 'run_id': plan['run_id'],
         'rows': len(test), 'score_columns': 18, 'timings': timings, 'worker_seconds': time.monotonic() - started,
         'completed_monotonic': time.monotonic(),
         'plan_sha256': initial_plan_hash, 'component_receipts': receipts, 'replay_status': replay['status'],
         'formal_submission_created': False, 'metrics_computed_from_truth': False, 'full_training_independence_claimed': False})


def stop_group(child, grace):
    """Signal only the worker session created by this supervisor, including descendants."""
    if child is None:
        return
    require(os.name == 'posix' and child.pid != os.getpgrp(), 'Unsafe child process-group identity')
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited before its descendants. Always kill the group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=20)


def process_rss(pid):
    try:
        root = psutil.Process(pid)
        return sum(process.memory_info().rss for process in [root] + root.children(recursive=True))
    except psutil.NoSuchProcess:
        return 0


def archive(out):
    destination = out / 'cache-artifacts.zip'
    temporary = out / 'cache-artifacts.zip.tmp'
    excluded = {'cache-artifacts.zip', 'result.json', 'artifacts.json', 'guard.json', 'state.json', '.run.lock'}
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in excluded and not path.name.endswith('.tmp'):
                bundle.write(path, path.relative_to(out).as_posix())
    temporary.replace(destination)


def supervise(here, out):
    require(os.name == 'posix', 'This supervisor is for Linux Kaggle only')
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), 'Refusing an occupied output directory')
    with (out / '.run.lock').open('x', encoding='utf-8') as stream:
        stream.write(stamp())
    child, error, detail, peak = None, None, None, 0
    stopped = []
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda number, frame: stopped.append(number))
    plan = None
    try:
        plan = read_json(here / 'plan.json')
        validate_plan(plan)
        deadline = started + plan['work_budget_seconds']
        env = os.environ.copy()
        env.update(NTL_OUTPUT_DIR=str(out), NTL_WORK_DEADLINE_MONOTONIC=str(deadline),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
        with (out / 'worker.log').open('w', encoding='utf-8') as stream, \
                (out / 'worker.log').open('r', encoding='utf-8', errors='replace') as mirror:
            child = subprocess.Popen([sys.executable, '-u', str(here / 'runner.py'), '--worker'],
                    env=env, cwd=here, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                rss = process_rss(child.pid)
                peak = max(peak, rss)
                save(out / 'guard.json', {'status': 'RUNNING', 'observed_at': stamp(), 'elapsed_seconds': elapsed,
                     'work_budget_seconds': plan['work_budget_seconds'], 'worker_pid': child.pid,
                     'worker_process_group': child.pid, 'rss_bytes': rss, 'peak_rss_bytes': peak})
                recent = mirror.read()
                if recent:
                    print(recent, end='', flush=True)
                if stopped:
                    error = 'SUPERVISOR_SIGNAL_' + str(stopped[0])
                    break
                if elapsed >= plan['work_budget_seconds']:
                    error = 'WORK_BUDGET_EXCEEDED'
                    break
                if plan.get('rss_limit_bytes') is not None and rss > plan['rss_limit_bytes']:
                    error = 'RSS_BUDGET_EXCEEDED'
                    break
                try:
                    child.wait(timeout=min(2, max(.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if error:
                stop_group(child, plan['shutdown_grace_seconds'])
            recent = mirror.read()
            if recent:
                print(recent, end='', flush=True)
        if not error:
            require(child.returncode == 0, 'Worker failed with exit ' + str(child.returncode))
            worker_result = read_json(out / 'worker-result.json')
            require(worker_result.get('status') == 'COMPLETE', 'Worker did not complete capture')
            require(worker_result['completed_monotonic'] <= deadline, 'Worker completed beyond its declared work deadline')
    except BaseException:
        error = error or 'CAPTURE_FAILED'
        detail = traceback.format_exc()
    finally:
        if child is not None:
            try:
                stop_group(child, plan['shutdown_grace_seconds'] if plan else 10)
            except BaseException:
                error = error or 'CHILD_CLEANUP_FAILED'
                detail = (detail or '') + '\n' + traceback.format_exc()
    if error and not (out / 'failure.json').exists():
        save(out / 'failure.json', {'status': 'ERROR', 'cause': error, 'traceback': detail, 'observed_at': stamp()})
    if error == 'CAPTURE_FAILED' and (out / 'failure.json').exists():
        issue = read_json(out / 'failure.json').get('cause')
        if issue in {'CONDITIONAL_COVERAGE_UNSATISFIED', 'CAPTURE_INVENTORY_MISMATCH', 'NEEDS_ADDITIONAL_ATLAS_SCORES'}:
            error = issue
    phase(out, 'archive', prior_error=error)
    archive_record = {'status': 'NOT_CREATED'}
    remaining = started + 28800 - time.monotonic() - 120
    if remaining > 0:
        archive_child = None
        try:
            archive_child = subprocess.Popen([sys.executable, str(here / 'runner.py'), '--archive', str(out)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True)
            while True:
                require(not stopped, 'Supervisor signal during archive')
                left = started + 28800 - time.monotonic() - 120
                require(left > 0, 'Archive exceeded finalization deadline')
                try:
                    _, stderr = archive_child.communicate(timeout=min(2, left))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(archive_child.returncode == 0, 'Archive failed: ' + stderr)
            archive_record = {'status': 'COMPLETE', 'path': 'cache-artifacts.zip',
                              'sha256': bounded_sha(out / 'cache-artifacts.zip', started + 28790, stopped),
                              'bytes': (out / 'cache-artifacts.zip').stat().st_size}
        except BaseException:
            if archive_child is not None:
                stop_group(archive_child, 1)
            error = error or 'ARCHIVE_FAILED'
            detail = (detail or '') + '\n' + traceback.format_exc()
    else:
        error = error or 'NO_ARCHIVE_TIME_REMAINING'
    if error and not (out / 'failure.json').exists():
        save(out / 'failure.json', {'status': 'ERROR', 'cause': error, 'traceback': detail, 'observed_at': stamp()})
    stable_entries = []
    try:
        for path in sorted(out.rglob('*')):
            if (path.is_file() and path.name not in {'result.json', 'artifacts.json', 'guard.json', 'state.json', '.run.lock'}
                    and not path.name.endswith('.tmp')):
                stable_entries.append({'path': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size,
                                       'sha256': bounded_sha(path, started + 28790, stopped)})
    except BaseException:
        error = error or 'FINALIZATION_BUDGET_OR_IO_FAILURE'
        detail = (detail or '') + '\n' + traceback.format_exc()
        stable_entries = None
    if stopped or time.monotonic() >= started + 28790:
        error = error or 'FINALIZATION_TIME_OR_SIGNAL_BOUNDARY'
    terminal = {'status': 'ERROR' if error else 'COMPLETE', 'cause': error, 'observed_at': stamp(),
                'scope': 'exploratory_external', 'experiment_id': 'EXP-029', 'run_id': plan.get('run_id') if plan else None,
                'elapsed_seconds': time.monotonic() - started, 'work_budget_seconds': 28200, 'platform_wall_seconds': 28800,
                'peak_rss_bytes': peak, 'child_exit': child.returncode if child else None, 'archive': archive_record,
                'formal_submission_created': False, 'automatic_retry': False, 'gpu_runtime_release': 'platform termination, not Colab unassign'}
    if detail:
        terminal['traceback'] = detail
    save(out / 'result.json', terminal)
    if stable_entries is not None:
        stable_entries.append({'path': 'result.json', 'bytes': (out / 'result.json').stat().st_size,
                               'sha256': sha(out / 'result.json')})
        save(out / 'artifacts.json', {'status': 'COMPLETE_INVENTORY', 'files': stable_entries})
    else:
        save(out / 'artifacts.json', {'status': 'INCOMPLETE_FINALIZATION', 'files': []})
    save(out / 'guard.json', terminal)
    save(out / 'state.json', {'status': terminal['status'], 'phase': 'finished', 'cause': error, 'observed_at': stamp()})
    print('FULL_STACK_CAPTURE_TERMINAL ' + json.dumps(terminal), flush=True)
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)
    if error:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--gpu-preflight', type=Path)
    args = parser.parse_args()
    if args.gpu_preflight is not None:
        save(args.gpu_preflight, gpu_preflight(read_json(HERE / 'plan.json'), read_json(HERE / 'preflight_manifest.json')))
        return
    if args.archive is not None:
        archive(args.archive)
        return
    out = Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results'))
    if args.worker:
        try:
            worker(HERE, out)
        except BaseException:
            if not (out / 'failure.json').exists():
                error = sys.exc_info()[1]
                save(out / 'failure.json', {'status': 'ERROR', 'cause': getattr(error, 'status', 'EXECUTION_ERROR'),
                                           'observed_at': stamp(), 'traceback': traceback.format_exc()})
            raise
    else:
        supervise(HERE, out)


if __name__ == '__main__':
    main()
