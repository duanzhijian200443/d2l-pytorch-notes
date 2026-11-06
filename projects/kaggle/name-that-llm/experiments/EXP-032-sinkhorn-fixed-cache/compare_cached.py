"""Prepare the fixed .45/.30 CPU comparison; incomplete captures fail closed.

The CLI never launches models or contacts Kaggle. It requires a separately
retained SHA-256 for the downloaded EXP-031 artifact inventory, then verifies
the original capture before reading evaluation truth or calculating metrics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import types
import zipfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_EXPERIMENT = 'EXP-031'
SOURCE_RUN = 'exp031-20261101T110842Z'
SOURCE_PLAN_SHA = 'ef61b941d32c020b9d221fc40c23e6cfc30fd2f0db1ef0340a59a45c78fea401'
BASELINE_SHA = '6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8'
SOURCE_INFER_SHA = '022cafd2e6504dd7cf38c8675b9597fb4fbd4c9dd0be69d38918293d9caac9b3'
SOURCE_COMP_SHA = 'b6b6a7b8adbe50580c2d0de5a41d0764ae752fe729e1471872a9197ac956f222'
MATH_SHA256 = '42048f6ea4a17025179965da4452f0e8091d69146163a328872a903145e39956'
HELPERS = ('full_stack_cache', 'full_stack_observer', 'full_stack_replay')
DYNAMIC_FILES = {'artifacts.json', 'guard.json', 'state.json', '.run.lock'}
EXPECTED_OMISSIONS = ['lora24_public', 'lora24_public_twin', 'tokattn_q35']


class AcceptanceError(ValueError):
    """The fixed real capture or its binding is missing, changed, or incomplete."""


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    def bad_constant(value):
        raise AcceptanceError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                      parse_constant=bad_constant)


def safe_relative(name):
    require(isinstance(name, str), 'Artifact path must be a string')
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts
            and path.as_posix() == name and '\\' not in name and ':' not in name,
            'Unsafe artifact path: ' + name)
    return path


def _no_link(path):
    require(not path.is_symlink() and not (hasattr(path, 'is_junction') and path.is_junction()),
            'Linked input is not accepted: ' + str(path))


def verify_artifact_inventory(results, expected_sha256):
    """Verify every stable file against an external inventory digest, without models."""
    results = Path(results)
    require(results.is_dir(), 'Complete result directory does not exist')
    for parent in (results, *results.parents):
        _no_link(parent)
    require(is_digest(expected_sha256), 'An external lowercase artifacts.json SHA-256 is required')
    inventory_path = results / 'artifacts.json'
    require(inventory_path.is_file() and sha(inventory_path) == expected_sha256,
            'Artifact inventory is missing or differs from the external receipt')
    inventory = read_json(inventory_path)
    require(inventory.get('status') == 'COMPLETE_INVENTORY', 'Incomplete artifact finalization')
    entries = inventory.get('files')
    require(isinstance(entries, list) and bool(entries), 'Empty artifact inventory')
    mapped, names = {}, set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {'path', 'bytes', 'sha256'}, 'Invalid artifact entry')
        name = entry['path']; relative = safe_relative(name)
        require(name.casefold() not in names and relative.name not in DYNAMIC_FILES
                and not relative.name.endswith('.tmp'), 'Duplicate or excluded artifact entry')
        require(type(entry['bytes']) is int and entry['bytes'] >= 0 and is_digest(entry['sha256']),
                'Invalid artifact size/hash')
        names.add(name.casefold()); mapped[name] = entry
        path = results.joinpath(*relative.parts)
        require(path.is_file(), 'Missing stable artifact: ' + name)
        _no_link(path)
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Stable artifact bytes changed: ' + name)
    observed = set()
    for path in results.rglob('*'):
        _no_link(path)
        if path.is_file():
            require(not path.name.endswith('.tmp'), 'Unfinished temporary artifact')
            if path.name not in DYNAMIC_FILES:
                observed.add(path.relative_to(results).as_posix())
    require(observed == set(mapped), 'Unlisted or missing stable artifact')
    require('result.json' in mapped, 'Result must be a stable artifact')
    require(sha(inventory_path) == expected_sha256, 'Inventory changed during verification')
    return mapped


def _verify_archive(results, entries, record):
    require(record.get('status') == 'COMPLETE' and record.get('path') == 'cache-artifacts.zip',
            'Capture archive did not complete')
    expected = entries.get('cache-artifacts.zip')
    require(expected is not None and all(record.get(key) == expected[key] for key in ('bytes', 'sha256')),
            'Archive receipt differs')
    archived = set(entries) - {'cache-artifacts.zip', 'result.json'}
    with zipfile.ZipFile(results / 'cache-artifacts.zip') as bundle:
        infos = bundle.infolist()
        require(len(infos) == len(archived) and {item.filename for item in infos} == archived,
                'Archive file inventory differs')
        for info in infos:
            safe_relative(info.filename)
            entry = entries[info.filename]
            require(not info.is_dir() and info.file_size == entry['bytes'], 'Archive entry size differs')
            digest = hashlib.sha256()
            with bundle.open(info) as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b''):
                    digest.update(block)
            require(digest.hexdigest() == entry['sha256'], 'Archive entry bytes differ: ' + info.filename)


def _complete_header(results, source):
    require(Path(results).is_dir(), 'Complete result directory does not exist')
    require(source.get('experiment_id') == SOURCE_EXPERIMENT and source.get('run_id') == SOURCE_RUN
            and source.get('plan_sha256') == SOURCE_PLAN_SHA, 'This comparison only accepts the fixed EXP-031 run')
    result = read_json(Path(results) / 'result.json')
    require(result.get('scope') == 'exploratory_external', 'Synthetic/sample capture is not eligible')
    require(result.get('experiment_id') == SOURCE_EXPERIMENT and result.get('run_id') == SOURCE_RUN,
            'Wrong capture experiment/run')
    require(result.get('status') == 'COMPLETE' and result.get('cause') is None
            and result.get('child_exit') == 0 and not result.get('traceback'), 'Capture did not complete successfully')
    require(not (Path(results) / 'failure.json').exists(), 'Capture retains a failure record')
    worker = read_json(Path(results) / 'worker-result.json')
    require(worker.get('status') == 'COMPLETE' and worker.get('scope') == 'exploratory_external'
            and worker.get('run_id') == SOURCE_RUN and worker.get('plan_sha256') == SOURCE_PLAN_SHA,
            'Worker is incomplete or bound to another run')
    require(worker.get('rows') == 729 and worker.get('score_columns') == 18, 'Wrong worker dimensions')
    for key in ('formal_submission_created', 'metrics_computed_from_truth', 'full_training_independence_claimed'):
        require(worker.get(key) is False, 'Invalid capture scope flag: ' + key)
    return result, worker


def verify_real_artifacts(results, protocol, artifacts_sha256):
    """Deep acceptance of files/runtime declarations; pure replay is a subsequent gate.

    This verifies the retained evidence, not a cryptographic attestation of GPU
    execution. Large input weights retain the capture's CPU-hash/version limit.
    """
    results = Path(results)
    source = protocol['source']
    result, worker = _complete_header(results, source)
    entries = verify_artifact_inventory(results, artifacts_sha256)
    mandatory = {'inputs/plan.json', 'inputs/preflight_manifest.json', 'inputs/runtime-build.json',
                 'inputs/availability.json', 'inputs/test.csv', 'capture-context.json',
                 'environment-before-setup.json', 'environment.json', 'mounted-input-check.json',
                 'input-availability-check.json', 'required-input-paths.json', 'captured-inventory.json',
                 'component-receipts.json', 'replay.json', 'worker-result.json', 'worker.log',
                 'setup-result.json', 'model-args.json', 'main-command.json', 'comp_logit.npy',
                 'comp_meta.json', 'v31_diagnostics.json', 'submission.parquet', 'cache-artifacts.zip',
                 'cache/main/manifest.json', 'cache/comp/manifest.json', 'logs/main.log', 'logs/comp.log',
                 'logs/gpu_preflight.log', 'logs/gpu_after_setup.log', 'logs/setup.log', 'logs/model_args.log'}
    require(mandatory <= set(entries), 'Incomplete capture stage/artifact set')
    require(read_json(results / 'guard.json') == result, 'Guard/result terminal records differ')
    state = read_json(results / 'state.json')
    require(state.get('status') == 'COMPLETE' and state.get('phase') == 'finished' and state.get('cause') is None,
            'Capture final state is incomplete')
    require(result.get('work_budget_seconds') == 28200 and result.get('platform_wall_seconds') == 28800
            and type(result.get('elapsed_seconds')) in (int, float) and 0 < result['elapsed_seconds'] < 28800,
            'Capture exceeded the fixed budget')
    require(result.get('formal_submission_created') is False and result.get('automatic_retry') is False,
            'Capture submission/retry contract differs')
    require(sha(results / 'inputs/plan.json') == SOURCE_PLAN_SHA, 'Captured plan hash differs')
    plan = read_json(results / 'inputs/plan.json')
    require(plan['experiment_id'] == SOURCE_EXPERIMENT and plan['run_id'] == SOURCE_RUN
            and plan['scope'] == 'exploratory_external' and plan['panel_rows'] == 729, 'Captured plan identity differs')
    require(source.get('input_hashes') == plan['input_hashes'], 'Protocol payload source hashes differ')
    require(sha(bound_input({'path': source['plan_path'], 'sha256': source['plan_sha256']})) == SOURCE_PLAN_SHA,
            'Local pre-launch plan differs')
    readback_path = bound_input(source['readback_evidence'])
    readback = read_json(readback_path)
    require(readback.get('status') == 'PASS' and readback.get('source_equal') is True
            and readback.get('attachments') == 31 and source.get('version_label') == 'v2'
            and source.get('notebook_ref') == 'zhjianduan/name-that-llm-cache-capture-031',
            'Fixed V2 source readback is not bound')
    metadata = read_json(readback_path.parent / 'kernel-metadata.json')
    notebook_path = readback_path.parent.joinpath(*safe_relative(metadata['code_file']).parts)
    require(sha(notebook_path) == source['notebook_sha256']
            and metadata.get('id') == source['notebook_ref'] and metadata.get('is_private') is True
            and metadata.get('enable_gpu') is True and metadata.get('enable_internet') is False
            and metadata.get('docker_image') == plan['container'], 'V2 source/environment readback differs')
    require(len(plan['expected_main_members']) == 16 and len(set(plan['expected_main_members'])) == 16
            and len(plan['expected_comp_members']) == 20 and len(set(plan['expected_comp_members'])) == 20
            and plan['allowed_omissions'] == EXPECTED_OMISSIONS, 'Wrong predeclared member inventory')
    for key in ('preflight_manifest_sha256', 'availability_sha256'):
        require(source[key] == plan[key], 'Protocol/source binding differs: ' + key)
    require(source['runner_sha256'] == plan['input_hashes']['runner.py'], 'Runner source binding differs')
    for name, digest in plan['input_hashes'].items():
        if name.endswith('.py'):
            captured = 'code/' + name
        elif name == 'panel/test.csv':
            captured = 'inputs/test.csv'
        elif name in ('preflight_manifest.json', 'availability.json', 'runtime-build.json'):
            captured = 'inputs/' + name
        elif name == 'ATTRIBUTION.md':
            continue  # The original runner did not export this non-executable payload file.
        else:
            raise AcceptanceError('Unexpected original payload input: ' + name)
        require(captured in entries and entries[captured]['sha256'] == digest, 'Captured source/input changed: ' + name)
    build = read_json(results / 'inputs/runtime-build.json')
    require(build.get('status') == 'BUILT_NOT_EXECUTED' and build.get('baseline_sha256') == BASELINE_SHA
            and build.get('models_executed') is False, 'Runtime source lineage differs')
    require(all(plan['input_hashes'].get(key) == value for key, value in build['files'].items()),
            'Runtime build source hashes differ')
    preflight = read_json(results / 'inputs/preflight_manifest.json')
    require(preflight.get('status') == 'PASS' and preflight.get('attachment_bindings') == plan['attachment_bindings']
            and len(plan['attachment_bindings']) == 31 and len(preflight['files']) == 531,
            'CPU input manifest/31 attachments differ')
    require(any(item['sha256'] == plan['train_sha256'] and PurePosixPath(item['path']).name == 'train.csv'
                for item in preflight['files']), 'Official training bytes are absent from CPU audit')
    mounted = read_json(results / 'mounted-input-check.json')
    small = [item for item in preflight['files'] if item['bytes'] <= 16 * 1024**2]
    require(mounted.get('files') == 531 and mounted.get('small_files_rehashed') == len(small)
            and mounted.get('large_files_version_bound') == 531 - len(small), 'Mounted input audit is incomplete')
    availability = read_json(results / 'inputs/availability.json')
    check = read_json(results / 'input-availability-check.json')
    require(check.get('status') == 'PASS' and check.get('cpu') == availability and check.get('gpu') == availability
            and check.get('availability_sha256') == plan['availability_sha256']
            and check.get('input_policy_sha256') == plan['input_hashes']['input_policy.py'],
            'CPU/GPU availability was not equal to the frozen input policy')
    require(read_json(results / 'required-input-paths.json') == availability, 'Resolved input evidence differs')
    require(availability['expected_comp_members'] == plan['expected_comp_members']
            and availability['allowed_omissions'] == EXPECTED_OMISSIONS
            and all(item.get('model_type') != 'qwen3_5' for item in check.get('config_model_types', [])),
            'Unexpected optional backbone/member')
    for name in ('environment-before-setup.json', 'environment.json'):
        environment = read_json(results / name)
        require(environment.get('requested_container') == plan['container'], 'Container declaration differs')
        devices = environment.get('devices')
        require(isinstance(devices, list) and len(devices) == 2
                and [device.get('index') for device in devices] == [0, 1]
                and all(re.search(r'\bT4\b', device.get('name', '').upper()) for device in devices),
                'The actual-device receipt is not T4x2')
        for package, version in plan['required_versions'].items():
            key = 'sklearn' if package == 'scikit-learn' else package
            require(environment['versions'].get(key) == version, 'Runtime package version differs: ' + package)
    context = read_json(results / 'capture-context.json')
    require(context.get('scope') == 'exploratory_external' and context.get('launch_ready') is True
            and context.get('run_id') == SOURCE_RUN and context.get('plan_sha256') == SOURCE_PLAN_SHA,
            'Synthetic/unprepared/wrong capture context')
    for key in ('expected_main_members', 'expected_comp_members', 'expected_family_gates', 'allowed_omissions',
                'preflight_manifest_sha256', 'availability_sha256', 'attachment_bindings', 'panel_sha256'):
        require(context.get(key) == plan[key], 'Context differs from the frozen plan: ' + key)
    receipts = read_json(results / 'component-receipts.json')
    require(set(receipts) == {'main', 'comp'} and receipts == worker['component_receipts'], 'Worker/component receipts differ')
    for component, receipt in receipts.items():
        require(receipt.get('run_id') == SOURCE_RUN and receipt.get('component') == component
                and receipt.get('scope') == 'exploratory_external' and is_digest(receipt.get('manifest_sha256')),
                'Invalid external component receipt')
        require(entries['cache/' + component + '/manifest.json']['sha256'] == receipt['manifest_sha256'],
                'Component manifest differs from external receipt')
        prefix = 'CACHE_COMPONENT_COMMITTED '
        logged = [json.loads(line[len(prefix):]) for line in (results / ('logs/' + component + '.log')).read_text(
            encoding='utf-8').splitlines() if line.startswith(prefix)]
        require(logged == [receipt], 'Missing/duplicate/different component commit log receipt')
    replay = read_json(results / 'replay.json')
    _accept_replay(replay, plan)
    require(worker.get('replay_status') == replay['status'], 'Worker replay status differs')
    diagnostics = read_json(results / 'v31_diagnostics.json')
    require(diagnostics.get('method') == 'hierarchical_beta_bernoulli_typography_v2'
            and diagnostics.get('trained_rows') == 2699 and diagnostics.get('tested_rows') == 729
            and diagnostics.get('sinkhorn') == .45 and diagnostics.get('sinkhorn_target_mix') == .35,
            'Final baseline diagnostics differ')
    _verify_archive(results, entries, result['archive'])
    return {'plan': plan, 'context': context, 'receipts': receipts, 'worker': worker, 'result': result,
            'entries': entries, 'captured_replay': replay, 'artifacts_sha256': artifacts_sha256}


def _accept_replay(replay, plan):
    require(replay.get('accepted') is True and replay.get('status') == 'PASS_CACHED_BASELINE_MATH'
            and replay.get('input_scope') == 'exploratory_external' and replay.get('rows') == 729
            and replay.get('all_checked_argmax_equal') is True, 'Real full-stack replay did not pass')
    for key, expected in (('main_members', 'expected_main_members'), ('comp_members', 'expected_comp_members'),
                          ('family_gates', 'expected_family_gates')):
        values = replay.get(key, [])
        require(len(values) == len(plan[expected]) and set(values) == set(plan[expected]), 'Replay member/gate coverage differs')
    coverage = replay.get('atlas_coverage')
    require(isinstance(coverage, list) and bool(coverage), 'No real ATLAS coverage evidence')
    for item in coverage:
        require(type(item.get('requested_rows')) is int and item['requested_rows'] >= 0
                and item.get('completed_rows') == item['requested_rows']
                and item.get('missing_row_indices') == [] and item.get('missing_ids') == [],
                'ATLAS requested rows are incomplete; no comparison is allowed')


@contextmanager
def captured_helpers(results, plan):
    """Execute only the pinned pure cache/replay helpers, never inference code."""
    previous, modules = {}, {}
    try:
        for name in HELPERS:
            path = results / 'code' / (name + '.py')
            content = path.read_bytes()
            require(hashlib.sha256(content).hexdigest() == plan['input_hashes'][name + '.py'], 'Replay helper changed')
            previous[name] = sys.modules.get(name)
            module = types.ModuleType(name); module.__file__ = str(path)
            sys.modules[name] = module
            exec(compile(content, str(path), 'exec'), module.__dict__)
            modules[name] = module
        yield modules
        for name in HELPERS:
            require(sha(results / 'code' / (name + '.py')) == plan['input_hashes'][name + '.py'], 'Replay helper changed at close')
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _one(events, name):
    values = [payload for event, payload in events if event == name]
    require(len(values) == 1, 'Missing/duplicate event: ' + name)
    return values[0]


def local_module(path, name):
    content = Path(path).read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(content, str(path), 'exec'), module.__dict__)
    return module


def read_parquet_predictions(path, python=None):
    if python is None:
        frame = pd.read_parquet(path)
    else:
        require(Path(python).is_file(), 'Explicit Parquet Python does not exist')
        code = ("import json,sys,pandas as pd; f=pd.read_parquet(sys.argv[1]); "
                "print(json.dumps({'columns':f.columns.tolist(),'data':f.values.tolist()},allow_nan=False))")
        child = subprocess.run([str(python), '-B', '-X', 'utf8', '-c', code, str(path)], check=True,
                               capture_output=True, text=True, encoding='utf-8', timeout=60)
        value = json.loads(child.stdout)
        frame = pd.DataFrame(value['data'], columns=value['columns'])
    require(list(frame.columns) == ['id', 'model'] and all(isinstance(value, str) for value in frame.id)
            and all(isinstance(value, str) for value in frame.model), 'Final Parquet schema differs')
    return frame


def bound_input(spec):
    relative = safe_relative(spec['path'])
    path = ROOT.joinpath(*relative.parts)
    require(is_digest(spec['sha256']) and path.is_file() and sha(path) == spec['sha256'],
            'Pinned local input is missing or changed: ' + spec['path'])
    return path


def _same_scores(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            'Nonfinite/misaligned baseline array: ' + label)
    require(actual.dtype == expected.dtype and expected.dtype in (np.dtype('float32'), np.dtype('float64')),
            'Baseline array dtype differs: ' + label)
    tolerance = 2e-6 if expected.dtype == np.dtype('float32') else 2e-10
    require(np.allclose(actual, expected, rtol=tolerance, atol=tolerance), 'Recomputed .45 differs from cached ' + label)
    if actual.ndim == 2 and actual.shape[1] == 18:
        require(np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1)), 'Baseline argmax differs: ' + label)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def verify_control(control, main_events, parquet, ids, labels):
    """Reject a .45 reconstruction before permitting the .30 branch."""
    sink = _one(main_events, 'sinkhorn_applied')
    typo = _one(main_events, 'typography')
    checks = {}
    for field, value in (('balanced_probability', sink['balanced_probability']),
                         ('predicted_marginal', sink['predicted_marginal']), ('train_prior', sink['train_prior']),
                         ('target_marginal', sink['target']), ('pre_typography_logits', typo['logits_before']),
                         ('typography_evidence', typo['evidence']), ('confidence_gate', typo['confidence_gate']),
                         ('length_gate', typo['length_gate']), ('coverage_gate', typo['coverage_gate']),
                         ('final_gate', typo['gate']), ('final_logits', typo['logits_after'])):
        checks[field] = _same_scores(control[field], value, field)
    require(parquet.id.tolist() == list(ids), 'Final Parquet ID/order differs')
    prediction = [labels[int(index)] for index in control['argmax']]
    require(parquet.model.tolist() == prediction, 'Recomputed .45 does not match final Parquet predictions')
    return {'status': 'PASS', 'argmax_equal': True, 'maximum_absolute_errors': checks,
            'tolerances': {'float32': {'rtol': 2e-6, 'atol': 2e-6}, 'float64': {'rtol': 2e-10, 'atol': 2e-10}},
            'parquet_rows_equal': True}


def validate_protocol(protocol):
    require(protocol.get('schema_version') == 1 and protocol.get('experiment_id') == 'EXP-032'
            and protocol.get('scope') == 'exploratory_external_fixed_cache', 'Wrong comparison protocol')
    arms = protocol.get('arms', [])
    require([(arm.get('id'), arm.get('sinkhorn'), arm.get('role')) for arm in arms]
            == [('sinkhorn_045', .45, 'baseline'), ('sinkhorn_030', .30, 'candidate')], 'Changed fixed comparison arms')
    frozen = protocol['frozen_math']
    for key, value in {'only_changed_parameter': 'sinkhorn', 'sinkhorn_target_mix': .35,
                       'sinkhorn_iterations': 100, 'typography_scale': 1.5,
                       'typography_clip': [-.3, .3], 'typography_kappa': 24.0}.items():
        require(frozen.get(key) == value, 'Changed frozen math: ' + key)
    bootstrap = protocol['metrics']['bootstrap']
    require(bootstrap.get('draws') == 2000 and bootstrap.get('seed') == 42
            and bootstrap.get('groups_per_draw') == 56 and bootstrap.get('paired') is True,
            'Changed bootstrap protocol')
    require(set(protocol['inputs']) == {'blind_panel', 'official_train', 'truth_sidecar'}, 'Changed local input set')
    bound_input(protocol['implementation']['metrics_module'])
    require(sha(ROOT / 'src/sinkhorn_compare.py') == MATH_SHA256, 'Frozen mathematical helper changed')


def compare(results, protocol, artifacts_sha256, parquet_python=None):
    """Return in-memory results; callers must explicitly publish to a fresh directory."""
    validate_protocol(protocol)
    accepted = verify_real_artifacts(results, protocol, artifacts_sha256)
    results = Path(results); plan = accepted['plan']
    panel_path = bound_input(protocol['inputs']['blind_panel'])
    train_path = bound_input(protocol['inputs']['official_train'])
    require(sha(panel_path) == plan['panel_sha256'] and sha(train_path) == plan['train_sha256'],
            'Local data does not match captured data')
    # Match the original infer.py CSV parsing and text fillna exactly.
    panel = pd.read_csv(panel_path)
    train = pd.read_csv(train_path)
    panel['text'] = panel.text.fillna('')
    train['text'] = train.text.fillna('')
    require(list(panel.columns) == ['id', 'text'] and len(panel) == 729 and panel.id.is_unique
            and panel.text.str.strip().ne('').all(), 'Invalid blind panel')
    require({'text', 'model'} <= set(train.columns) and len(train) == 2699, 'Invalid official training table')
    parquet = read_parquet_predictions(results / 'submission.parquet', parquet_python)
    with captured_helpers(results, plan) as helpers:
        labels = list(helpers['full_stack_cache'].LABELS)
        require(labels == protocol['evaluation']['output_labels'], 'Protocol output-label order differs')
        snapshots = {name: helpers['full_stack_cache'].read_cache(results / 'cache', name,
            panel.id.tolist(), panel.text.tolist(), labels, required_notes=['event-index'],
            manifest_sha256=accepted['receipts'][name]['manifest_sha256']) for name in ('main', 'comp')}
        for component, snapshot in snapshots.items():
            runtime = snapshot.metadata.get('runtime_metadata', {})
            require(snapshot.metadata.get('run_id') == SOURCE_RUN and snapshot.metadata.get('scope') == 'exploratory_external'
                    and snapshot.metadata.get('context') == accepted['context']
                    and snapshot.metadata.get('context_sha256') == sha(results / 'capture-context.json')
                    and not any(runtime.get(key) for key in ('smoke', 'tiny', 'fake_test', 'sim')),
                    'Cache identity/context is synthetic or differs from the real run')
            require(runtime.get('source_sha256') == (SOURCE_INFER_SHA if component == 'main' else SOURCE_COMP_SHA)
                    and runtime.get('train_rows') == 2699 and runtime.get('data_dir') == '/kaggle/working/panel',
                    'Component source/data declaration differs')
        events = {name: helpers['full_stack_observer'].decode_events(snapshot) for name, snapshot in snapshots.items()}
        replay = helpers['full_stack_replay'].verify_replay(events['main'], events['comp'],
                    plan['expected_main_members'], plan['expected_comp_members'])
        _accept_replay(replay, plan)
        captured = accepted['captured_replay']
        require(replay['atlas_coverage'] == captured['atlas_coverage'], 'Replayed ATLAS coverage differs')
        inventory = read_json(results / 'captured-inventory.json')
        require(inventory.get('main') == replay['main_members'] and inventory.get('comp') == replay['comp_members']
                and inventory.get('family_gates') == replay['family_gates']
                and inventory.get('comp_eligible_rows') == int(_one(events['comp'], 'comp_before_gates')['group_mask'].sum()),
                'Persisted member/group inventory differs from actual events')
        comp_meta = read_json(results / 'comp_meta.json')
        loaded_comp = _one(events['main'], 'comp_loaded')
        comp_array = np.load(results / 'comp_logit.npy', allow_pickle=False)
        require(comp_meta == loaded_comp['comp_meta'] and comp_meta.get('labels') == labels
                and comp_meta.get('ids') == panel.id.tolist() and comp_meta.get('family_gates_applied') is True,
                'Persisted comp row/label metadata differs')
        require(comp_array.dtype == np.float32 and comp_array.shape == (len(panel), 18)
                and np.array_equal(comp_array, loaded_comp['logit'])
                and np.array_equal(comp_array, _one(events['comp'], 'component_end')['serialized_logit']),
                'Persisted float32 comp array differs from both caches')
        member = _one(events['main'], 'main_members')
        args = member['args']
        require(args['sinkhorn'] == .45 and args['sinkhorn_target_mix'] == .35 and not args['sim']
                and args['data'] == '/kaggle/working/panel', 'Original inference parameters differ')
        for key in ('comp_w', 'atlas_margin', 'sk_div', 'perpassage', 'post'):
            require(args[key] == protocol['frozen_math'][key], 'Captured fixed argument differs: ' + key)
        require(float(args['atlas_w']) == protocol['frozen_math']['atlas_w'], 'Captured ATLAS weight differs')
        train_targets = np.array([labels.index(value) for value in train.model], dtype=np.int64)
        require(np.array_equal(member['train_targets'], train_targets), 'Official train labels/order differ from capture')
        pre = _one(events['main'], 'pre_sinkhorn')['probability']
        # The mathematical helper receives no evaluation truth or prompt IDs.
        math_module = local_module(ROOT / 'src/sinkhorn_compare.py', '_exp032_sinkhorn_math')
        require(math_module.SOURCE_SHA256 == SOURCE_INFER_SHA
                and sha(ROOT / math_module.SOURCE_PATH) == SOURCE_INFER_SHA, 'Mathematical reference source differs')
        control = math_module.compute_sinkhorn_variant(pre, train_targets, train.text.tolist(), panel.text.tolist(), variant='control', labels=labels)
        control_check = verify_control(control, events['main'], parquet, panel.id.tolist(), labels)
        candidate = math_module.compute_sinkhorn_variant(pre, train_targets, train.text.tolist(), panel.text.tolist(), variant='candidate', labels=labels)
    # Evaluation truth is first loaded after both branches and baseline parity.
    truth_path = bound_input(protocol['inputs']['truth_sidecar'])
    truth = pd.read_csv(truth_path, dtype=str, keep_default_na=False)
    require(list(truth.columns) == ['id', 'source_id', 'prompt_id', 'model', 'text_sha256']
            and truth.id.is_unique and truth.id.tolist() == panel.id.tolist(), 'Truth sidecar ID/order differs')
    hashes = [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in panel.text]
    require(truth.text_sha256.tolist() == hashes and len(truth) == 729 and truth.prompt_id.nunique() == 56,
            'Truth sidecar text/group binding differs')
    metrics_module = local_module(HERE / 'metrics_utils.py', '_exp032_metrics')
    require(sorted(truth.model.unique()) == sorted(metrics_module.COVERED_LABELS), 'Truth is not the fixed 17 covered classes')
    evaluation = protocol['evaluation']
    require(evaluation['rows'] == len(truth) and evaluation['groups'] == truth.prompt_id.nunique()
            and evaluation['labels'] == list(metrics_module.COVERED_LABELS)
            and evaluation['class_counts'] == truth.model.value_counts().to_dict(), 'Frozen evaluation composition differs')
    predictions = truth[['id', 'source_id', 'prompt_id', 'model']].copy()
    predictions['prediction_045'] = metrics_module.prediction_labels(control['final_logits'])
    predictions['prediction_030'] = metrics_module.prediction_labels(candidate['final_logits'])
    metrics = metrics_module.compare_predictions(predictions)
    source_ids = truth.set_index('id').source_id.to_dict()
    for error in metrics['errors']:
        error['source_id'] = source_ids[error['id']]
    # Recheck the externally bound inventory after all reads; no outputs live under results.
    verify_artifact_inventory(results, artifacts_sha256)
    for spec in protocol['inputs'].values():
        bound_input(spec)
    return {'accepted': accepted, 'replay': replay, 'control_check': control_check,
            'control': control, 'candidate': candidate, 'predictions': predictions, 'metrics': metrics}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    parser.add_argument('--artifacts-sha256', required=True, help='External digest retained after the real EXP-031 download')
    parser.add_argument('--parquet-python', type=Path, help='Existing interpreter with pandas/pyarrow; no installation is attempted')
    parser.add_argument('--output', type=Path, required=True, help='A new local analysis directory, separate from the capture')
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'Refusing an existing comparison output directory')
    require(not args.output.resolve().is_relative_to(args.results.resolve()), 'Comparison output cannot be inside source results')
    protocol_hash = sha(args.protocol)
    code_paths = [Path(__file__), HERE / 'metrics_utils.py', ROOT / 'src/sinkhorn_compare.py']
    code_hashes = {str(path.relative_to(ROOT)): sha(path) for path in code_paths}
    protocol = read_json(args.protocol)
    try:
        outcome = compare(args.results, protocol, args.artifacts_sha256, args.parquet_python)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=False)
        save_json(args.output / 'failure.json', {'status': 'BLOCKED', 'experiment_id': 'EXP-032',
                  'error_type': type(error).__name__, 'reason': str(error), 'metrics_published': False,
                  'source_run_id': SOURCE_RUN, 'protocol_sha256': protocol_hash})
        raise
    require(sha(args.protocol) == protocol_hash and all(sha(ROOT / name) == digest for name, digest in code_hashes.items()),
            'Comparison code/protocol changed while running')
    args.output.mkdir(parents=True, exist_ok=False)
    outcome['predictions'].to_csv(args.output / 'predictions.csv', index=False)
    arrays = {variant + '_' + key: value for variant in ('control', 'candidate')
              for key, value in outcome[variant].items() if isinstance(value, np.ndarray)}
    arrays['ids'] = outcome['predictions'].id.to_numpy(dtype=str)
    arrays['labels'] = np.asarray(outcome['control']['metadata']['labels'], dtype=str)
    np.savez_compressed(args.output / 'scores.npz', **arrays)
    save_json(args.output / 'summary.json', outcome['metrics']['summary'])
    for key in ('per_class', 'per_prompt', 'errors'):
        columns = ['id', 'source_id', 'prompt_id', 'model', 'prediction_045', 'prediction_030', 'change'] if key == 'errors' else None
        pd.DataFrame(outcome['metrics'][key], columns=columns).to_csv(args.output / (key + '.csv'), index=False)
    save_json(args.output / 'analysis-manifest.json', {
        'status': 'PASS', 'scope': 'exploratory_external', 'source_experiment': SOURCE_EXPERIMENT,
        'run_id': SOURCE_RUN, 'plan_sha256': SOURCE_PLAN_SHA, 'artifacts_sha256': args.artifacts_sha256,
        'protocol_sha256': protocol_hash, 'code_sha256': code_hashes,
        'source': protocol['source'], 'input_bindings': protocol['inputs'],
        'replay': outcome['replay'], 'control_check': outcome['control_check'],
        'large_weight_limit': 'CPU full hashes plus fixed attachment versions and GPU path/size; no local weight rehash',
        'runtime_attestation_limit': 'Retained source/device/log evidence verified; not a cryptographic execution attestation',
        'no_training': True, 'no_network': True, 'evaluation_truth_used_for_math': False})
    save_json(args.output / 'result.json', {
        'status': 'COMPLETE', 'experiment_id': 'EXP-032', 'scope': 'exploratory_external',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'source_run_id': SOURCE_RUN,
        'control': .45, 'candidate': .30, 'metrics': outcome['metrics'],
        'control_math': outcome['control']['metadata'], 'candidate_math': outcome['candidate']['metadata'],
        'independent_holdout_claimed': False, 'official_lb_gain_claimed': False,
        'code_sha256': code_hashes, 'protocol_sha256': protocol_hash})
    saved = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(args.output.iterdir()) if path.is_file()]
    save_json(args.output / 'artifacts.json', {'files': saved})
    print(json.dumps({'status': 'COMPLETE', 'output': str(args.output), 'rows': len(outcome['predictions'])}))


if __name__ == '__main__':
    main()
