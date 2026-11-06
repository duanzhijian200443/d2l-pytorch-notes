"""One-shot official Kaggle SDK controller for EXP046. Never submits predictions.

save-only -> saved V1 plus readback; run -> exactly one V2 execution request.
An intent is durably recorded before each write and is never cleared/retried.
Read-only commands preserve separate receipts. Uses only this cloud directory.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

HERE = Path(__file__).resolve().parent
REF = 'zhjianduan/name-that-llm-a1-public-stylegram-046'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
MACHINE = 'NvidiaTeslaT4'
EXTRA = 'zhjianduan/ntl-stylegram-cv-inputs-v1/1'
SOURCE_KEYS = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources')
TIMEOUT_SECONDS = 32400
OUTPUT_PATTERN = r'^(submission\.parquet|submission_receipt\.json|input_preflight\.json|availability\.json|a1_synthetic_parity\.json|reproduction_environment\.json|comp-process\.log|a1_training_receipt\.json|comp_meta\.json)$'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def state():
    path = HERE / 'execution-state.json'
    value = read(path) if path.exists() else {'experiment_id': 'EXP-046', 'notebook_ref': REF}
    require(value.get('notebook_ref', REF) == REF, 'execution-state.json belongs to another notebook')
    return value


def update(**changes):
    value = state()
    value.update(changes, observed_at=now(), notebook_ref=REF)
    save(HERE / 'execution-state.json', value)
    return value


@contextmanager
def command_lock():
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.control.lock').open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def canonical_notebook(path):
    """Canonical full JSON, matching SDK source-array join/output clearing.

    Preserve all notebook/cell metadata and execution_count. Only normalize JSON
    formatting/key order, actual source newlines, joined source lists and the
    output arrays the SDK itself clears before saving.
    """
    value = read(path)
    require(value.get('nbformat') == 4 and isinstance(value.get('cells'), list), 'Expected nbformat4 notebook')
    for cell in value['cells']:
        source = cell.get('source', '')
        if isinstance(source, list):
            require(all(isinstance(line, str) for line in source), 'Invalid notebook cell source')
            source = ''.join(source)
        require(isinstance(source, str), 'Invalid notebook cell source')
        cell['source'] = source.replace('\r\n', '\n').replace('\r', '\n')
        if cell.get('cell_type') == 'code' and 'outputs' in cell:
            cell['outputs'] = []
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def normalized_ref(ref, *, version_required):
    require(isinstance(ref, str), 'Attachment ref must be text')
    parts = ref.split('/')
    require(len(parts) in (2, 3) and all(parts), 'Invalid dataset/kernel reference: ' + ref)
    if len(parts) == 3:
        require(parts[2].isdigit() and int(parts[2]) > 0, 'Invalid attachment version: ' + ref)
    else:
        require(not version_required, 'Unpinned requested attachment: ' + ref)
    return '/'.join(parts[:2])


def local_package():
    folder = HERE / 'kernel'
    metadata = read(folder / 'kernel-metadata.json')
    require(metadata.get('id') == REF, 'Local notebook ref mismatch')
    for key, expected in {'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
                          'enable_internet': False, 'machine_shape': MACHINE,
                          'docker_image': IMAGE, 'docker_image_pinning_type': 'original',
                          'language': 'python', 'kernel_type': 'notebook'}.items():
        require(metadata.get(key) == expected, 'Local frozen setting mismatch: ' + key)
    for key in SOURCE_KEYS:
        require(isinstance(metadata.get(key), list) and len(set(metadata[key])) == len(metadata[key]), 'Invalid/duplicate input set: ' + key)
    require(sum(len(metadata[key]) for key in SOURCE_KEYS) == 32, 'Expected exactly 32 attached inputs including competition')
    require(metadata['competition_sources'] == ['name-that-llm'] and EXTRA in metadata['dataset_sources'], 'Competition/fixed Stylegram input changed')
    for key in ('dataset_sources', 'kernel_sources'):
        refs = [normalized_ref(ref, version_required=True) for ref in metadata[key]]
        require(len(refs) == len(set(refs)), 'Duplicate normalized input: ' + key)
    for ref in metadata['model_sources']:
        require(isinstance(ref, str) and len(ref.split('/')) == 5 and ref.split('/')[-1].isdigit()
                and int(ref.split('/')[-1]) > 0, 'Model input must retain its explicit version')
    name = metadata['code_file']
    require(isinstance(name, str) and Path(name).name == name and '\\' not in name and ':' not in name and name.endswith('.ipynb'), 'Unsafe notebook code filename')
    code = folder / name
    bindings = {'metadata_sha256': sha(folder / 'kernel-metadata.json'), 'source_bytes_sha256': sha(code),
                'source_canonical_json_sha256': canonical_notebook(code), 'input_count': 32,
                'ref': REF, 'controller_sha256': sha(__file__)}
    return folder, metadata, bindings


def verify_readback(expected, actual, expected_code, actual_code):
    require(actual.get('id') in (REF, '/code/' + REF), 'Wrong saved notebook ref')
    checks = {}
    for key in ('title', 'language', 'kernel_type', 'is_private', 'enable_gpu', 'enable_tpu',
                'enable_internet', 'docker_image', 'machine_shape'):
        require(actual.get(key) == expected.get(key), 'Saved setting mismatch: ' + key)
        checks[key] = {'expected': expected.get(key), 'actual': actual.get(key), 'match': True}
    require(actual['docker_image'] == IMAGE and actual['machine_shape'] == MACHINE, 'Wrong actual image/machine shape')
    require(sum(len(actual.get(key, [])) for key in SOURCE_KEYS) == 32, 'Saved input count differs from 32')
    bindings = []
    for key in ('dataset_sources', 'kernel_sources'):
        requested = expected[key]
        saved = actual.get(key, [])
        normalized = [normalized_ref(ref, version_required=False) for ref in saved]
        require(len(set(normalized)) == len(normalized), 'Duplicate saved normalized attachment')
        expected_by_ref = {normalized_ref(ref, version_required=True): ref for ref in requested}
        require(set(normalized) == set(expected_by_ref), 'Saved attachment set differs: ' + key)
        for ref in saved:
            exact = expected_by_ref[normalized_ref(ref, version_required=False)]
            if len(ref.split('/')) == 3:
                require(ref == exact, 'Saved explicit attachment version changed')
            bindings.append({'kind': key, 'requested': exact, 'saved': ref,
                             'version_binding_source': 'saved metadata exact version' if len(ref.split('/')) == 3
                             else 'original pinned save request; SDK pull metadata omits version; remote bytes not certified here'})
    for key in ('model_sources', 'competition_sources'):
        require(sorted(actual.get(key, [])) == sorted(expected[key]), 'Saved exact attachment differs: ' + key)
        for ref in actual[key]:
            bindings.append({'kind': key, 'requested': ref, 'saved': ref, 'version_binding_source': 'saved metadata exact reference'})
    expected_hash, actual_hash = canonical_notebook(expected_code), canonical_notebook(actual_code)
    require(expected_hash == actual_hash, 'Saved canonical notebook JSON differs')
    return {'checks': checks, 'input_bindings': bindings, 'input_count': 32,
            'source_canonical_json_sha256': actual_hash, 'source_bytes_sha256': sha(actual_code),
            'image_field_verified': True, 'machine_shape_field_verified': True,
            'gpu_count_limit': 'NvidiaTeslaT4 is the requested T4x2 shape; physical GPU count requires runtime receipt.',
            'pinning_limit': 'SDK pull omits docker_image_pinning_type; local request is original and actual returned image digest matches.'}


def unique_observation(prefix):
    return HERE / f'{prefix}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}-{uuid.uuid4().hex[:6]}'


def readback(api, version):
    folder, expected, binding = local_package()
    destination = unique_observation(f'readback-v{version}')
    destination.mkdir(exist_ok=False)
    save(destination / 'intent.json', {'ref': REF, 'version': version, 'operation': 'kernels_pull', 'request_count': 1,
                                     'local_bindings': binding, 'observed_at': now()}, exclusive=True)
    api.kernels_pull(f'{REF}/{version}', str(destination), metadata=True, quiet=True)
    actual = read(destination / 'kernel-metadata.json')
    code_name = actual.get('code_file')
    require(isinstance(code_name, str) and Path(code_name).name == code_name and '\\' not in code_name and ':' not in code_name, 'Unsafe pulled code filename')
    details = verify_readback(expected, actual, folder / expected['code_file'], destination / code_name)
    require(local_package()[2] == binding, 'Local package changed during readback')
    receipt = {'status': 'PASS', 'ref': REF, 'version': version, 'observed_at': now(), 'local_bindings': binding,
               'metadata_sha256': sha(destination / 'kernel-metadata.json'), **details}
    save(destination / 'verified.json', receipt, exclusive=True)
    receipts = state().get('readback_receipts', {})
    receipts[str(version)] = {'path': str((destination / 'verified.json').relative_to(HERE)),
                              'sha256': sha(destination / 'verified.json')}
    update(readback_receipts=receipts)
    return receipt


def response_dict(response):
    return response.to_dict() if hasattr(response, 'to_dict') else {'text': str(response)}


def require_saved(response, version):
    require(not response.error, 'Kaggle rejected save: ' + str(response.error))
    for key in ('invalid_dataset_sources', 'invalid_kernel_sources', 'invalid_model_sources', 'invalid_competition_sources', 'invalid_tags'):
        require(not getattr(response, key, None), 'Kaggle rejected input/tag: ' + key)
    require(response.ref in (REF, '/code/' + REF) and response.version_number == version,
            'Unexpected saved notebook/version; do not retry')


def assert_unused(api):
    for page in range(1, 11):
        items = api.kernels_list(user='zhjianduan', mine=True, page=page, page_size=100)
        require(items is not None, 'Cannot establish unused notebook destination')
        require(not any(item is not None and item.ref in (REF, '/code/' + REF) for item in items), 'Notebook destination already exists; no overwrite')
        if len(items) < 100:
            return
    raise ValueError('Owned notebook inventory incomplete; save not dispatched')


def validated_saved_package(binding):
    current = state()
    require(current.get('saved_version') == 1 and current.get('save_request_count') == 1, 'V1 save acknowledgement required before run')
    item = current.get('readback_receipts', {}).get('1')
    require(item is not None, 'Saved V1 readback PASS is required before run')
    path = HERE / item['path']
    require(sha(path) == item['sha256'], 'Saved V1 readback receipt changed')
    receipt = read(path)
    require(receipt['status'] == 'PASS' and receipt['ref'] == REF and receipt['version'] == 1
            and receipt['local_bindings'] == binding, 'Saved package differs from current launch package')
    pulled = read(path.parent / 'kernel-metadata.json')
    require(sha(path.parent / 'kernel-metadata.json') == receipt['metadata_sha256'], 'Saved V1 metadata evidence changed')
    require(sha(path.parent / pulled['code_file']) == receipt['source_bytes_sha256'], 'Saved V1 source evidence changed')


def push_once(api, run):
    folder, metadata, binding = local_package()
    operation, version = ('run', 2) if run else ('save-only', 1)
    intent_path = HERE / f'{operation}-intent.json'
    require(not intent_path.exists(), f'{operation} already has an intent; outcome may be unknown, never retry')
    current = state()
    count_key = 'run_request_count' if run else 'save_request_count'
    require(current.get(count_key, 0) == 0, operation + ' request already recorded')
    if run:
        validated_saved_package(binding)
    else:
        assert_unused(api)
    intent = {'operation': operation, 'expected_version': version, 'no_run': not run, 'ref': REF,
              'request_count': 1, 'local_bindings': binding, 'observed_at': now(),
              'machine_shape': MACHINE, 'timeout_seconds': TIMEOUT_SECONDS}
    save(intent_path, intent, exclusive=True)
    update(**{count_key: 1}, status=operation.upper().replace('-', '_') + '_IN_FLIGHT',
           gpu_may_be_active=run, ack_outcome='UNKNOWN_UNTIL_RESPONSE')
    # Recheck after durable intent. Any local failure consumes the intent too;
    # this controller deliberately offers no reset or automatic recovery path.
    require(local_package()[2] == binding, 'Local package changed before SDK dispatch')
    response = api.kernels_push(str(folder), timeout=TIMEOUT_SECONDS, acc=MACHINE, no_run=not run)
    save(HERE / f'{operation}-response.json', {'observed_at': now(), 'response': response_dict(response)}, exclusive=True)
    require_saved(response, version)
    update(**({'run_version': version} if run else {'saved_version': version}),
           status='RUN_ACKNOWLEDGED' if run else 'SAVED_NOT_RUN', ack_outcome='ACKNOWLEDGED', gpu_may_be_active=run)
    receipt = readback(api, version)
    update(status='RUN_V2_READBACK_VERIFIED' if run else 'SAVED_V1_READBACK_VERIFIED',
           gpu_may_be_active=run, ack_outcome='ACKNOWLEDGED')
    return {'status': 'PASS', 'operation': operation, 'version': version, 'request_count': 1,
            'saved_readback': receipt, 'note': 'No competition submission is performed by this controller.'}


def pick_version(explicit):
    current = state()
    value = explicit if explicit is not None else current.get('run_version', current.get('saved_version'))
    require(type(value) is int and value in (1, 2), 'Specify --version 1 or 2, or establish a saved acknowledgement')
    return value


def observe_status(api, version):
    destination = unique_observation(f'status-v{version}').with_suffix('.json')
    save(destination.with_suffix('.intent.json'), {'ref': REF, 'version': version, 'operation': 'kernels_status',
            'request_count': 1, 'observed_at': now()}, exclusive=True)
    result = api.kernels_status(f'{REF}/{version}')
    status_name = str(getattr(result.status, 'name', result.status)).split('.')[-1].upper()
    receipt = {'ref': REF, 'version': version, 'observed_at': now(), 'request_count': 1,
               'status': status_name, 'response': response_dict(result)}
    save(destination, receipt, exclusive=True)
    update(last_status={'version': version, 'status': status_name, 'path': str(destination.relative_to(HERE)), 'observed_at': receipt['observed_at']})
    return receipt


def download(api, version):
    require(version == 2, 'Only executed V2 outputs are downloaded')
    observed = observe_status(api, version)
    require(observed['status'] == 'COMPLETE', 'V2 is not COMPLETE; output download not started')
    output = HERE / f'outputs-v{version}'
    require(not output.exists(), 'Output directory already exists; no overwrite or implicit download retry')
    output.mkdir()
    save(output / 'download-intent.json', {'ref': REF, 'version': version, 'request_count': 1, 'observed_at': now()}, exclusive=True)
    # SDK automatically follows all output pages with the same filter. The
    # notebook log is returned separately by the API, outside this file filter.
    paths, token = api.kernels_output(f'{REF}/{version}', str(output), file_pattern=OUTPUT_PATTERN,
                                     quiet=True, force=False)
    require(not token, 'Output pagination incomplete')
    files = []
    for name in paths:
        supplied = Path(name)
        path = (supplied if supplied.is_absolute() else output / supplied).resolve()
        require(path.is_relative_to(output.resolve()) and path.is_file(), 'Downloaded file outside output root')
        relative = path.relative_to(output.resolve()).as_posix()
        require(re.fullmatch(OUTPUT_PATTERN, relative) is not None or relative == REF.split('/')[-1]+'.log',
                'Unexpected downloaded artifact outside allowlist')
        files.append({'path': path.relative_to(output.resolve()).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    require(files and len({item['path'] for item in files}) == len(files), 'Empty/duplicate downloaded inventory')
    receipt = {'status': 'DOWNLOADED', 'ref': REF, 'version': version, 'observed_at': now(),
               'file_filter': OUTPUT_PATTERN, 'sdk_automatic_log_included': True, 'files': files}
    save(HERE / f'outputs-v{version}-manifest.json', receipt, exclusive=True)
    update(downloaded_version=version, outputs_path=str(output.relative_to(HERE)), outputs_manifest_sha256=sha(HERE / f'outputs-v{version}-manifest.json'))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('save-only', 'run', 'readback', 'status', 'download'))
    parser.add_argument('--version', type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    with command_lock():
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            if args.command in ('save-only', 'run'):
                require(args.version is None, 'Version is fixed automatically for write commands')
                result = push_once(api, args.command == 'run')
            else:
                version = pick_version(args.version)
                result = {'readback': readback, 'status': observe_status, 'download': download}[args.command](api, version)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except BaseException as error:
            message = re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[redacted]', str(error))
            failure = {'status': 'REQUIRES_REVIEW', 'command': args.command, 'observed_at': now(),
                       'error_type': type(error).__name__, 'message': message,
                       'automatic_retry': False, 'state': state()}
            save(unique_observation(args.command + '-failure').with_suffix('.json'), failure, exclusive=True)
            update(last_control_failure={key: failure[key] for key in ('command', 'observed_at', 'error_type', 'message')})
            print(json.dumps({key: failure[key] for key in ('status', 'command', 'message', 'automatic_retry')}, ensure_ascii=False), flush=True)
            raise SystemExit(1)


if __name__ == '__main__':
    main()
