"""Map existing Drive assets into an isolated, Kaggle-shaped local input tree.

Only small configuration/CSV files are read. Model files stay on Drive; the
runtime receives file symlinks. No GPU, model loading, downloads, or pip installs.
"""
from pathlib import Path
import csv
import glob
import hashlib
import json
import os


def _relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe relative path: {value}')
    return path


def _files(source):
    source = Path(source)
    if source.is_file():
        yield Path('.'), source
    elif source.is_dir():
        for directory, subdirs, names in os.walk(source, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if d not in {'.ipynb_checkpoints', '__pycache__'})
            for name in sorted(names):
                file = Path(directory) / name
                if file.is_file():
                    yield file.relative_to(source), file


def plan_inputs(manifest, drive_kaggle):
    """Resolve renamed folders first; retain unmoved notebook sidecar files."""
    drive_kaggle = Path(drive_kaggle)
    plan, report, errors = {}, [], []
    for item in manifest['inputs']:
        target = _relative(item['target'])
        legacy = drive_kaggle / _relative(item['legacy'])
        selected, chosen = {}, []
        replacements = item.get('relocations', [])
        complete = next((r for r in replacements if r.get('replace_root')), None)
        if complete and (drive_kaggle / _relative(complete['source'])).exists():
            sources = [complete]
        else:
            sources = replacements + [{'source': item['legacy'], 'destination': ''}]
        for relocation in sources:
            source = drive_kaggle / _relative(relocation['source'])
            destination = target / _relative(relocation.get('destination', ''))
            found = 0
            for relative, file in _files(source):
                key = destination / relative
                # Relocations precede the legacy tree; never modify the Drive tree.
                if key not in selected:
                    selected[key] = file
                    found += 1
            if found:
                chosen.append({'source': str(source), 'files': found})
        content = [k for k in selected if not k.name.startswith('.')]
        if not content:
            errors.append(f"{item['ref']}: 没有可用文件；检查整理后的目录与旧缓存")
        for key, source in selected.items():
            if key in plan and plan[key] != source:
                errors.append(f'输入路径冲突: {key}')
            plan[key] = source
        report.append({'ref': item['ref'], 'cached_version': item['version'],
                       'target': str(target), 'files': len(selected), 'sources': chosen})
    if errors:
        raise FileNotFoundError('Drive 输入缺失：\n' + '\n'.join(errors))
    return plan, report


def create_aliases(plan, input_root):
    """Create only local directories/file links; refuse unrelated existing paths."""
    input_root = Path(input_root)
    owner = input_root / '.atlas20-drive-aliases.json'
    if input_root.is_symlink():
        raise RuntimeError(f'别名根目录不能是符号链接: {input_root}')
    if input_root.exists() and any(input_root.iterdir()) and not owner.is_file():
        raise RuntimeError(f'拒绝覆盖非本程序创建的目录: {input_root}')
    input_root.mkdir(parents=True, exist_ok=True)
    if owner.is_file():
        if json.loads(owner.read_text()) != {'owner': 'atlas20-drive-v1'}:
            raise RuntimeError('别名目录来源不匹配')
    else:
        owner.write_text(json.dumps({'owner': 'atlas20-drive-v1'}))
    for relative, source in sorted(plan.items(), key=lambda x: str(x[0])):
        target = input_root / _relative(relative)
        parent = input_root
        for part in target.relative_to(input_root).parts[:-1]:
            parent = parent / part
            if parent.is_symlink():
                raise RuntimeError(f'禁止通过目录链接写入源文件: {parent}')
            parent.mkdir(exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f'已有别名指向其他输入，使用新的 INPUT_ROOT: {target}')
        elif target.exists():
            raise RuntimeError(f'拒绝覆盖已有文件: {target}')
        else:
            target.symlink_to(source.resolve())


def validate_inputs(input_root, manifest):
    """Check files required by the unchanged ATLAS20 member configuration."""
    root = Path(input_root)
    missing = []
    for rule in manifest['required']:
        matches = [Path(p) for p in glob.glob(str(root / rule['pattern']), recursive=True)]
        matches = [p for p in matches if p.is_file() and p.stat().st_size > 0]
        if len(matches) < rule.get('minimum', 1):
            missing.append(f"{rule['name']}: {rule['pattern']}")
    configs, shard_count = {}, 0
    # Verify every indexed shard exists; never accept a folder containing only config.json.
    config_paths = set()
    for pattern in manifest['model_config_patterns']:
        config_paths.update(glob.glob(str(root / pattern), recursive=True))
    for cfg in sorted(config_paths):
        directory = Path(cfg).parent
        data = json.loads(Path(cfg).read_text())
        if 'model_type' not in data:
            continue
        configs[str(directory.relative_to(root))] = data.get('model_type')
        index = directory / 'model.safetensors.index.json'
        if index.is_file():
            shards = set(json.loads(index.read_text()).get('weight_map', {}).values())
            if not shards:
                missing.append(f'{directory}: 空的权重索引')
            for shard in shards:
                path = directory / _relative(shard)
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(f'权重分片缺失: {path}')
            shard_count += len(shards)
        elif not any((directory / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
            # Adapter configs use adapter_config.json and are checked separately.
            missing.append(f'权重文件缺失: {directory}')
    verified = []
    for check in manifest.get('small_file_hashes', []):
        path = root / _relative(check['path'])
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != check['sha256']:
                missing.append(f'历史数据哈希不匹配: {check["path"]}')
            else:
                verified.append(check['path'])
        else:
            missing.append(f'历史数据缺失: {check["path"]}')
    llama = root / 'datasets/zhjianduan/llama31-8b-base-verified'
    for name, size in manifest.get('llama_shard_bytes', {}).items():
        path = llama / name
        if not path.is_file() or path.stat().st_size != size:
            missing.append(f'Llama 分片大小不匹配: {name}')
    if missing:
        raise FileNotFoundError('输入检查未通过，尚未加载模型：\n' + '\n'.join(missing))
    test = root / 'competitions/name-that-llm/test.csv'
    with test.open(encoding='utf-8-sig', newline='') as stream:
        test_rows = sum(1 for _ in csv.DictReader(stream))
    return {'status': 'READY', 'model_configs': configs, 'indexed_shards': shard_count,
            'small_file_hashes_verified': verified, 'test_rows': test_rows,
            'all_model_content_hashes_verified': False,
            'scope': 'path, required-file, index and selected historical CSV checks; no inference'}


def prepare_drive_inputs(manifest, drive_kaggle, input_root, report_path):
    plan, inputs = plan_inputs(manifest, drive_kaggle)
    create_aliases(plan, input_root)
    result = validate_inputs(input_root, manifest)
    result['inputs'] = inputs
    result['input_groups'] = len(inputs)
    result['linked_files'] = len(plan)
    result['baseline_input_bytes_identical'] = None
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
