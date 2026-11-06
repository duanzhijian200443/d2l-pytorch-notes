"""Build a new EXP-029 runtime without editing or executing frozen sources."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / 'reviews/own-scheme-readiness-20261030/baseline-v2.ipynb'
BASELINE_SHA256 = '6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8'
PACKAGE = ROOT / 'cloud/full-stack-cache'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def data_dir_assignment(tree):
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'DATA_DIR' for target in node.targets)]
    require(len(nodes) == 1, 'Expected one non-SMOKE DATA_DIR assignment')
    return nodes[0]


def remap_comp_input(source):
    """The only change after EXP-028 instrumentation is DATA_DIR's value."""
    before = ast.parse(source)
    after = copy.deepcopy(before)
    original_value = copy.deepcopy(data_dir_assignment(before).value)
    require(isinstance(original_value, ast.Call) and isinstance(original_value.func, ast.Name)
            and original_value.func.id == 'next', 'Unexpected original DATA_DIR binding')
    data_dir_assignment(after).value = ast.parse(
        "pathlib.Path(os.environ['NTL_CAPTURE_DATA']).resolve(strict=True)", mode='eval').body
    recovered = copy.deepcopy(after)
    data_dir_assignment(recovered).value = original_value
    require(ast.dump(recovered, include_attributes=False) == ast.dump(before, include_attributes=False),
            'Non-path comp algorithm changed')
    return '# EXP-029: only the non-SMOKE input directory is rebound.\n' + ast.unparse(after) + '\n'


def runtime_sources(notebook):
    cell2 = notebook['cells'][2]['source']
    cell2 = ''.join(cell2) if isinstance(cell2, list) else cell2
    cell12 = notebook['cells'][12]['source']
    cell12 = ''.join(cell12) if isinstance(cell12, list) else cell12
    tree = ast.parse(cell12)
    command_index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'cmd' for target in node.targets))
    prefix = ast.Module(body=tree.body[:command_index + 1], type_ignores=[])
    model_prefix = ast.unparse(prefix)
    setup = '''"""Run the frozen cell2 discovery; fail closed on its offline installer."""
import ast, json, os
from pathlib import Path
SOURCE = SOURCE_LITERAL
tree = ast.parse(SOURCE)
installs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
    and node.func.value.id == 'subprocess' and node.func.attr == 'run']
assert len(installs) == 1
assert not any(keyword.arg == 'check' for keyword in installs[0].keywords)
installs[0].keywords.append(ast.keyword(arg='check', value=ast.Constant(value=True)))
ast.fix_missing_locations(tree)
values = {'__name__': '__main__'}
exec(compile(tree, 'frozen-baseline-cell2.py', 'exec'), values)
keys = ['HEADS','EXTRA','POOL','B8','B14','B25','I8','I14','A8','A14','A25',
        'N8','N8C','N14','N14C','ATLAS','BL','AL','A8S7','B25_14',
        'N14B','N14BC','Q25_14','Q25_14C','DEB','AQ8V5','AQ14V5','ALL8V5']
output = {key: values[key] for key in keys}
output.update(DATA=os.environ['NTL_CAPTURE_DATA'], FAST_COMMIT=False,
              original_FAST_COMMIT=bool(values['FAST_COMMIT']),
              original_DATA=values['DATA'], offline_installer_check=True)
destination = Path(os.environ['NTL_SETUP_RESULT'])
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\\n', encoding='utf-8')
temporary.replace(destination)
print('CAPTURE_SETUP_COMPLETE', flush=True)
'''.replace('SOURCE_LITERAL', repr(cell2))
    model_args = '''"""Evaluate only the original cell12 model/command construction, never run it."""
import json, os
from pathlib import Path
values = json.loads(Path(os.environ['NTL_SETUP_RESULT']).read_text(encoding='utf-8'))
values['COMP_SEC'] = float(os.environ['NTL_COMP_SECONDS'])
values['json'] = json
exec(compile(SOURCE_LITERAL, 'frozen-baseline-cell12-prefix.py', 'exec'), values)
output = {'models': values['MODELS'], 'command': values['cmd'],
          'original_atlas_budget': values['ATLAS_BUDGET'],
          'original_atlas_cli_budget': float(values['cmd'][values['cmd'].index('--atlas_budget')+1]),
          'comp_seconds': values['COMP_SEC']}
destination = Path(os.environ['NTL_MODEL_ARGS_RESULT'])
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\\n', encoding='utf-8')
temporary.replace(destination)
print('CAPTURE_MODEL_ARGS_READY', flush=True)
'''.replace('SOURCE_LITERAL', repr(model_prefix))
    return setup, model_args, {'cell2_sha256': sha(cell2.encode()), 'cell12_sha256': sha(cell12.encode()),
                                'model_command_ast_sha256': sha(ast.dump(prefix).encode())}


def build(output):
    require(not output.exists(), 'Refusing an existing runtime destination')
    baseline = BASELINE.read_bytes()
    require(sha(baseline) == BASELINE_SHA256, 'Frozen baseline changed')
    package_manifest = json.loads((PACKAGE / 'package-manifest.json').read_text(encoding='utf-8'))
    require(package_manifest['status'] == 'LOCAL_ENGINEERING_PACKAGE', 'Unexpected EXP-028 package status')
    for entry in package_manifest['files']:
        content = (ROOT / entry['path']).read_bytes()
        require(sha(content) == entry['sha256'] and len(content) == entry['bytes'], 'Frozen package input changed: ' + entry['path'])
    generated = PACKAGE / 'generated'
    sources = {name: (generated / name).read_bytes() for name in (
        'infer.py', 'gauss.py', 'full_stack_cache.py', 'full_stack_observer.py', 'full_stack_replay.py')}
    original_comp = (generated / 'comp_stack.py').read_bytes()
    sources['comp_stack.py'] = remap_comp_input(original_comp.decode('utf-8')).encode('utf-8')
    setup, args, source_receipt = runtime_sources(json.loads(baseline))
    sources.update({'setup.py': setup.encode(), 'model_args.py': args.encode()})
    for name, content in sources.items():
        compile(content, name, 'exec')
    output.mkdir(parents=True, exist_ok=False)
    for name, content in sources.items():
        with (output / name).open('xb') as stream:
            stream.write(content)
    receipt = {'schema_version': 1, 'status': 'BUILT_NOT_EXECUTED', 'baseline_sha256': BASELINE_SHA256,
               'package_manifest_sha256': sha((PACKAGE / 'package-manifest.json').read_bytes()),
               'build_runtime_sha256': sha(Path(__file__).read_bytes()), **source_receipt,
               'original_instrumented_comp_sha256': sha(original_comp),
               'comp_change': "Only DATA_DIR=pathlib.Path(os.environ['NTL_CAPTURE_DATA']).resolve(strict=True)",
               'setup_changes': ['offline pip subprocess check=True', 'export DATA from NTL_CAPTURE_DATA',
                                 'export FAST_COMMIT=False; do not invoke the original placeholder branch'],
               'model_command_change': 'None in builder; runner records global-budget and private output-path overrides',
               'files': {name: sha(content) for name, content in sources.items()},
               'models_executed': False}
    (output / 'runtime-build.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.output_dir)
    print(json.dumps({'status': receipt['status'], 'files': len(receipt['files']), 'models_executed': False}))


if __name__ == '__main__':
    main()
