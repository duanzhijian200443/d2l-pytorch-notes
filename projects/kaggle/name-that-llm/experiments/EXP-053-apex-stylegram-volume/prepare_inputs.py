"""Fetch fixed attachment inventories or rebuild EXP053 local inputs. Never trains or submits."""
from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import json
import os
import re
import types
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = HERE / 'inputs'
STAGING = INPUTS / 'source-snapshot'
SOURCE = next((ROOT / 'research/solutions').glob('SRC-068-*'))
SRC_SHA = '07cb21ed6f9b2b2d0920756a8fea64ec5a5d37637d7ea4fad07ab6eb85ce5140'
GLOBS = ('ntl_extra_passages_claude.csv', 'gen_passages.csv', 'ntl_generations.csv', 'ntl_proxy_generations.csv', 'ntl_generated*.csv', 'passages_clean.csv')
LOCAL = {
    'seyitkaangunes/ntl-extra-api-passages': 'data/public-pools/seyitkaangunes/ntl-extra-api-passages/v2/ntl_extra_passages_claude.csv',
    'lucasboesen/ntl-free-proxy-generations': 'data/community-factcheck-20261102/lucas-v12/gen_passages.csv',
    'dophuchung/ntl-proxy-generations': 'data/public-pools/dophuchung/ntl-proxy-generations/v1/ntl_proxy_generations.csv',
    'prvsiyan/prvsiyans-extended-name-that-llm-dataset': 'data/open-pool-v13-audit/baseline-prvs-v7/passages_clean.csv',
}
KNOWN_HASHES = {
    'data/public-pools/seyitkaangunes/ntl-extra-api-passages/v2/ntl_extra_passages_claude.csv': '710dbe484fda23f0dee9b67c20e5bef9376bf754af05584ad142bc17845cd4b9',
    'data/community-factcheck-20261102/lucas-v12/gen_passages.csv': 'b855fc128375984111d51ed4ae6bf23ca0fbec667b58d5ad2370bdb310495271',
    'data/public-pools/dophuchung/ntl-proxy-generations/v1/ntl_proxy_generations.csv': '593708f7fd55636bd270824c0e129447b2a5808b1e224bb3dbf43bf2383fdf9c',
    'data/open-pool-v13-audit/baseline-prvs-v7/passages_clean.csv': '502653b8cc053bc1ef08920d0ddda892e8b656f14229e38e0755d8a2f079eaf9',
    'train.csv': '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707',
    'validation/prompt-group-v1/folds.csv': 'e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551',
    'experiments/EXP-029-full-stack-capture/panel/test.csv': '8eeec7f64e924f0fd79dca57d324d42d269f400bdfbfff76d60b4c168d9ae6f2',
    'experiments/EXP-029-full-stack-capture/labels-local-only.csv': 'f5425fbae56355e72537074cf5eef514c0319cd67489fc63c36fe07b2ff2302b',
    'experiments/EXP-052-stylegram-four-model-volume/data/four120.csv': 'a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d',
    'experiments/EXP-052-stylegram-four-model-volume/data/four240.csv': '694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01',
    'data/open-pool-v13-audit/v13/ntl_open_pool.parquet': '38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5',
    'data/public-pools/dhruvtibarewal/nlm-matched-clean/v3/nlm_matched_clean.parquet': '000f8d541f7c4c5942a18c4fcf557aa01756882931307414cf2530ac21980482',
}


def need(value, msg):
    if not value:
        raise ValueError(msg)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf8')).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'Existing fixed artifact differs: ' + str(path))
        return
    with path.open('xb') as target:
        target.write(data)


def save(path, value):
    publish(path, (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf8'))


def entry(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}


def name_matches(name):
    return any(fnmatch.fnmatchcase(Path(name).name, p) for p in GLOBS)


def list_attachment(item):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    typ, ref = item['type'], item['ref']
    version = item.get('current_version_at_observation') or item.get('requested_version')
    files = []
    token = None
    seen_tokens = set()
    while True:
        with api.build_kaggle_client() as k:
            if typ == 'dataset':
                f = k.datasets.dataset_api_client.list_dataset_files
                q = f.__globals__['ApiListDatasetFilesRequest']()
                q.owner_slug, q.dataset_slug = ref.split('/')
                q.dataset_version_number = version
            elif typ == 'kernel':
                f = k.kernels.kernels_api_client.list_kernel_files
                q = f.__globals__['ApiListKernelFilesRequest']()
                q.user_name, q.kernel_slug = ref.split('/')
                q.version_label = 'v' + str(version)
            else:
                f = k.models.model_api_client.list_model_instance_version_files
                q = f.__globals__['ApiListModelInstanceVersionFilesRequest']()
                q.owner_slug, q.model_slug, framework, q.instance_slug, number = ref.split('/')
                need(framework == 'Transformers' and int(number) == version, 'Unexpected model ref')
                q.framework = type(q.framework).MODEL_FRAMEWORK_TRANSFORMERS
                q.version_number = version
            q.page_size = 100
            if token:
                q.page_token = token
            d = json.loads(f(q).to_json())
        for a in d.get('datasetFiles', d.get('files', [])):
            files.append({'name': a['name'], 'size': int(a.get('totalBytes', a.get('size', 0))), 'created_at': a.get('creationDate')})
        token = d.get('nextPageToken')
        if not token:
            break
        need(token not in seen_tokens, 'Repeated pagination cursor')
        seen_tokens.add(token)
    return {'type': typ, 'ref': ref, 'version': version, 'listed_files': len(files),
            'file_listing_sha256': sha(json.dumps(files, sort_keys=True)),
            'loader_files': [f for f in files if name_matches(f['name'])],
            'binding': 'historical card V13' if ref == 'crtproduction/name-that-llm-open-pool' else
                       'version explicitly in model ref' if typ == 'model' else 'public snapshot selected; unbound to historical hidden run'}


def fetch():
    """Use official SDK to list only names/sizes and download matching passage CSVs."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    inventory_file = STAGING / 'attachment-inventory.json'
    if inventory_file.exists():
        inventory = read(inventory_file)
    else:
        public_check = read(SOURCE / 'public-input-check.json')
        with ThreadPoolExecutor(max_workers=4) as executor:
            inventory = list(executor.map(list_attachment, public_check['items']))
        need(len(inventory) == 31, 'Incomplete inventory')
        save(inventory_file, inventory)
    if (STAGING / 'loader-file-receipt.json').exists():
        for row in read(STAGING / 'loader-file-receipt.json'):
            need(entry(ROOT / row['path'])['sha256'] == row['sha256'], 'Previously fetched file changed')
        return
    entries = []
    for item in inventory:
        for f in item['loader_files']:
            name = f['name']
            need(item['type'] == 'dataset' and '/' not in name, 'Unrecognized extra source path needs manual review')
            if item['ref'] in LOCAL and Path(LOCAL[item['ref']]).name == name:
                path = ROOT / LOCAL[item['ref']]
                need(sha(path.read_bytes()) == KNOWN_HASHES[LOCAL[item['ref']]], 'Frozen reusable source changed')
            else:
                slug = item['ref'].replace('/', '--') + '-v' + str(item['version'])
                path = STAGING / slug / name
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    api.dataset_download_file(item['ref'] + '/' + str(item['version']), name, path=str(path.parent), quiet=True, unzip=True)
            need(path.stat().st_size == f['size'], 'File size differs from fixed listing')
            entries.append(dict(entry(path), ref=item['ref'], version=item['version'], name=name, binding=item['binding']))
    save(STAGING / 'loader-file-receipt.json', entries)
    print('Attachment inventory complete:', len(inventory), 'refs,', len(entries), 'matching files')


def norm(t):
    return ' '.join(unicodedata.normalize('NFKC', str(t)).casefold().split())


def main_key(t):
    return ' '.join(unicodedata.normalize('NFKC', str(t)).split())


def comp_text(t):
    return re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', str(t).strip())).strip()


def build():
    import pandas as pd
    need(sha((SOURCE / 'source.ipynb').read_bytes()) == SRC_SHA, 'Fixed source hash differs')
    for path, expected in KNOWN_HASHES.items():
        need(sha((ROOT / path).read_bytes()) == expected, 'Frozen input changed: ' + path)
    entries = read(STAGING / 'loader-file-receipt.json')
    for row in entries:
        need(sha((ROOT / row['path']).read_bytes()) == row['sha256'], 'Raw source hash differs')
    notebook = read(SOURCE / 'source.ipynb')
    code = [''.join(cell['source']).split('\n', 1)[1] for cell in notebook['cells'] if ''.join(cell.get('source', [])).startswith('%%writefile comp_stack.py')]
    need(len(code) == 1, 'Cannot identify author comp_stack.py')
    tree = ast.parse(code[0])
    fun = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_extra_passages']
    need(len(fun) == 1 and 'EXTRA_MIN_LABELS, EXTRA_SHARE = 12, 0.5' in code[0], 'Cannot bind fixed loader')
    labels = sorted(pd.read_csv(ROOT / 'train.csv').model.unique())
    def fake_glob(pattern, recursive=True):
        suffix = pattern.rsplit('/', 1)[-1]
        return [str(ROOT / r['path']) for r in entries if fnmatch.fnmatchcase(r['name'], suffix)]
    ns = {'pd': pd, 're': re, 'glob': types.SimpleNamespace(glob=fake_glob), 'LABELS': labels, 'EXTRA_MIN_LABELS': 12}
    exec(compile(ast.Module(body=fun, type_ignores=[]), '<fixed-SRC068-data-loader>', 'exec'), ns)
    shared = ns['load_extra_passages']()
    need(list(shared.columns) == ['prompt_id', 'model', 'text'] and len(shared) and shared.groupby('prompt_id').model.nunique().min() >= 12, 'Invalid loader output')
    publish(INPUTS / 'shared.csv', shared.to_csv(index=False, lineterminator='\n').encode('utf8'))
    panel_path = ROOT / 'experiments/EXP-029-full-stack-capture/panel/test.csv'
    eval_path = ROOT / 'experiments/EXP-029-full-stack-capture/labels-local-only.csv'
    publish(INPUTS / 'panel.csv', panel_path.read_bytes())
    publish(INPUTS / 'local-only/evaluation.csv', eval_path.read_bytes())
    panel = pd.read_csv(panel_path)
    ev = pd.read_csv(eval_path)
    need(list(panel.columns) == ['id', 'text'] and len(panel) == len(ev) == 729 and panel.id.is_unique and ev.id.is_unique, 'Invalid panel')
    ev = ev.set_index('id').loc[panel.id].reset_index()
    need(panel.text.map(sha).tolist() == ev.text_sha256.tolist(), 'Blinded panel-text order mismatch')
    original = pd.read_csv(ROOT / 'train.csv')
    folds = pd.read_csv(ROOT / 'validation/prompt-group-v1/folds.csv')
    matched = pd.read_parquet(ROOT / 'data/public-pools/dhruvtibarewal/nlm-matched-clean/v3/nlm_matched_clean.parquet')
    v13 = pd.read_parquet(ROOT / 'data/open-pool-v13-audit/v13/ntl_open_pool.parquet')
    safe = v13.loc[v13.model.isin(labels) & ~v13.flag_train_prompt_regen & ~v13.flag_gemini37_routing_risk]
    seen = set(original.text.map(main_key)); contributions = {'official': len(seen)}
    for name, frame in [('matched', matched), ('open_pool_v13', safe)]:
        before = len(seen); seen.update(x for x in frame.loc[frame.model.isin(labels), 'text'].map(main_key) if x); contributions[name] = len(seen) - before
    need(len(seen) == 200478 and contributions['open_pool_v13'] == 173336, 'Main projection drift')
    overlaps = {}
    panel_norm = panel.text.map(norm)
    flags = ev.copy()
    refs = {'official': original, 'shared_extra': shared, 'matched_raw': matched, 'open_pool_v13': safe,
            'four120': pd.read_csv(ROOT / 'experiments/EXP-052-stylegram-four-model-volume/data/four120.csv'),
            'four240': pd.read_csv(ROOT / 'experiments/EXP-052-stylegram-four-model-volume/data/four240.csv')}
    for name, frame in refs.items():
        hit = panel_norm.isin(set(frame.text.map(norm))); flags[name + '_normalized_hit'] = hit
        overlaps[name] = {'raw_text_rows': int(panel.text.isin(set(frame.text)).sum()), 'normalized_text_rows': int(hit.sum()),
                          'affected_groups': int(flags.loc[hit, 'prompt_id'].nunique())}
    flags['apex_main_svc_fit_exact_key_hit'] = panel.text.map(main_key).isin(seen)
    official_fold = folds.groupby('prompt_id').fold.first().to_dict()
    rawmap = {}
    for pattern in GLOBS:
        for r in entries:
            if not fnmatch.fnmatchcase(r['name'], pattern):
                continue
            raw = pd.read_csv(ROOT / r['path'], keep_default_na=False)
            label_col = 'label' if r['name'] in ['ntl_generations.csv', 'ntl_proxy_generations.csv'] else 'model'
            finish = 'finish_reason' if r['name'] == 'ntl_generations.csv' else 'finish' if pattern == 'ntl_generated*.csv' else None
            if finish and finish in raw:
                raw = raw[raw[finish].eq('stop')]
            for rr in raw.to_dict('records'):
                pid = ('pv_' if r['name'] == 'passages_clean.csv' else '') + str(rr['prompt_id'])
                key = (pid, rr[label_col], comp_text(rr['text']))
                link = str(rr.get('source_prompt_id', '')) if rr.get('split') == 'train_regen' else ''
                rawmap.setdefault(key, (r['ref'], link))
    audit = []
    for i, row in shared.iterrows():
        ref, link = rawmap[(row.prompt_id, row.model, row.text)]
        audit.append({'shared_row': int(i), 'prompt_id': row.prompt_id, 'model': row.model, 'source_ref': ref,
                      'normalized_text_sha256': sha(norm(row.text)), 'known_regen_official_prompt': link,
                      'official_fold_to_isolate': int(official_fold[link]) if link in official_fold else None})
    audit = pd.DataFrame(audit)
    publish(INPUTS / 'local-only/shared-source-audit.csv', audit.to_csv(index=False, lineterminator='\n').encode('utf8'))
    near = {'cp0049': 'v22_v22_28_', 'cp0029': 'v22_v22_11_'}
    semantic = {}
    for arm in ['four120', 'four240']:
        present = set(refs[arm].prompt_id)
        near_flags = flags.source_id.map(lambda x: any('codex:' + q in present and x.startswith(v) for q, v in near.items()))
        flags[arm + '_known_near_prompt'] = near_flags
        semantic[arm] = {'training_groups': [q for q in near if 'codex:' + q in present],
                         'panel_rows': int(near_flags.sum()), 'panel_groups': int(flags.loc[near_flags, 'prompt_id'].nunique())}
    flags['entire_group_has_main_train_overlap'] = flags.groupby('prompt_id').apex_main_svc_fit_exact_key_hit.transform('any')
    publish(INPUTS / 'local-only/panel-overlap.csv', flags.to_csv(index=False, lineterminator='\n').encode('utf8'))
    receipt = {'status': 'PASS_CURRENT_ATTACHMENT_RECONSTRUCTION_DIAGNOSTIC_PANEL', 'source': 'SRC-068 V1 / 364709540',
        'source_sha256': SRC_SHA, 'author_loader_sha256': sha(ast.get_source_segment(code[0], fun[0])),
        'attachment_inventory': entry(STAGING / 'attachment-inventory.json'), 'loader_files': entries,
        'attachment_version_policy': 'Use the individually fixed public source versions observed in SRC068 public-input-check. Historical fixed V1 API supplies dataset/kernel refs without version numbers; Open Pool has V13 card/log evidence. Remaining current data versions are a reproducible comparison snapshot, not a byte-identical reconstruction of the hidden author run.',
        'shared': dict(entry(INPUTS / 'shared.csv'), rows=len(shared), groups=int(shared.prompt_id.nunique()),
                       classes=shared.model.value_counts().to_dict(), contributors=audit.source_ref.value_counts().to_dict(),
                       known_regen_rows=int(audit.known_regen_official_prompt.ne('').sum()),
                       known_regen_groups=int(audit.loc[audit.known_regen_official_prompt.ne(''), 'known_regen_official_prompt'].nunique())),
        'exp031_comparison': {'rows': 10469, 'groups': 716, 'delta_rows': len(shared)-10469, 'delta_groups': int(shared.prompt_id.nunique())-716,
                             'reason': 'Apex current public Lucas V12, Abdullah V9 compared with EXP031 verified Lucas V7, Abdullah V2 mirrors; actual prvs bytes match V7 for both.'},
        'official_inputs': [entry(ROOT / 'train.csv'), entry(ROOT / 'validation/prompt-group-v1/folds.csv')],
        'panel': dict(entry(INPUTS / 'panel.csv'), rows=len(panel), groups=int(ev.prompt_id.nunique()), truth_labels=int(ev.model.nunique()),
                      byte_identical_to_exp029_input=True, columns=['id', 'text']),
        'local_only_evaluation': [entry(INPUTS / 'local-only/evaluation.csv'), entry(INPUTS / 'local-only/panel-overlap.csv'), entry(INPUTS / 'local-only/shared-source-audit.csv')],
        'panel_overlaps': overlaps, 'semantic_overlap_prior_review_scope': semantic,
        'apex_main_svc': {'source_weights': {'official': 1, 'matched': .5, 'pool': .35}, 'log_score_coefficient': 12,
                          'pre_dedup_safe_pool': len(safe), 'fit_text_projection': contributions, 'total_fit_projection': len(seen),
                          'panel_exact_fit_key_hits': int(flags.apex_main_svc_fit_exact_key_hit.sum()),
                          'groups_with_any_hit': int(flags.loc[flags.apex_main_svc_fit_exact_key_hit, 'prompt_id'].nunique()),
                          'rows_in_groups_with_any_hit': int(flags.entire_group_has_main_train_overlap.sum()),
                          'hidden_actual_fit_execution_verified': False},
        'diagnostic_limits': ['729 is an inspected, 17-label panel with direct Apex main-SVC training overlap; no generalization evaluation.',
            'Known near-task review is inherited from EXP042, not an exhaustive new semantic search.',
            'Shared EXTRA known train_regen rows require explicit fold isolation for development OOF; other semantic links may remain.',
            'Unknown historical attachment bytes, upstream model training membership and hidden ATLAS routing remain unverified.'],
        'packaging': {'allowlisted_local_files': ['shared.csv', 'panel.csv'], 'exclude_directory': 'local-only', 'exclude_source_staging': 'source-snapshot'},
        'no_training_or_submission': True}
    save(INPUTS / 'input-receipt.json', receipt)
    print(json.dumps({'shared': receipt['shared'], 'panel_overlap': overlaps, 'main': receipt['apex_main_svc']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['fetch', 'build'])
    args = parser.parse_args()
    if args.action == 'fetch':
        try:
            fetch()
        except Exception as exc:
            # Avoid upstream exception strings containing temporary signed URLs.
            raise SystemExit('Fixed fetch stopped: ' + type(exc).__name__) from None
    else:
        build()
