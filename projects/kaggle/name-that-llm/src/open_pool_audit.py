"""Fixed-file Open Pool audit. Reads data; never trains, generates or submits.

Exact means UTF-8 bytes. Normalized means NFKC + casefold + whitespace collapse;
neither establishes semantic novelty or checkpoint-training independence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def text_hash(text, normalized=True):
    text = normalize(text) if normalized else str(text)
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def main_key(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text or ''))).strip()


def comp_key(text):
    return re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', str(text).strip())).strip()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def verified_path(root, spec):
    path = root / spec['path']
    if not path.is_file() or path.stat().st_size != spec['bytes'] or sha_file(path) != spec['sha256']:
        raise ValueError('Input missing or changed: ' + spec['path'])
    return path


def read_table(path):
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, keep_default_na=False)


def add_hashes(df):
    df = df.copy()
    df['text_sha256'] = df.text.map(lambda x: text_hash(x, False))
    df['normalized_text_sha256'] = df.text.map(text_hash)
    return df


def conflict_mask(df, key='normalized_text_sha256'):
    return df.groupby(key).model.transform('nunique').gt(1)


def panel_checks(pool):
    """An empty canonical ID is unknown, never one shared prompt."""
    valid = pool.prompt_canon.fillna('').ne('')
    groups = pool.loc[valid].groupby('prompt_canon').agg(
        rows=('id', 'size'), labels=('model', 'nunique'), sources=('source_dataset', 'nunique'),
        panel_declared_min=('panel_labels', 'min'), panel_declared_max=('panel_labels', 'max'))
    groups['mismatch'] = groups.labels.ne(groups.panel_declared_min) | groups.labels.ne(groups.panel_declared_max)
    mismatch = pool.prompt_canon.map(groups.mismatch).fillna(False).astype(bool)
    return groups, mismatch


def reconstruct_comp(raw_tables, labels):
    """Mirror the fixed load_extra_passages data-only transform, preserving order."""
    parts = []
    for spec, raw in raw_tables:
        d = raw.copy()
        finish = spec.get('finish_column')
        if finish and finish in d:
            d = d.loc[d[finish].eq('stop')]
        d = d.rename(columns={spec['label_column']: 'model'})[['prompt_id', 'model', 'text']].copy()
        d['prompt_id'] = spec.get('prompt_prefix', '') + d.prompt_id.astype(str)
        d['input_name'] = spec['name']
        parts.append(d)
    result = pd.concat(parts, ignore_index=True)
    result = result.loc[result.model.isin(labels)].copy()
    result['text'] = result.text.astype(str).map(comp_key)
    result = result.loc[result.text.str.split().str.len().between(30, 300)]
    result = result.drop_duplicates(['model', 'text']).drop_duplicates(['prompt_id', 'model'])
    return result.loc[result.groupby('prompt_id').model.transform('nunique').ge(12)].reset_index(drop=True)


def reconstruct_main(official, matched, pool, labels):
    parts = [official[['text', 'model']].assign(input_name='official')]
    seen = set(official.text.map(main_key))
    for name, data in [('matched', matched), ('pool_v5', pool)]:
        data = data.loc[data.model.isin(labels), ['text', 'model']].copy()
        data['text'] = data.text.fillna('').astype(str)
        keys = data.text.map(main_key)
        keep = []
        for k in keys:
            yes = bool(k) and k not in seen
            keep.append(yes)
            if yes:
                seen.add(k)
        parts.append(data.loc[keep].assign(input_name=name))
    return pd.concat(parts, ignore_index=True)


def propagate_group_hits(pool, row_hits):
    canons = set(pool.loc[row_hits & pool.prompt_canon.ne(''), 'prompt_canon'])
    return pool.prompt_canon.ne('') & pool.prompt_canon.isin(canons)


def output_path(root, config_path, cfg, override=None):
    out = (root / (override or cfg['output_dir'])).resolve()
    experiment = Path(config_path).resolve().parent
    if not out.is_relative_to(experiment) or out == experiment:
        raise ValueError('Output must be a child directory of this experiment')
    return out


def verify(config_path, output_dir=None):
    """Read-only integrity verification; never creates, replaces or downloads files."""
    config_path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(config_path.read_text(encoding='utf8'))
    for spec in cfg['inputs'].values():
        verified_path(root, spec)
    out = output_path(root, config_path, cfg, output_dir)
    receipt = json.loads((out / 'audit-complete.json').read_text(encoding='utf8'))
    if receipt['config_sha256'] != sha_file(config_path) or receipt['code_sha256'] != sha_file(__file__):
        raise ValueError('Code/config differs from completed audit')
    for spec in receipt['outputs']:
        verified_path(root, spec)
    manifest_checked = False
    if output_dir is None:
        manifest = json.loads((config_path.parent / 'manifest.json').read_text(encoding='utf8'))
        for spec in manifest['files']:
            verified_path(root, spec)
        manifest_checked = True
    print(json.dumps({'status': 'PASS_READ_ONLY_INTEGRITY', 'inputs': len(cfg['inputs']),
                      'outputs': len(receipt['outputs']), 'manifest_checked': manifest_checked,
                      'training_approved': False}, ensure_ascii=False))


def run(config_path, output_dir=None):
    config_path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(config_path.read_text(encoding='utf8'))
    out = output_path(root, config_path, cfg, output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Audit output already exists. Use --verify or a new --output-dir; no overwrite allowed.')
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg['inputs']
    paths = {name: verified_path(root, s) for name, s in specs.items()}
    code_hash = sha_file(__file__)
    config_hash = sha_file(config_path)
    print('All input hashes verified', flush=True)

    pool = add_hashes(read_table(paths['open_pool']))
    original_columns = [c for c in pool if c not in ('text_sha256', 'normalized_text_sha256')]
    prompts = read_table(paths['prompts'])
    quarantine = add_hashes(read_table(paths['quarantine']))
    official = add_hashes(read_table(paths['official']))
    a1 = add_hashes(read_table(paths['a1']))
    matched = read_table(paths['matched'])
    old_pool = read_table(paths['pool_v5'])
    pool_v2 = read_table(paths['pool_v2'])
    labels = cfg['labels']
    if len(pool) != 190588 or len(original_columns) != 23 or set(pool.model) != set(labels):
        raise ValueError('Unexpected fixed V13 primary shape or labels')
    raw_tables = [(s, read_table(paths[s['name']])) for s in cfg['comp_sources']]
    comp = reconstruct_comp(raw_tables, labels)
    if len(comp) != 10469 or comp.prompt_id.nunique() != 716:
        raise ValueError('Actual shared EXTRA reconstruction differs from accepted runtime 10469/716')
    main = reconstruct_main(official, matched, old_pool, labels)

    # Check the reimplemented comp transform against only the frozen function AST;
    # never execute the training/inference module or imports.
    import fnmatch
    import types
    fixed_code = paths['baseline_comp_code'].read_text(encoding='utf8')
    tree = ast.parse(fixed_code)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_extra_passages']
    if len(fn) != 1:
        raise ValueError('Cannot isolate frozen data loader')
    def fake_glob(pattern, recursive=True):
        suffix = pattern.split('/**/', 1)[1]
        return sorted(str(paths[s['name']]) for s in cfg['comp_sources'] if fnmatch.fnmatch(paths[s['name']].name, suffix))
    ns = {'glob': types.SimpleNamespace(glob=fake_glob), 'pd': pd, 're': re, 'LABELS': labels, 'EXTRA_MIN_LABELS': 12}
    exec(compile(ast.Module(body=fn, type_ignores=[]), '<isolated fixed data loader>', 'exec'), ns)
    pd.testing.assert_frame_equal(comp[['prompt_id', 'model', 'text']], ns['load_extra_passages']())

    refs = {'official_train': official, 'a1_style10073': a1,
            'matched_raw': add_hashes(matched), 'pool_v5_raw': add_hashes(old_pool), 'pool_v2_raw': add_hashes(pool_v2),
            'main_runtime': add_hashes(main), 'shared_extra10469': add_hashes(comp)}
    for s, d in raw_tables:
        refs['source_' + s['name']] = add_hashes(d.rename(columns={s['label_column']: 'model'}))
    # A1 replaces Stylegram only; the full shared EXTRA still feeds GPT/Gemini/GLM gates.
    gates = comp.loc[comp.model.str.startswith(('gpt', 'gemini', 'glm'))]
    claude = raw_tables[0][1].rename(columns={raw_tables[0][0]['label_column']: 'model'})
    refs['a1_runtime_union'] = add_hashes(pd.concat([main, a1, gates, claude], ignore_index=True)[['text', 'model']])
    refs['available_raw_union'] = add_hashes(pd.concat(
        [official, a1, matched, old_pool, pool_v2] + [d.rename(columns={s['label_column']: 'model'}) for s, d in raw_tables],
        ignore_index=True)[['text', 'model']])
    flags = pool[['id', 'model', 'source_dataset', 'prompt_canon', 'prompt_evidence',
                  'panel_labels', 'flag_kb_proxy', 'flag_gemini37_routing_risk',
                  'text_sha256', 'normalized_text_sha256']].copy()
    overlap = []
    for name, ref in refs.items():
        exact = pool.text_sha256.isin(set(ref.text_sha256))
        norm = pool.normalized_text_sha256.isin(set(ref.normalized_text_sha256))
        pairs = set(zip(ref.normalized_text_sha256, ref.model))
        same_label = pd.Series([pair in pairs for pair in zip(pool.normalized_text_sha256, pool.model)], index=pool.index)
        conflict = norm & ~same_label
        # Also flag texts whose reference includes ANY other label, even if this label exists too.
        label_sets = ref.groupby('normalized_text_sha256').model.agg(set).to_dict()
        any_other = pd.Series([bool(label_sets.get(h, set()) - {m}) for h, m in zip(pool.normalized_text_sha256, pool.model)], index=pool.index)
        flags['in_' + name] = norm
        flags['label_conflict_' + name] = any_other
        overlap.append({'reference': name, 'reference_rows': len(ref),
                        'reference_unique_normalized_texts': int(ref.normalized_text_sha256.nunique()),
                        'pool_exact_rows': int(exact.sum()), 'pool_normalized_rows': int(norm.sum()),
                        'pool_normalized_same_label_rows': int(same_label.sum()),
                        'pool_normalized_no_matching_label_rows': int(conflict.sum()),
                        'pool_normalized_any_conflicting_label_rows': int(any_other.sum()),
                        'pool_not_in_reference_rows': int((~norm).sum())})
    pd.DataFrame(overlap).to_csv(out / 'overlap.csv', index=False)
    print('Reference joins complete', flush=True)

    historical = read_table(paths['historical_row_flags'])
    route_index = historical.loc[historical.route.fillna('').ne(''), ['text_hash', 'route', 'requested_model_id', 'served_model_id', 'identity_mismatch']].copy()
    route_index = route_index.drop_duplicates()
    route_joins = pool[['id', 'normalized_text_sha256', 'source_dataset', 'model']].merge(
        route_index, left_on='normalized_text_sha256', right_on='text_hash', how='inner')
    route_joins.to_parquet(out / 'historical-route-evidence-joins.parquet', index=False)
    flags['has_historical_route_claim'] = pool.id.isin(set(route_joins.id))
    flags['historical_identity_mismatch_claim'] = pool.id.isin(set(route_joins.loc[route_joins.identity_mismatch, 'id']))
    known_official_hashes = set(historical.loc[historical.linked_official_prompt, 'text_hash'])
    prvs = raw_tables[-1][1]
    if 'split' in prvs:
        known_official_hashes |= set(prvs.loc[prvs['split'].eq('train_regen'), 'text'].map(text_hash))
    flags['known_official_prompt_text_link'] = pool.normalized_text_sha256.isin(known_official_hashes)
    official_hit = flags.in_official_train | flags.known_official_prompt_text_link
    flags['known_official_prompt_group_link'] = propagate_group_hits(pool, official_hit)
    flags['known_a1_runtime_prompt_link'] = propagate_group_hits(pool, flags.in_a1_runtime_union)
    flags['novel_text_to_audited_runtime'] = ~flags.in_a1_runtime_union
    flags['novel_text_to_available_raw'] = ~flags.in_available_raw_union

    groups, mismatch = panel_checks(pool)
    flags['panel_label_mismatch'] = mismatch
    flags['prompt_unknown'] = pool.prompt_canon.eq('')
    flags['prompt_source_only'] = pool.prompt_canon.str.startswith('s')
    flags['prompt_centroid_inferred'] = pool.prompt_evidence.eq('centroid')
    flags['duplicate_exact_text'] = pool.text_sha256.duplicated(keep=False)
    flags['duplicate_normalized_text'] = pool.normalized_text_sha256.duplicated(keep=False)
    flags['within_pool_label_conflict'] = conflict_mask(pool)
    flags['duplicate_prompt_model'] = pool.prompt_canon.ne('') & pool.duplicated(['prompt_canon', 'model'], keep=False)
    flags['missing_required'] = pool[['id', 'text', 'model', 'source_dataset']].isna().any(axis=1) | pool[['id', 'text', 'model', 'source_dataset']].fillna('').eq('').any(axis=1)
    invalid_weight_classes = set(pool.groupby('model').match_weight.sum().loc[lambda x: x.le(0)].index)
    flags['invalid_match_weight'] = ~np.isfinite(pool.match_weight) | pool.match_weight.lt(0) | pool.model.isin(invalid_weight_classes)
    flags['upstream_full_train_selection_exposure'] = True
    flags['official_train_fitted_match_weight'] = True
    flags['independently_verified_served_model'] = False
    flags['approved_for_training'] = False

    groups.reset_index().to_csv(out / 'prompt-groups.csv', index=False)
    pr = prompts.copy()
    pr['normalized_prompt_sha256'] = pr.prompt_text.map(text_hash)
    alias_hashes = set(pr.loc[pr.duplicated('normalized_prompt_sha256', keep=False), 'normalized_prompt_sha256'])
    alias_canons = set(pr.loc[pr.normalized_prompt_sha256.isin(alias_hashes), 'prompt_canon'])
    flags['duplicate_prompt_text_across_canons'] = pool.prompt_canon.isin(alias_canons)
    prompt_join = pr.merge(groups[['rows', 'labels']], left_on='prompt_canon', right_index=True, how='left', validate='one_to_one')
    prompt_join['panel_mismatch'] = prompt_join.panel_labels.ne(prompt_join.labels)
    prompt_join['passages_mismatch'] = prompt_join.passages.ne(prompt_join.rows)
    prompt_join.drop(columns='prompt_text').to_csv(out / 'prompt-text-index.csv', index=False)

    # Existing A1 canonical ancestor bridges: text matches can expose splits across new canons.
    bridge = pool[['id', 'normalized_text_sha256', 'prompt_canon']].merge(
        a1[['normalized_text_sha256', 'canonical_prompt']].drop_duplicates(), on='normalized_text_sha256', how='inner')
    bridge.to_csv(out / 'a1-canonical-bridges.csv', index=False)
    ancestor_many = bridge.loc[bridge.prompt_canon.ne('')].groupby('canonical_prompt').prompt_canon.nunique()
    many_ancestors = set(ancestor_many[ancestor_many.gt(1)].index)
    split_canons = set(bridge.loc[bridge.canonical_prompt.isin(many_ancestors), 'prompt_canon']) - {''}
    flags['a1_ancestor_split_across_canons'] = pool.prompt_canon.isin(split_canons)

    blocked = (flags.missing_required | flags.within_pool_label_conflict | flags.label_conflict_available_raw_union |
               flags.known_official_prompt_text_link | flags.known_official_prompt_group_link |
               flags.panel_label_mismatch | flags.invalid_match_weight)
    flags['increment_screen_only'] = flags.novel_text_to_audited_runtime & ~blocked
    flags['higher_traceability_increment_screen_only'] = (flags.increment_screen_only &
        pool.prompt_canon.str.startswith('p') & pool.prompt_evidence.isin(['own_text', 'own_bank', 'other_text']) &
        ~flags.a1_ancestor_split_across_canons & ~flags.duplicate_prompt_text_across_canons)
    flags['new_recovered_prompt_increment_screen_only'] = flags.higher_traceability_increment_screen_only & ~flags.known_a1_runtime_prompt_link
    flags.to_parquet(out / 'row-risk-index.parquet', index=False)

    by_label = flags.groupby('model').agg(rows=('id', 'size'),
        already_in_runtime=('in_a1_runtime_union', 'sum'), novel_text_upper_bound=('novel_text_to_audited_runtime', 'sum'),
        not_in_available_raw_upper_bound=('novel_text_to_available_raw', 'sum'),
        proxy_author_flag=('flag_kb_proxy', 'sum'), unknown_prompt=('prompt_unknown', 'sum'),
        source_only_prompt=('prompt_source_only', 'sum'), centroid_prompt=('prompt_centroid_inferred', 'sum'),
        increment_screen_only=('increment_screen_only', 'sum'),
        higher_traceability_screen_only=('higher_traceability_increment_screen_only', 'sum'))
    by_label['new_recovered_prompt_screen_only'] = flags.groupby('model').new_recovered_prompt_increment_screen_only.sum()
    weight = pool.groupby('model').match_weight.agg(['min', 'max', 'mean', 'sum'])
    weight['effective_sample_size'] = pool.groupby('model').match_weight.apply(lambda x: float(x.sum() ** 2 / x.pow(2).sum()))
    weight['effective_share'] = weight.effective_sample_size / pool.groupby('model').size()
    by_label.join(weight).to_csv(out / 'per-label.csv')
    source_summary = flags.groupby('source_dataset').agg(rows=('id', 'size'),
        novel_text_upper_bound=('novel_text_to_audited_runtime', 'sum'),
        proxy_author_flag=('flag_kb_proxy', 'sum'),
        increment_screen_only=('increment_screen_only', 'sum'),
        higher_traceability_screen_only=('higher_traceability_increment_screen_only', 'sum'))
    source_summary.to_csv(out / 'per-source.csv')
    candidate_counts = {}
    for name, column in [('C1-recovered-prompt-increment', 'higher_traceability_increment_screen_only'),
                         ('C2-new-recovered-prompt-increment', 'new_recovered_prompt_increment_screen_only')]:
        selected = flags.loc[flags[column], ['id', 'model', 'source_dataset', 'prompt_canon', 'prompt_evidence',
                         'normalized_text_sha256', 'flag_kb_proxy', 'known_a1_runtime_prompt_link', 'approved_for_training']]
        selected.to_csv(out / (name + '-candidate-index.csv'), index=False)
        candidate_counts[name] = {'rows': len(selected), 'groups': int(selected.prompt_canon.nunique()),
                                 'per_label': selected.model.value_counts().to_dict(), 'approved_for_training': False}
    pool.groupby(['source_dataset', 'model', 'flag_kb_proxy', 'origin_hint'], dropna=False).size().rename('rows').reset_index().to_csv(out / 'source-route-claims.csv', index=False)

    # Attribution sets are claims, not independently verified routes or unique origins.
    attribution_sets = pool.also_in.map(lambda x: set(str(x).split(';')) - {''})
    attr_mismatch = pd.Series([len(s) != n for s, n in zip(attribution_sets, pool.n_sources)])
    primary_missing = pd.Series([s not in all_s for s, all_s in zip(pool.source_dataset, attribution_sets)])
    quarantine_overlap = pool.normalized_text_sha256.isin(set(quarantine.normalized_text_sha256))
    summary = {
        'experiment_id': cfg['experiment_id'], 'status': 'AUDIT_COMPLETE_NOT_TRAINING_APPROVED',
        'observed_at': datetime.now(timezone.utc).isoformat(), 'rows': len(pool), 'columns': len(original_columns),
        'labels': sorted(set(pool.model)), 'null_counts': {c: int(pool[c].isna().sum()) for c in original_columns},
        'blank_string_counts': {c: int(pool[c].eq('').sum()) for c in original_columns if pd.api.types.is_string_dtype(pool[c])},
        'id_duplicate_rows': int(pool.id.duplicated(keep=False).sum()),
        'risk_counts': {c: int(flags[c].sum()) for c in flags if pd.api.types.is_bool_dtype(flags[c])},
        'prompt_evidence_counts': pool.prompt_evidence.value_counts().to_dict(),
        'nonempty_prompt_groups': len(groups), 'full_18_label_groups': int(groups.labels.eq(18).sum()),
        'prompt_text_rows': len(prompts), 'prompt_text_duplicates_across_canons': len(alias_hashes),
        'prompt_table_missing_groups': int(prompt_join.rows.isna().sum()),
        'prompt_table_panel_mismatch': int(prompt_join.panel_mismatch.sum()),
        'prompt_table_passages_mismatch': int(prompt_join.passages_mismatch.sum()),
        'a1_canonical_ancestors_split_across_pool_canons': len(many_ancestors),
        'source_count': int(pool.source_dataset.nunique()), 'attribution_count_mismatch': int(attr_mismatch.sum()),
        'primary_source_missing_from_attributions': int(primary_missing.sum()),
        'word_count_mismatch': int(pool.text.str.split().str.len().ne(pool.n_words).sum()),
        'word_count_range': [int(pool.n_words.min()), int(pool.n_words.max())],
        'license_claims': pool.source_license.value_counts().to_dict(),
        'quarantine_rows': len(quarantine), 'quarantine_labels': quarantine.model.value_counts().to_dict(),
        'quarantine_main_normalized_overlap_rows': int(quarantine_overlap.sum()),
        'main_runtime_rows': len(main), 'main_runtime_counts': main.input_name.value_counts().to_dict(),
        'shared_extra_actual_rows': len(comp), 'shared_extra_actual_groups': int(comp.prompt_id.nunique()),
        'shared_extra_source_rows': comp.input_name.value_counts().to_dict(),
        'shared_loader_exact_ast_parity': True, 'overlap': overlap,
        'candidate_inventories': candidate_counts,
        'numeric_quality': {c: {'nonfinite': int((~np.isfinite(pool[c])).sum()),
                                'min': float(pool[c].min()), 'max': float(pool[c].max())}
                            for c in ['max_cos_train', 'max_sim_train_prompt', 'style_score', 'match_weight']},
        'runtime_file_coverage': cfg['runtime_file_coverage'],
        'scope_limits': cfg['scope_limits'], 'code_sha256': code_hash, 'config_sha256': config_hash,
        'training_executed': False, 'cloud_mutation': False,
    }
    json_write(out / 'summary.json', summary)
    refs['main_runtime'].drop(columns='text').to_parquet(out / 'main-runtime-index.parquet', index=False)
    refs['shared_extra10469'].drop(columns='text').to_parquet(out / 'shared-extra10469-index.parquet', index=False)
    for spec in specs.values():
        verified_path(root, spec)
    if sha_file(__file__) != code_hash or sha_file(config_path) != config_hash:
        raise ValueError('Code/config changed during audit')
    output_files = [{'path': f.relative_to(root).as_posix(), 'bytes': f.stat().st_size, 'sha256': sha_file(f)}
                    for f in sorted(out.iterdir()) if f.is_file() and f.name != 'audit-complete.json']
    json_write(out / 'audit-complete.json', {'status': 'PASS_AUDIT_INTEGRITY_ONLY', 'inputs_rechecked_after_run': True,
               'code_sha256': code_hash, 'config_sha256': config_hash, 'outputs': output_files,
               'not_training_approval': True})
    print(json.dumps({'rows': len(pool), 'main_rows': len(main), 'shared_extra': len(comp),
                      'runtime_overlap': int(flags.in_a1_runtime_union.sum()),
                      'novel_upper_bound': int(flags.novel_text_to_audited_runtime.sum()),
                      'risk_counts': summary['risk_counts']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--verify', action='store_true', help='Only verify existing inputs, manifest and outputs')
    parser.add_argument('--output-dir', help='New project-relative output directory for a repeat audit')
    args = parser.parse_args()
    if args.verify:
        verify(args.config, args.output_dir)
    else:
        run(args.config, args.output_dir)
