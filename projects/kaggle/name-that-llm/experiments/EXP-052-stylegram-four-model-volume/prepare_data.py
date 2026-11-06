"""Freeze EXP052's complete four-answer panels using metadata alone. No training or network."""
import csv
import hashlib
import io
import itertools
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = {'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-6-astra'}
LT = {'gpt-5.6-luna', 'gpt-5.6-terra'}
FIELDS = ['row_id', 'origin', 'prompt_id', 'model', 'text', 'base_prompt', 'text_sha256', 'job_key', 'split', 'source_protocol']
SEED = 42
LENGTHS = (80, 120, 160, 200)
SOURCE_HASHES = {
    'experiments/EXP-040-codex300-concurrency2/prompts.jsonl': '7a8c35d4418711ce721be6e16aed10fe233044b72e1e94c93221b50e3253fd41',
    'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl': '2d6c691016a6ae6f789d5c6a625a207a3ec89e5093bf804c5052b11c90a6125b',
    'experiments/EXP-041-aligned-channel-repeat-audit/metrics/training-eligibility.jsonl': 'e84bb7ab68584911be74d49f4377a28df99fb6fe216f45a63b6eb0dc6336f645',
    'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl': 'cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722',
    'experiments/EXP-038-paired-cloud-precheck-recovery/collection/combined-records.jsonl': '2c560d93f9cce57e42607886cb54449f84e88858a9370d3f175bcf441aea86e7',
    'cloud/luna-terra120-042/payload/new120.csv': 'cbb0f2227f7748d9172f04d23c1653403103a192dbb7503d897153d24bb00acd',
    'train.csv': '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707',
    'cloud/luna-terra120-042/payload/extra.csv': '1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e',
    'validation/prompt-group-v1/folds.csv': 'e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551',
    'experiments/EXP-008-public-pool-audit/outputs/row_flags.parquet': 'dd0ef1c8b006f90fb5f6c493b0b5d4322563af61105d590eac10dcadaf26e6f8',
    'experiments/EXP-050-open-pool-v13-increment-audit/metrics/row-risk-index.parquet': '9eefca76e503e52b74995e1d33edd54b69fb21d86c04d2de1fdb7656389990c6',
    'experiments/EXP-050-open-pool-v13-increment-audit/metrics/main-runtime-index.parquet': '15e53cbe64ee2e2793afc0e92ae8bad49145cca5e09e72aaac92389e88d20762',
    'experiments/EXP-050-open-pool-v13-increment-audit/metrics/shared-extra10469-index.parquet': 'eea95b19972878f81a9f1a02bc39f49366406f0ae22197ee5d8b79d93a938e17',
}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf8')).hexdigest()


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def inputs():
    for name, expected in SOURCE_HASHES.items():
        require(digest((ROOT / name).read_bytes()) == expected, 'Frozen source drift: ' + name)


def by_suffix(suffix):
    hits = [ROOT / name for name in SOURCE_HASHES if name.endswith(suffix)]
    require(len(hits) == 1, 'Ambiguous source ' + suffix)
    return hits[0]


def jsonl(suffix):
    return [json.loads(line) for line in by_suffix(suffix).read_text(encoding='utf8').splitlines()]


def table(suffix):
    with by_suffix(suffix).open(encoding='utf8', newline='') as source:
        return list(csv.DictReader(source))


def encode(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf8')


def select_groups(metadata):
    """Find the minimum seeded hash cost under exact category and nearest length margins."""
    strata = defaultdict(list)
    for item in metadata:
        strata[item['category']].append(item)
    require(len(strata) == 10 and all(len(s) == 6 for s in strata.values()), 'Expected ten strata of six train groups')
    require(Counter(x['target_words'] for x in metadata) == dict(zip(LENGTHS, [17, 18, 16, 9])), 'Unexpected training length counts')
    # DP keeps the least-cost candidate for each reachable length-count vector.
    states = {(0, 0, 0, 0): (0, ())}
    for category in sorted(strata):
        next_states = {}
        for three in itertools.combinations(sorted(strata[category], key=lambda x: x['prompt_group']), 3):
            ids = tuple(x['prompt_group'] for x in three)
            counts = tuple(sum(x['target_words'] == target for x in three) for target in LENGTHS)
            cost = sum(int(digest(f'{SEED}:{group}'), 16) for group in ids)
            for old_counts, (old_cost, old_ids) in states.items():
                key = tuple(a + b for a, b in zip(old_counts, counts))
                if any(x > cap for x, cap in zip(key, (9, 9, 8, 5))):
                    continue
                option = old_cost + cost, old_ids + ids
                if key not in next_states or option < next_states[key]:
                    next_states[key] = option
        states = next_states
    acceptable = [(states[k], k) for k in [(8, 9, 8, 5), (9, 9, 8, 4)] if k in states]
    require(acceptable, 'No feasible nearest integer split for the frozen strata')
    (cost, groups), counts = min(acceptable)
    return set(groups), dict(zip(LENGTHS, counts)), str(cost)


def ancestry(old, combined, paired):
    bases = {}
    for row in old:
        group = row['prompt_group']
        require(bases.setdefault(group, row['prompt']) == row['prompt'], 'Conflicting historical ancestor')
    splits, base_splits = defaultdict(set), defaultdict(set)
    for row in [*old, *combined, *paired]:
        group = row['prompt_group']
        base = bases[group]
        require(row.get('base_prompt', base) == base, 'Protocol rewrite changed base ancestor')
        splits[group].add(row['split'])
        base_splits[digest(norm(base))].add(row['split'])
    require(all(len(s) == 1 for s in [*splits.values(), *base_splits.values()]), 'Cross-protocol ancestor crosses splits')
    held = sorted(h for h, split in base_splits.items() if split == {'holdout'})
    require(len(held) == 50, 'Old protocol holdout ancestor count mismatch')
    return held, bases


def materialize(source, eligibility, train_meta, held):
    index = {r['job_key']: r for r in source}
    require(len(index) == len(source) == len(eligibility) == 300, 'Source/eligibility grain mismatch')
    require(set(index) == {e['job_key'] for e in eligibility}, 'Incomplete eligibility join')
    rows = []
    for e in eligibility:
        r = index[e['job_key']]
        group, model = r['prompt_group'], r['model_requested']
        require((e['prompt_group'], e['model'], e['split']) == (group, model, r['split']), 'Eligibility identity mismatch')
        if r['split'] == 'holdout':
            require(e['assessment'] == 'RESERVED_HOLDOUT' and group not in train_meta, 'Holdout incorrectly classified')
            continue
        require(r['split'] == 'train' and group in train_meta and model in LABELS, 'Invalid train identity')
        expected = 'CONTROLLED_TRAINING_CANDIDATE' if model in LT else 'EXPLORATORY_TRAINING_CANDIDATE_NO_PAIRED_CHANNEL_EVIDENCE'
        require(e['assessment'] == expected and e['formal_pool_accepted'] is False, 'Unexpected training eligibility')
        m = train_meta[group]
        require(all(r[k] == m[k] for k in ['base_prompt', 'category', 'target_words', 'split', 'protocol_version']), 'Prompt metadata mismatch')
        require(r['model_observation_scope'] == 'requested_only' and digest(norm(r['base_prompt'])) not in held, 'Bad provenance or holdout')
        require(e['attempt_path_project'] == r['attempt_path_project'], 'Raw source path mismatch')
        raw = (ROOT / r['attempt_path_project'] / 'final.txt').read_bytes()
        require(raw == r['text'].encode('utf8') and digest(raw) == r['text_sha256'] == r['raw_final_sha256'], 'Raw answer bytes mismatch')
        rows.append({'row_id': 'codex:' + r['job_key'], 'origin': 'codex_subscription_aligned_v1',
                     'prompt_id': 'codex:' + group, 'model': model, 'text': r['text'], 'base_prompt': r['base_prompt'],
                     'text_sha256': r['text_sha256'], 'job_key': r['job_key'], 'split': 'train', 'source_protocol': r['protocol_version']})
    return sorted(rows, key=lambda r: (r['prompt_id'], r['model']))


def validate_panel(rows, n_groups):
    panels = defaultdict(list)
    for row in rows:
        panels[row['prompt_id']].append(row)
    require(len(panels) == n_groups and all(len(group) == 4 and {r['model'] for r in group} == LABELS for group in panels.values()), 'Incomplete four-answer panel')
    require(len({r['row_id'] for r in rows}) == len(rows), 'Duplicate source ID')
    require(len({digest(norm(r['base_prompt'])) for r in rows}) == n_groups, 'Duplicate or inconsistent base ancestor')
    require(len({digest(norm(r['text'])) for r in rows}) == len(rows), 'Duplicate normalized answer')


def overlap_audit(rows, old, paired):
    exact = {r['text_sha256'] for r in rows}
    normalized = {digest(norm(r['text'])) for r in rows}
    checks = {}
    def check(name, raw_hashes, norm_hashes):
        raw_hits = None if raw_hashes is None else len(exact.intersection(raw_hashes))
        norm_hits = len(normalized.intersection(norm_hashes))
        require(raw_hits in (None, 0) and norm_hits == 0, 'Overlap with ' + name)
        checks[name] = {'raw_overlap': raw_hits, 'normalized_overlap': norm_hits}
    for name, source in [('official2699', table('train.csv')), ('public10073', table('extra.csv')),
                         ('old_codex1000', old), ('paired_diagnostic78', paired)]:
        check(name, {digest(r['text']) for r in source}, {digest(norm(r['text'])) for r in source})
    for name, suffix in [('open_pool_v13', 'row-risk-index.parquet'), ('main_runtime', 'main-runtime-index.parquet'),
                         ('shared_extra10469', 'shared-extra10469-index.parquet')]:
        index = pq.read_table(by_suffix(suffix), columns=['text_sha256', 'normalized_text_sha256'])
        check(name, set(index['text_sha256'].to_pylist()), set(index['normalized_text_sha256'].to_pylist()))
    historic = pq.read_table(by_suffix('row_flags.parquet'), columns=['text_hash'])
    check('EXP008_six_source_index', None, set(historic['text_hash'].to_pylist()))
    return checks


def publish(path, data):
    """No overwrite: retry recovers absent files and verifies already published bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Existing frozen artifact differs: ' + str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged.write(data)
        tmp = Path(staged.name)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(path.read_bytes() == data, 'Concurrent freeze mismatch')
    finally:
        tmp.unlink()


def main():
    inputs()
    metadata = jsonl('prompts.jsonl')
    train_meta = {m['prompt_group']: m for m in metadata if m['split'] == 'train'}
    require(len(metadata) == 75 and len(train_meta) == 60, 'Original split counts changed')
    # Freeze group IDs from prompt metadata only, before reading any answer records.
    groups, by_length, rank = select_groups(list(train_meta.values()))
    old = jsonl('pilot-r1/collection/records.jsonl')
    source = jsonl('concurrency2/collection/combined-records.jsonl')
    paired = jsonl('recovery/collection/combined-records.jsonl')
    eligibility = jsonl('training-eligibility.jsonl')
    held, bases = ancestry(old, source, paired)
    require(all(m['base_prompt'] == bases[g] for g, m in train_meta.items()), 'Prompt bank ancestor mismatch')
    c = materialize(source, eligibility, train_meta, held)
    b = [r for r in c if r['prompt_id'].removeprefix('codex:') in groups]
    validate_panel(b, 30)
    validate_panel(c, 60)
    require(set(tuple(r.values()) for r in b) < set(tuple(r.values()) for r in c), 'B is not strictly nested in C')
    require(encode([r for r in c if r['model'] in LT]) == by_suffix('new120.csv').read_bytes(), 'EXP042 LT120 bytes differ')
    checks = overlap_audit(c, old, paired)
    files = {'four120.csv': (b, 30), 'four240.csv': (c, 60)}
    outputs = {name: {'sha256': digest(encode(rows)), 'rows': len(rows), 'groups': n,
                      'per_model': dict(sorted(Counter(r['model'] for r in rows).items()))} for name, (rows, n) in files.items()}
    gate = {'schema_version': 1, 'experiment_id': 'EXP-052', 'status': 'PASS_FROZEN_BEFORE_EXP052_OOF',
            'selection': {'seed': SEED, 'strata': 'three complete groups per original category',
                          'length_target': 'nearest integer to half each full-bank target length count; (8,9,8,5) or (9,9,8,4)',
                          'tie_break': 'minimize sum of unsigned SHA256("42:" + prompt_group) across feasible sets; lexical IDs for exact tie',
                          'selected_hash_sum_decimal': rank, 'input': 'prompt_group, category, target_words, split only; no OOF input',
                          'four120_groups': sorted(groups), 'four120_target_words': by_length,
                          'four120_category_counts': dict(sorted(Counter(train_meta[g]['category'] for g in groups).items())),
                          'four240_groups': sorted(train_meta), 'four240_target_words': dict(sorted(Counter(m['target_words'] for m in train_meta.values()).items())),
                          'four240_category_counts': dict(sorted(Counter(m['category'] for m in train_meta.values()).items()))},
            'outputs': outputs, 'source_hashes': SOURCE_HASHES, 'raw_final_bytes_verified_rows': len(c),
            'subset_all_columns_equal': True, 'four240_lt_projection_byte_identical_to_exp042': True,
            'all_240_raw_and_normalized_response_hashes_unique': True, 'overlap_checks_for_four240_superset': checks,
            'cross_protocol_group_sources': {'old_EXP023': len(old), 'aligned_EXP040': len(source), 'paired_EXP038': len(paired)},
            'split_isolation': 'all source IDs and normalized base-prompt hashes occur in one split only',
            'holdout_prompt_group_hashes': held, 'old_protocol_holdout_groups': len(held),
            'aligned_holdout_rows_excluded': sum(e['split'] == 'holdout' for e in eligibility),
            'holdout_answer_rows_in_outputs': 0, 'paired_diagnostic_answer_rows_in_outputs': 0,
            'model_provenance': 'requested_only, backend snapshot unknown',
            'eligibility': 'Luna/Terra controlled candidates; Sol/Astra exploratory without paired channel evidence',
            'scope': 'Exact and normalized text plus known ancestor checks; semantic independence not certified.',
            'prepared_data_only': True, 'training_started_by_preparer': False, 'source_model_calls': 0,
            'prepare_code_sha256': digest(Path(__file__).read_bytes())}
    inputs()
    for name, (rows, _) in files.items():
        publish(HERE / 'data' / name, encode(rows))
    publish(HERE / 'metrics/data-gate.json', (json.dumps(gate, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
    print(json.dumps({'status': gate['status'], 'outputs': outputs, 'B_length_counts': by_length,
                      'data_gate_sha256': digest((HERE / 'metrics/data-gate.json').read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
