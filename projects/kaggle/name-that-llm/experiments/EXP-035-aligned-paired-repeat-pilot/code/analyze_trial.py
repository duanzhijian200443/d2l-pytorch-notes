"""Offline paired descriptive analysis; writing requires explicit --analyze.

Reads terminal collection artifacts, never generates, cleans, retries or trains.
Lexical distances measure surface overlap, not semantic quality or causality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import statistics


EXP = Path(__file__).resolve().parents[1]
ROUTES = ('codex_subscription', 'kaggle_model_proxy')
MODELS = ('gpt-5.6-luna', 'gpt-5.6-terra')
OLD_CODEX = 'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'
OLD_KAGGLE = 'pending-handoff-kaggle-proxy-pilot-20261101T084539Z/authorization-v2-results/responses.jsonl'
OLD_HASHES = {
    OLD_CODEX: 'cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722',
    OLD_KAGGLE: '227ed0103dee52a9a6bea6680a77e8e6409c7e289e99d404f311337db2cac269',
}
SEED = 20261101
BOOTSTRAPS = 10000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode_jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]


def mean(values):
    return statistics.mean(values) if values else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*q
    low = int(position)
    high = min(low+1, len(values)-1)
    return values[low]+(position-low)*(values[high]-values[low])


def lexical_units(text):
    tokens = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
    # Very short responses use their whole token tuple instead of collapsing
    # every response shorter than three tokens into the same empty set.
    return {tuple(tokens[i:i+3]) for i in range(len(tokens)-2)} if len(tokens) >= 3 else {tuple(tokens)}


def trigram_distance(a, b):
    left, right = lexical_units(a), lexical_units(b)
    return 1-len(left & right)/len(left | right)


def words(text):
    return len(re.findall(r'\S+', text))


def pair_measures(left, right):
    return {'lexical_distance': trigram_distance(left['text'], right['text']),
            'absolute_word_count_gap': abs(words(left['text'])-words(right['text'])),
            'exact_text_equal': left['text'] == right['text']}


def paired_cells(jobs, rows):
    expected = defaultdict(list)
    for job in jobs:
        expected[(job['prompt_id'], job['model_label'])].append(job)
    by_id = {row['job_id']: row for row in rows}
    complete, incomplete = [], []
    for (prompt, model), planned in sorted(expected.items()):
        saved = [by_id[j['job_id']] for j in planned if j['job_id'] in by_id]
        missing = [{'route': j['route'], 'repeat_id': j['repeat_id'], 'job_id': j['job_id']}
                   for j in planned if j['job_id'] not in by_id]
        if missing:
            incomplete.append({'prompt_id': prompt, 'model_label': model,
                               'saved': len(saved), 'missing': missing})
            continue
        lookup = {(r['route'], r['repeat_id']): r for r in saved}
        if len(saved) != 4 or set(lookup) != {(route, rep) for route in ROUTES for rep in (1, 2)}:
            raise ValueError('Invalid complete cell: '+prompt+'|'+model)
        within = {route: pair_measures(lookup[(route, 1)], lookup[(route, 2)]) for route in ROUTES}
        cross = [dict(left_repeat=i, right_repeat=j,
                      **pair_measures(lookup[(ROUTES[0], i)], lookup[(ROUTES[1], j)]))
                 for i in (1, 2) for j in (1, 2)]
        row = {'prompt_id': prompt, 'model_label': model, 'within_route': within,
               'cross_route_pairs': cross}
        for metric in ('lexical_distance', 'absolute_word_count_gap'):
            w = mean([within[route][metric] for route in ROUTES])
            c = mean([p[metric] for p in cross])
            row['mean_within_'+metric] = w
            row['mean_cross_'+metric] = c
            row['cross_minus_within_'+metric] = c-w
        complete.append(row)
    return complete, incomplete


def clustered_summary(cells):
    """Average model cells within prompt first, then resample whole prompts."""
    grouped = defaultdict(list)
    for cell in cells:
        grouped[cell['prompt_id']].append(cell)
    keys = ['mean_within_'+m for m in ('lexical_distance', 'absolute_word_count_gap')]
    keys += ['mean_cross_'+m for m in ('lexical_distance', 'absolute_word_count_gap')]
    keys += ['cross_minus_within_'+m for m in ('lexical_distance', 'absolute_word_count_gap')]
    prompts = [{'prompt_id': prompt, 'complete_model_cells': len(items),
                **{key: mean([r[key] for r in items]) for key in keys}}
               for prompt, items in sorted(grouped.items())]
    output = {'independent_prompt_groups': len(prompts), 'complete_cells': len(cells),
              'prompt_level_values': prompts, 'weighting': 'equal prompts; equal available model cells within prompt',
              'bootstrap_seed': SEED, 'bootstrap_resamples': BOOTSTRAPS, 'metrics': {}}
    for key in keys:
        values = [p[key] for p in prompts]
        interval = None
        if len(values) >= 2:
            rng = random.Random(SEED)
            resampled = [mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAPS)]
            interval = [quantile(resampled, .025), quantile(resampled, .975)]
        output['metrics'][key] = {'mean': mean(values), 'prompt_cluster_bootstrap_95_interval': interval}
    return output


def profile(rows, format_flags):
    flags = Counter()
    lengths = []
    deviations = 0
    conforming = 0
    for row in rows:
        current = format_flags(row['text'])
        flags.update(current)
        conforming += not current
        count = words(row['text'])
        lengths.append(count)
        target = row.get('target_words')
        deviations += target is not None and not .75*target <= count <= 1.25*target
    return {'responses': len(rows), 'format_screen_pass': conforming,
            'format_screen_pass_rate': conforming/len(rows) if rows else None,
            'format_flag_counts': dict(sorted(flags.items())),
            'word_count_mean': mean(lengths), 'word_count_median': statistics.median(lengths) if lengths else None,
            'word_count_min': min(lengths) if lengths else None, 'word_count_max': max(lengths) if lengths else None,
            'target_word_deviation_over_25_percent': deviations,
            'response_flags': dict(Counter(flag for row in rows for flag in row.get('flags', []))),
            'model_observation_scopes': dict(Counter(row.get('model_observation_scope', 'not_recorded') for row in rows))}


def historical_rows(codex, kaggle, jobs):
    cells = {(j['prompt_id'], j['model_label']) for j in jobs}
    output = []
    for route, source in ((ROUTES[0], codex), (ROUTES[1], kaggle)):
        seen = set()
        for row in source:
            model = row.get('model_requested') if route == ROUTES[0] else row.get('request_model', '').removeprefix('openai/')
            key = (row['prompt_id'], model)
            if key not in cells:
                continue
            if key in seen or sha(row['text'].encode('utf-8')) != row['text_sha256']:
                raise ValueError('Historical duplicate key or text hash mismatch')
            seen.add(key)
            output.append({**row, 'route': route, 'model_label': model})
        if seen != cells:
            raise ValueError('Historical paired cells are incomplete')
    for prompt, model in cells:
        pair = [r for r in output if r['prompt_id'] == prompt and r['model_label'] == model]
        if pair[0]['prompt'] != pair[1]['prompt']:
            raise ValueError('Historical prompt mismatch')
    return output


def source_profiles(rows, format_flags):
    result = {}
    for route in ROUTES:
        selected = [r for r in rows if r['route'] == route]
        result[route] = {'all_models': profile(selected, format_flags),
                         'by_model': {model: profile([r for r in selected if r['model_label'] == model], format_flags)
                                      for model in MODELS}}
    return result


def analyze(experiment=EXP):
    experiment = Path(experiment).resolve()
    root = experiment.parents[1]
    sources = {}

    def read_bytes(path):
        path = path.resolve()
        data = path.read_bytes()
        sources[path.relative_to(root).as_posix()] = sha(data)
        return data

    def read(path):
        return json.loads(read_bytes(path).decode('utf-8-sig'))

    state = read(experiment/'collection/state.json')
    if state['status'] == 'RUNNING' or state.get('active'):
        raise ValueError('Collection still active; wait for terminal state before analysis')
    audit = read(experiment/'metrics/collection-audit.json')
    if audit.get('run_id') != state['run_id']:
        raise ValueError('Audit belongs to a different run')
    jobs = decode_jsonl(read_bytes(experiment/'request-plan.jsonl'))
    records_path = experiment/'collection/records.jsonl'
    rows = decode_jsonl(read_bytes(records_path)) if records_path.exists() else []
    planned = {j['job_id']: j for j in jobs}
    if len({r['job_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate saved job identity')
    for row in rows:
        job = planned[row['job_id']]
        for key in ('route', 'repeat_id', 'prompt_id', 'prompt_group', 'model_label', 'user_message_sha256'):
            if row[key] != job[key]:
                raise ValueError('Saved identity differs from request plan')
        text = row['text'].encode('utf-8')
        if sha(text) != row['text_sha256'] or read_bytes(experiment/row['attempt_path']/'final.txt') != text:
            raise ValueError('Saved response text differs from final artifact')
    if len(rows) != state['saved_total'] or audit['responses'] != len(rows):
        raise ValueError('Saved counts disagree')
    if state['status'] == 'COMPLETE' and (len(rows) != 80 or audit['status'] != 'COLLECTED_UNVALIDATED'):
        raise ValueError('Complete collection lacks complete audit')

    checker_path = root/'experiments/EXP-034-explicit-format-alignment/code/prepare_alignment.py'
    read_bytes(checker_path)
    spec = importlib.util.spec_from_file_location('frozen_format_rules', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    old = {}
    for relative, expected_hash in OLD_HASHES.items():
        data = read_bytes(root/relative)
        if sha(data) != expected_hash:
            raise ValueError('Historical source changed: '+relative)
        old[relative] = decode_jsonl(data)
    historical = historical_rows(old[OLD_CODEX], old[OLD_KAGGLE], jobs)
    complete, incomplete = paired_cells(jobs, rows)
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row['text_sha256']].append({key: row[key] for key in ('job_id', 'prompt_id', 'model_label', 'route', 'repeat_id')})
    attempts = []
    for path in sorted((experiment/'collection/attempts').glob('*/result.json')):
        result = read(path)
        attempts.append({key: result.get(key) for key in ('job_id', 'route', 'status', 'reasons', 'cost_usd', 'stage', 'error_type', 'http_status')})
    data = {'schema_version': 1, 'experiment_id': 'EXP-035', 'observed_at': datetime.now(timezone.utc).isoformat(),
            'run_id': state['run_id'], 'collection_status': state['status'], 'audit_status': audit['status'],
            'responses': len(rows), 'requests_sent': state['requests_sent_total'],
            'confirmed_cost_usd': state['confirmed_cost_usd'], 'cost_known': state['cost_known'],
            'analysis_scope': 'descriptive surface-form and repeat-variation diagnostics; no classifier or causal inference',
            'definitions': {
                'lexical_distance': '1 minus Jaccard similarity of sets of lowercase Unicode word-token trigrams; regex \\b\\w+\\b. Under three tokens uses one whole-token tuple. Range0 identical surface units,1 no shared units. Not semantic difference.',
                'word_count': 'Whitespace-separated nonempty tokens, matching collection convention.',
                'cross_minus_within': 'Per cell: mean of four cross-route pair distances minus equal mean of the two within-route repeat distances.',
                'uncertainty': 'Percentile bootstrap of whole base-prompt groups after equal averaging available model cells; 10000 resamples, seed20261101; exploratory small-sample interval, not equivalence test.',
                'format_screen': 'Frozen EXP034 format_flags, including assistant-preface candidates; rule-negative is not full human content compliance.'},
            'aligned_profiles': source_profiles(rows, checker.format_flags),
            'historical_unaligned_profiles': source_profiles(historical, checker.format_flags),
            'complete_cells': complete, 'incomplete_cells': incomplete,
            'overall_paired': clustered_summary(complete),
            'by_model_paired': {model: clustered_summary([c for c in complete if c['model_label'] == model]) for model in MODELS},
            'identical_text_groups': [{'sha256': h, 'members': members} for h, members in sorted(duplicates.items()) if len(members) > 1],
            'identical_text_extra_rows': sum(len(members)-1 for members in duplicates.values()),
            'attempt_results': attempts,
            'limitations': [
                'Only10 independent original prompt groups, two repeats per route; estimates may be unstable.',
                'All successful raw records retained, including duplicates and flagged format/length results; paired distances use complete cells only, with incomplete cells listed.',
                'Missing or failed cells may cause selection bias; bootstrap does not correct it.',
                'Historical unaligned samples are not a simultaneous randomized format-control arm; cannot isolate a causal effect of format instructions.',
                'Channel differences may reflect hidden context, backend snapshots, output caps, timing or random variation; cannot identify built-in prompt contamination.',
                'Codex model labels may be requested_only; matching label text does not prove identical backend snapshot.',
                'No evidence here establishes training benefit, leaderboard improvement or raw-API equivalence.'],
            'source_sha256': sources}
    # Refuse a mixed snapshot if any read artifact changed during computation.
    for relative, expected in sources.items():
        if sha((root/relative).read_bytes()) != expected:
            raise ValueError('Source changed during analysis: '+relative)
    return data


def render_markdown(data):
    names = {'codex_subscription': 'Codex订阅', 'kaggle_model_proxy': 'Kaggle Model Proxy'}
    def number(value):
        return '未知' if value is None else f'{value:.4f}'
    lines = ['# EXP-035 双渠道重复对照：描述性结果', '',
             f"采集状态 `{data['collection_status']}`；保存{data['responses']}条，发送{data['requests_sent']}次；已确认Kaggle费用${data['confirmed_cost_usd']}，费用完整性={data['cost_known']}。",
             f"完整题目×模型单元{len(data['complete_cells'])}/20，不完整{len(data['incomplete_cells'])}；参与成对汇总的独立原题组{data['overall_paired']['independent_prompt_groups']}个。",
             ('存在费用未知的请求；已确认金额不是本次总费用，不能把未知部分当作0。' if not data['cost_known'] else '费用字段完整。'),
             ('没有完整配对单元，无法估计跨渠道差异或同渠道随机波动。' if not data['complete_cells'] else '以下成对估计仅覆盖完整单元。'), '',
             '## 格式与长度', '',
             '| 渠道 | 阶段 | 条数 | 格式筛查通过 | 含换行 | 含加粗 | 平均词数 | 词数偏离目标>25% |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for route in ROUTES:
        for label, key in [('旧未对齐', 'historical_unaligned_profiles'), ('新已对齐', 'aligned_profiles')]:
            p = data[key][route]['all_models']
            lines.append(f"| {names[route]} | {label} | {p['responses']} | {p['format_screen_pass']}/{p['responses']} | {p['format_flag_counts'].get('line_break',0)} | {p['format_flag_counts'].get('bold',0)} | {number(p['word_count_mean'])} | {p['target_word_deviation_over_25_percent']} |")
    lines += ['', '旧样本只是历史参照，格式变化不能解释为同时随机化实验得到的因果效果。格式筛查是规则检查，不等于内容全部合规。', '',
              '## 同渠道与跨渠道波动', '',
              '词汇距离是词组三元组的Jaccard距离，0表示表面词组相同，1表示没有共享词组；不是语义差异百分比。每单元先计算两个同渠道重复距离与四个跨渠道距离，再按原题组汇总。', '',
              '| 范围 | 同渠道词汇距离 | 跨渠道词汇距离 | 跨−同差值 | 按题组bootstrap 95%区间 |',
              '|---|---:|---:|---:|---|']
    for label, summary in [('两模型按题等权', data['overall_paired'])]+list(data['by_model_paired'].items()):
        m = summary['metrics']; difference = m['cross_minus_within_lexical_distance']
        interval = difference['prompt_cluster_bootstrap_95_interval']
        ci = '不足2个题组，未估计' if interval is None else f"[{number(interval[0])}, {number(interval[1])}]"
        lines.append(f"| {label} | {number(m['mean_within_lexical_distance']['mean'])} | {number(m['mean_cross_lexical_distance']['mean'])} | {number(difference['mean'])} | {ci} |")
    m = data['overall_paired']['metrics']
    lines += ['', f"平均绝对词数差：同渠道{number(m['mean_within_absolute_word_count_gap']['mean'])}词，跨渠道{number(m['mean_cross_absolute_word_count_gap']['mean'])}词。",
              f"完整文本完全相同的额外行数为{data['identical_text_extra_rows']}；这些真实重复均保留。", '',
              '## 解释边界', '',
              *['- '+item for item in data['limitations']], '',
              '完整逐单元、逐模型、缺失请求、失败结果、格式标记和输入哈希见 [paired-descriptive.json](paired-descriptive.json)。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=EXP)
    parser.add_argument('--analyze', action='store_true', help='Read terminal artifacts and write only paired-descriptive JSON/Markdown')
    args = parser.parse_args()
    if not args.analyze:
        print('No analysis written. Use --analyze after collection reaches a terminal state.')
        return 0
    data = analyze(args.experiment)
    metrics = args.experiment/'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics/'paired-descriptive.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (metrics/'paired-descriptive.md').write_text(render_markdown(data), encoding='utf-8')
    print(json.dumps({'responses': data['responses'], 'complete_cells': len(data['complete_cells']),
                      'prompt_groups': data['overall_paired']['independent_prompt_groups'],
                      'collection_status': data['collection_status'], 'written': ['metrics/paired-descriptive.json','metrics/paired-descriptive.md']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
