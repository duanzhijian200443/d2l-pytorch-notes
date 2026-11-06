"""Replay Apex layers on the fixed 729-row diagnostic panel with explicit cache boundaries."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import special

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src import full_stack_replay as engine

SRC = ROOT / 'research/solutions/SRC-067-synthreaper-sovereign-gold-apex-v1/source.ipynb'
SOURCE_SHA = 'ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd'
MODULES = {'comp_stack.py': '4a4b800a5380094d78da66d140d224f8a3974d1e3012086783d5bf81b1264514',
           'infer.py': '63112c88d43f9d69b3132243315e0db28c40b4bf0f42bac651f73ef07a67f9bb',
           'sibling_style.py': '8a7f3b3dad7cbe1755b66e8455a157894cac92b8983fc805f5ed6286ea915e66'}
CACHE = ROOT / 'cloud/full-stack-capture-r1/outputs-v2/results'
CACHE_SHA = '1732849196220d023716a2dc912d660282c94022c2df370c69d9a6c2899bac63'
OLD = ROOT / 'experiments/EXP-045-public-data-full-stack'
PRIOR = ROOT / 'experiments/EXP-032-sinkhorn-fixed-cache'
ARMS = ('zero', 'four120', 'four240')
LABELS = list(engine.LABELS)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load(path, key):
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_modules():
    require(sha(SRC) == SOURCE_SHA, 'Fixed SRC067 byte binding differs')
    cells = read(SRC)['cells']
    modules = {}
    for cell in cells:
        first, _, body = ''.join(cell['source']).partition('\n')
        if first.startswith('%%writefile '):
            name = first.split()[-1]
            if name in MODULES:
                require(hashlib.sha256(body.encode()).hexdigest() == MODULES[name] and name not in modules,
                        'Generated source module differs: ' + name)
                modules[name] = body
    require(set(modules) == set(MODULES), 'Missing author module')
    launch = ''.join(cells[16]['source'])
    for argument in ("'--svc_w', '12'", "'--sinkhorn', '0.60'", "'--sinkhorn_target_mix', '0.35'",
                     "'--post', '0'", "'--atlas_w', '0.20'", "'--atlas_margin', '0.3'", "'--comp_w', '0.15'"):
        require(argument in launch, 'Apex argument differs: ' + argument)
    return modules


def block(source, first, stop):
    require(source.count(first) == 1 and source.count(stop) == 1, 'Ambiguous source boundary: ' + first)
    i = source.index(first)
    j = source.index(stop, i + len(first))
    return textwrap.dedent(source[i:j])


def verify_shared_loader(inputs, modules):
    # The shared-input preparer used the byte-identical SRC068 loader; the SVC
    # fits and replay explicitly use SRC067. Bind both, without equating notebooks.
    source = ROOT / 'research/solutions/SRC-068-chyibo-name-that-llm-0663-fork-v1/source.ipynb'
    expected = '07cb21ed6f9b2b2d0920756a8fea64ec5a5d37637d7ea4fad07ab6eb85ce5140'
    require(inputs['source'] == 'SRC-068 V1 / 364709540' and inputs['source_sha256'] == sha(source) == expected,
            'Invalid fixed shared-loader source')
    cells = [''.join(c['source']).partition('\n')[2] for c in read(source)['cells']
             if ''.join(c['source']).startswith('%%writefile comp_stack.py\n')]
    require(cells == [modules['comp_stack.py']], 'SRC067 and SRC068 comp module bytes differ')
    funcs = [n for n in ast.parse(cells[0]).body if isinstance(n, ast.FunctionDef) and n.name == 'load_extra_passages']
    require(len(funcs)==1 and hashlib.sha256(ast.get_source_segment(cells[0],funcs[0]).encode()).hexdigest()
            == inputs['author_loader_sha256'], 'Shared loader function bytes differ')


def capture():
    cmp = load(PRIOR / 'compare_cached.py', 'exp053_cache_reader')
    math = load(OLD / 'math_replay.py', 'exp053_reference_math')
    protocol = read(PRIOR / 'protocol.json')
    cmp.validate_protocol(protocol)
    accepted = cmp.verify_real_artifacts(CACHE, protocol, CACHE_SHA)
    panel = pd.read_csv(cmp.bound_input(protocol['inputs']['blind_panel']), dtype={'id':str})
    train = pd.read_csv(cmp.bound_input(protocol['inputs']['official_train']), dtype={'id':str, 'prompt_id':str})
    panel['text'] = panel.text.fillna('')
    train['text'] = train.text.fillna('')
    with cmp.captured_helpers(CACHE, accepted['plan']) as helpers:
        decoded = {}
        for key in ('main', 'comp'):
            snap = helpers['full_stack_cache'].read_cache(CACHE/'cache', key, panel.id.tolist(),
                panel.text.tolist(), LABELS, required_notes=['event-index'],
                manifest_sha256=accepted['receipts'][key]['manifest_sha256'])
            decoded[key] = helpers['full_stack_observer'].decode_events(snap)
    main = engine._Events(decoded['main'], 'main')
    comp = engine._Events(decoded['comp'], 'comp')
    ident = engine._identity(main)
    require(ident == engine._identity(comp) and ident['ids'] == panel.id.tolist() and len(panel) == 729,
            'Cross-stream row/hash identity differs')
    checks = engine._Checks()
    legacy_logit, _, _ = engine._replay_comp(comp, len(panel), accepted['plan']['expected_comp_members'], checks)
    legacy_p, _, divisor, args = math._main_base(main, len(panel), checks, accepted['plan']['expected_main_members'])
    old_pre, _ = math._mix_comp(legacy_p, legacy_logit)
    checks.compare('legacy.pre_atlas', main.one('pre_atlas')['probability'], old_pre, scores=True)
    atlas = math._atlas_cache(main, len(panel), ident['ids'], args, old_pre, checks)
    checks.compare('legacy.pre_sinkhorn', main.one('pre_sinkhorn')['probability'],
                   math._apply_atlas(old_pre, atlas['completed_rows'], atlas), scores=True)
    return SimpleNamespace(cmp=cmp, math=math, protocol=protocol, panel=panel, train=train, main=main, comp=comp,
                           main_members=main.one('main_members')['members'], divisor=int(divisor), atlas=atlas,
                           legacy_logit=legacy_logit, id=ident, parity_checks=checks.stages)


def source_comp(modules, cache, probability, gates=None):
    """Execute only the published member addition and conditional family-gate functions."""
    module = modules['comp_stack.py']
    selected = {'add_member', '_np_logsoftmax', '_apply_conditional_family_gate'}
    nodes = [node for node in ast.parse(module).body if isinstance(node, ast.FunctionDef) and node.name in selected]
    require({node.name for node in nodes} == selected, 'Missing author comp functions')
    families = {f:np.flatnonzero([s.startswith(f) for s in LABELS]) for f in ('gpt','glm','gemini','claude')}
    gates = cache.comp.one('comp_before_gates')['gates'] if gates is None else gates
    for key in engine.GATE_ORDER:
        family = 'claude' if key == 'claude_lora' else key
        require(np.asarray(gates[key]).shape == (len(cache.panel),len(families[family])) and
                np.isfinite(gates[key]).all(), 'Missing/nonfinite gate logits: ' + key)
    env = dict(np=np, time=__import__('time'), T0=0, TE=cache.panel.text.tolist(), GATES=gates,
               FAMILY_IDX=families, LOGIT=np.zeros((len(cache.panel),18)), print=lambda *a, **k: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<fixed Apex comp functions>', 'exec'), env)
    weight_style = []
    for _, item in cache.comp.by_name['member']:
        is_style = item['key'] == 'stylegram'
        if is_style:
            weight_style.append(item['weight'])
            require(item['rows'] is None, 'Stylegram row scope differs')
        env['add_member'](item['key'], probability if is_style else item['probability'],
                          rows=item['rows'], weight=item['weight'])
    require(weight_style == [1.516], 'Stylegram missing, repeated or wrong weight')
    before = env['LOGIT'].copy()
    gate_audit = {}
    for key, family, lam in [('claude','claude',1.5),('claude_lora','claude',1.),('gpt','gpt',.25),
                             ('gemini','gemini',.25),('glm','glm',.25)]:
        idx = families[family]
        base = env['LOGIT'][:, idx].copy()
        sorted_p = np.sort(special.softmax(base, axis=1), axis=1)
        strength = min(.45, lam/(1+lam)) * np.clip((.45 - (sorted_p[:,-1]-sorted_p[:,-2]))/.45, 0, 1)
        env['_apply_conditional_family_gate'](key, family, lam, key == 'claude_lora')
        require(np.allclose(special.logsumexp(env['LOGIT'][:,idx],axis=1), special.logsumexp(base,axis=1),
                            rtol=0, atol=1e-10), 'Family mass changed')
        gate_audit[key] = {'active_rows':int(np.count_nonzero(strength)), 'mean_strength':float(strength.mean())}
    return before, env['LOGIT'].astype(np.float32), gate_audit


def check_author_baseline(modules, cache):
    style = next(item['probability'] for _, item in cache.comp.by_name['member'] if item['key'] == 'stylegram')
    _, logit, _ = source_comp(modules, cache, style)
    require(np.array_equal(logit, cache.legacy_logit), 'Apex gate arithmetic differs on original cache')


def main_probability(cache, weighted_svc):
    members = dict(cache.main_members)
    require('svc4' in members, 'Missing main SVC slot')
    members['svc4'] = weighted_svc
    divisor = sum(abs(v).max() > 0 for v in members.values())
    require(divisor == cache.divisor and divisor > 0, 'Main divisor changed')
    return engine._softmax_original(sum(members.values()) / divisor)


def strict_route(cache, probabilities):
    requested, margins = cache.math._requested(probabilities)
    missing = np.setdiff1d(requested, cache.atlas['completed_rows'])
    info = dict(requested_count=len(requested), covered_count=len(requested)-len(missing),
                missing_count=len(missing), requested_ids=cache.panel.id.iloc[requested].tolist(),
                missing_ids=cache.panel.id.iloc[missing].tolist(), all_requested_covered=len(missing)==0)
    pre_sinkhorn = cache.math._apply_atlas(probabilities, requested, cache.atlas) if len(missing)==0 else None
    return info, margins, pre_sinkhorn


def author_post(modules, probabilities, train, panel, diag_path):
    infer = modules['infer.py']
    env = dict(np=np, re=re, _v38sp=special, P=probabilities.copy(), train=train, test=panel, LABELS=LABELS,
               y=train.model.map({s:i for i,s in enumerate(LABELS)}).to_numpy(dtype=int),
               a=SimpleNamespace(sinkhorn=.60, sinkhorn_target_mix=.35))
    sinkhorn = block(infer, '    Q = P.copy()\n', '_v38_before_hard = tot.copy()')
    exec(compile(sinkhorn, '<fixed Apex Sinkhorn>', 'exec'), env)
    env['_v38_before_hard'] = env['tot'].copy()
    typography = block(infer, '    _marks = [', '    # V40A: explicitly preserve')
    exec(compile(typography, '<fixed Apex typography>', 'exec'), env)
    reference = env['_safe_logits'].copy()
    sibling = {}
    exec(compile(modules['sibling_style.py'], '<fixed Apex sibling_style.py>', 'exec'), sibling)
    try:
        final = sibling['apply_grouped_style_residual'](reference, train.text.values, panel.text.values, LABELS,
                  env['y'], train.prompt_id.values, max_alpha=.08, diagnostics_path=str(diag_path))
        residual = 'EXECUTED'
    except Exception as error:
        final = reference.copy()
        residual = 'SOURCE_FALLBACK_' + type(error).__name__
    require(np.isfinite(final).all(), 'Nonfinite postprocessing')
    return env['tot'], reference, final, residual


def read_style(folder, digest, cache, inputs):
    summary_file = folder / 'summary.json'
    require(sha(summary_file) == digest, 'Wrong style training result digest')
    summary = read(summary_file)
    state = read(folder / 'state.json')
    require(summary['status'] == state['status'] == 'COMPLETE' and state['summary_sha256'] == digest and
            summary['experiment_id'] == 'EXP-053' and summary['source_sha256'] == SOURCE_SHA
            and summary['comp_stack_sha256'] == MODULES['comp_stack.py'] and summary['labels'] == LABELS
            and summary['trained_svms'] == 36 and summary['family_gate_svm_fits'] == 3 and summary['gpu_used'] is False,
            'Wrong style run identity')
    job_config = ROOT / 'cloud/apex-stylegram-053/style/experiment-config.json'
    require(summary['config_sha256'] == sha(job_config) and read(job_config)['input_hashes'] == summary['input_hashes'] and
            summary['shared_sha256'] == inputs['shared']['sha256'] == sha(HERE / 'inputs/shared.csv')
            and summary['input_hashes']['panel.csv'] == sha(HERE/'inputs/panel.csv'), 'Style inputs differ')
    style = {}
    for arm in ARMS:
        item = summary['outputs']['panel'][arm]
        require(item['path'] == f'panel/{arm}.npz' and sha(folder/item['path']) == item['sha256'], 'Style hash/arm mismatch')
        with np.load(folder/item['path'], allow_pickle=False) as z:
            require(z['ids'].tolist() == cache.id['ids'] and z['labels'].tolist() == LABELS and
                    z['arm'].item() == arm and z['config_sha256'].item() == summary['config_sha256']
                    and np.all(z['fold'] == -1), 'Wrong full-training panel matrix')
            p = z['probabilities']
            engine._prob(p, len(cache.panel), 'Apex Stylegram panel probabilities')
            style[arm] = p.copy()
    gates = dict(cache.comp.one('comp_before_gates')['gates'])
    gate_file = folder/'panel/family_gates.npz'
    require(summary['family_gates']['path'] == 'panel/family_gates.npz' and
            sha(gate_file) == summary['family_gates']['sha256'], 'Family gate artifact hash differs')
    with np.load(gate_file, allow_pickle=False) as z:
        require(z['ids'].tolist() == cache.id['ids'] and z['labels'].tolist() == LABELS and
                z['config_sha256'].item() == summary['config_sha256'] and
                z['shared_sha256'].item() == summary['shared_sha256'], 'Family gate identity differs')
        for family in ('gpt', 'gemini', 'glm'):
            idx = [i for i, label in enumerate(LABELS) if label.startswith(family)]
            require(z[family+'_indices'].tolist() == idx and z[family].shape == (len(cache.panel),len(idx))
                    and np.isfinite(z[family]).all(), 'Invalid family score or class order')
            gates[family] = z[family].copy()
    return style, gates, summary


def read_main(folder, digest, cache):
    path = folder / 'main-result.json'
    require(sha(path) == digest, 'Wrong main result digest')
    receipt = read(path)
    require(receipt['status'] == 'COMPLETE' and receipt['source_sha256'] == SOURCE_SHA and
            receipt['coefficient'] == 12 and receipt['pool_rows'] == 190588 and
            receipt['fit_log'].startswith('svc4 fit rows=200478 ') and sha(folder/'main.npz') == receipt['score_sha256'],
            'Wrong main fit or unchanged-version binding')
    with np.load(folder/'main.npz',allow_pickle=False) as z:
        require(z['ids'].tolist() == cache.id['ids'] and z['labels'].tolist() == LABELS,'Main score row/label identity differs')
        lp = z['weighted_logp']
        require(lp.dtype.kind == 'f' and lp.shape == (len(cache.panel),18) and np.isfinite(lp).all()
                and np.allclose(special.logsumexp(lp/12.,axis=1),0,rtol=0,atol=1e-5),
                'Main source-weighted log probability mismatch')
    return lp


def evaluate(scores, cache, inputs):
    truth = pd.read_csv(cache.cmp.bound_input(cache.protocol['inputs']['truth_sidecar']), dtype={'id':str,'prompt_id':str})
    require(truth.id.tolist() == cache.id['ids'] and truth.text_sha256.tolist() == cache.id['text_sha256'] and
            truth.model.nunique() == 17 and truth.prompt_id.nunique() == 56, 'Panel truth binding differs')
    overlap_file = HERE/'inputs/local-only/panel-overlap.csv'
    item = next(v for v in inputs['local_only_evaluation'] if v['path'].endswith('/panel-overlap.csv'))
    require(sha(overlap_file) == item['sha256'], 'Panel overlap evidence differs')
    flags = pd.read_csv(overlap_file, dtype={'id':str, 'prompt_id':str})
    require(flags.id.tolist() == truth.id.tolist() and flags.model.tolist() == truth.model.tolist() and
            flags.prompt_id.tolist() == truth.prompt_id.tolist() and int(flags.apex_main_svc_fit_exact_key_hit.sum()) == 698,
            'Overlap does not align with panel')
    table = flags.copy()
    metrics = {}
    changes = {}
    for key, scores_matrix in scores.items():
        pred = np.asarray(LABELS)[scores_matrix.argmax(axis=1)]
        correct = pred == truth.model.to_numpy()
        table[key] = pred
        recall = pd.Series(correct).groupby(truth.model).mean()
        metrics[key] = {'correct':int(correct.sum()), 'rows':len(correct),'accuracy':float(correct.mean()),
                        'covered_17_class_macro_recall':float(recall.mean()),
                        'scope':'descriptive exposed panel only' if key.startswith(('main','pre_atlas','post_','final'))
                                else 'inspected external panel; feature/model exposure not exhaustively known'}
    for stage in ('style', 'comp', 'pre_atlas', 'post_sinkhorn', 'post_typography', 'final'):
        for left,right in [('zero','four120'),('zero','four240'),('four120','four240')]:
            a,b = stage+'_'+left, stage+'_'+right
            if a not in table or b not in table:
                continue
            aok,bok = table[a].eq(truth.model),table[b].eq(truth.model)
            diff = table[a].ne(table[b])
            changes[b+'_minus_'+a] = dict(changed=int(diff.sum()), fixed=int((~aok&bok).sum()),
                        new_errors=int((aok&~bok).sum()), both_wrong_changed=int((diff&~aok&~bok).sum()),
                        net_correct=int(bok.sum()-aok.sum()), delta_covered_macro_recall=
                        metrics[b]['covered_17_class_macro_recall']-metrics[a]['covered_17_class_macro_recall'])
    return table, metrics, changes


def run(style_folder, style_digest, main_folder, main_digest, output):
    require(not output.exists(), 'Fresh output directory required')
    modules, cache = source_modules(), capture()
    check_author_baseline(modules, cache)
    inputs = read(HERE/'inputs/input-receipt.json')
    verify_shared_loader(inputs, modules)
    style, gates, style_summary = read_style(style_folder,style_digest,cache,inputs)
    main_p = main_probability(cache, read_main(main_folder, main_digest, cache)) if main_folder else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp)/'report'
        staging.mkdir()
        scores, arrays, routes, gate_stats, residual = {}, {}, {}, {}, {}
        if main_p is not None:
            scores['main_v13_w12'] = main_p
        for arm in ARMS:
            scores['style_'+arm] = style[arm]
            before, comp, gate_stats[arm] = source_comp(modules,cache,style[arm],gates=gates)
            arrays['comp_before_gates_'+arm] = before
            scores['comp_'+arm] = comp
            if main_p is None:
                routes[arm] = {'status':'NOT_COMPUTED_NO_AUTHENTIC_MAIN_V13_SVC'}
                residual[arm] = 'NO_FULL_PROBABILITIES'
                continue
            mixed, comp_p = cache.math._mix_comp(main_p, comp)
            scores['pre_atlas_'+arm] = mixed
            arrays['comp_probability_'+arm] = comp_p
            info, margin, after = strict_route(cache, mixed)
            arrays['route_margin_'+arm] = margin
            routes[arm] = info
            if after is None:
                residual[arm] = 'NOT_RUN_INCOMPLETE_ATLAS'
                continue
            arrays['pre_sinkhorn_'+arm] = after
            sk,ty,final,residual[arm] = author_post(modules,after,cache.train,cache.panel,staging/('v40a-'+arm+'.json'))
            scores['post_sinkhorn_'+arm], scores['post_typography_'+arm],scores['final_'+arm] = sk,ty,final
        # Instantiate truth only after inference and routing finish.
        table, metrics, changes = evaluate(scores,cache,inputs)
        table.to_csv(staging/'predictions.csv',index=False)
        np.savez_compressed(staging/'scores.npz',ids=np.asarray(cache.id['ids']),labels=np.asarray(LABELS),
                            text_sha256=np.asarray(cache.id['text_sha256']),**arrays,**scores)
        write(staging/'coverage.json',{'legacy_atlas_completed_rows':len(cache.atlas['completed_rows']),
                                       'arms':routes,'missing_rows_imputed':False})
        summary = dict(status='COMPLETE_CONDITIONAL_EXPOSED_PANEL_REPLAY',source='SRC-067',
            fixed_source_sha256=SOURCE_SHA, version=1, script_version_id=364684268, cache_artifacts_sha256=CACHE_SHA,
            cache_parity_stages=len(cache.parity_checks), old_comp_bitwise_source_parity=True,
            main_inventory=list(cache.main_members), member_count_divisor=cache.divisor,
            comp_inventory=[item['key'] for _,item in cache.comp.by_name['member']],
            style_result_sha256=style_digest, style_shared_sha256=style_summary['shared_sha256'],
            main_result_sha256=main_digest, main_refit_used=main_p is not None,
            parameters={'svc4':12,'stylegram':1.516,'comp':.15,'atlas':.2,'atlas_margin':.3,
                        'sinkhorn':.6,'target_mix':.35,'typography':1.2,'v40a_max_alpha':.08},
            lineage={'stylegram':'recomputed full official training + snapshot shared data + arm additions',
                     'family_gates_refit_current_shared':['gpt','gemini','glm'],
                     'family_gate_scores_from_exp031':['claude','claude_lora'],
                     'all_other_comp_and_main_members_except_main_svc4':'fixed EXP031 captures; not certified historical Apex attachment bytes',
                     'nonzero_comp_candidates_absent_from_capture':['lora24_public', 'lora24_public_twin', 'tokattn_q35'],
                     'omission_policy':'No inference, renormalization, or substitute cache for absent optional members. Their execution on the author hidden rerun is unknown.',
                     'atlas_scores':'only completed original EXP031 rows',
                     'shared_comparison':inputs['exp031_comparison'],
                     'attachment_version_policy':inputs['attachment_version_policy']},
            contamination={'panel_overlaps':inputs['panel_overlaps'],'apex_main_svc':inputs['apex_main_svc'],
                'known_near_prompt_scope':inputs['semantic_overlap_prior_review_scope'],
                'boundaries':'Main branch trained on 698 of 729 panel answers. All main-related metrics descriptive. '
                             'Remaining 31 rows are not an independent evaluation. Upstream heavy-model training membership unknown.'},
            gates=gate_stats, residual_status=residual, metrics=metrics, changes=changes,
            final_metrics={arm:metrics.get('final_'+arm) for arm in ARMS},
            limits='Conditional on reused older heavy members and Claude gates. This panel cannot establish validation or leaderboard gain.')
        write(staging/'summary.json',summary)
        write(staging/'artifacts.json',{'code_sha256':sha(Path(__file__)),'input_receipt_sha256':sha(HERE/'inputs/input-receipt.json'),
            'module_hashes':MODULES,'files':[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(staging.iterdir())]})
        require(not output.exists(),'Output already exists')
        staging.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--style-results',type=Path)
    parser.add_argument('--style-summary-sha256')
    parser.add_argument('--main-results',type=Path)
    parser.add_argument('--main-receipt-sha256')
    parser.add_argument('--output',type=Path,default=HERE/'replay-results')
    args = parser.parse_args()
    if args.preflight:
        modules, cache = source_modules(), capture()
        check_author_baseline(modules,cache)
        print(json.dumps({'status':'PASS_CACHE_AND_SOURCE', 'rows':len(cache.panel),'atlas_completed':len(cache.atlas['completed_rows']),
                          'legacy_main_divisor':cache.divisor,'verified_cache_stages':len(cache.parity_checks),
                          'author_gate_legacy_reproduction':'bitwise_equal','module_hashes':MODULES}))
        return
    require(args.style_results is not None and args.style_summary_sha256, 'Style result path and receipt digest required')
    require((args.main_results is None) == (args.main_receipt_sha256 is None), 'Main folder and digest required together')
    summary = run(args.style_results,args.style_summary_sha256,args.main_results,args.main_receipt_sha256,args.output)
    print(json.dumps({k:summary[k] for k in ('status','metrics','changes','residual_status')},ensure_ascii=False))


if __name__ == '__main__':
    main()
