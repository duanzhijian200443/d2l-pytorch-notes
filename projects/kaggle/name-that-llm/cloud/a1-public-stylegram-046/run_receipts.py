"""Fail-closed observers of frozen comp/main logs; no model or math changes.

Log evidence is meaningful only together with the hash-frozen producer source,
zero subprocess exit status and output schema/ID/finite-array checks by caller.
"""
import ast
from collections import Counter
import hashlib
import re


COMP_WEIGHTS = {
    'stylegram': 1.516, 'stylometry_lgb': .063, 'glm9b_twin': 1.5*.110,
    'gauss_glm9b_L8': 1.5*.110, 'q3_8b_twin': 1.5*.219, 'gauss_q3_8b_L22': 1.5*.219,
    'q25_7b_twin': 1.5*.141, 'q3_4b_twin': 1.5*.083, 'gauss_q3_4b_L4': 1.5*.083,
    'debv3l': .05, 'debv3l_twin': .25, 'gauss_debv3l_L24': .25,
    'electl': .032, 'desk': .132, 'lora24_seed7': .113, 'lora24_claude': .177,
    'lora24_claude_twin': .15, 'lora20_q25_instyle': .17, 'tokattn_deb': .185, 'ultrabert': .095,
}
CONDITIONAL = {k for k in COMP_WEIGHTS if k.endswith('_twin') or k.startswith('gauss_')}
GATES = {'claude', 'claude_lora', 'gpt', 'gemini', 'glm'}
ALLOWED_OMISSIONS = ['lora24_public', 'lora24_public_twin', 'tokattn_q35']
MAIN_BASE = ['tfidf', 'svc4', 'distort', 'q14v5', 'q8v5', 'll8v5', 'n14b', 'q25_14', 'deb']
ADAPTERS = {k: [f'{k}_f{i}' for i in range(5)]+[k] for k in ['q14v5','q8v5','ll8v5']}
ADAPTERS.update(n14b=['lora_qwen3_14b'], q25_14=['lora_qwen25_14b'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def one(pattern, text, name):
    found = re.findall(pattern, text, re.MULTILINE)
    require(len(found) == 1, name+' missing or duplicated')
    return found[0]


def header(text, n_rows):
    require(isinstance(text, str) and bool(text), 'Empty log')
    require(type(n_rows) is int and n_rows > 0, 'Invalid test row count')
    require('Traceback (most recent call last)' not in text, 'Traceback in log')


def verify_comp_log(text, n_rows):
    header(text, n_rows)
    rows, gpus, smoke = one(r'^train \(2699, 4\)  test \((\d+), 2\)  gpus (\d+)  smoke (\w+)\s*$', text, 'Comp data header')
    require(int(rows) == n_rows and gpus == '2' and smoke == 'False', 'Comp data/GPU/smoke mismatch')
    groups = one(r'^test groups \(content \+ GLM L24\): (\d+) rows \| singletons (\d+)  2-3: (\d+)  4-18: (\d+)  >18: (\d+)\s*$', text, 'Comp grouping')
    total, singles, small, medium, large = map(int, groups)
    require(total == n_rows and singles+small+medium+large == total, 'Comp grouping row count mismatch')
    eligible = small+medium
    expected = set(COMP_WEIGHTS) if eligible else set(COMP_WEIGHTS)-CONDITIONAL
    members = re.findall(r'^\s*\+ (\w+)\s+weight ([0-9.]+) on (\d+) rows\s+\([0-9]+s\)\s*$', text, re.MULTILINE)
    require(Counter(k for k, _, _ in members) == Counter({k:1 for k in expected}), 'Comp member set missing, unexpected or duplicated')
    for key, weight, count in members:
        require(weight == format(COMP_WEIGHTS[key], '.3f'), 'Comp weight mismatch: '+key)
        require(int(count) == (eligible if key in CONDITIONAL else n_rows), 'Comp member row coverage mismatch: '+key)
    # The three historical omissions have two explicit log lines; the public
    # twin is implicitly absent with its parent. No additional skip is accepted.
    skips = [line.strip() for line in text.splitlines() if re.search(r'\bskip(?:ped)?\b', line, re.I)]
    allowed = [r'SKIP lora24_public: checkpoint not attached \(\*\*/lora24/final/COMPLETE\.json\)',
               r'OPTIONAL HEAD SKIPPED tokattn_q35: head=True backbone=False']
    require(len(skips) == 2 and all(sum(bool(re.fullmatch(p, line)) for line in skips) == 1 for p in allowed), 'Unexpected comp skip/degradation')
    gates = re.findall(r'^\s*family gate applied (\w+): mean strength=([0-9.]+), active rows=(\d+)\s*$', text, re.MULTILINE)
    require(Counter(k for k, _, _ in gates) == Counter({k:1 for k in GATES}), 'Five family gates missing, unexpected or duplicated')
    for key, strength, count in gates:
        require(0 <= float(strength) <= .45005 and 0 <= int(count) <= n_rows, 'Invalid gate receipt: '+key)
    one(r'^COMP STACK DONE [0-9]+s\s*$', text, 'Comp completion')
    return {'status':'PASS', 'log_sha256':hashlib.sha256(text.encode()).hexdigest(), 'test_rows':n_rows,
            'eligible_comp_rows':eligible, 'members':[{'name':k,'displayed_weight':w,'rows':int(n)} for k,w,n in members],
            'member_count':len(members), 'family_gates':sorted(GATES), 'allowed_omissions':ALLOWED_OMISSIONS,
            'conditional_members_not_applicable':sorted(CONDITIONAL) if not eligible else [],
            'scope':'Frozen-source log execution evidence; caller must additionally validate exit code and artifacts.'}


def verify_main_log(text, n_rows):
    header(text, n_rows)
    # Only producer log() lines enter the grammar; weight-loading progress is
    # intentionally ignored. Known failure messages still fail over the whole log.
    require(not re.search(r'COMP SKIPPED|ATLAS SKIPPED|atlas invalid scores|atlas budget checkpoint|atlas probe stopped|WARNING non-finite rows', text), 'Main skipped, truncated or invalid numerical result')
    lines = '\n'.join(re.sub(r'^\[\s*[0-9]+s\]\s*', '', line) for line in text.splitlines() if re.match(r'^\[\s*[0-9]+s\]',line))
    require(int(one(r'^train 2699 test (\d+)$',lines,'Main data header')) == n_rows, 'Main data rows mismatch')
    nclusters, histogram = one(r'^test clusters (\d+) sizes (\[[0-9, ]+\])$',lines,'Main grouping')
    hist = ast.literal_eval(histogram)
    require(all(type(v) is int and v >= 0 for v in hist) and sum(hist) == int(nclusters)
            and sum(i*v for i,v in enumerate(hist)) == n_rows, 'Main cluster histogram mismatch')
    gauss_rows = sum(i*v for i,v in enumerate(hist) if 2 <= i <= 18)
    done = re.findall(r'^(\w+) done(?: gpu.*)?$',lines,re.MULTILINE)
    require(Counter(done) == Counter(MAIN_BASE), 'Main base members missing, unexpected or duplicated')
    for key, expected in ADAPTERS.items():
        observed = re.findall(r'^l24 '+re.escape(key)+r': adapter (\S+) done \(bf16 redo batches [0-9]+\)$',lines,re.MULTILINE)
        repeats = 2 if n_rows < 50 else 1  # frozen tiny-test in-sample reproduction check
        require(Counter(observed) == Counter(expected*repeats), 'Main adapter set mismatch: '+key)
    gaussian = re.findall(r'^gaussian head (\w+): (\d+) grouped test rows, weight ([0-9.]+)$',lines,re.MULTILINE)
    require(Counter(k for k,_,_ in gaussian) == Counter(['q14v5','q8v5']), 'Main Gaussian heads missing/duplicated')
    for key, count, weight in gaussian:
        require(int(count) == gauss_rows and float(weight) == {'q14v5':.7,'q8v5':.44}[key], 'Main Gaussian coverage/weight mismatch')
    one(r'^comp stack loaded \([0-9]+s\)$',lines,'Main comp loaded')
    request, total = map(int, one(r'^atlas on (\d+)/(\d+) rows \(top-2 margin < 0\.3\)$',lines,'ATLAS routing'))
    require(total == n_rows and 0 <= request <= total, 'ATLAS request rows mismatch')
    complete, target = map(int, one(r'^atlas completed (\d+)/(\d+) rows before budget$',lines,'ATLAS completion'))
    require(complete == target == request, 'ATLAS incomplete coverage')
    bundle, mixed, target, weight = one(r'^atlas (\S+) mixed (\d+)/(\d+) rows at ([0-9.]+)(?: gpu.*)?$',lines,'ATLAS mix')
    require(int(mixed) == int(target) == request and float(weight) == .2, 'ATLAS mix coverage/weight mismatch')
    require(bundle.endswith('/ntl-atlas-qwen3-8b-v1/atlas_adapter/atlas_adapter'), 'Unexpected ATLAS bundle')
    one(r'^wrote /kaggle/working/submission\.parquet .+$',lines,'Main submission write')
    changed = int(one(r'^V31 Bayesian typography completed; changed from pre-Bayes: (\d+)$',lines,'Typography completion'))
    require(0 <= changed <= n_rows, 'Typography count invalid')
    return {'status':'PASS','log_sha256':hashlib.sha256(text.encode()).hexdigest(), 'test_rows':n_rows,
            'base_members_observed':MAIN_BASE,'gaussian_heads_observed':['q14v5_gauss','q8v5_gauss'],
            'gaussian_rows':gauss_rows,'adapter_counts':{k:len(v) for k,v in ADAPTERS.items()},
            'twin_evidence':'No distinct twin-completion line exists; twin computation remains inferred from hash-frozen successful l24 paths, not independently observed.',
            'comp_loaded':True,'atlas_requested_rows':request,'atlas_completed_rows':complete,'atlas_weight':.2,
            'typography_changed_rows':changed,'scope':'Frozen-source logs plus caller exit/artifact validation; not a new model run.'}
