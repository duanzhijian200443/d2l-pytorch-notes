"""Generate observer-only copies of the exact EXP-004/009 pipeline sources.

No source script is imported or executed. Removing the marked observer
statements and unwrapping return_array must reproduce the original AST.
An existing destination or an unpinned/already instrumented source is rejected.
The original clocks, data paths, model calls and ATLAS request set are retained.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / 'experiments/EXP-009-baseline-data-lineage'
EXPECTED = {
    'main': ('infer.py', '022cafd2e6504dd7cf38c8675b9597fb4fbd4c9dd0be69d38918293d9caac9b3'),
    'comp': ('comp_stack.py', 'b6b6a7b8adbe50580c2d0de5a41d0764ae752fe729e1471872a9197ac956f222'),
}
BASELINE_PATH = ROOT / 'reviews/own-scheme-readiness-20261030/baseline-v2.ipynb'
BASELINE_SHA256 = '6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ast_identity(tree):
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def observer_call(node, method):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == '_cache'
            and node.func.attr == method)


class StripObserver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if (node.module == 'full_stack_observer' and len(node.names) == 1
                and node.names[0].name == 'FullStackObserver' and node.names[0].asname == '_cache_observer'):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == '_cache' and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == '_cache_observer' and node.value.func.attr == 'from_env'):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if observer_call(node.value, 'capture'):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if observer_call(node, 'return_array'):
            require(len(node.args) == 2 and not node.keywords, 'Malformed return_array wrapper')
            return self.visit(node.args[1])
        return self.generic_visit(node)


def strip_observer_calls(tree):
    return StripObserver().visit(copy.deepcopy(tree))


def verify_ast_equivalence(original, generated):
    left = ast.parse(original)
    right = strip_observer_calls(ast.parse(generated))
    require(ast_identity(left) == ast_identity(right), 'Observer removal does not recover original algorithm AST')
    return True


def instrument_source(source, component):
    """Return (generated Python text, receipt), without filesystem writes."""
    require(component in EXPECTED, 'Unknown component')
    data = source.encode('utf-8') if isinstance(source, str) else bytes(source)
    filename, expected = EXPECTED[component]
    require(sha(data) == expected, 'Refusing unpinned or already instrumented ' + filename)
    text = data.decode('utf-8')
    tree = ast.parse(text, filename=filename)
    require(not any(isinstance(node, ast.Name) and node.id in ('_cache', '_cache_observer')
                    for node in ast.walk(tree)), 'Observer name collision')
    insertions, anchors = {}, []

    def inject(line, kind, statement, where='after'):
        matches = [node for node in ast.walk(tree) if isinstance(node, kind) and node.lineno == line]
        require(len(matches) == 1, f'Expected one {kind.__name__} at {filename}:{line}')
        target = matches[0]
        before, after = insertions.setdefault(id(target), ([], []))
        nodes = ast.parse(statement).body
        (before if where == 'before' else after).extend(nodes)
        anchors.append({'line': line, 'end_line': target.end_lineno, 'node': kind.__name__,
                        'placement': where, 'anchor_ast_sha256': sha(ast_identity(target).encode('utf-8')),
                        'observer_statement': statement})

    if component == 'main':
        inject(74, ast.Expr,
            "_cache = _cache_observer.from_env('main', test['id'].astype(str).tolist(), test['text'].tolist(), LABELS, "
            + repr({'source_sha256': expected})[:-1]
            + ", 'args': vars(a), 'data_dir': a.data, 'sim': a.sim, 'fake_test': a.fake_test, 'train_rows': len(train)})")
        inject(230, ast.Assign, "_cache.capture('main_svc4_margin', margins=_d, classes=_svc.classes_)")
        inject(514, ast.Assign,
            "_cache.capture('main_members', members=LP, total=tot, specs=specs, args=vars(a), groups=Gte, train_targets=y)")
        inject(546, ast.Assign,
            "_cache.capture('atlas_batch', bundle=bundle, requested_rows=rows, local_rows=_idx, global_rows=rows[_idx], "
            "probability_native=_batch, native_labels=_alab, completed_mask=scored)")
        inject(567, ast.Expr,
            "_cache.capture('comp_loaded', logit=_cl, probability=COMP_P, comp_meta=_cm, path=a.comp_path)")
        inject(561, ast.If,
            "_cache.capture('comp_status', probability=COMP_P, requested_weight=a.comp_w, path=a.comp_path)")
        inject(575, ast.AugAssign,
            "_cache.capture('main_probability', probability=P, normalized_logits=lpn, divisor=wsum)")
        inject(576, ast.If, "_cache.capture('pre_atlas', probability=P)")
        inject(579, ast.Assign,
            "_cache.capture('atlas_requested', rows=rows, margin=_margin, probability=P, bundles=_bundles, "
            "weights=_aw, budget=a.atlas_budget, probe=a.atlas_probe)")
        inject(585, ast.Assign,
            "_cache.capture('atlas_returned', bundle=_bd, weight=_w, requested_rows=rows, probabilities=A_, "
            "completed_mask=scored, completed_rows=completed_rows)")
        inject(592, ast.Assign, "_cache.capture('pre_sinkhorn', probability=P)")
        inject(606, ast.Assign,
            "_cache.capture('sinkhorn_applied', probability=P, balanced_probability=Q, logits=tot, "
            "predicted_marginal=_pred_marg, train_prior=_train_prior, target=_target, target_mix=_rho, weight=a.sinkhorn)")
        inject(607, ast.Assign,
            "_cache.capture('postprocessing_base', logits=_v38_before_hard, post=a.post, sim=a.sim)")
        inject(672, ast.Assign,
            "_cache.capture('typography', evidence=_evidence, confidence_gate=_confidence_gate, length_gate=_length_gate, "
            "coverage_gate=_coverage_gate, gate=_gate, logits_before=_raw, logits_after=_safe_logits, marks=_marks, "
            "word_count=_word_count, active_markers=_active_markers, train_rows=len(train), train_class_counts=_freq, "
            "train_marker_hits=_hits, test_marker_matrix=_Xte, kappa=_kappa, feature_weights=_fw, rates=_rate, global_rate=_global_rate)")
        end = "_cache.capture('component_end', output_path=a.out, typography_applied=not bool(a.sim))"
        margin_call = None
    else:
        inject(95, ast.Expr,
            "_cache = _cache_observer.from_env('comp', test['id'].astype(str).tolist(), test['text'].tolist(), LABELS, "
            + repr({'source_sha256': expected})[:-1]
            + ", 'smoke': SMOKE, 'tiny': TINY, 'fake_test': os.environ.get('NTL_FAKE_TEST'), "
            "'drop_q35': os.environ.get('NTL_DROP_Q35'), 'data_dir': os.environ.get('NTL_DATA') if SMOKE else str(DATA_DIR), 'train_rows': len(train)})")
        inject(103, ast.Assign,
            "_cache.capture('member', key=key, probability=prob, log_probability=lp, weight=w, rows=rows)")
        inject(289, ast.Assign,
            "_cache.capture('style_margin_context', role='plain', rows=np.arange(len(TE)), classes=svc_plain.classes_, evaluation=True)", 'before')
        inject(302, ast.Assign,
            "_cache.capture('style_margin_context', role='centred', rows=np.flatnonzero(rows_style), classes=svc_centred.classes_, evaluation=True)", 'before')
        inject(307, ast.Expr,
            "_cache.capture('style_routed', probability=P_style, groups=G_style, mask=rows_style)", 'before')
        inject(947, ast.If,
            "_cache.capture('comp_before_gates', logit=LOGIT, gates=GATES, groups=G_TE, group_mask=GROUP_ROWS, "
            "style_groups=G_style, style_mask=rows_style)", 'before')
        inject(943, ast.Assign,
            "_cache.capture('family_gate_applied', key=gate_key, family=family, indices=idx, raw_score=GATES[gate_key], "
            "gate_log_probability=gate_lp, base=base, strength=strength, family_mass=family_mass, logit=LOGIT)")
        end = ("_cache.capture('component_end', logit=LOGIT, serialized_logit=LOGIT.astype(np.float32), "
               "output_path='/kaggle/working/comp_logit.npy', meta_path='/kaggle/working/comp_meta.json')")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and node.lineno == 236
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'decision_function']
        require(len(calls) == 1, 'Expected one original svc_probs decision_function call')
        margin_call = calls[0]
        anchors.append({'line': 236, 'end_line': 236, 'node': 'Call', 'placement': 'identity_return_wrapper',
                        'anchor_ast_sha256': sha(ast_identity(margin_call).encode('utf-8')),
                        'observer_statement': "_cache.return_array('style_margin', ORIGINAL_CALL)"})

    class Insert(ast.NodeTransformer):
        def visit(self, node):
            identity = id(node)
            changed = super().visit(node)
            if identity in insertions:
                before, after = insertions[identity]
                return [*copy.deepcopy(before), changed, *copy.deepcopy(after)]
            return changed

        def visit_Call(self, node):
            self.generic_visit(node)
            if node is margin_call:
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Name(id='_cache', ctx=ast.Load()), attr='return_array', ctx=ast.Load()),
                    args=[ast.Constant(value='style_margin'), node], keywords=[]), node)
            return node

    generated_tree = Insert().visit(tree)
    import_position = 1 if ast.get_docstring(generated_tree, clean=False) is not None else 0
    while (import_position < len(generated_tree.body) and isinstance(generated_tree.body[import_position], ast.ImportFrom)
           and generated_tree.body[import_position].module == '__future__'):
        import_position += 1
    generated_tree.body[import_position:import_position] = ast.parse(
        'from full_stack_observer import FullStackObserver as _cache_observer').body
    generated_tree.body.extend(ast.parse(end).body)
    ast.fix_missing_locations(generated_tree)
    generated = '# Generated observer-only copy. Do not edit; see instrumentation-receipt.json.\n' + ast.unparse(generated_tree) + '\n'
    verify_ast_equivalence(text, generated)
    compile(generated, filename, 'exec')
    receipt = {'component': component, 'source_filename': filename, 'source_sha256': expected,
               'generated_sha256': sha(generated.encode('utf-8')), 'algorithm_ast_equal_after_removing_observers': True,
               'algorithm_ast_sha256': sha(ast_identity(ast.parse(text)).encode('utf-8')),
               'anchors': anchors, 'module_end_observer': end,
               'original_script_executed': False, 'clock_or_data_paths_changed': False}
    return generated, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Refusing an existing output directory')
    baseline = BASELINE_PATH.read_bytes()
    require(sha(baseline) == BASELINE_SHA256, 'Frozen baseline notebook changed')
    notebook = json.loads(baseline)
    rendered, receipts = {}, []
    for component, (filename, _) in EXPECTED.items():
        data = (args.source_dir / filename).read_bytes()
        cell = notebook['cells'][6 if component == 'main' else 8]['source']
        source = ''.join(cell) if isinstance(cell, list) else cell
        require(''.join(source.splitlines(keepends=True)[1:]) == data.decode('utf-8').replace('\r\n', '\n'),
                'EXP-009 source differs from frozen notebook cell')
        rendered[filename], receipt = instrument_source(data, component)
        receipts.append(receipt)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, generated in rendered.items():
        with (args.output_dir / filename).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(generated)
    receipt = {'schema_version': 1, 'status': 'GENERATED_NOT_EXECUTED', 'baseline_sha256': BASELINE_SHA256,
               'instrument_sha256': sha(Path(__file__).read_bytes()), 'components': receipts,
               'observer_module': 'full_stack_observer', 'observer_module_copied': False,
               'limits': ['AST equality is not a real-model runtime equality result.',
                          'Observer IO can change wall-clock-limited ATLAS coverage; no clock is adjusted.',
                          'Only original requested ATLAS rows are scored; future requests need coverage verification.']}
    with (args.output_dir / 'instrumentation-receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'output_dir': str(args.output_dir),
                      'generated_files': sorted(rendered), 'models_executed': False}))


if __name__ == '__main__':
    main()
