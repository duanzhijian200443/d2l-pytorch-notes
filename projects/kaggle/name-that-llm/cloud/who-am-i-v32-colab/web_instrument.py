"""Add stage persistence to frozen V32 AST without editing its mathematical statements."""
import ast
import copy


def parse_statements(source):
    return ast.parse(source).body


def delta_return():
    return ast.parse("return {k: LP[k] for k in LP if k not in _v32_before}").body[0]


class ContinueToReturn(ast.NodeTransformer):
    def visit_Continue(self, node):
        return ast.copy_location(delta_return(), node)

    # A continue within a nested loop or function belongs to that scope.
    def visit_For(self, node):
        return node

    def visit_While(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node


def instrument_infer(source):
    tree = ast.parse(source)
    result = []
    text_stages = {'a.tfidf_w > 0': 'tfidf', 'a.svc_w > 0': 'svc4', 'a.distort_w > 0': 'distort'}
    found = {'initialization': 0, 'models': 0, 'memoized': 0, 'text': 0}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'T0' for t in node.targets):
            result.append(node)
            result.extend(parse_statements("""
from web_checkpoint import Checkpoints
from sklearn.linear_model import LogisticRegression
def _v32_replay_seconds(seconds):
    global T0
    T0 -= seconds
_v32_ck = Checkpoints(os.environ['V32_CHECKPOINT_ROOT'], os.environ['V32_CHECKPOINT_CONTEXT'], _v32_replay_seconds)
"""))
            found['initialization'] += 1
        elif isinstance(node, ast.If) and ast.unparse(node.test) in text_stages:
            stage = text_stages[ast.unparse(node.test)]
            function = ast.parse(f'def _v32_compute_{stage}():\n    pass').body[0]
            function.body = parse_statements('_v32_before = set(LP)') + [node, delta_return()]
            result.append(function)
            result.extend(parse_statements(f"LP.update(_v32_ck.call('text/{stage}', _v32_compute_{stage}))"))
            found['text'] += 1
        elif isinstance(node, ast.FunctionDef) and node.name in {'l24_predict', 'atlas_scores'}:
            node.decorator_list.insert(0, ast.parse(f"_v32_ck.memoized('{node.name}')", mode='eval').body)
            result.append(node)
            found['memoized'] += 1
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'spec' and isinstance(node.iter, ast.Name) and node.iter.id == 'specs':
            function = ast.parse('def _v32_compute_model(spec):\n    pass').body[0]
            converted = [ContinueToReturn().visit(copy.deepcopy(part)) for part in node.body]
            function.body = parse_statements('_v32_before = set(LP)') + converted + [delta_return()]
            result.append(function)
            node.body = parse_statements("LP.update(_v32_ck.call('model/' + spec['name'], lambda: _v32_compute_model(spec)))")
            result.append(node)
            found['models'] += 1
        else:
            result.append(node)
    assert found == {'initialization': 1, 'models': 1, 'memoized': 2, 'text': 3}, found
    tree.body = result
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'
