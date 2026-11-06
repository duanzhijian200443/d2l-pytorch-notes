"""No-GPU tests for the EXP-031 lifecycle/input-policy-only runner change."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import input_policy as P


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('capture_r1_runner_under_test', HERE / 'runner.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
OLD = ROOT / 'cloud/full-stack-capture/runner.py'
MANIFEST = ROOT / 'cloud/full-stack-capture/preflight-output-v1/preflight/preflight_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_fixture():
    hashes = {name: 'a' * 64 for name in R.RUNTIME_FILES + ['runner.py', 'panel/test.csv',
                                                          'preflight_manifest.json', 'availability.json']}
    hashes['panel/test.csv'] = R.PANEL_SHA256
    return {'schema_version': 1, 'experiment_id': 'EXP-031', 'run_id': 'synthetic-policy-only',
            'scope': 'exploratory_external', 'train_sha256': R.TRAIN_SHA256, 'panel_sha256': R.PANEL_SHA256,
            'panel_rows': 729, 'work_budget_seconds': 28200, 'platform_wall_seconds': 28800,
            'shutdown_grace_seconds': 10, 'expected_main_members': list(R.MAIN_MEMBERS),
            'expected_comp_members': list(P.EXPECTED_COMP_MEMBERS), 'expected_family_gates': list(R.FAMILY_GATES),
            'allowed_omissions': list(P.ALLOWED_OMISSIONS), 'attachment_bindings': [{'fixture': i} for i in range(31)],
            'preflight_manifest_sha256': hashes['preflight_manifest.json'],
            'availability_sha256': hashes['availability.json'], 'input_hashes': hashes}


class CompatibilityTests(unittest.TestCase):
    def test_old_source_is_still_the_frozen_parent(self):
        self.assertEqual(sha(OLD), '923c05137561bc0571b64975c809ee67b2fce192d7165299c00c63e1de6ceb76')

    def test_unrelated_functions_and_lifecycle_are_ast_identical(self):
        old = ast.parse(OLD.read_text(encoding='utf-8'))
        new = ast.parse((HERE / 'runner.py').read_text(encoding='utf-8'))
        old_functions = {node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        new_functions = {node.name: node for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        class NormalizeIdentity(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value == 'EXP-031':
                    node.value = 'EXP-029'
                return node

        for name in old_functions.keys() - {'validate_plan', 'required_optional_inputs', 'worker'}:
            with self.subTest(function=name):
                normalized = NormalizeIdentity().visit(copy.deepcopy(new_functions[name]))
                self.assertEqual(ast.dump(normalized), ast.dump(old_functions[name]))

    def test_worker_diff_contains_only_input_policy_and_identity_changes(self):
        old = ast.parse(OLD.read_text(encoding='utf-8'))
        new = ast.parse((HERE / 'runner.py').read_text(encoding='utf-8'))
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'worker')
                     for tree in (old, new)]

        class NormalizePolicy(ast.NodeTransformer):
            def visit_Expr(self, node):
                call = node.value
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'save'
                        and len(call.args) > 1 and isinstance(call.args[1], ast.Call)
                        and isinstance(call.args[1].func, ast.Name)
                        and call.args[1].func.id in ('required_optional_inputs', 'verify_input_policy')):
                    return None
                return self.generic_visit(node)

            def visit_Tuple(self, node):
                node.elts = [value for value in node.elts if not (isinstance(value, ast.Constant) and value.value == 'availability.json')]
                return self.generic_visit(node)

            def visit_Dict(self, node):
                pairs = [(key, value) for key, value in zip(node.keys, node.values)
                         if not (isinstance(key, ast.Constant) and key.value in ('availability_sha256', 'input_policy_sha256'))]
                node.keys = [key for key, _ in pairs]
                node.values = [value for _, value in pairs]
                return self.generic_visit(node)

            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    node.value = node.value.replace('predeclared20-member', 'predeclared22-member')
                return node

        normalized = [NormalizePolicy().visit(copy.deepcopy(function)) for function in functions]
        self.assertEqual(ast.dump(normalized[0]), ast.dump(normalized[1]))

    def test_plan_uses_shared_twenty_member_policy(self):
        plan = plan_fixture()
        R.validate_plan(plan)
        self.assertEqual(len(R.COMP_MEMBERS), 20)
        self.assertEqual(R.COMP_MEMBERS, list(P.EXPECTED_COMP_MEMBERS))
        self.assertEqual(plan['allowed_omissions'], ['lora24_public', 'lora24_public_twin', 'tokattn_q35'])

    def test_old_identity_inventory_and_missing_bindings_are_rejected(self):
        mutations = [('experiment_id', 'EXP-029'),
                     ('expected_comp_members', list(P.EXPECTED_COMP_MEMBERS) + ['lora24_public', 'lora24_public_twin']),
                     ('allowed_omissions', ['tokattn_q35']), ('availability_sha256', None)]
        for key, value in mutations:
            plan = plan_fixture(); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.validate_plan(plan)
        for name in ('input_policy.py', 'availability.json'):
            plan = plan_fixture(); del plan['input_hashes'][name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                R.validate_plan(plan)

    def test_budget_contract_and_cap_are_unchanged(self):
        for key in ('work_budget_seconds', 'platform_wall_seconds', 'shutdown_grace_seconds'):
            plan = plan_fixture(); plan[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.validate_plan(plan)
        self.assertEqual(R.cap_atlas_budget(1000, 300), 180)
        with self.assertRaisesRegex(ValueError, 'BUDGET_INSUFFICIENT'):
            R.cap_atlas_budget(1000, 299)


class FrozenPolicyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sha(MANIFEST) != 'de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5':
            raise AssertionError('Accepted CPU file manifest changed')
        cls.manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        cls.paths = [entry['path'] for entry in cls.manifest['files']]
        cls.availability = P.resolve_input_policy(cls.paths)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ntl-r1-policy-test-')
        self.here = Path(self.temp.name).resolve()
        self.assertEqual(self.here.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.here.name.startswith('ntl-r1-policy-test-'))
        self.addCleanup(self.temp.cleanup)
        (self.here / 'input_policy.py').write_bytes((HERE / 'input_policy.py').read_bytes())
        (self.here / 'availability.json').write_text(json.dumps(self.availability), encoding='utf-8')
        self.plan = plan_fixture()
        self.plan['input_hashes']['input_policy.py'] = sha(self.here / 'input_policy.py')
        self.plan['availability_sha256'] = self.plan['input_hashes']['availability.json'] = sha(self.here / 'availability.json')
        self.module_location = patch.object(P, '__file__', str(self.here / 'input_policy.py'))
        self.module_location.start()
        self.addCleanup(self.module_location.stop)

    def test_absent_public_passes_for_real_audited_paths(self):
        actual = R.verify_input_policy(self.here, self.plan, self.manifest, list(reversed(self.paths)))
        self.assertEqual(actual, self.availability)
        self.assertEqual(actual['input_file_count'], 531)
        self.assertEqual(actual['resolved_paths']['lora24_public'], [])

    def test_new_public_or_missing_mandatory_gpu_input_fails(self):
        missing = self.availability['resolved_paths']['tokattn_deb'][0]
        cases = [self.paths + ['/kaggle/input/new/lora24/final/COMPLETE.json'],
                 [path for path in self.paths if path != missing]]
        for paths in cases:
            with self.subTest(paths=len(paths)), self.assertRaises(P.InputPolicyError):
                R.verify_input_policy(self.here, self.plan, self.manifest, paths)

    def test_unlisted_gpu_file_and_changed_cpu_inventory_fail(self):
        changed = self.paths + ['/kaggle/input/new/unlisted.txt']
        with self.assertRaises(P.InputPolicyError):
            R.verify_input_policy(self.here, self.plan, self.manifest, changed)
        manifest = copy.deepcopy(self.manifest)
        manifest['files'].append({'path': '/kaggle/input/new/unlisted.txt'})
        with self.assertRaises(P.InputPolicyError):
            R.verify_input_policy(self.here, self.plan, manifest, self.paths)

    def test_plan_and_availability_must_match_order_and_omissions(self):
        for key in ('expected_comp_members', 'allowed_omissions'):
            plan = copy.deepcopy(self.plan); plan[key].reverse()
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.verify_input_policy(self.here, plan, self.manifest, self.paths)

    def test_policy_code_and_receipt_are_hash_bound(self):
        (self.here / 'input_policy.py').write_text('# changed fixture', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            R.verify_input_policy(self.here, self.plan, self.manifest, self.paths)

    def test_q35_model_type_is_rejected_even_without_a_q35_path(self):
        config = next(path for path in self.paths if path.endswith('/config.json'))
        original_reader = R.read_json
        def read_config(path):
            return {'model_type': 'qwen3_5'} if str(path) == config else original_reader(path)
        audit = self.here / 'input-availability-check.json'
        with patch.object(R, 'read_json', side_effect=read_config), self.assertRaisesRegex(ValueError, 'Undeclared Q35'):
            R.verify_input_policy(self.here, self.plan, self.manifest, self.paths, audit_path=audit)
        receipt = json.loads(audit.read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertIn({'path': config, 'model_type': 'qwen3_5'}, receipt['config_model_types'])

    def test_success_and_failure_keep_cpu_gpu_availability_receipt(self):
        path = self.here / 'input-availability-check.json'
        R.verify_input_policy(self.here, self.plan, self.manifest, self.paths, audit_path=path)
        receipt = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['cpu'], receipt['gpu'])
        changed = self.paths + ['/kaggle/input/new/unlisted.txt']
        with self.assertRaises(P.InputPolicyError):
            R.verify_input_policy(self.here, self.plan, self.manifest, changed, audit_path=path)
        receipt = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertNotEqual(receipt['cpu']['input_paths_sha256'], receipt['gpu']['input_paths_sha256'])

    def test_policy_rejection_occurs_before_any_child_or_gpu_probe(self):
        out = self.here / 'results'; out.mkdir()
        (self.here / 'plan.json').write_text(json.dumps(self.plan), encoding='utf-8')
        build = {'baseline_sha256': R.BASELINE_SHA256, 'status': 'BUILT_NOT_EXECUTED', 'files': {}}
        values = {'plan.json': self.plan, 'runtime-build.json': build, 'preflight_manifest.json': self.manifest}
        with patch.object(R, 'read_json', side_effect=lambda path: values[Path(path).name]), \
                patch.object(R, 'verify_payload'), patch.object(R.shutil, 'copyfile'), \
                patch.object(R, 'verify_cpu_manifest', return_value={}), \
                patch.object(R, 'verify_input_policy', side_effect=P.InputPolicyError('policy mismatch')), \
                patch.object(R, 'command') as command:
            with self.assertRaisesRegex(P.InputPolicyError, 'policy mismatch'):
                R.worker(self.here, out)
        command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
