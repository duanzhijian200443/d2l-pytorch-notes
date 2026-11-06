"""Pure-path rejection tests anchored to the actual audited531-file manifest."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from input_policy import (ALLOWED_OMISSIONS, EXPECTED_COMP_MEMBERS, InputPolicyError,
                          assert_input_policy_equal, match_input_glob, resolve_input_policy)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "cloud/full-stack-capture/preflight-output-v1/preflight/preflight_manifest.json"
MANIFEST_SHA256 = "de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5"


class InputPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        content = MANIFEST.read_bytes()
        if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
            raise AssertionError("The actual CPU manifest changed")
        manifest = json.loads(content)
        if manifest["status"] != "PASS" or len(manifest["files"]) != 531:
            raise AssertionError("Expected the accepted531-file CPU manifest")
        cls.paths = [entry["path"] for entry in manifest["files"]]
        cls.policy = resolve_input_policy(cls.paths)

    def test_actual_inventory_resolves_twenty_and_original_omissions(self):
        self.assertEqual(self.policy["expected_comp_members"], list(EXPECTED_COMP_MEMBERS))
        self.assertEqual(len(EXPECTED_COMP_MEMBERS), 20)
        self.assertEqual(self.policy["allowed_omissions"], list(ALLOWED_OMISSIONS))
        self.assertEqual(self.policy["resolved_paths"]["lora24_public"], [])
        self.assertFalse(self.policy["weights_changed"])
        self.assertTrue(self.policy["group_coverage"]["requires_at_least_one_eligible_group"])
        self.assertEqual(self.policy["input_file_count"], 531)
        self.assertEqual(len(self.policy["resolved_paths"]["ultrabert"]), 5)
        self.assertTrue(all(path.startswith("/kaggle/input/") for values in self.policy["resolved_paths"].values() for path in values))

    def test_input_order_does_not_change_frozen_policy(self):
        reordered = resolve_input_policy(reversed(self.paths))
        assert_input_policy_equal(reordered, self.policy)
        self.assertEqual(reordered, self.policy)

    def test_missing_mandatory_inputs_and_exact_lora_siblings_fail(self):
        keys = ["lora24_seed7", "lora24_claude/heads.safetensors", "lora20_q25_instyle/norms.safetensors",
                "tokattn_deb", "backbone_deberta", "atlas_bundle", "head_q14v5", "ultrabert/fold3/model.safetensors"]
        for key in keys:
            removed = self.policy["resolved_paths"][key][0]
            with self.subTest(key=key), self.assertRaises(InputPolicyError):
                resolve_input_policy([path for path in self.paths if path != removed])

    def test_public_marker_suddenly_appearing_is_rejected(self):
        with self.assertRaisesRegex(InputPolicyError, "lora24_public"):
            resolve_input_policy(self.paths + ["/kaggle/input/datasets/new/source/lora24/final/COMPLETE.json"])

    def test_new_named_q35_backbone_is_rejected(self):
        for directory in ("qwen3_5", "qwen3.5", "Qwen-3.5-9B"):
            with self.subTest(directory=directory), self.assertRaisesRegex(InputPolicyError, "Q35"):
                resolve_input_policy(self.paths + ["/kaggle/input/models/owner/" + directory + "/config.json"])

    def test_duplicate_and_case_colliding_paths_are_rejected(self):
        path = self.paths[0]
        for duplicate in (path, str(Path(path).parent).replace("\\", "/") + "/" + Path(path).name.upper()):
            with self.subTest(path=duplicate), self.assertRaises(InputPolicyError):
                resolve_input_policy(self.paths + [duplicate])

    def test_a_second_matching_checkpoint_is_ambiguous(self):
        duplicate = "/kaggle/input/datasets/other/ntl-qwen3-8b-lora-claude-head/final/COMPLETE.json"
        with self.assertRaisesRegex(InputPolicyError, "ambiguous"):
            resolve_input_policy(self.paths + [duplicate])

    def test_paths_must_be_canonical_absolute_input_files(self):
        bad = ["relative/file", "/kaggle/working/file", "/kaggle/input", "/kaggle/input/file/",
               "/kaggle/input/a/../file", "/kaggle/input/a/./file", "/kaggle/input//file",
               "C:\\file", "/kaggle/input/line\nbreak", None]
        for path in bad:
            with self.subTest(path=path), self.assertRaises(InputPolicyError):
                resolve_input_policy(self.paths + [path])

    def test_double_star_supports_zero_one_and_many_directories(self):
        pattern = "**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json"
        for middle in ("", "nested/", "a/b/"):
            self.assertTrue(match_input_glob("/kaggle/input/datasets/u/ntl-qwen3-8b-lora-claude-head/" + middle + "final/COMPLETE.json", pattern))
        self.assertTrue(match_input_glob("/kaggle/input/final/COMPLETE.json", "**/final/COMPLETE.json"))
        self.assertFalse(match_input_glob("/kaggle/input/a/b/config.json", "a/*.json"))
        self.assertTrue(match_input_glob("/kaggle/input/a/b/config.json", "a/**/*.json"))

    def test_full_inventory_digest_detects_noncritical_loss_and_addition(self):
        irrelevant = next(path for path in self.paths if path.endswith("custom.css"))
        mutations = [[path for path in self.paths if path != irrelevant], self.paths + ["/kaggle/input/new/unlisted.txt"]]
        for paths in mutations:
            actual = resolve_input_policy(paths)
            with self.assertRaisesRegex(InputPolicyError, "frozen CPU"):
                assert_input_policy_equal(actual, self.policy)

    def test_partial_shard_loss_is_rejected_by_cpu_policy_comparison(self):
        weights = self.policy["resolved_paths"]["backbone_qwen3_8b/weights"]
        self.assertGreater(len(weights), 1)
        actual = resolve_input_policy([path for path in self.paths if path != weights[0]])
        with self.assertRaises(InputPolicyError):
            assert_input_policy_equal(actual, self.policy)

    def test_policy_comparison_rejects_field_and_json_type_changes(self):
        changed = copy.deepcopy(self.policy); changed["schema_version"] = True
        self.assertEqual(changed["schema_version"], self.policy["schema_version"])
        with self.assertRaises(InputPolicyError):
            assert_input_policy_equal(changed, self.policy)

    def test_invalid_objects_cannot_count_as_matching_policies(self):
        for value in (None, {}, [], {"schema_version": 1}):
            with self.subTest(value=value), self.assertRaises(InputPolicyError):
                assert_input_policy_equal(value, value)
        changed = copy.deepcopy(self.policy); changed["expected_comp_members"].reverse()
        with self.assertRaises(InputPolicyError):
            assert_input_policy_equal(changed, self.policy)


if __name__ == "__main__":
    unittest.main()
