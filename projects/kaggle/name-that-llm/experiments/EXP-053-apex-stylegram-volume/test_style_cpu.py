"""CPU-only synthetic checks. No competition fit or external calls."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import style_cpu as cpu

ROOT = Path(__file__).resolve().parents[2]


def fixture(labels):
    topics = ("garden flowers soil", "solar orbit planet", "kitchen baker bread", "music piano violin")
    def rows(prefix, groups, use_labels):
        return pd.DataFrame([{"id": f"{prefix}-{g}-{j}", "prompt_id": f"{prefix}-{g}",
                              "model": label, "text": f"{topics[g % len(topics)]} {prefix} with vocabulary{j} and expression{j}. "}
                             for g in range(groups) for j, label in enumerate(use_labels)])
    test = rows("test", 3, labels)
    test["text"] = [topics[0]] * 12 + [topics[1]] * 20 + [topics[2]] * 22
    return rows("train", 4, labels), rows("public", 2, labels), rows("new", 3, sorted(cpu.NEW_MODELS)), test


class StyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = next((ROOT / "research/solutions").glob("SRC-067*/source.ipynb"))
        cls.body = cpu.author_module(cls.source)
        cls.labels = cpu.author_namespace(cls.body)["LABELS"]
        cls.train, cls.shared, cls.added, cls.test = fixture(cls.labels)

    def test_direct_author_matches_incremental_zero_path(self):
        no_rows = self.added.iloc[:0]
        author, detail = cpu.fit(self.body, self.train, self.shared, no_rows, self.test, direct_author=True)
        local, local_detail = cpu.fit(self.body, self.train, self.shared, no_rows, self.test, direct_author=False)
        for key in author:
            np.testing.assert_array_equal(author[key], local[key])
        self.assertEqual(int(author["centred_mask"].sum()), 12)
        self.assertEqual(detail["models"], local_detail["models"])
        self.assertEqual(detail["models"]["svc_plain"]["settings"]["dual"], "auto")

    def test_added_rows_change_scores_but_not_inferred_grouping(self):
        ref, _ = cpu.fit(self.body, self.train, self.shared, self.added.iloc[:0], self.test, direct_author=True)
        increment, detail = cpu.fit(self.body, self.train, self.shared, self.added, self.test, direct_author=False)
        np.testing.assert_array_equal(ref["group_ids"], increment["group_ids"])
        np.testing.assert_array_equal(ref["centred_mask"], increment["centred_mask"])
        self.assertEqual(detail["added_class_counts"], dict.fromkeys(cpu.NEW_MODELS, 3))
        self.assertGreater(np.max(np.abs(increment["scores"] - ref["scores"])), 1e-6)

    def test_original_three_family_gates(self):
        matrices, detail = cpu.fit_family_gates(self.body, self.train, self.shared, self.test)
        self.assertEqual(detail["svm_fits"], 3)
        for fam, expected in (("gpt", [11, 12, 13, 14, 15]), ("gemini", [5, 6, 7, 8]), ("glm", [9, 10])):
            self.assertEqual(matrices[fam + "_indices"].tolist(), expected)
            self.assertEqual(matrices[fam].shape, (len(self.test), len(expected)))
            self.assertEqual(detail["sources"][fam]["added_rows"], 0)
        np.testing.assert_allclose(matrices["glm"][:, 0], -matrices["glm"][:, 1])

    def test_wrong_source_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ipynb"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                cpu.author_module(path)

    def test_exact_shared_regens_removed_only_in_their_heldout_fold(self):
        directory = Path(__file__).resolve().parent / "inputs"
        public = cpu.table(directory / "shared.csv")
        folds = cpu.table(ROOT / "cloud/format120-044/payload/folds.csv")
        folds["fold"] = folds.fold.astype(int)
        regen = cpu.table(directory / "regen-fold-map.csv")
        groups = cpu.validate_regen_mapping(regen, folds, public)
        sizes = [len(public) - len(cpu.training_shared(public, groups, fold)[0]) for fold in range(5)]
        self.assertEqual(sizes, [36, 54, 36, 72, 54])
        full, excluded = cpu.training_shared(public, groups, -1)
        self.assertTrue(full.equals(public))
        self.assertEqual(excluded, [])
        regen.loc[0, "official_fold"] = str((int(regen.loc[0, "official_fold"]) + 1) % 5)
        with self.assertRaisesRegex(ValueError, "official fold"):
            cpu.validate_regen_mapping(regen, folds, public)


if __name__ == "__main__":
    unittest.main()
