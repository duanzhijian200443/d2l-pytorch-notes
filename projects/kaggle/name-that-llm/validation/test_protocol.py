import unittest
from pathlib import Path

import numpy as np

from protocol import load_split, select_sparse_rows, validate_fold_table, validate_score_cache


class ValidationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.folds, cls.labels = load_split(Path(__file__).resolve().parents[1])

    def test_same_prompt_cannot_cross_folds(self):
        broken = self.folds.copy()
        broken.loc[0, "fold"] = (broken.loc[0, "fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "Prompt crosses folds"):
            validate_fold_table(self.train, broken, self.labels)

    def test_duplicate_id_rejected(self):
        broken = self.folds.copy()
        broken.loc[0, "id"] = broken.loc[1, "id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_fold_table(self.train, broken, self.labels)

    def test_changed_text_rejected(self):
        broken = self.train.copy()
        broken.loc[0, "text"] += " changed"
        with self.assertRaisesRegex(ValueError, "Changed text"):
            validate_fold_table(broken, self.folds, self.labels)

    def test_label_order_and_row_order_rejected(self):
        predictions = self.folds.copy()
        scores = np.zeros((len(predictions), len(self.labels)))
        predictions["predicted_model"] = self.labels[0]
        with self.assertRaisesRegex(ValueError, "label order"):
            validate_score_cache(self.train, self.folds, predictions, scores, self.labels, self.labels[::-1])
        with self.assertRaisesRegex(ValueError, "row order"):
            validate_score_cache(self.train, self.folds, predictions.iloc[::-1], scores, self.labels, self.labels)

    def test_sparse_selection_is_order_invariant_and_label_blind(self):
        a = select_sparse_rows(self.folds, 2)
        changed = self.folds.sample(frac=1, random_state=9).copy()
        changed["model"] = "hidden"
        b = select_sparse_rows(changed, 2)
        self.assertEqual(a.id.tolist(), b.id.tolist())
        self.assertTrue(a.groupby("prompt_id").size().eq(2).all())
        self.assertNotEqual(a.id.tolist(), select_sparse_rows(self.folds, 2, repeat=1).id.tolist())


if __name__ == "__main__":
    unittest.main()
