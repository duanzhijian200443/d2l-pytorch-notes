import json
import unittest

import pandas as pd

from audit import OUT, ROOT, RAW, SPECS, canonical_model, fingerprint, prompt_identity


class AuditRules(unittest.TestCase):
    def test_normalization_does_not_erase_punctuation(self):
        self.assertEqual(fingerprint("Hello   WORLD\n"), fingerprint("hello world"))
        self.assertNotEqual(fingerprint("a—b"), fingerprint("a-b"))

    def test_model_mapping_keeps_nearby_versions_distinct(self):
        self.assertEqual(canonical_model("openai/gpt-5.5-2026-04-23"), "gpt-5.5")
        self.assertNotEqual(canonical_model("google/gemini-3.7-flash"), canonical_model("google/gemini-3.8-flash"))
        self.assertTrue(canonical_model("unknown/gpt-5.5").startswith("UNKNOWN_PROVIDER"))

    def test_official_prompt_lineage_requires_evidence(self):
        row = {"source_prompt_id": "", "split": "", "prompt_key": "p034"}
        self.assertEqual(prompt_identity("pool-v5", row), "")
        row["prompt_key"] = "pukkei37:pv:tr_p034"
        self.assertEqual(prompt_identity("pool-v5", row), "official:p034")

    def test_shared_prompt_bank_is_one_group(self):
        row = {"source_prompt_id": "", "split": "", "prompt_key": "g0001"}
        self.assertEqual(prompt_identity("seyit-v2", row), prompt_identity("abdullah-v2", row))


class ArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pd.read_parquet(OUT / "row_flags.parquet")
        cls.candidates = pd.read_csv(OUT / "initial_candidate_index.csv", keep_default_na=False)
        cls.manifest = json.loads((ROOT / "validation/prompt-group-v1/manifest.json").read_text(encoding="utf-8"))

    def test_candidate_rows_unique_valid_and_no_known_official_links(self):
        c = self.candidates
        self.assertTrue(c.row_id.is_unique and c.text_hash.is_unique)
        self.assertTrue(c.model.isin(self.manifest["labels"]).all())
        self.assertFalse(c.hold_for_review.any())
        self.assertFalse(c.official_shingle_review.any())
        self.assertTrue(c.related_official_prompts.eq("").all())

    def test_clean_regenerations_protected(self):
        rows = self.rows
        regen = rows[rows.table.eq("prvs-clean-v6") & rows.split.eq("train_regen")]
        self.assertEqual(len(regen), 252)
        self.assertTrue(regen.linked_official_prompt.all())
        self.assertFalse(regen.initial_pilot_selected.any())
        fold_of = pd.read_csv(ROOT / "validation/prompt-group-v1/folds.csv").groupby("prompt_id").fold.first().to_dict()
        for row in regen.itertuples():
            self.assertIn(str(fold_of[row.source_prompt_id]), row.exclude_validation_folds.split("|"))

    def test_matching_text_cannot_lose_official_prompt_exclusion(self):
        linked = self.rows.groupby("text_hash").linked_official_prompt
        self.assertTrue(linked.nunique().eq(1).all())

    def test_candidate_indices_recover_the_original_text_and_label(self):
        for table, ref, version, filename in SPECS:
            chosen = self.candidates[self.candidates.table.eq(table)]
            if chosen.empty:
                continue
            source = pd.read_csv(RAW / ref / f"v{version}" / filename, low_memory=False)
            recovered = source.iloc[chosen.row_number.astype(int)]
            label = "model" if "model" in recovered else "label"
            self.assertEqual(recovered.text.map(fingerprint).tolist(), chosen.text_hash.tolist())
            self.assertEqual(recovered[label].tolist(), chosen.model.tolist())


if __name__ == "__main__":
    unittest.main()
