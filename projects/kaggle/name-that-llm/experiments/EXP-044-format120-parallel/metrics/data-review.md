# EXP044 data preparation gate

PASS_KNOWN_EXACT_FIXED_RAW_CLEAN_120.

Both inputs contain the frozen EXP043 same60-prompt old120 selection, in the original selection-list order. Raw uses actual EXP023 text and old job keys. Clean equals the accepted EXP043 transform exactly. All metadata, IDs, labels, prompts and raw hashes are identical across arms; only text and text_sha256 may differ.

There are60 prompts and60 rows per model (Luna/Terra); 86 rows change text. No source rows were dropped or remapped. Both arms inherit identical EXP042 per-fold weights, with the old10073 public counts frozen.

The50 old reserved prompt hashes are stored in data-review.json for runner configuration. Exact and normalized response overlap with official2699/public10073/old and derived holdouts is zero in both views; exact and normalized known public/150 reconstructed official prompt intersections are zero. Internal duplicates and different-job cross-view normalized collisions are zero.

All frozen EXP043 manifest entries and independent acceptance, old source hash and EXP042 reference schema/population were checked before output. Only the two120-row CSVs are exported. No800-row pool or holdout prose is copied. This gate does not establish exhaustive semantic isolation, provider identity, or classifier utility. EXP027 related topic groups remain unsuitable for an independent-validation claim.

Preparation does not train or touch remote services. Full hashes and checks accompany this report.
