# EXP045 public candidate cache input review

PASS_BOUND_REORDERED_PUBLIC_CANDIDATE_MARGINS. Both candidates can be reused on the729 cached texts without retraining.

EXP027 source IDs/order differ from EXP031 blind IDs/order: all729 positions change. The frozen EXP032 truth-sidecar mapping provides a unique source_id to blind-id bridge. Exact response text,18-column label order,56 canonical prompt groups and17 true classes match after reordering. Row truth is used only for integrity review and is absent from the mathematical payload.

The output NPZ contains ids (blind panel order), source_ids (integrity-only), text_sha256, labels (18 prediction-column names), stylegram_scores and svc4_scores (729x18 finite float64 unstandardized margins). It contains no per-row model/true/prompt fields. The accepted Stylegram group IDs/mask match EXP027 predictions; after reordering the inferred partition and698 centred-row mask also match the original full-stack Stylegram inference partition.

Training provenance remains official2699 + public10073 with50% per-class Stylegram weighting, and official2699 + raw/public5400 with .35 per-row svc4 weighting and balanced class weights. This candidate svc4 is distinct from the original full-stack main svc4. No self-generated data is included.

The original Stylegram slot uses1.516 * log(clip(softmax(raw_margin),1e-9,1)). EXP027 blend_scores already contains row z-score mixing and must not substitute for raw margins in the recipe-only arm. Replace the pre-gate member contribution and replay downstream gates/routing; do not subtract/add into already-gated comp_logit.

All19 EXP027 manifest entries,18 lightweight output artifact entries and567 full-stack artifact entries were reread and hash-verified. Source hashes, permutation and output hash are in input-review.json.

This panel has already been evaluated. It covers17 true labels and56 prompts, with Gemini3.7 absent as truth but retained among18 output columns. Complete checkpoint training independence is unknown. Reuse supports an exploratory fixed-cache comparison, not new independent confirmation, clean full-stack OOF, or exact historical0.662 reproduction.
