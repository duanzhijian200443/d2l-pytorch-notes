# Qwen3.8-27B LoRA + Spectral Ensemble

This candidate combines the public multi-model attribution method with a separately trained Qwen3.8-27B GSQ3 LoRA readout. The own-model probability share is **7.5%**, after the ATLAS mixture and before marginal balancing.

The public reference is [mercurydust's 0.660 submission, version 2](https://www.kaggle.com/code/mercurydust/name-that-llm-submission?scriptVersionId=363008892). Its leaderboard score belongs to that reference version; this candidate requires its own competition evaluation.

The core includes source-weighted, deduplicated four-view SVC, matched-clean TF-IDF logistic regression at weight 0.25, distorted-text logistic regression at weight 0.10, and the reference V5/two-stage frozen model heads. ModernBERT is excluded from this reference recipe. Core scores divide by the number of active vote matrices before probability conversion.

The complementary public stack contributes 15% of the first probability mixture. Its family gates rerank siblings while preserving family probability mass. The DeBERTa token-attention head remains active; the Qwen3.5 token-attention head is disabled to match the scored reference's effective backbone availability. ATLAS contributes 20% on rows selected from the core/complementary confidence margin.

Marginal balancing uses strength 0.45 and a target containing 65% of the predicted marginal plus 35% of the smoothed official-training prior. The final writer applies 37-marker hierarchical Beta-Bernoulli typography with confidence, passage-length and marker-coverage gates. It applies no final hard punctuation or Hungarian override.

The own GSQ adapter uses the complete 119899-row public corpus followed by two official-data epochs. Its inference retains every text token, uses batch size one, and assigns rows across two T4 GPUs. The established fixed attachment paths, isolated GPU processes and sequential temporary checkpoint staging are retained.

Credits: Roman Tamrazov for the parent ensemble and Bayesian typography; mercurydust for the 0.660 integration and revised gates, source weighting, adaptive balancing and hierarchical typography; Joseph Adamski for ATLAS; the original public model, feature and adapter authors credited in the visible modules. Algorithm source is included below; datasets contain weights, data and fitted numeric states.

This variant adopts the 20% ATLAS mixture from [Omer Faruk Merey, scored version 1 (0.663)](https://www.kaggle.com/code/omerfarukmerey/0-663-0-660-v31-atlas-20?scriptVersionId=363213496). ATLAS selection and predictions are unchanged; only its probability share changes. Its source's wall-clock cutoff is not used here. The already retained sequential checkpoint staging and GPU pooling serve the same performance purpose without changing the model recipe.


## Exact public sample preview

The exact three public sample records produce a CPU-only format preview. Their IDs, byte lengths and text fingerprints must all match. Every other test input runs the full model pipeline.


## Fixed inputs

Model weights and numerical fitted states use explicit input paths. The core sparse cache contains source-weighted, deduplicated four-view SVC, matched-clean TF-IDF and distorted-text logistic regression. The complementary cache contains stylegram plain and centred heads plus Claude, GPT, Gemini and GLM family gates. These states score test passages without refitting. The frozen dense readouts retain their official-training feature lineage; Gaussian states preserve feature specification, canonical class order and ordered adapter chains.

The numerical cache is distributed as twelve lossless binary file parts for reliable attachment upload. Before inference, the visible preparation cell joins these data parts, verifies archive paths and semantic metadata, and extracts the fitted numerical states. No model algorithm or runtime source is encoded in the parts. The exact public-format preview skips this preparation.


## Reference grouping and CPU readouts

Content groups define the core centred and Gaussian heads. The three sparse score matrices and model score matrices each enter the active-vote divisor before the first probability mixture.

## V5 and two-stage feature extraction


## Complementary stack and family gates

The 15% complementary mixture consumes all four fitted style family gates and the Claude LoRA gate before saving complementary logits.

## ATLAS and custom model architectures


## Full-text GSQ3 inference


## Sequential stages and final mixture

Core and complementary probabilities mix at 85/15. Selected rows then receive 20% ATLAS. The own full-text GSQ head contributes 7.5% to every row after ATLAS. Adaptive balancing and hierarchical typography follow this mixture.

## Input preparation and execution

Each core model's base, adapter, head and metadata files are copied from explicit immutable inventories to temporary storage and removed after feature extraction releases the model. Hidden-state tensors are released after pooling each batch and before an overflow retry, so one batch's states do not overlap the next forward or adapter update. The six V5 checkpoints retain their original order. Every GPU child exits before the next stage.

The complementary GPU child and its fitted readouts are active at 15% mixture share. Each complementary backbone is copied sequentially to temporary storage from its fixed file inventory. A LoRA trunk remains staged across its adapters, while each adapter and tokenizer directory is staged and released in the original order. The five family gates update complementary logits before they are saved. DeBERTa token-attention is active and stages its backbone and head; Qwen3.5 token-attention is disabled. ATLAS then stages its own base and adapter bundle. Models are released before temporary files are removed.

The GSQ base is then copied once to temporary storage, shared read-only by two isolated T4 workers and removed after both workers finish. GSQ retains every passage and its complete token sequence.

Final probabilities receive adaptive marginal balancing and the 37-marker hierarchical typography correction. The last writer saves the corrected per-passage argmax and verifies the saved IDs, row order and label schema.
