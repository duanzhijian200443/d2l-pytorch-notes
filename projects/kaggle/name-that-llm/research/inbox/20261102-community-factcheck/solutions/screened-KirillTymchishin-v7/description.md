# Independent LoRA + Trained27

This local candidate combines the exact public One Swap V1 recipe (measured
0.664 on its author's account) with Vladislav Lavrentev's already trained
Qwen3.8-27B GSQ3 classifier. Its own result has not been measured.

| Component | Fixed probability share |
|---|---:|
| Complete V31 / Open Pool / ATLAS20 reference | 93% |
| Completed, aligned trained GSQ27 predictions | 7% |

The reference retains 15% complementary evidence, 20% ATLAS on selected rows,
Sinkhorn strength 0.45 / target mix 0.35, and the original 37-marker typography.
The GSQ plain head is mixed before one final balance and typography pass.
No 32B model, adapter training, new data generation or parameter search runs.

Each core LoRA adapter starts from exact immutable FP16 weights on local disk.
Sequential addition/subtraction cannot restore FP16 weights reliably. This
mechanical fix has CPU evidence; an accuracy gain is still unproven.

The genuine base is saved first. It has a six-hour guard; the trained 27B expert
can use the remaining global 8.5-hour target, leaving time for final output.
Only a completed all-row expert can replace the base. These are guards, not a
measured nine-hour runtime guarantee. **This file is local and not submitted.**


## 1. Exact public fixture preview

Only exact IDs, text lengths and fingerprints skip model imports. Every other input runs genuine inference.


### Exact fixture identity and format preview

Source: `preview.py`.


### Shared-prompt Gaussian readout

Source: `gauss.py`.


### Full public core, Gaussian views and ATLAS20

Source: `infer.py`.


### Original complementary ensemble

Source: `comp_stack.py`.


### Resolve the pinned public reference assets

Source: `setup.py`.


### Restore each LoRA member from exact frozen weights

Source: `immutable_weights.py`.


### Align ID and class order; mix 93/7 once

Source: `friend27_mix.py`.


### Trained27 worker: local_stage.py

Source: `local_stage.py`.


### Trained27 worker: run_stage.py

Source: `run_stage.py`.


### Bounded base-first execution and explicit completion evidence

Source: `pipeline.py`.


### Trained27 worker: gsq/gsq_infer_model.py

Source: `gsq/gsq_infer_model.py`.


### Trained27 worker: gsq/gsq_infer_layers.py

Source: `gsq/gsq_infer_layers.py`.


### Trained27 worker: gsq/delta_turing.py

Source: `gsq/delta_turing.py`.


### Trained27 worker: gsq/adapter.py

Source: `gsq/adapter.py`.


### Trained27 worker: gsq/infer.py

Source: `gsq/infer.py`.


### Trained27 worker: stage27.py

Source: `stage27.py`.


### Original V31 balancing and hierarchical typography

Source: `v31_math.py`.


## 2. Fixed trained-27B input contract

The classifier keeps its original 64 layers, rank-16 tail-eight adapter, EOS 248044, tokenizer, plain head, quantized base and training-completion checks. A 27B adapter is never loaded into a different model family.


## 3. Real current-test inference

A failed or incomplete optional expert is reported as base-only; no partial prediction or benchmark weight is accepted.


## Credits, licences and evidence

The public baseline is Zbigniew Kirchner's [One Swap V1](https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public?scriptVersionId=364073396), derived from omerfarukmerey's ATLAS20, mercurydust's V31, Roman Tamrazov's typography and Dhruv Tibarewal's nlm recipe. The trained GSQ27 model and its exact runtime come from [Vladislav Lavrentev](https://www.kaggle.com/code/lightningv08/0-662-qwen3-8-27b-lora-weighted-ensemble). All original authorship and component licences remain applicable: public code Apache 2.0, the Open Pool CC BY-SA 4.0, Llama 3.1 Community License (**Built with Llama**), and individual model/data notices in the pinned inputs. Additional source credits remain in the visible modules.

`friend27-completion.json` states whether the trained expert actually completed.
The exact sample preview proves format only. CPU mechanics, real hardware
feasibility, paired quality and a complete hidden runtime are distinct checks.

