# Raw public data attribution

prvsiyan (2026), prvsiyan's Extended Name That LLM Dataset, fixed Kaggle dataset version 6.

Source: https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/6

The pinned source README states CC BY 4.0 for the generated passages and the author-written new prompts. This payload contains every one of the 5,400 passages with CSV split=new, preserving their original text and order. It includes no train_regen, proxy_v1, pilot_range, or new_kimi_nothink rows and no reconstructed competition prompt texts.

The CSV row_id is prvs-raw-v6:<zero-based original passages.csv row number>; id and call_id retain the source identifiers. text_sha256 is SHA-256 of the exact UTF-8 text, without normalization. The source's clean/funnel, punctuation, word-range, and markdown filters are not used for selection.

Intended evidence scope: isolated svc4 fixed-external-data development OOF. The generation protocol used official train length statistics, so this is not a strictly untouched validation source. Supplied API logs verify internal consistency only. The 5,376 texts already present in the EXP-009 main-svc4 projection must not be called new additions to ATLAS20.

The source README and input_audit.json retain provenance and limitations. No training, competition submission, or publication was performed by this preparation script.
