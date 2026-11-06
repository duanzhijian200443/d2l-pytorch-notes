# EXP046 input compatibility review

PASS_LOCAL_INPUT_COMPATIBILITY_PREPARATION. Actual32-attachment runtime preflight is still pending.

EXP031 preserves the EXP004 V2 container, seven model refs and eight notebook refs. Two public data refs use private mirrors: Abdullah V2 to zhjianduan/ntl-generations V1, Lucas V7 to zhjianduan/ntl-free-proxy-generations V1. The three consumed CSVs were compared with local original files and match historical mounted sizes and full hashes. This does not establish every historical EXP004 checkpoint version/byte.

Read-only Kaggle MCP fixed-V1 inventory confirms the added Stylegram dataset contains exactly extra.csv, folds.csv, protocol.json, runner.py and stylegram_core.py. Local ZIP member hashes match the new extra-inputs.json. The original loader's six public CSV globs match none of these names. Simulating531 old paths plus five new paths leaves every model resolver result,20-member composition and allowed omission unchanged. The whole-path count and digest necessarily change.

input_preflight.py checks all528 old noncompetition paths/sizes, full hashes for398 files up to16MiB (576874614 bytes), and only sizes for130 large files (190244277214 bytes). Large weights are not rehashed; pinned versions and historical full-hash evidence remain their provenance. Official train.csv is fully hashed. Hidden test.csv is checked only for exact id,text schema and nonempty unique IDs/text; neither its public-sample size/hash nor the old sample_submission.parquet bytes are imposed.

The added dataset must appear at exactly one of datasets/zhjianduan/ntl-stylegram-cv-inputs-v1 or ntl-stylegram-cv-inputs-v1. Its five files are checked strictly by size and SHA. Unrelated extra.csv files cannot be selected. Return keys include data_dir, extra_dir, extra_path and stylegram_core_path. Receipt writes only on success and cannot target the input mount.

Root must copy input_preflight.py, extra-inputs.json and the fixed EXP031 inputs/preflight_manifest.json into the Notebook payload. Call run_preflight with the two manifest paths and /kaggle/working/input_preflight.json. Import the returned core path explicitly; never run the mounted CV runner. Preserve original global EXTRA and family gate data. The returned official DATA remains the real competition mount, never the external diagnostic panel.

All11 local fixture tests passed, including4000 hidden rows, same-size tampering, duplicate extra mounts, old shortmount support and malformed/duplicate hidden IDs. No real model files were opened by tests; no GPU, training, dependency installation, remote writes or weight download occurred.
