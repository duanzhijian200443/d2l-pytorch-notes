"""Evaluate only the original cell12 model/command construction, never run it."""
import json, os
from pathlib import Path
values = json.loads(Path(os.environ['NTL_SETUP_RESULT']).read_text(encoding='utf-8'))
values['COMP_SEC'] = float(os.environ['NTL_COMP_SECONDS'])
values['json'] = json
exec(compile("W_N14, W_N8 = (1.0, 0.5)\nL = [2, 4, 6, 8, 12, 16, 20, 24]\nMODELS = [{'name': 'q14v5', 'type': 'l24', 'path': B14, 'adapters': AQ14V5, 'keep': 27, 'layers': L, 'w': 3.0, 'bs': 8, 'max_len': 384, 'twin': 0.35, 'gauss': 0.7}, {'name': 'q8v5', 'type': 'l24', 'path': B8, 'adapters': AQ8V5, 'keep': 24, 'layers': L, 'w': 2.0, 'bs': 16, 'max_len': 384, 'twin': 0.35, 'gauss': 0.44, 'dp': True}, {'name': 'll8v5', 'type': 'l24', 'path': BL, 'adapters': ALL8V5, 'keep': 24, 'layers': L, 'w': 1.0, 'bs': 16, 'max_len': 384, 'twin': 0.35, 'dp': True}, {'name': 'n14b', 'type': 'l24', 'path': B14, 'adapters': [N14B], 'keep': N14BC['keep'], 'layers': N14BC['layers'], 'w': 2.5, 'bs': 8, 'max_len': N14BC['max_len'], 'twin': 0.35, 'no_final_norm': True, 'eos_id': N14BC['eos_id']}, {'name': 'q25_14', 'type': 'l24', 'path': B25_14, 'adapters': [Q25_14], 'keep': Q25_14C['keep'], 'layers': Q25_14C['layers'], 'w': 1.0, 'bs': 8, 'max_len': Q25_14C['max_len'], 'twin': 0.35, 'no_final_norm': True, 'eos_id': Q25_14C['eos_id']}, {'name': 'deb', 'type': 'enc', 'ckpt': DEB, 'w': 1.0}]\nimport pandas as pd\nATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)\nimport subprocess, sys\ncmd = [sys.executable, 'infer.py', '--data', str(DATA), '--models', json.dumps(MODELS), '--perpassage', '1', '--extra', str(EXTRA), '--extra_w', '0.5', '--tfidf_w', '0.25', '--distort_w', '0.10', '--svc_w', '1.5', '--pool', str(POOL), '--adapt_w', '1.0', '--adapt_thr', '0.15', '--sinkhorn', '0.45', '--sinkhorn_target_mix', '0.35', '--post', '0', '--atlas_dir', str(ATLAS), '--atlas_base', str(B8), '--atlas_w', '0.20', '--atlas_margin', '0.3', '--atlas_bs', '4', '--atlas_chunk', '3', '--atlas_budget', str(max(180, ATLAS_BUDGET)), '--comp_path', '/kaggle/working/comp_logit.npy', '--comp_w', '0.15', '--out', '/kaggle/working/submission.parquet', '--cache', '/kaggle/working/cache']", 'frozen-baseline-cell12-prefix.py', 'exec'), values)
output = {'models': values['MODELS'], 'command': values['cmd'],
          'original_atlas_budget': values['ATLAS_BUDGET'],
          'original_atlas_cli_budget': float(values['cmd'][values['cmd'].index('--atlas_budget')+1]),
          'comp_seconds': values['COMP_SEC']}
destination = Path(os.environ['NTL_MODEL_ARGS_RESULT'])
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
temporary.replace(destination)
print('CAPTURE_MODEL_ARGS_READY', flush=True)
