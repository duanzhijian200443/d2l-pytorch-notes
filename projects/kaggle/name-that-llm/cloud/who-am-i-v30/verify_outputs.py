"""Validate downloaded Kaggle outputs against frozen V30 inputs and sample predictions."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, required=True)
parser.add_argument('--mode', choices=['pincheck', 'full'], required=True)
args = parser.parse_args()
out = args.directory.resolve()
lock = json.loads((ROOT / 'input-lock.json').read_text(encoding='utf-8'))
pinned = json.loads((out / 'pinned_inputs_verified.json').read_text(encoding='utf-8'))
assert pinned['status'] == 'PASS'
actual = {item['input']: item['sha256'] for item in pinned['files']}
assert actual == lock['pinned_csv_sha256'], (actual, lock['pinned_csv_sha256'])
environment = json.loads((out / 'reproduction_environment.json').read_text(encoding='utf-8'))
assert environment['mode'] == args.mode
log = (out / 'name-that-llm-who-am-i-v30-repro.log').read_text(encoding='utf-8', errors='replace')
if log.lstrip().startswith('['):
    log = ''.join(event['data'] for event in json.loads(log))
assert 'PapermillExecutionError' not in log and 'Traceback (most recent call last)' not in log
verification = {'status': 'PASS', 'mode': args.mode, 'historical_csv_hashes_match': True,
                'public_score_verified': False}
if args.mode == 'pincheck':
    assert 'COMPLEMENTARY COMPONENT INPUTS PASS' in log
    verification['complementary_components_present'] = True
else:
    import pyarrow.parquet as pq
    report = json.loads((out / 'reproduction_result.json').read_text(encoding='utf-8'))
    assert report['status'] == 'COMPLETE' and report['upstream_version'] == 30
    assert environment['gpu_count'] == 2
    assert all('T4' in name for name in environment['gpu_names'])
    submission = pq.read_table(out / 'submission.parquet')
    assert set(submission.column_names) == {'id', 'model'}
    rows = submission.to_pylist()
    reference = json.loads((ROOT / 'outputs/upstream-v30-targets/sample-predictions.json').read_text(encoding='utf-8'))
    original_log = json.loads((ROOT / 'outputs/upstream-v30-targets/name-that-llm-who-am-i.log').read_text(encoding='utf-8'))
    original_text = ''.join(event['data'] for event in original_log)
    member_pattern = r'^\s*\+\s+(\S+)\s+weight\s+([0-9.]+)'
    original_members = dict(re.findall(member_pattern, original_text, re.M))
    actual_members = dict(re.findall(member_pattern, log, re.M))
    assert original_members and actual_members == original_members, (actual_members, original_members)
    assert 'SKIP lora24_public: checkpoint not attached' in log
    assert len(rows) == report['test_rows'] == len(reference)
    assert [row['id'] for row in rows] == [row['id'] for row in reference]
    differences = [{'id': row['id'], 'actual': row['model'], 'upstream': expected['model']}
                   for row, expected in zip(rows, reference) if row['model'] != expected['model']]
    verification.update({'sample_rows': len(rows), 'sample_prediction_differences': differences,
                         'logged_complement_members_match': True,
                         'submission_sha256': hashlib.sha256((out / 'submission.parquet').read_bytes()).hexdigest()})
    if differences:
        verification['status'] = 'SAMPLE_PREDICTIONS_DIFFER'
(out / 'local_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(verification, indent=2))
if verification['status'] != 'PASS':
    raise SystemExit(1)
