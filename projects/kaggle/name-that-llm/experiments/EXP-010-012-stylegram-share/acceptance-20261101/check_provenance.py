"""Reconcile Drive backups, fixed-version downloads and original verifier evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))

def main():
    src = EXP / 'download-v2/results'
    manifest = read(HERE/'download-manifest.json')
    assert manifest['file_count'] == 251
    for f in manifest['files']:
        p = EXP/'download-v2'/f['path']
        assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256'], f['path']
    counts = {'oof_csv': len(list(src.glob('oof-share*.csv'))),
              'oof_npy': len(list(src.glob('oof-share*.npy'))),
              'fold_npz': len(list(src.glob('fold*-share*.npz'))),
              'sparse_npz': len(list(src.glob('sparse-*.npz')))}
    assert counts == dict(oof_csv=3, oof_npy=3, fold_npz=15, sparse_npz=225)
    provenance = read(HERE/'drive-provenance.json')
    compared = []
    for f in provenance['files']:
        p = HERE/f['local_path']
        assert p.read_bytes() == (src/f['name']).read_bytes(), f['name']
        compared.append({**f, 'bytes': p.stat().st_size, 'sha256': sha(p),
                         'kaggle_byte_identical': True})
    downloaded_code = HERE/'source-v2/name-that-llm-stylegram-share-cv.py'
    assert downloaded_code.read_bytes() == (EXP/'kernel/bootstrap.py').read_bytes()
    meta = read(HERE/'source-v2/kernel-metadata.json')
    assert meta['is_private'] and not any(meta[k] for k in ('enable_gpu','enable_tpu','enable_internet'))
    assert meta['dataset_sources'] == ['zhjianduan/ntl-stylegram-cv-inputs-v1']
    assert meta['competition_sources'] == ['name-that-llm']
    protocol = read(EXP/'payload/protocol.json')
    assert sha(EXP/'payload/protocol.json') == '3983ceb881750b9b138ee59a3c03ba1acb2dd0f04c2b56bbe96ec682446ce50c'
    for name, digest in protocol['input_hashes'].items():
        assert sha(EXP/'payload'/name) == digest, name
    assert sha(ROOT/'validation/prompt-group-v1/folds.csv') == protocol['input_hashes']['folds.csv']
    assert sha(ROOT/'train.csv') == protocol['train_sha256']
    expected_verifier = 'ea6fa44926c036da45e694b3b2fe1b64f32ecb0b44200f5496eb9a4c1d907bc2'
    assert sha(EXP/'verify_results.py') == expected_verifier
    verified = read(EXP/'analysis/verified.json')
    assert verified['status'] == 'PASS'
    assert verified['source_result_sha256'] == sha(src/'result.json')
    partial = read(EXP/'acceptance-20261101-partial/csv_diagnostics.json')
    for row in partial['dense']:
        actual = next(x for x in verified['dense'] if x['share'] == row['share'])
        assert actual['balanced_accuracy'] == row['ba'] and actual['correct'] == row['correct']
    for row in partial['paired']:
        actual = next(x for x in verified['paired'] if x['share'] == row['share'])
        assert all(actual[key] == value for key, value in row.items())
    for name in ('result.json', 'environment.json'):
        # DOT copied browser JSON text; insignificant trailing newline may differ.
        archived = ROOT/'2026-11-01_Stylegram-V2_run364307522_三组结果归档'/name
        assert read(archived) == read(src/name), name
    log = (EXP/'download-v2/name-that-llm-stylegram-share-cv.log').read_text(encoding='utf-8')
    events = json.loads(log)
    messages = ''.join(x.get('data','') for x in events)
    assert messages.count('FIT COMPLETE ') == 15
    assert messages.count('EXPERIMENT COMPLETE ') == 1
    assert 'ConvergenceWarning' not in messages and 'Traceback (most recent call last)' not in messages
    assert not (src/'failure.json').exists()
    assert read(src/'state.json')['status'] == 'COMPLETE'
    baseline = ROOT/'tracking/records/experiments/EXP-004.json'
    assert sha(baseline) == 'f0cef27c942819d1f3ca828e579e19419752f68024041792613bac508a79b50f'
    result = {'status': 'PASS', 'observed_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'fixed V2 source/input/output integrity and Drive backup reconciliation; no training',
              'notebook_ref': manifest['notebook_ref'], 'remote_status': 'COMPLETE',
              'script_version_id_reported_by_dot': 364307522,
              'script_version_id_note': 'Retained DOT archive identifier; official CLI fixed-v2 response does not expose global scriptVersionId.',
              'source_exact': True, 'source_sha256': sha(downloaded_code),
              'original_verifier_unchanged': True, 'original_verifier_sha256': expected_verifier,
              'input_hashes_verified': True, 'counts': counts, 'all_download_hashes_verified': 251,
              'drive_files': compared, 'dot_csv_analysis_matches': True,
              'log_fit_complete_records': 15, 'log_experiment_complete_records': 1,
              'baseline_record_unchanged': True, 'baseline_record_sha256': sha(baseline),
              'training_or_submission_performed': False}
    (HERE/'provenance-check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':'PASS', 'counts':counts, 'drive_byte_matches':len(compared), 'source_exact':True}))

if __name__ == '__main__':
    main()
