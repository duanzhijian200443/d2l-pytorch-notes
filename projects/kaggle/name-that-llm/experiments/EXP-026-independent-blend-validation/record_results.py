"""Archive the verified fixed V2 result and update only this experiment's records."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLOUD = ROOT / 'cloud/fixed-blend-external'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(path):
    return path.relative_to(ROOT).as_posix()

def main():
    accepted = read(CLOUD / 'analysis/verified.json')
    remote = read(CLOUD / 'complete-v2-readback.json')
    meta = remote['info']['metadata']
    assert accepted['status'] == 'PASS' and accepted['stable_artifacts_verified'] == 18
    assert remote['status']['status'] == 'COMPLETE' and meta['current_version_number'] == 2
    assert meta['is_private'] and not meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['dataset_data_sources'] == ['zhjianduan/ntl-fixed-blend-external-inputs-026']
    assert meta['competition_data_sources'] == ['name-that-llm']
    assert remote['info']['blob']['source'] == (CLOUD / 'kernel/bootstrap.py').read_text(encoding='utf-8')
    manifest = read(HERE / 'manifest.json')
    for item in manifest['inputs'] + manifest['code'] + [manifest['config'], manifest['protocol']]:
        assert digest(ROOT / item['path']) == item['sha256'], item['path']
    result = read(CLOUD / 'outputs-v2/results/result.json')
    guard = read(CLOUD / 'outputs-v2/results/guard.json')
    assert result['status'] == guard['status'] == 'COMPLETE'
    assert digest(CLOUD / 'verify_outputs.py') == accepted['verifier_sha256']
    stamp = datetime.now(timezone.utc).isoformat()
    archive = []
    prediction_names = {'component-and-blend-scores.npz', 'stylegram-scores.npz', 'predictions.csv', 'confusion.csv'}
    files = list((CLOUD / 'outputs-v2').rglob('*')) + list((CLOUD / 'analysis').glob('*'))
    for source in sorted(p for p in files if p.is_file()):
        if source.parent == CLOUD / 'analysis':
            target = HERE / 'metrics' / source.name
        elif source.name in prediction_names:
            target = HERE / 'predictions' / source.name
        elif source.suffix == '.log' or source.name == '.run.lock':
            target = HERE / 'logs' / source.name
        else:
            target = HERE / 'metrics/runtime' / source.name
        data = source.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists() or target.read_bytes() == data, f'Existing archive differs: {target}'
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
        assert digest(target) == digest(source)
        archive.append({'source': rel(source), 'archive': rel(target), 'sha256': digest(target), 'bytes': len(data)})
    write(HERE / 'metrics/archive-manifest.json', {'archived_at_utc': stamp, 'experiment': 'EXP-026',
        'notebook_ref': meta['ref'], 'version': 2, 'cli_download_exit_code': 0,
        'download_command': ['kaggle', 'kernels', 'output', meta['ref'] + '/2', '-p', 'cloud/fixed-blend-external/outputs-v2', '-q'],
        'complete_readback': rel(CLOUD / 'complete-v2-readback.json'), 'files': archive})
    write(HERE / 'metrics/data-use.json', {'experiment': 'EXP-026', 'observed_at_utc': stamp,
        'evaluation_sha256': accepted['evaluation_sha256'], 'rows': 5200, 'groups': 1300,
        'use': 'First scored external check of pre-fixed EXP-024 alpha=0.10 full-data refit recipe',
        'selection_fixed_before_fit_and_score': True, 'retuned_on_external_result': False,
        'future_status': 'Inspected evaluation data; no longer untouched for later candidate selection or confirmation',
        'not_full_18_class_validation': True})
    write(HERE / 'metrics/verification-tests.json', {'observed_at_utc': stamp, 'status': 'PASS', 'tests': 12,
        'command': 'python -B -X utf8 -m unittest discover -s cloud/fixed-blend-external -p test_verify_outputs.py -v',
        'verifier_sha256': digest(CLOUD / 'verify_outputs.py'), 'tests_sha256': digest(CLOUD / 'test_verify_outputs.py'),
        'note': 'Includes near-tie argmax flip rejection; verifier and tests were corrected before real acceptance; frozen model protocol unchanged.'})
    comparison = accepted['recomputed_comparison']
    metrics = accepted['recomputed_metrics']
    record_path = ROOT / 'tracking/records/experiments/EXP-026.json'
    record = read(record_path)
    record['observed_at'] = stamp
    record['execution'].update(status='COMPLETE', result_status='PASS', observed_at=stamp,
        completed_at=guard['observed_at'], accepted_at=accepted['verified_at_utc'])
    record['validation'].update(covered_label_macro_recall=metrics['blend']['macro_recall'],
        component_metrics=metrics, comparison=comparison,
        per_class_recall_path=rel(HERE / 'metrics/per_class.csv'),
        external_predictions_path=rel(HERE / 'predictions/predictions.csv'),
        acceptance_path=rel(HERE / 'metrics/verified.json'),
        independent_review_path=rel(HERE / 'metrics/independent-review.json'))
    assert record['validation']['balanced_accuracy'] is None and record['submission'] is None
    record['timing_note'] = 'Runner 386.083s; guard 389.756s (6m29.8s), sampled peak RSS 5.549 GiB; 3 SVM fits 213.845s total. Excludes queue/setup/download/verification. CPU only.'
    record['conclusion'] = '外部四类检查验收PASS：固定10%融合宏召回0.199230769，相对Stylegram0.188846154提高1.0385个百分点，修正73错、新增19错；配对prompt 95%区间[+0.6923,+1.4038]个百分点。仅支持该四类来源上的增益，保留候选。'
    record['limitations'] = '仅四GPT标签/Dolly来源/固定生成设置及已核known-ancestry与exact隔离；语义重叠与服务标签未独立认证。全量重训轻量成员，不是完整ATLAS20、18类验证或榜单结果；区间不含重新训练不确定性，外部数据已被查看。'
    record['next_action'] = '保留固定10%候选；覆盖其余14类/其他来源，并在可核实训练隔离的完整集成预测上检验增量。该外部集不再作为后续调参后的新独立证明；不改EXP-004正式0.662、不提交。'
    extra = [HERE / 'record_results.py', HERE / 'metrics/archive-manifest.json', HERE / 'metrics/data-use.json',
        HERE / 'metrics/verification-tests.json', HERE / 'metrics/verified.json', HERE / 'metrics/independent-review.json',
        HERE / 'metrics/per_class.csv', HERE / 'metrics/comparison.csv', HERE / 'predictions/predictions.csv',
        CLOUD / 'verify_outputs.py', CLOUD / 'test_verify_outputs.py', CLOUD / 'complete-v2-readback.json']
    for path in extra:
        if rel(path) not in record['evidence']:
            record['evidence'].append(rel(path))
    write(record_path, record)
    state_path = CLOUD / 'execution-state.json'
    state = read(state_path)
    state.update(status='COMPLETE', result_status='PASS', observed_at=stamp, phase='verified',
        completed_at=guard['observed_at'], accepted_at=accepted['verified_at_utc'],
        completed_fits_observed=['stylegram_plain', 'stylegram_centred', 'svc4'],
        readback_path=rel(CLOUD / 'complete-v2-readback.json'), output_directory=rel(CLOUD / 'outputs-v2'),
        verification_path=rel(HERE / 'metrics/verified.json'),
        guard_seconds=guard['elapsed_seconds'], peak_rss_bytes=guard['peak_rss_bytes'],
        next_action='Terminal: no rerun, new GPU, tuning, or submission')
    write(state_path, state)
    ideas_path = ROOT / 'tracking/ideas.json'
    ideas = read(ideas_path)
    for idea in ideas:
        if idea['id'] == 'IDEA-004':
            idea.update(phase='fixed_blend_external_four_class_verified', observed_at=stamp,
                next_action=record['next_action'])
            if 'EXP-026' not in idea.setdefault('completed_experiments', []):
                idea['completed_experiments'].append('EXP-026')
            idea['active_experiments'] = [x for x in idea.get('active_experiments', []) if x != 'EXP-026']
    write(ideas_path, ideas)
    print(json.dumps({'status': 'RECORDED', 'archived_files': len(archive), 'comparison': comparison}, ensure_ascii=False))

if __name__ == '__main__':
    main()
