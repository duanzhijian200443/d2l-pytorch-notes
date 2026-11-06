"""Record already verified results locally. No training or remote mutations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
REL = 'experiments/EXP-010-012-stylegram-share'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    v = read(EXP/'analysis/verified.json')
    integrity = read(HERE/'provenance-check.json')
    independent = read(HERE/'independent-review.json')
    assert all(x['status']=='PASS' for x in (v, integrity, independent))
    assert independent['source_result_sha256'] == v['source_result_sha256']
    assert independent['review_script_sha256'] == sha(HERE/'independent_review.py')
    for row in independent['dense']:
        actual = next(x for x in v['dense'] if x['share']==row['share'])
        assert all(actual[k] == value for k, value in row.items())
    for row in independent['paired']:
        actual = next(x for x in v['paired'] if x['share']==row['share'])
        assert all(actual[k] == value for k, value in row.items())
    now = datetime.now(timezone.utc).isoformat()
    baseline = ROOT/'tracking/records/experiments/EXP-004.json'
    old_baseline_hash = sha(baseline)
    assert old_baseline_hash == integrity['baseline_record_sha256']
    result = read(EXP/'download-v2/results/result.json')
    state = read(EXP/'execution-state.json')
    # Retain previous observation as history rather than leaving it as current.
    history = state.setdefault('observation_history', [])
    if state['status'] != 'COMPLETE':
        history.append({k: state.get(k) for k in ('status','observed_at','phase','last_verified_log','completed_fold_shares','estimated_total_note','status_source')})
    for key in ('estimated_total_note','completed_fold_shares','last_verified_log'):
        state.pop(key, None)
    state.update(status='COMPLETE', observed_at=now, phase='verified', result_status='PASS',
                 finished_at=result['observed_at'], runtime_seconds=result['total_seconds'],
                 status_source='Official Kaggle CLI/API fixed V2 COMPLETE + original verifier + independent cache/provenance checks',
                 result_directory=REL+'/download-v2/results',
                 verification_path=REL+'/analysis/verified.json',
                 acceptance_path=REL+'/acceptance-20261101/acceptance-status.json',
                 reported_script_version_id=364307522,
                 reported_script_version_id_source='DOT archive; fixed V2 confirmed, global ID not exposed by CLI',
                 no_training_or_submission_during_acceptance=True)
    write(EXP/'execution-state.json', state)
    for ident, share in [('EXP-010',.5),('EXP-011',.25),('EXP-012',.75)]:
        path = ROOT/f'tracking/records/experiments/{ident}.json'
        record = read(path)
        metric = next(x for x in v['dense'] if x['share']==share)
        record['observed_at'] = now
        record['source']['reported_script_version_id'] = 364307522
        record['source']['reported_script_version_id_note'] = integrity['script_version_id_note']
        record['execution'].update(status='COMPLETE', observed_at=now, result_status='PASS',
                                   completed_at=result['observed_at'])
        record['environment']['versions'] = {k:v['environment'][k] for k in ('python','numpy','pandas','scipy','sklearn')}
        record['environment']['cpu_count'] = v['environment']['cpu_count']
        record['validation'].update(
            balanced_accuracy=metric['balanced_accuracy'],
            scope='Isolated stylegram fixed prompt-group-v1 5-fold development OOF; not full ATLAS20 or leaderboard',
            per_class_recall_path=REL+'/analysis/per_class.csv',
            oof_path=REL+f'/download-v2/results/oof-share{share}.csv',
            scores_path=REL+f'/download-v2/results/oof-share{share}.npy',
            leakage_audit='PASS within checked scope: per-fold features, known official-prompt links removed, 0 normalized text overlap, text-only validation clustering. Unknown semantic overlap and upstream selection bias remain unverified.',
            acceptance_path=REL+'/acceptance-20261101/acceptance-status.json')
        record['timing_note'] = 'Shared sweep EXP-010/011/012: main program 3822.3624 seconds for all 30 SVM fits; not per-share runtime. Acceptance only reads cached outputs; no retraining.'
        if share == .5:
            record['conclusion'] = '完整产物验收PASS；独立stylegram OOF BA=0.4682426050，在三组主指标中最高，保留50%设置。未证明统计显著领先，未改变ATLAS20正式0.662基线。'
        else:
            pair = next(x for x in v['paired'] if x['share']==share)
            record['validation']['paired_comparison'] = pair
            record['conclusion'] = f"完整产物验收PASS；OOF BA={metric['balanced_accuracy']:.10f}，较50%变化{100*pair['delta_ba']:+.3f}个百分点；prompt配对区间跨0，未获得替换50%的依据。不是榜单成绩。"
        record['next_action'] = '保留50%作为本成员开发对照；不按单次逐类得失事后挑份额。主svc4权重实验先解决公共池来源与验证隔离，另行登记，不从本结果自动启动。'
        for evidence in ('report.md','analysis/verified.json','acceptance-20261101/acceptance-status.json',
                         'acceptance-20261101/provenance-check.json','acceptance-20261101/independent-review.json',
                         'acceptance-20261101/download-manifest.json','download-v2/name-that-llm-stylegram-share-cv.log'):
            if REL+'/'+evidence not in record['evidence']:
                record['evidence'].append(REL+'/'+evidence)
        write(path, record)
    ideas_path = ROOT/'tracking/ideas.json'
    ideas = read(ideas_path)
    idea = next(x for x in ideas if x['id']=='IDEA-003')
    idea.update(status='planned', observed_at=now,
                phase='stylegram_complete_next_svc4_audit_planned',
                completed_experiments=['EXP-010','EXP-011','EXP-012'],
                next_action='Stylegram 0.25/0.50/0.75五折已完整验收，主指标保留0.50；下一阶段先补主svc4公共池来源/官方prompt关联/上游清洗的隔离证据，再提出单变量权重实验。尚未启动主svc4训练。')
    write(ideas_path, ideas)
    assert sha(baseline)==old_baseline_hash
    write(HERE/'acceptance-status.json', {
        'status':'PASS', 'completeness':'COMPLETE_FOR_DECLARED_STYLEGRAM_PROTOCOL',
        'observed_at':now, 'notebook_ref':'zhjianduan/name-that-llm-stylegram-share-cv/2',
        'original_verifier':'analysis/verified.json', 'provenance':'provenance-check.json',
        'independent_review':'independent-review.json', 'drive_oof_byte_identical':True,
        'counts':integrity['counts'], 'ledger_updated':['EXP-010','EXP-011','EXP-012'],
        'EXP004_unchanged':True, 'EXP004_record_sha256':old_baseline_hash,
        'training':False, 'gpu':False, 'api_generation':False, 'competition_submission':False,
        'verdict':'Keep 50% for isolated stylegram; no established gain from 25% or 75%.',
        'limits':['Not full ATLAS20 OOF or leaderboard', 'Unknown semantic public-prompt overlap and upstream selection bias not ruled out',
                  'Bootstrap conditional on fitted models and development folds, not independent repeats',
                  'Saved decision scores are not calibrated probabilities; SVM fitted weights were not saved']})
    print('Acceptance PASS; EXP-010/011/012 recorded; EXP-004 unchanged; no new training')

if __name__=='__main__':
    main()
