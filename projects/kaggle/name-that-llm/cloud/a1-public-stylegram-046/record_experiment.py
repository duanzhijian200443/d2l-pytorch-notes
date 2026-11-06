"""Update only EXP046's authoritative local index from its immutable package/control state."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'experiments/EXP-046-atlas20-a1-submission'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rel(p):return p.relative_to(ROOT).as_posix()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    existing=read(ROOT/'tracking/records/experiments/EXP-046.json')
    if (existing.get('submission') or {}).get('status')=='COMPLETE':
        print('Completed formal result preserved; refresh it through the read-only ledger workflow.')
        return
    now=datetime.now(timezone.utc).isoformat()
    package=read(EXP/'metrics/package-preflight.json')
    metadata=read(HERE/'kernel/kernel-metadata.json')
    config={'schema_version':1,'experiment_id':'EXP-046','baseline_id':'EXP-004','candidate_from':'EXP-045 A1',
        'scope':'single_private_code_competition_submission','recipe':{'official_rows':2699,'official_groups':150,
        'extra_rows':10073,'extra_groups':694,'extra_sha256':'1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e',
        'extra_share':.5,'C':.5,'dual':True,'max_iter':4000,'seed':42,'dtype':'float64',
        'post_margin_normalization':False,'self_generated_data':False,'format_cleaning':False},
        'ensemble':{'atlas_w':.20,'comp_w':.15,'sinkhorn':.45,'sinkhorn_target_mix':.35,'typography_scale':1.5},
        'preserve':['original other main/comp models','original family-gate training pool','original main svc4','original dynamic ATLAS routing'],
        'validation':{'scope':'Kaggle hidden rerun; no local panel uploaded','prior_local_A0_A1_macro_recall':.6010754296632905,
        'prior_local_prediction_changes':0,'prior_local_rows':729,'prior_local_is_clean_oof':False},
        'environment':{'platform':'Kaggle','accelerator':'T4 x2','internet':False,'container':metadata['docker_image']},
        'budget':{'save_only':1,'quick_commit_runs':1,'formal_submissions':1,'max_hidden_runtime_seconds':32400,'api_usd':0,'colab_cu':0},
        'authority':'User explicitly requested Kaggle submission of the annotated A1 comparison in this chat.',
        'quick_commit':'Small public placeholder only when competition-rerun environment key is absent; actual hidden rerun always full pipeline.'}
    cp=EXP/'config.json'
    if cp.exists() and read(cp).get('experiment_id')=='EXP-046':
        assert read(cp)==config,'Frozen experiment configuration changed'
    else:save(cp,config)
    source_files=[HERE/'kernel/a1.ipynb',HERE/'kernel/kernel-metadata.json',HERE/'build_package.py',HERE/'runtime_preflight.py',
        HERE/'run_receipts.py',HERE/'control.py',EXP/'a1_stylegram.py',EXP/'patch_comp.py',EXP/'input_preflight.py',EXP/'extra-inputs.json',EXP/'config.json']
    manifest={'schema_version':1,'experiment_id':'EXP-046','status':'FROZEN_LOCAL_PACKAGE',
        'files':[{'path':rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in source_files],
        'attachment_count':32,'attachments':{k:metadata[k] for k in ['dataset_sources','kernel_sources','model_sources','competition_sources']},
        'baseline_source':{'path':'reviews/own-scheme-readiness-20261030/baseline-v2.ipynb','sha256':package['baseline_sha256']},
        'runtime_content_check':'398 small original files+5 extra files+official train hashed;130 large weights size checked and pinned; hidden test is schema checked, not locked to public rows.',
        'historical_limitation':'The EXP031 pinned mirrors match their source CSV hashes, but every historical EXP004 weight byte is not certified.'}
    save(EXP/'manifest.json',manifest)
    path=ROOT/'tracking/records/experiments/EXP-046.json'; record=read(path)
    record.update(observed_at=now,hypothesis='A1已整理公共池的Stylegram配方可能在Kaggle隐藏集产生收益；本地729条最终类别无变化，不预设榜单提升。',
        changes=['仅替换Stylegram为官方2699+固定公共10073/share0.5；原family gates及其他集成设置保留。',
        '新增固定输入/环境预检、hidden rerun强制完整路径及comp子进程失败传播。'],
        environment=dict(config['environment'],seed=42),conclusion='候选本地构建与数值/输入测试已通过；正式分数以绑定提交的实时结果为准。',
        limitations=manifest['historical_limitation']+' Public quick commit只检查环境/输入/小合成数值，不证明完整隐藏推理已成功。',
        next_action='按固定版本完成快速提交运行核验，正式提交一次并查询结果。')
    record['source'].update(notebook_ref=metadata['id'],local_path=rel(HERE/'kernel/a1.ipynb'),sha256=sha(HERE/'kernel/a1.ipynb'))
    record['execution'].update(scope='code_submission_candidate',status='READY_LOCAL',observed_at=now)
    state_path=HERE/'execution-state.json'
    if state_path.exists():
        state=read(state_path);record['execution']['control_state_path']=rel(state_path)
        record['execution']['status']=state.get('status',state.get('phase','CONTROL_STATE_AVAILABLE'))
        record['source']['notebook_version']=state.get('run_version',state.get('saved_version'))
        record['source']['script_version_id']=state.get('script_version_id')
        observed=state.get('last_status',{})
        if observed.get('version')==state.get('run_version') and observed.get('status'):
            record['execution'].update(status=observed['status'],observed_at=observed['observed_at'])
    submission_path=HERE/'submission-ack.json'
    if submission_path.exists():
        record['submission']=read(submission_path)
        record['execution']['status']='SUBMITTED'
        record['execution']['observed_at']=record['submission']['observed_at']
        record['conclusion']='已通过私有V2快速运行及云端输入/环境/格式核验，正式提交一次；当前尚无新分数，不宣称提升。'
        record['next_action']='等待Kaggle隐藏集重跑完成，核对当前submission ID的正式分数。'
    record['inputs'].update(manifest_path=rel(EXP/'manifest.json'),validation_split_id=None,content_hashes_verified=False)
    record['validation'].update(scope='Prior EXP045 exploratory cached panel; not a leaderboard score or clean OOF.',balanced_accuracy=None)
    ev=[rel(p) for p in [EXP/'config.json',EXP/'manifest.json',EXP/'report.md',EXP/'metrics/package-preflight.json',EXP/'metrics/submission-policy.json',
        EXP/'metrics/input-review.json',EXP/'metrics/comp-patch.json']]
    ev += [rel(p) for p in [EXP/'metrics/final-review.json',EXP/'metrics/post-submit-review.json',HERE/'execution-state.json',submission_path,
        HERE/'quick-commit-verification.json',HERE/'submission-binding.json'] if p.exists()]
    record['evidence']=ev
    save(path,record)
    (EXP/'report.md').write_text('''# EXP046：A1公共数据Stylegram正式提交

本实验接续EXP045 A1：在完整ATLAS20中，只替换Stylegram为已冻结公共数据配方。保留EXP004其余模型与融合设置，不加入自制120条、不清除文本换行加粗、不采用A2/A3。

本地729条探索集上A0/A1均为0.601075宏召回、488条正确，最终类别无变化；这不是干净OOF或正式榜单分数。此次向Kaggle提交代码，在隐藏集重新计算所有需要的模型预测，不上传本地729条预测缓存。

固定输入为EXP031的31附件加公共Stylegram输入V1，合计32。两个历史公共CSV镜像已核对字节，不能宣称所有历史模型权重都逐字节复现。详细参数见config.json，源代码/附件绑定见manifest.json，独立审查见metrics。

快速保存运行使用小公开样例占位，仅用于云端输入、T4×2环境、数值自检与输出格式检查。只要KAGGLE_IS_COMPETITION_RERUN键存在，无论它的值或样本数如何，都执行完整算法。完整运行仍遵守9小时上限。

当前远端进展与具体版本以cloud/a1-public-stylegram-046/execution-state.json和提交回执为准。未获得绑定的COMPLETE分数前，不宣称提升。
''',encoding='utf-8')
    if submission_path.exists():
        sub=read(submission_path)
        with (EXP/'report.md').open('a',encoding='utf-8') as stream:
            stream.write('\n## 已提交\n\n正式提交ID：'+str(sub['id'])+'；自有Notebook V2；scriptVersionId：'+str(sub['script_version_id'])+'。\n\n')
            stream.write('提交时间：'+sub['submitted_at']+'。当前状态：'+sub['status']+'；Public分数尚未返回。\n\n')
            stream.write('[固定提交版本]('+sub['url']+')。详见submission-ack.json及submission-binding.json。\n')
    print('EXP046 record updated:',record['execution']['status'])

if __name__=='__main__':main()
