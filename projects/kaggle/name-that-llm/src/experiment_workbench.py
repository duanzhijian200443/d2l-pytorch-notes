"""Thin project entrypoint: read-only cloud observations and frozen local jobs."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from workbench_core import *

OPERATIONS=['list','doctor','status','collect','verify','report','check','prepare','run-local']
JOB_SPECS={
    ('EXP-049','prepare'):('cloud/a1-sinkhorn030-049/prepare_verify.py',['prepare'],120),
    ('EXP-049','verify-package'):('src/workbench_local.py',['verify-package','EXP-049'],120),
    ('EXP-049','run-local'):('experiments/EXP-049-a1-sinkhorn030-ready/run_local.py',[],300),
    ('EXP-049','verify-results'):('experiments/EXP-049-a1-sinkhorn030-ready/verify_results.py',[],300),
    ('EXP-050','verify-results'):('src/open_pool_audit.py',['--verify','--config','experiments/EXP-050-open-pool-v13-increment-audit/config.json'],300),
}

def child_env(cfg):
    env=dict(os.environ)
    env.pop('PYTHONOPTIMIZE',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',CUDA_VISIBLE_DEVICES='-1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    proxy=cfg.get('proxy')
    if proxy:
        require(re.fullmatch(r'http://127\.0\.0\.1:\d{1,5}',proxy) is not None,'INVALID_LOCAL_PROXY')
        env['HTTP_PROXY']=env['HTTPS_PROXY']=proxy
    return env

def interpreter(cfg,role):
    value=cfg['interpreters'].get(role)
    if value=='current':return sys.executable
    require(isinstance(value,str) and Path(value).is_file(),'INTERPRETER_NOT_AVAILABLE')
    return value

def backend(cfg,p,operation,role):
    folder=new_observation(p['id'],operation)
    script=ROOT/'src'/('workbench_parquet.py' if operation=='verify' else 'workbench_remote.py')
    args=[interpreter(cfg,role),'-B','-X','utf8',str(script)]
    if operation!='verify':args.append(operation)
    args += [p['id'],folder.relative_to(ROOT).as_posix()]
    atomic_json(folder/'intent.json',{'operation':operation,'experiment_id':p['id'],'attempted_at':now(),'remote_mutations':False},exclusive=True)
    try:
        result=subprocess.run(args,cwd=ROOT,env=child_env(cfg),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,shell=False,timeout=600 if operation=='collect' else 120)
        require((folder/'backend.json').is_file(),'BACKEND_RETURNED_NO_RECEIPT')
        value=read(folder/'backend.json')
        require(result.returncode==0 or value.get('status')=='ERROR','BACKEND_EXIT_MISMATCH')
    except Exception as error:value=safe_failure(error)
    remember(p['id'],operation,value,folder/'receipt.json')
    return value

def local_job(cfg,p,job):
    with local_lock(p['id']):return _local_job(cfg,p,job)

def _local_job(cfg,p,job):
    ident=p['id'];require((ident,job) in JOB_SPECS,'LOCAL_JOB_NOT_ALLOWED')
    script,args,timeout=JOB_SPECS[(ident,job)]
    expected=p.get('local_code_hashes',{}).get(script)
    require(expected is not None and sha(inside(script))==expected,'LOCAL_JOB_CODE_CHANGED')
    for dependency,digest in p.get('local_code_hashes',{}).items():
        require(sha(inside(dependency))==digest,'LOCAL_JOB_DEPENDENCY_CHANGED')
    folder=new_observation(ident,'prepare' if job in ('prepare','verify-package') else 'run-local')
    atomic_json(folder/'intent.json',{'job':job,'script':script,'sha256':expected,'timeout_seconds':timeout,'scope':'audited fixed local operation; environment GPU hiding is not a security sandbox','attempted_at':now()},exclusive=True)
    summary_path=inside(p['local_summary']) if job=='verify-results' and p.get('local_summary') else None
    summary_before=sha(summary_path) if summary_path and summary_path.is_file() else None
    try:
        role='parquet' if ident=='EXP-050' else 'analysis'
        proc=subprocess.run([interpreter(cfg,role),'-B','-X','utf8',str(inside(script)),*args],cwd=ROOT,
                            env=child_env(cfg),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=False,timeout=timeout)
        require(proc.returncode==0,'LOCAL_JOB_FAILED')
        value={'status':'PASS_FIXED_LOCAL_JOB','job':job,'scope':'local_only_no_training_or_remote_launch','observed_at':now(),'exit_code':0}
        if summary_path:
            require(summary_before is not None and sha(summary_path)==summary_before,'SUMMARY_CHANGED_DURING_VERIFICATION')
            value['verified_summary']={'path':p['local_summary'],'sha256':summary_before}
    except Exception as error:value=safe_failure(error)
    operation='prepare' if job in ('prepare','verify-package') else 'run-local'
    remember(ident,operation,value,folder/'receipt.json');return value

def offline_verify(cfg,p):
    require(p['id'] in ('EXP-049','EXP-050'),'NO_REGISTERED_LOCAL_VERIFIER')
    folder=new_observation(p['id'],'verify')
    if p['id']=='EXP-049':
        result=local_job(cfg,p,'verify-package')
        if result['status']=='ERROR':
            remember(p['id'],'verify',result,folder/'receipt.json');return result
    result=local_job(cfg,p,'verify-results')
    remember(p['id'],'verify',result,folder/'receipt.json');return result

def confirmed_score(p,record):
    ident=p['id']
    last=historical_score(record)
    confirmed=latest(ident,'confirmed-score')
    if confirmed:
        remote=p.get('remote') or {}
        require(confirmed.get('submission_id')==remote.get('submission_id') and confirmed.get('script_version_id')==remote.get('script_version_id'),'CONFIRMED_SCORE_IDENTITY_CHANGED')
        time_key=lambda v:datetime.fromisoformat((v or '1970-01-01T00:00:00+00:00').replace('Z','+00:00'))
        if last is None or time_key(confirmed.get('observed_at'))>=time_key(last.get('observed_at')):
            last={'value':confirmed['score'],'observed_at':confirmed['observed_at']}
    return last

def make_report(p,record):
    ident=p['id'];lines=['# '+ident+' · '+record['title'],'',f'报告生成时间：{now()}。证据时间单独列出；本命令不查询远端。','']
    last=confirmed_score(p,record)
    if last:lines+=['已确认的正式 Public LB：**'+str(last['value'])+'**；确认时间：'+str(last['observed_at'])+'。','']
    else:lines+=['本实验尚无已确认的正式 Public LB。','']
    baseline_id=record.get('baseline_id')
    if p.get('remote') and last and baseline_id:
        require(re.fullmatch(r'EXP-\d{3}',baseline_id) is not None,'INVALID_BASELINE_ID')
        baseline=historical_score(read(inside('tracking/records/experiments/'+baseline_id+'.json')))
        if baseline:
            delta=last['value']-baseline['value']
            lines+=['相对台账对照 '+baseline_id+' 的 '+str(baseline['value'])+'：分数差 '+format(delta,'+.6f')+
                    '（'+format(delta*100,'+.4f')+' 个百分点）。这是两次正式结果的数值差，不能单独证明改动稳定有效。','']
    observed=latest(ident,'status')
    if observed:
        lines+=['最近一次只读查询：`'+observed.get('status','UNKNOWN')+'`；查询时间：'+str(observed.get('observed_at'))+'。']
        if observed.get('version_verified') and observed.get('status')=='COMPLETE' and observed.get('score') is not None:
            lines+=['本次已绑定版本的正式 Public LB：**'+str(observed['score'])+'**。']
        if not observed.get('query_ok') or not observed.get('version_verified'):
            lines+=['本次查询不能确认当前正式结果；上方历史确认分数保留。']
        if observed.get('commit_status'):lines+=['保存运行状态：`'+observed['commit_status']+'`（与正式评分状态分开）。']
        lines+=['']
    elif p.get('remote'):lines+=['本次尚未实时查询；台账状态为历史快照。','']
    else:lines+=['这是本地审计/候选准备；没有已启动的远端版本。','']
    checked=latest(ident,'verify') or latest(ident,'run-local')
    if checked:
        lines+=['最近核验：`'+checked['status']+'`；时间：'+str(checked.get('observed_at'))+'。']
        if checked.get('status')=='PASS_COMMIT_SAMPLE_ONLY':lines+=['核验范围：**公开样例占位输出、输入政策、T4×2环境与小合成自检**。不是隐藏集预测，也不证明完整模型/ATLAS执行覆盖。']
        if checked.get('status')=='ERROR':lines+=['请检查该观察收据中的 error_code；旧结果未覆盖。']
        lines+=['']
    summary_path=p.get('local_summary')
    summary_binding=(checked or {}).get('verified_summary')
    summary_valid=bool(summary_path and checked and checked.get('status')=='PASS_FIXED_LOCAL_JOB' and summary_binding
        and summary_binding.get('path')==summary_path and inside(summary_path).is_file()
        and sha(inside(summary_path))==summary_binding.get('sha256'))
    if summary_valid:
        s=read(inside(summary_path));lines+=['本地结果摘要：`'+str(s.get('status','UNCLASSIFIED'))+'`。']
        if ident=='EXP-049':
            a,b=s['baseline_045'],s['candidate_030']
            lines+=['已看过的729条/17真类缓存：0.45为'+str(a['correct'])+'条正确，0.30为'+str(b['correct'])+'条正确。',
                    'Macro recall：'+format(a['macro_recall'],'.8f')+' → '+format(b['macro_recall'],'.8f')+
                    '；变化 '+format((b['macro_recall']-a['macro_recall'])*100,'+.5f')+' 个百分点。',
                    '配对prompt bootstrap区间与逐行变化见详细报告；差值区间跨0，尚无稳定收益证据。',
                    '**这是固定外部缓存诊断，不是干净OOF或正式榜单收益。候选尚未启动GPU。**']
        if ident=='EXP-050':
            risk=s['risk_counts']
            lines+=['主池 '+str(s['rows'])+' 条；与A1已审计运行用数重合 '+str(risk['in_a1_runtime_union'])+
                    ' 条，未命中 '+str(risk['novel_text_to_audited_runtime'])+' 条。',
                    '**仅数据审计，不等于训练准入或提分。** 正文未命中只表示已知输入之外的上界，不能证明所有模型权重未见。']
        lines+=['']
    elif summary_path:
        lines+=['本地摘要当前没有匹配的成功核验及哈希绑定，暂不展示数值。请先执行 verify。','']
    lines+=['详细报告：['+p['report']+']('+inside(p['report']).as_posix()+')','',
            '状态查询、保存样例、静态核查、缓存诊断和正式评分各自独立。没有新版成绩或缺失输入时不得补零或继承标题分数。','']
    path=WORK/'reports'/(ident+'.md');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines),encoding='utf-8');return path

def doctor(cfg):
    for role in ['analysis','kaggle','parquet']:
        path=interpreter(cfg,role);print(role+': '+path)
    print('远端能力：只读查询与小文件下载。未提供保存、启动、取消、提交或生成数据命令。')

def main(argv=None):
    parser=argparse.ArgumentParser(description='Name That LLM 固定实验入口；不启动云端任务或提交比赛。')
    parser.add_argument('command',choices=OPERATIONS);parser.add_argument('experiment',nargs='?');args=parser.parse_args(argv)
    try:
        cfg=read(CONFIG)
        if args.command=='list':
            for p in cfg['experiments']:
                _,_,r=profile(p['id']); score=confirmed_score(p,r)
                print(p['id']+' | '+','.join(p.get('aliases',[]))+' | '+r['title']+' | 正式分数 '+(str(score['value']) if score else '未确认'))
            return 0
        if args.command=='doctor':doctor(cfg);return 0
        require(args.experiment is not None,'EXPERIMENT_ID_REQUIRED');cfg,p,r=profile(args.experiment)
        command=args.command;result={}
        if command in ('status','collect'):
            require(bool(p.get('remote')),'EXPERIMENT_HAS_NO_SAVED_REMOTE_VERSION')
            result=backend(cfg,p,command,'kaggle')
        elif command=='verify':result=backend(cfg,p,'verify','parquet') if p.get('remote') else offline_verify(cfg,p)
        elif command=='check':
            if p.get('remote'):
                result=backend(cfg,p,'status','kaggle')
                if result.get('query_ok') and result.get('version_verified'):
                    previous=latest(p['id'],'collect')
                    if previous and previous.get('status')=='COLLECTED_COMMIT_OUTPUTS':verify_files(previous,inside(previous['files_path']));result=previous
                    else:result=backend(cfg,p,'collect','kaggle')
                    if result.get('status')!='ERROR':result=backend(cfg,p,'verify','parquet')
            else:result=offline_verify(cfg,p)
        elif command=='prepare':
            require(p['id']=='EXP-049','NO_REGISTERED_PREPARER')
            result=local_job(cfg,p,'verify-package' if inside(p['package_ready_receipt']).is_file() else 'prepare')
        elif command=='run-local':
            require(p['id']=='EXP-049','LOCAL_RUN_NOT_ALLOWED_USE_VERIFY')
            existing=inside(p['local_summary']).is_file()
            result=offline_verify(cfg,p) if existing else local_job(cfg,p,'run-local')
            if not existing and result.get('status')!='ERROR':result=offline_verify(cfg,p)
            if existing:print('固定诊断结果已存在：已核验复用，没有重算或覆盖。')
        report=make_report(p,r);print('报告：'+str(report));print('本次操作：'+result.get('status','OFFLINE_REPORT'))
        return 2 if result.get('status')=='ERROR' or result.get('status','').startswith('UNKNOWN') else 0
    except Exception as error:
        print('未通过：'+safe_failure(error)['error_code']);return 2

if __name__=='__main__':raise SystemExit(main())
