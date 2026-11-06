"""Archive and report the accepted sole V2 external 17-class run; never train."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CLOUD=ROOT/'cloud/fixed-blend-multiclass'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rel=lambda p:p.relative_to(ROOT).as_posix()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    accepted=read(CLOUD/'analysis/verified.json')
    remote=read(CLOUD/'complete-v2-readback.json');meta=remote['info']['metadata']
    assert accepted['status']=='PASS' and accepted['stable_artifacts_verified']==18
    assert meta['current_version_number']==2 and remote['status']['status']=='COMPLETE'
    assert meta['is_private'] and not meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['dataset_data_sources']==['zhjianduan/ntl-fixed-blend-multiclass-inputs-027']
    assert meta['competition_data_sources']==['name-that-llm']
    assert remote['info']['blob']['source']==(CLOUD/'kernel/bootstrap.py').read_text()
    manifest=read(HERE/'manifest.json')
    for entry in manifest['inputs']+manifest['code']+[manifest['config'],manifest['protocol']]:
        assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    for path,digest in read(HERE/'logs/protected-files-before.json')['files'].items():
        assert sha(ROOT/path)==digest,path
    assert sha(CLOUD/'verify_outputs.py')==accepted['verifier_sha256']
    assert read(HERE/'metrics/independent-review.json')['status']=='PASS'
    result=read(CLOUD/'outputs-v2/results/result.json');guard=read(CLOUD/'outputs-v2/results/guard.json')
    stamp=datetime.now(timezone.utc).isoformat()
    archived=[]
    prediction_names={'component-and-blend-scores.npz','stylegram-scores.npz','predictions.csv','confusion.csv'}
    files=list((CLOUD/'outputs-v2').rglob('*'))+list((CLOUD/'analysis').glob('*'))
    for source in sorted(p for p in files if p.is_file()):
        if source.parent==CLOUD/'analysis':target=HERE/'metrics'/source.name
        elif source.name in prediction_names:target=HERE/'predictions'/source.name
        elif source.suffix=='.log' or source.name=='.run.lock':target=HERE/'logs'/source.name
        else:target=HERE/'metrics/runtime'/source.name
        data=source.read_bytes();target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists() or target.read_bytes()==data,'Archive differs: '+str(target)
        if not target.exists():
            with target.open('xb') as f:f.write(data)
        assert sha(source)==sha(target)
        archived.append({'source':rel(source),'archive':rel(target),'sha256':sha(target),'bytes':len(data)})
    write(HERE/'metrics/archive-manifest.json',{'observed_at_utc':stamp,'notebook_ref':meta['ref'],'version':2,
        'cli_command':['kaggle','kernels','output',meta['ref']+'/2','-p','cloud/fixed-blend-multiclass/outputs-v2','-q'],
        'cli_exit_code':0,'complete_readback':rel(CLOUD/'complete-v2-readback.json'),'files':archived})
    write(HERE/'metrics/data-use.json',{'experiment':'EXP-027','observed_at_utc':stamp,'rows':729,'groups':56,'covered_labels':17,
        'evaluation_sha256':accepted['evaluation_sha256'],'alpha':.1,'retuned_on_results':False,
        'future_status':'Inspected evaluation set; later tuned candidates cannot claim first independent confirmation on it',
        'scope':'Known/exact two-member isolation only; upstream official eightgram filtering and publisher identity limits retained'})
    write(HERE/'metrics/verification-tests.json',{'status':'PASS','runner_tests':16,'verifier_tests':20,
        'runner_sha256':sha(CLOUD/'runner.py'),'runner_tests_sha256':sha(CLOUD/'test_runner.py'),
        'verifier_sha256':sha(CLOUD/'verify_outputs.py'),'verifier_tests_sha256':sha(CLOUD/'test_verify_outputs.py'),
        'observed_at_utc':stamp,'no_test_training':True,
        'verifier_test_note':'20 tests passed before the metadata-only V1-to-V2 declaration correction; mathematical checks unchanged. Real result verification uses the final recorded verifier hash.'})
    metrics=accepted['recomputed_metrics'];comparison=accepted['recomputed_comparison']
    delta=comparison['delta_macro_recall'];ci=comparison['paired_prompt_bootstrap']['ci95']
    ci_text='未形成区间（存在缺类抽样）' if ci is None else f'[{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] 个百分点'
    if ci is not None and ci[0]>0:decision='本来源支持正向增益，保留固定候选；仍需完整集成增量证据。'
    elif ci is not None and ci[1]<0:decision='本来源呈明确负向证据，不升级候选；需要重新审视适用范围，不能用当前测试集调参后再称独立验证。'
    else:decision='本来源尚不支持稳定增益，暂不升级候选；不依据此结果继续扫权重。'
    net=metrics['blend']['correct']-metrics['stylegram']['correct']
    conclusion=f'17类外部检查验收PASS；Stylegram {metrics["stylegram"]["macro_recall"]:.9f}，svc4 {metrics["svc4"]["macro_recall"]:.9f}，固定10%融合 {metrics["blend"]["macro_recall"]:.9f}。差值{delta*100:+.4f}个百分点，95%区间{ci_text}。'+decision
    record_path=ROOT/'tracking/records/experiments/EXP-027.json';r=read(record_path)
    r['observed_at']=stamp;r['execution'].update(status='COMPLETE',result_status='PASS',observed_at=stamp,
        completed_at=guard['observed_at'],accepted_at=accepted['verified_at_utc'])
    r['validation'].update(balanced_accuracy=None,scope='17-covered-label macro recall with unrestricted18-way predictions; not full18 or ATLAS20',
        covered_label_macro_recall=metrics['blend']['macro_recall'],component_metrics=metrics,comparison=comparison,
        per_class_recall_path=rel(HERE/'metrics/per_class.csv'),external_predictions_path=rel(HERE/'predictions/predictions.csv'),
        acceptance_path=rel(HERE/'metrics/verified.json'),independent_review_path=rel(HERE/'metrics/independent-review.json'))
    r['timing_note']=f'CPU runner {result["total_seconds"]:.3f}s; guard {guard["elapsed_seconds"]:.3f}s; sampled RSS {guard["peak_rss_bytes"]/1024**3:.3f}GiB. Excludes queue/setup/download/verification.'
    r['conclusion']=conclusion;r['limitations']='56题/17类且不均衡，五个少数类各16至19条；源数据经官方8gram后筛，服务身份仅发布者声明，语义重叠未穷尽；没有完整集成隔离缓存，不能宣称榜单改进。'
    r['next_action']='依据本次跨来源证据决定是否保留候选；正式0.662不变。完整集成先补成员训练谱系与gate前缓存，再按integration-plan的固定分离对照执行，不自动启GPU或提交。'
    assert r['submission'] is None
    for p in [HERE/'record_results.py',HERE/'metrics/verified.json',HERE/'metrics/independent-review.json',
              HERE/'metrics/archive-manifest.json',HERE/'metrics/data-use.json',HERE/'metrics/verification-tests.json',
              HERE/'metrics/per_class.csv',HERE/'metrics/comparison.csv',HERE/'predictions/predictions.csv',
              CLOUD/'complete-v2-readback.json',CLOUD/'verify_outputs.py',CLOUD/'test_verify_outputs.py']:
        if rel(p) not in r['evidence']:r['evidence'].append(rel(p))
    write(record_path,r)
    state=read(CLOUD/'execution-state.json');state.update(status='COMPLETE',result_status='PASS',phase='verified',
        observed_at=stamp,completed_at=guard['observed_at'],accepted_at=accepted['verified_at_utc'],
        readback_path=rel(CLOUD/'complete-v2-readback.json'),verification_path=rel(HERE/'metrics/verified.json'),
        output_directory=rel(CLOUD/'outputs-v2'),guard_seconds=guard['elapsed_seconds'],peak_rss_bytes=guard['peak_rss_bytes'],
        next_action='Terminal; no rerun, retune, new GPU, or submission')
    write(CLOUD/'execution-state.json',state)
    p=ROOT/'tracking/ideas.json';ideas=read(p)
    for idea in ideas:
        if idea['id']=='IDEA-004':
            idea.update(observed_at=stamp,phase='fixed_blend_multisource_checks_complete',next_action=decision+' 完整集成接入需要分开配方、分数尺度与追加成员，见EXP027 integration-plan。')
            if 'EXP-027' not in idea.setdefault('completed_experiments',[]):idea['completed_experiments'].append('EXP-027')
            idea['active_experiments']=[x for x in idea.get('active_experiments',[]) if x!='EXP-027']
    write(p,ideas)
    rows=['| 成员 | 17类宏召回 | 答对/729 |','|---|---:|---:|']
    for name,label in [('stylegram','Stylegram'),('svc4','svc4'),('blend','固定10%融合')]:
        m=metrics[name];rows.append(f'| {label} | {m["macro_recall"]:.9f} | {m["correct"]} |')
    per=['| 类别 | 样本数 | Stylegram召回 | 融合召回 |','|---|---:|---:|---:|']
    for a,b in zip(metrics['stylegram']['per_class'],metrics['blend']['per_class']):
        assert a['model']==b['model']
        per.append(f'| {a["model"]} | {a["support"]} | {a["recall"]:.4%} | {b["recall"]:.4%} |')
    report=f'''# EXP-027 · 固定10%融合的多家族外部检查

**运行与验收完成。{decision}** 正式EXP-004的0.662未改，没有提交比赛。本次验证两个轻量成员，不是完整ATLAS20。

## 主要结果

{chr(10).join(rows)}

本次729条、56个规范化题干组，覆盖17个真类，预测始终从18类选择。指标是被覆盖17类召回率的等权平均；样本数不均衡，不能用答对总数/729代替该指标，也不称完整18类BA。

融合相对Stylegram差值 **{delta*100:+.4f} 个百分点**；修正{comparison['fixed_errors']}条错、新增{comparison['new_errors']}条错，净多对{net:+d}条，改判{comparison['prediction_disagreements']}条。按56个题组配对bootstrap2000次、seed42，95%区间为 **{ci_text}**。任何抽样缺失某个覆盖类时整个区间留空，不丢弃或补抽；本次无效抽样数为{comparison['paired_prompt_bootstrap']['invalid_draws']}。区间条件于固定模型与本来源，不包含重新训练或其他来源不确定性。

## 数据与范围

SRC-063 earthmaker固定V1，749条原文来自20题pilot和36题v22。评分前预先排除20条服务版本无法确认的gemini-3.7-flash；其余729条原文全部保留，无新增长度/风格/得分筛选。每条按batch、题目银行、template与目标词数重建请求；用NFKC/casefold/空白归一化task正文跨批次合组，共56组，只有16组完整覆盖17类。fable、glm、glm-flash、kimi、qwen五类只有16～19条，且全部来自pilot OpenRouter，其余类别含更多proxy样本。

与两成员真实训练回答、763个已知训练祖先/正文、150个已知官方重建题及EXP026归并后的1299正文的相关exact交集均为0。原数据作者已用官方8gram和少量格式规则作后筛，因此不是严格未触碰的holdout；未知语义重叠、真实服务权重和完整集成各checkpoint训练暴露仍未核全。详见[数据门槛](metrics/data-gate.json)、[来源资料](../../research/datasets/SRC-063-earthmaker-own-prompts-v1/notes.md)。本选集已经评分，后续不再作为调参后的全新独立确认，见[使用记录](metrics/data-use.json)。

## 保持不变的模型与实际运行

官方2699+Stylegram公共10073训练plain/centred两个SVM，公共share=.5；官方2699+svc4 raw公共5400训练一个SVM，公共每行权重=.35。原算法、训练输入/core字节、参数、18类顺序及sklearn1.6.1保持固定。两个成员每行population z-score后固定.9/.1组合，不是校准概率；没有根据本轮结果调整系数。真prompt与标签只用于计分和bootstrap，Stylegram推断分组只看无标签文本。

Kaggle私有CPU2线程、断网、固定容器，V1只保存、V2唯一实际执行。Runner {result['total_seconds']:.3f}秒，guard {guard['elapsed_seconds']:.3f}秒，采样峰值RSS {guard['peak_rss_bytes']/1024**3:.3f}GiB；内部3300秒/平台3600秒/12GiB保护均通过。GPU、API美元、Colab CU为0。耗时不含排队/启动/下载/本地核验，API异常last_run_time不用于计算时长。

18项稳定产物哈希、实际训练成员/权重/参数/依赖、逐行18维分数/argmax、918格混淆矩阵、宏召回与区间均已核验；另有独立复算。16项runner测试与20项验收器测试通过。未保存拟合模型和矩阵本体，归档不能独立重建训练内部过程。

## 逐类结果

{chr(10).join(per)}

逐类小样本差异不另作显著性声称，不能依此按真实标签动态挑成员。与EXP026的四类Dolly来源结果应分开解释，不合并成一个跨来源“总分”。

## 完整集成接入检查

原ATLAS20主栈已经包含matched/pool版svc4；本候选raw版不是同一模型。.1也不是完整集成占比，alpha=0的z-score本身都会改变Stylegram置信度尺度。原Stylegram公共10325行与当前10073行不同，不能把三项差别混成单一提升。

[接入方案](integration-plan.md)定义四个固定对照：原配方raw、仅替换为当前Stylegram配方raw、仅标准化、再加入10%raw svc4。重成员只计算一次，必须缓存gate之前的各成员和原始gate分数，再分别重放后处理。ATLAS请求取各臂请求并集缓存，单臂仍按原路由使用。目前本地无可核验完整集成OOF或该外部集全成员缓存，上游多份权重已见官方训练数据，不能直接冒充干净OOF。本轮未启动该GPU流程。

## 证据与后续

{conclusion}

[固定配置](config.json) · [冻结清单](manifest.json) · [验收](metrics/verified.json) · [独立复算](metrics/independent-review.json) · [逐行预测](predictions/predictions.csv) · [归档哈希](metrics/archive-manifest.json) · [远端终态](../../cloud/fixed-blend-multiclass/complete-v2-readback.json)。

EXP026另有题组计数勘误：1300源组归并1299，净54条不变，修正95%区间为[+0.6736,+1.3857]个百分点，见[旧实验勘误](../EXP-026-independent-blend-validation/metrics/prompt-unit-correction.json)。保留原产物与旧值，不覆盖原训练或模型字节。
'''
    (HERE/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'status':'RECORDED','archived_files':len(archived),'conclusion':conclusion},ensure_ascii=False))

if __name__=='__main__':main()
