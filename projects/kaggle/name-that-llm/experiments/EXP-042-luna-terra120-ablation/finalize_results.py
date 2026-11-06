"""Publish the verified EXP042 result; no fitting, network, or input changes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CLOUD=ROOT/'cloud/luna-terra120-042'
OUT=CLOUD/'outputs-v2/results'
ANALYSIS=HERE/'analysis-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rel(p):return p.relative_to(ROOT).as_posix()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    s=read(ANALYSIS/'summary.json');v=read(ANALYSIS/'verification.json');g=read(OUT/'guard.json')
    r=read(OUT/'result.json');ind=read(HERE/'metrics/independent-result-check.json')
    assert v['status']=='PASS' and g['status']==r['status']=='COMPLETE'
    assert ind['status']=='PASS_INDEPENDENT_ARITHMETIC'
    assert v['historical_control']['matrix_byte_values_equal'] and v['historical_control']['prediction_disagreements']==0
    now=datetime.now(timezone.utc).isoformat();a=s['dense']['control'];b=s['dense']['add120'];paired=s['paired']
    ci=[x*100 for x in paired['bootstrap']['ci95']];delta=paired['delta_ba']*100
    sp=read(ANALYSIS/'sparse_summary.json')
    sparse_rows=[]
    for m in (2,3,6):
        scenes=[x for x in sp['scenarios'] if x['m']==m]
        assert len(scenes)==5
        sparse_rows.append({'m':m,'control_mean':float(np.mean([x['control']['balanced_accuracy'] for x in scenes])),
            'add120_mean':float(np.mean([x['add120']['balanced_accuracy'] for x in scenes])),
            'delta_mean':float(np.mean([x['delta_ba'] for x in scenes])),
            'positive_repeats':sum(x['delta_ba']>0 for x in scenes),
            'negative_repeats':sum(x['delta_ba']<0 for x in scenes)})
    pd.DataFrame(sparse_rows).to_csv(ANALYSIS/'sparse_aggregate.csv',index=False)
    predictions=pd.read_csv(ANALYSIS/'predictions.csv')
    rows=[]
    for label in ('gpt-5.6-luna','gpt-5.6-terra'):
        d={'model':label}
        for arm in ('control','add120'):
            hit=predictions[arm].eq(label);truth=predictions.model.eq(label)
            d[arm]={'predicted':int(hit.sum()),'true_positive':int((hit&truth).sum()),
                'false_positive':int((hit&~truth).sum()),'recall':float((hit&truth).sum()/truth.sum()),
                'precision':float((hit&truth).sum()/hit.sum())}
        rows.append(d)
    save(ANALYSIS/'target_class_tradeoff.json',{'scope':'Descriptive post-hoc diagnosis; not a new primary endpoint','classes':rows})
    table='\n'.join(f"| 每题{x['m']}条 | {x['control_mean']:.6f} | {x['add120_mean']:.6f} | {x['delta_mean']*100:+.4f} | {x['positive_repeats']} / {x['negative_repeats']} |" for x in sparse_rows)
    headline=f'''**验收完成：加入120条后，没有看到总体稳定提升，保留原基线。**

运行于北京时间2026-11-01 23:21:37结束，CPU墙钟2896.151秒（约48分16秒），20个SVM、两臂五折及150份稀疏缓存均完成。211个稳定产物哈希、实际固定依赖环境及独立统计复算通过；对照组与EXP010的完整OOF、五折及75份对照稀疏缓存数值完全一致。

## 核心结果

| 指标 | 不加120条 | 加入120条 | 差值 |
|---|---:|---:|---:|
| 18类OOF Balanced Accuracy | {a['balanced_accuracy']:.6f} | {b['balanced_accuracy']:.6f} | {delta:+.4f}个百分点 |
| 普通准确率 | {a['accuracy']:.6f} | {b['accuracy']:.6f} | {(b['accuracy']-a['accuracy'])*100:+.4f}个百分点 |
| 答对条数 / 2699 | {a['correct']} | {b['correct']} | -2 |

29条预测改变，其中修正6错、新增8错，其余15条仍错。按150个官方prompt配对bootstrap，2000次、seed42，95%差值区间为 **[{ci[0]:+.4f}, {ci[1]:+.4f}]个百分点**，没有缺类抽样。五折净正确数依次-1、0、0、-3、+2。这不支持稳定提升，也不足以认定稳定退化；不能把“不显著”解释为等价。

![主指标与差值区间](figures/ablation-result.png)

## Luna/Terra 的收益与代价

| 类别 | 正确识别 | 召回率 | 其他模型被错报为该类 |
|---|---|---|---|
| Luna | 69→71 | 46.00%→47.33% | 82→90 |
| Terra | 50→52 | 33.33%→34.67% | 70→75 |

两类召回各升1.33个百分点，但precision分别从45.70%降至44.10%、41.67%降至40.94%。候选更倾向输出这两个标签，更多误报抵消了召回收益；其余16类合计净少对6条。该诊断不能证明系统提示污染或某个单一机制，因为本轮同时包含新增文本带来的词表/IDF变化和既定的两类总权重增加。

## 稀疏压力测试

以下均为原固定抽样方案，每个repeat先拼接五折再计算18类BA，最后对五个repeat求平均。

| 场景 | 不加：平均BA | 加入：平均BA | 平均差值（百分点） | 升 / 降次数 |
|---|---:|---:|---:|---|
{table}

次要场景有小幅正向均值，但各档都是3升2降；这5次抽样共享同一批官方题组和已拟合模型，不是5次独立训练。它们不能后验替代预先固定的完整OOF主终点。

## 决策与限制

保留原Stylegram和完整ATLAS20正式0.662基线，这120条暂不并入正式训练池，也不据此扩大采集。数据和全部预测保留；本轮只说明这套数据、权重及分类器配方未证明总体收益，不能断言Codex数据普遍无效、有害或渠道污染已被证实。

未读取新60条holdout用于训练或评分，未评分有近似题重叠的EXP027。官方五折是已经查看过的开发验证；单成员BA不等于整栈榜单分数。Bootstrap条件于这些已拟合模型，不包括重新训练的不确定性。没有重跑、继续调参、GPU、API生成或正式提交。

[统计摘要](analysis-v1/summary.json) · [逐类结果](analysis-v1/dense_perclass.csv) · [逐行预测](analysis-v1/predictions.csv) · [完整验收](analysis-v1/verification.json) · [独立复算](metrics/independent-result-check.json) · [下载收据](metrics/download-receipt.json)。
'''
    rp=HERE/'report.md';old=rp.read_text(encoding='utf-8');design=old[old.index('## 假设与主要改动'):]
    design=design.split('## 执行与结果')[0]
    rp.write_text('# EXP-042：Luna/Terra 120条增量实验结果\n\n'+headline+'\n## 冻结设计记录\n\n'+design,encoding='utf-8')
    state=read(CLOUD/'execution-state.json')
    state.update(status='COMPLETE_VERIFIED',observed_at=now,completed_at_utc=g['observed_at'],
        output_directory=rel(OUT),artifacts_sha256=v['external_artifacts_sha256'],
        verification_path=rel(ANALYSIS/'verification.json'),analysis_path=rel(ANALYSIS/'summary.json'),
        cpu_wall_seconds=g['elapsed_seconds'],result_status='NO_STABLE_PRIMARY_GAIN',formal_submission_created=False)
    save(CLOUD/'execution-state.json',state)
    path=ROOT/'tracking/records/experiments/EXP-042.json';rec=read(path)
    rec['observed_at']=now;rec['execution'].update(status='COMPLETE',observed_at=now,outputs_verified=True,
        actual_environment_verified=True,trained_svms=20,cpu_wall_seconds=g['elapsed_seconds'])
    rec['validation'].update(balanced_accuracy=b['balanced_accuracy'],control_balanced_accuracy=a['balanced_accuracy'],
        delta_balanced_accuracy=paired['delta_ba'],delta_ci95=paired['bootstrap']['ci95'],
        per_class_recall_path=rel(ANALYSIS/'dense_perclass.csv'),oof_path=rel(OUT/'oof/add120.npz'),
        scope='Verified fixed official 18-class development OOF, isolated Stylegram; not full stack or leaderboard')
    rec['conclusion']='新增120条未证明总体收益：BA0.468243→0.467502，改对6/改错8，净少对2；95%差值区间[-0.3333,+0.2222]个百分点跨0。'
    rec['next_action']='保留原基线与数据，不正式入池、不据此扩采；次要稀疏正均值只作诊断，后续改配方需单独固定实验。'
    rec['timing_note']='CPU守护墙钟2896.151秒（约48分16秒），训练进程2892.316秒；峰值进程树RSS约4.82GiB，未使用GPU。'
    rec['evidence']=list(dict.fromkeys(rec['evidence']+[rel(p) for p in [HERE/'metrics/download-receipt.json',HERE/'metrics/independent-result-check.json',ANALYSIS/'verification.json',ANALYSIS/'summary.json',ANALYSIS/'target_class_tradeoff.json',OUT/'guard.json',OUT/'result.json']]))
    save(path,rec)
    ideas=read(ROOT/'tracking/ideas.json')
    for x in ideas:
        if x['id']=='IDEA-005':
            x.update(status='evaluated_no_stable_primary_gain',observed_at=now,
                next_action='EXP042固定加入120条对照已完成，主OOF净少对2条且区间跨0；保留原基线，不正式入池或扩采。后续若研究类别偏移，另立固定实验。',
                evidence_assessment='EXP041格式/原文审计通过；EXP042 fresh control完全复现，新增数据BA差-0.0741个百分点，95%CI[-0.3333,+0.2222]。目标两类召回增加但误报更多，稀疏均值小幅正且每档3升2降。',
                planning_scope='本轮CPU训练及完整验收完成；未扩大生成、启动GPU或正式提交。')
            x['related_experiments']=list(dict.fromkeys(x['related_experiments']+['EXP-042']))
            x['evidence']=list(dict.fromkeys(x['evidence']+[rel(HERE/'report.md')]))
    save(ROOT/'tracking/ideas.json',ideas)
    stable=[p for p in ANALYSIS.iterdir() if p.is_file()]
    save(HERE/'metrics/final-acceptance.json',{'status':'PASS_VERIFIED_NEGATIVE_OR_INCONCLUSIVE_PRIMARY_RESULT',
        'observed_at':now,'run_id':r['run_id'],'source_zip_sha256':read(HERE/'metrics/download-receipt.json')['zip_sha256'],
        'source_artifacts_sha256':v['external_artifacts_sha256'],'analysis':[{'path':rel(p),'sha256':sha(p)} for p in stable],
        'independent_check_sha256':sha(HERE/'metrics/independent-result-check.json'),
        'original_config_sha256':sha(HERE/'config.json'),'no_additional_training':True})
    print(json.dumps({'status':'FINALIZED','delta_ba_pp':delta,'ci95_pp':ci,'correct':[a['correct'],b['correct']]}))

if __name__=='__main__':main()
