"""Publish verified EXP044 results into its existing journal; no remote writes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CLOUD=ROOT/'cloud/format120-044'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    v=read(HERE/'metrics/verified.json')
    independent=read(HERE/'metrics/independent-result-review.json')
    assert v['status']=='PASS_FROZEN_RAW_CLEAN_OOF' and independent['status'].startswith('PASS')
    p=v['primary']; ci=v['paired_prompt_bootstrap']['ci95']
    g=read(CLOUD/'outputs-v2/results/guard.json')
    old=read(ROOT/'cloud/luna-terra120-042/outputs-v2/results/guard.json')
    remote=read(CLOUD/'outputs-v2/remote-file-list.json')
    assert remote['status']=='COMPLETE' and remote['version']==2 and remote['file_count']==365
    assert g['status']=='COMPLETE' and g['child_exit']==0 and g['tree_stopped'] and not g['termination_reason']
    seconds=g['cumulative_elapsed_seconds']; minutes=int(seconds//60); sec=seconds%60
    speed={'current_guard_seconds':seconds,'historical_exp042_guard_seconds':old['elapsed_seconds'],
           'observed_wall_reduction_fraction':1-seconds/old['elapsed_seconds'],
           'observed_wall_ratio':old['elapsed_seconds']/seconds,'peak_rss_gib':g['peak_rss_bytes']/1024**3,
           'parallel_workers_observed':2,'scope':'Similar-size20-fit/5-fold comparisons with different added texts/arms, not a controlled identical-input speed benchmark.'}
    save(HERE/'metrics/runtime-comparison.json',speed)
    pd.DataFrame(p['per_class']).to_csv(HERE/'metrics/per_class.csv',index=False)
    pd.DataFrame(v['sparse_means']).to_csv(HERE/'metrics/sparse_summary.csv',index=False)
    folds=[{'fold':f['fold'],'raw_ba':f['raw120']['balanced_accuracy'],'clean_ba':f['clean120']['balanced_accuracy'],
            'delta_ba':f['clean_minus_raw_ba'],'net_correct':f['net_correct']} for f in v['fold_comparisons']]
    pd.DataFrame(folds).to_csv(HERE/'metrics/per_fold.csv',index=False)
    sparse='\n'.join(f"| {r['m']} | {r['raw120_mean_ba']:.6f} | {r['clean120_mean_ba']:.6f} | {r['mean_delta_ba']*100:+.4f} |" for r in v['sparse_means'])
    text=f'''# EXP-044 · 旧120原文与去格式120：两路并行对照

**Kaggle固定第2版COMPLETE，完整产物验收与独立复算通过。去格式未显示稳定收益，暂不作为默认训练处理；并行运行达到预期耗时。**

## 本次比较

两组使用同60题、Luna/Terra各60条旧回答。raw120保留EXP023原文，clean120只应用EXP043已验收的换行/成对加粗移除。86行发生格式变化，34行不变；原文、标签、题组、行顺序、权重保持对应。两组共享官方2699行固定五折、原公共10073行/694组、Stylegram核心与参数；不使用保留集、不加入Sol/Astra、不改正式0.662基线。

## 主结果：18类Stylegram单成员OOF

| 组 | Balanced Accuracy | 普通准确率 | 答对/2699 |
|---|---:|---:|---:|
| 旧原文120 | {p['raw120']['balanced_accuracy']:.9f} | {p['raw120']['accuracy']:.9f} | {p['raw120']['correct']} |
| 旧去格式120 | {p['clean120']['balanced_accuracy']:.9f} | {p['clean120']['accuracy']:.9f} | {p['clean120']['correct']} |

去格式减原文为 **{p['clean_minus_raw_ba']*100:+.4f}个百分点**。共{p['prediction_disagreements']}条改判，修正{p['fixed_errors']}条错、引入{p['introduced_errors']}条新错，净少对2条。150题组配对bootstrap2000次、seed42，95%区间 **[{ci[0]*100:+.4f}, {ci[1]*100:+.4f}]个百分点**，跨0；不能认定清洗必然有害，也没有稳定提分依据。

五折净答对变化为-1、0、+1、0、-2。Luna仍为69/150、Terra仍为50/150，两类召回未改变。其他类别净变化是Sonnet+1、GLM5.3+1、Gemini3.6−2、Gemini3.7−2。

![主效果及不确定性](figures/format-result.png)

## 稀疏压力测试

每题只保留m条回答、每档五次固定抽样，重新做无标签聚类与中心化。

| m | 原文平均BA | 去格式平均BA | 差值（百分点） |
|---|---:|---:|---:|
{sparse}

m2略升，m3/m6略降，方向不一致。15种场景复用相同训练模型，不能当作15次独立实验。

## 运行与完整性

- 私有Notebook `zhjianduan/name-that-llm-format120-044`，固定V2；Kaggle输出版本标识364606984。MCP查询COMPLETE，365个远端文件；下载8498985字节结果ZIP，安全解包348文件。
- 十个fold/arm任务、二十个SVM、十份完整折缓存、150份稀疏折缓存、两份2699×18 OOF均完整；ID、标签、折、权重、路由、哈希、实际环境通过。
- 实际numpy2.1.3/pandas2.3.3/scipy1.16.3/sklearn1.6.1/Python3.13.15；云端串行/并行小样本分数差0。最高两个worker，guard退出0，子进程树已结束，无失败重试。
- 守护计时 **{minutes}分{sec:.1f}秒**，峰值RSS **{speed['peak_rss_gib']:.2f}GiB**。Notebook日志最后事件约28分39秒，平台截图约28分40秒；二者包含范围不同，不能混为纯训练时间。
- 相比EXP042守护48分16秒，本次墙钟约少{speed['observed_wall_reduction_fraction']*100:.1f}%。两轮训练数据/两臂不同，只能作为相近规模的运行参考，不能宣称严格同输入的1.71倍加速。
- 正式运行只执行一次，没有使用恢复；已完成job复用、损坏拒绝等续跑机制通过小样本测试，跨Kaggle会话恢复未在本轮实际执行。

## 决策与边界

保留原文和派生数据，暂不把去格式处理作为默认方案，不升级到正式集成。该结果不足以证明120条数量太少，也不能证明Codex数据普遍无效，或两种格式未来等价。若后续研究扩量，需另设固定协议和权重对照；本轮不自动训练400条或再采样。

这是已用于开发的Stylegram单成员OOF，不是完整ATLAS20的外部0.601结果，也不是Kaggle榜单分数。区间条件于当前已拟合模型，不包含重新训练或未观测来源的不确定性。正式EXP004/0.662保持不变。

## 证据

[配置](config.json) · [冻结清单](manifest.json) · [验收JSON](metrics/verified.json) · [独立复算](metrics/independent-result-review.json) · [运行时间](metrics/runtime-comparison.json) · [逐类](metrics/per_class.csv) · [逐折](metrics/per_fold.csv) · [下载收据](../../cloud/format120-044/outputs-v2/download-receipt.json) · [控制状态](../../cloud/format120-044/execution-state.json)。原冻结源码、数据与原始结果保持不变。
'''
    (HERE/'report.md').write_text(text,encoding='utf-8')
    now=datetime.now(timezone.utc).isoformat()
    state=read(CLOUD/'execution-state.json')
    state.update(status='COMPLETE_VERIFIED',observed_at=now,script_version_id=364606984,cloud_selftest_observed=True,
                 output_observation='Fixed V2 complete; all ten jobs and OOF/sparse predictions verified locally.',
                 output_directory='cloud/format120-044/outputs-v2/results',acceptance_path='experiments/EXP-044-format120-parallel/metrics/verified.json',
                 raw120_ba=p['raw120']['balanced_accuracy'],clean120_ba=p['clean120']['balanced_accuracy'],guard_wall_seconds=seconds,
                 observed_max_workers=2,peak_rss_bytes=g['peak_rss_bytes'],status_source='Kaggle MCP fixed v2 COMPLETE plus downloaded original artifacts')
    save(CLOUD/'execution-state.json',state)
    path=ROOT/'tracking/records/experiments/EXP-044.json'; record=read(path)
    record['observed_at']=now; record['source']['script_version_id']=364606984
    record['execution'].update(status='COMPLETE',observed_at=now,result_status=v['status'],run_id=v['run_id'],trained_svms=20,completed_jobs=10)
    record['environment'].update(versions=v['identity']['environment'],gpu_used=False,parallel_workers=2)
    record['validation'].update(balanced_accuracy=p['clean120']['balanced_accuracy'],control_balanced_accuracy=p['raw120']['balanced_accuracy'],
              scope='clean120 versus raw120 Stylegram18-class grouped development OOF, not full ensemble or leaderboard',
              per_class_recall_path='experiments/EXP-044-format120-parallel/metrics/per_class.csv',
              oof_path='cloud/format120-044/outputs-v2/results/aggregates/attempt-0001/oof-clean120.npz',
              paired_difference=p['clean_minus_raw_ba'],ci95=ci,acceptance_path='experiments/EXP-044-format120-parallel/metrics/verified.json')
    record['timing_note']=f'Guard {seconds:.3f}s, two CPU workers, peak{speed["peak_rss_gib"]:.3f}GiB; platform log~1719s. Compared EXP042 different arms, not matched speed benchmark.'
    record['conclusion']='完整及独立验收通过：去格式BA差−0.0741个百分点、净少对2条，95%区间跨0；无稳定收益，原文/基线保留。两路CPU守护28分14秒。'
    record['limitations']='已查看的单成员开发OOF，非整栈或LB；CI不含重训不确定性。稀疏场景相关；实际跨云续跑未执行。'
    record['next_action']='暂不采用默认去格式或自动扩量；保留全部产物。完整公共候选比较的4条ATLAS缺口另见EXP045，需另行授权模型推理。'
    record['evidence']=list(dict.fromkeys(record['evidence']+[p.relative_to(ROOT).as_posix() for p in
        (HERE/'metrics/verified.json',HERE/'metrics/independent-result-review.json',HERE/'metrics/runtime-comparison.json',
         HERE/'metrics/per_class.csv',HERE/'metrics/per_fold.csv',HERE/'metrics/sparse_summary.csv',
         CLOUD/'outputs-v2/download-receipt.json',CLOUD/'outputs-v2/remote-file-list.json',HERE/'finalize_results.py')]))
    save(path,record)
    ideas_path=ROOT/'tracking/ideas.json';ideas=read(ideas_path)
    for item in ideas:
        if item['id']=='IDEA-005':
            item['observed_at']=now;item['status']='evaluated_no_stable_primary_gain'
            item['next_action']='EXP042新增120与EXP044去格式均未提供稳定主指标收益；保留无新增基线，不自动扩采、扩量或将清洗设为默认。'
            item['evidence_assessment']='EXP042 aligned120主BA差−0.0741pp；EXP044旧clean120对旧raw120差−0.0741pp、净−2、95%CI[−0.2593,+0.1111]pp，target类召回不变、稀疏方向不一致。不同实验比较不可混为同一配对。'
            item['related_experiments']=list(dict.fromkeys(item['related_experiments']+['EXP-043','EXP-044']))
            item['evidence']=list(dict.fromkeys(item['evidence']+['experiments/EXP-044-format120-parallel/report.md','experiments/EXP-044-format120-parallel/metrics/verified.json']))
    save(ideas_path,ideas)
    print(json.dumps({'status':'COMPLETE_VERIFIED','raw_ba':p['raw120']['balanced_accuracy'],'clean_ba':p['clean120']['balanced_accuracy'],'guard_seconds':seconds}))

if __name__=='__main__':main()
