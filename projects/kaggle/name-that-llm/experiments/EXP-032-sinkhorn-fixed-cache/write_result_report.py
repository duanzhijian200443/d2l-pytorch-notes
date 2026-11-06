"""Write derived experiment reports from the accepted frozen comparison."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAPTURE = ROOT / 'cloud/full-stack-capture-r1'
RESULTS = CAPTURE / 'outputs-v2/results'
ANALYSIS = HERE / 'analysis-v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def rel(path):
    return path.relative_to(ROOT).as_posix()

def main():
    summary = read(ANALYSIS / 'summary.json')
    result = read(RESULTS / 'result.json')
    worker = read(RESULTS / 'worker-result.json')
    accepted = read(ANALYSIS / 'analysis-manifest.json')
    assert result['status'] == 'COMPLETE' and read(ANALYSIS/'result.json')['status'] == 'COMPLETE'
    assert accepted['control_check']['status'] == 'PASS'
    now = datetime.now(timezone.utc).isoformat()
    a, b = summary['baseline_045'], summary['candidate_030']
    delta = summary['delta_030_minus_045']['macro_recall'] * 100
    low, high = [x * 100 for x in summary['paired_prompt_bootstrap']['ci95']]
    report = f'''# EXP-032 固定缓存比较 Sinkhorn 0.45 与 0.30

**真实比较已完成：0.30 在本面板净多答对4条，但差值区间跨0，保留为候选，不替换正式0.662基线。**

这是729条、56个题组、17个真实类别的外部探索面板；预测始终保留18类。它不是官方18类验证分数、干净OOF或Kaggle榜单成绩。面板已在EXP027查看，完整集成成员的训练隔离未得到证明。

## 实际结果

| 指标 | 对照0.45 | 候选0.30 | 差值 |
|---|---:|---:|---:|
| 17类宏平均召回率 | {a['macro_recall']:.6f} | {b['macro_recall']:.6f} | +{delta:.4f}个百分点 |
| 样本准确率 | {a['accuracy']:.6f} | {b['accuracy']:.6f} | +{(b['accuracy']-a['accuracy'])*100:.4f}个百分点 |
| 答对条数 / 729 | {a['correct']} | {b['correct']} | +4 |

0.30修正5条、引入1条新错；共9条预测变化，涉及8题。按56个原题整组配对bootstrap，2000次、seed42，宏召回差值95%区间为 **[{low:+.4f}, {high:+.4f}]个百分点**，2000次均包含17类。区间条件于现有模型与面板，不包含重新训练的不确定性；跨0不等于两者等价。

![固定参数对照](figures/fixed-comparison.png)

## 哪些类别发生变化

- Gemini3.1多对1条；Gemini3.6多对2条；GPT5.5、Sol各多对1条。
- Terra少对1条（原本判对，改成Astra）；其他真实类别答对数不变。
- 本面板预先排除了未核实服务身份的Gemini3.7，因此没有该类真值，但始终保留其预测列。0.45判为3.7的22条降至0.30的19条，其中两条改判3.6后正确、另一条仍错。这个缺类、非均衡面板可能影响平衡强度结论，不能删除这一列或利用真值类别占比来调参数。
- 在这份小面板上Kimi为0/17、GLM5.3为2/16，需检查来源身份、任务差异和混淆方向；样本量很小，不能外推为官方测试集的类别表现。

## 验收与复现

真实来源为EXP031固定V2，run_id `exp031-20261101T110842Z`。原始565个归档成员与外部固定清单逐项核验；输入、源代码、成员、设备、完整ATLAS覆盖及缓存数学重放通过。0.45重建最终分数最大绝对误差1.78e-15、全部argmax与原Parquet一致，然后才计算0.30和读取标签。两臂只改变Sinkhorn强度，两侧分别重算标点置信度门控。

冻结协议、代码与当时config字节保持不变；准备状态属于开跑前快照。实际状态以本报告、[执行结果](analysis-v1/result.json)、[验收清单](analysis-v1/analysis-manifest.json)及台账为准。

- [统计摘要](analysis-v1/summary.json) · [逐类结果](analysis-v1/per_class.csv) · [逐行预测](analysis-v1/predictions.csv) · [错误变化](analysis-v1/errors.csv)
- [原缓存下载收据](../EXP-031-full-stack-capture-r1/metrics/download-receipt.json)
- [独立统计复核](metrics/independent-verification.json)

本地比较复用已下载预测，没有GPU、模型调用、生成数据或正式提交。本轮不增加其他参数扫描，不用已查看面板选择新的系数。

## 新数据与下一步

采集窗口已完成EXP041审计，本窗口再次复核23个绑定文件哈希、378份原文哈希和跨协议题组划分。300条新Codex回答为75题×4模型；240训练/60保留，其中Luna/Terra120、Sol/Astra120。78条双渠道面板为10题的诊断材料，继续单独保存。

建议下一项固定对照是轻量分类器不加新增数据/加入Luna与Terra120条，保持同一基线、数据权重规则与验证划分；Sol/Astra另作实验。所有旧新、渠道和重复回答按原题一起分组，60条保留集不训练。新增数据只有GPT家族，不能认为它能直接修复Kimi/GLM问题。Codex身份仍为requested_only、底层snapshot未知，来源标签必须保留。

这一建议尚未启动训练、重新生成或正式入池。详见[EXP041数据审计](../EXP-041-aligned-channel-repeat-audit/report.md)。
'''
    (HERE/'report.md').write_text(report,encoding='utf-8')
    p31=ROOT/'experiments/EXP-031-full-stack-capture-r1'
    r31=(p31/'report.md').read_text(encoding='utf-8')
    r31=r31.replace('状态：**Kaggle 第2版运行中，尚未验收真实模型结果**。','状态：**Kaggle 第2版完成，真实缓存验收通过**。')
    r31=r31.replace('实际设备与模型阶段仍需输出核验。','实际设备与模型阶段现已通过输出核验，详情见下方收尾记录。')
    r31=r31.replace('尚无准确率或提分结论。完成后下载并核验输入解析、组件分数、ATLAS覆盖、重放及Parquet格式，再用本地标签侧车计算诊断指标。','输入解析、组件分数、ATLAS覆盖、重放及Parquet格式已通过核验；固定后处理比较与诊断指标见EXP032。尚无新的榜单提分证据。')
    marker='\n## 实际完成与收尾（2026-11-01）\n'
    if marker in r31:r31=r31.split(marker)[0]
    r31+=marker+f'''
Kaggle官方CLI实时确认固定V2 COMPLETE。运行于北京时间22:09:21结束；守护总耗时{result['elapsed_seconds']:.3f}秒（约2小时57分35秒），setup523.070秒、comp3840.104秒、main6138.553秒。此为执行墙钟耗时，不是精确计费GPU小时。

实际T4×2；main16、comp20、family gate5均匹配预定清单；comp适用728/729行，ATLAS206/206请求行完成。143项缓存重放检查最大误差0。全部归档与稳定产物哈希通过EXP032入口验收，未发现新增遗漏或异常降级。public/public_twin/Q35按预先固定的原算法可选分支跳过。

最终result、guard与state确认完整结束；官方查询不再运行，账户quota返回reserved0。缓存已保存本地[输出](../../cloud/full-stack-capture-r1/outputs-v2/results/)，[下载收据](metrics/download-receipt.json)独立固定清单SHA。保持未正式提交、无重开GPU；0.45/0.30真实统计见[EXP032报告](../EXP-032-sinkhorn-fixed-cache/report.md)。
'''
    (p31/'report.md').write_text(r31,encoding='utf-8')
    s=read(CAPTURE/'execution-state.json')
    s.update(status='COMPLETE_VERIFIED',observed_at_utc=now,gpu_status='COMPLETE',gpu_may_be_active=False,
        completed_at_utc=result['observed_at'],actual_hardware_verified=True,outputs_path=rel(RESULTS),
        artifacts_sha256=accepted['artifacts_sha256'],acceptance_path=rel(ANALYSIS/'analysis-manifest.json'),
        completion_discrepancy='Resolved: user previously mistook another collection task for this run; this V2 now genuinely COMPLETE.')
    save(CAPTURE/'execution-state.json',s)
    for exp in ('031','032'):
        path=ROOT/f'tracking/records/experiments/EXP-{exp}.json';rec=read(path)
        rec['observed_at']=now;rec['execution']['observed_at']=now
        rec['execution']['status']='COMPLETE';rec['submission']=None
        if exp=='031':
            rec['execution'].update(actual_hardware_verified=True,outputs_verified=True,elapsed_seconds=result['elapsed_seconds'])
            rec['validation'].update(scope='Verified real full-stack exploratory prediction cache; not clean OOF',leakage_audit='FULL_STACK_TRAINING_INDEPENDENCE_UNKNOWN')
            rec['conclusion']='V2真实模型缓存完成验收：729行/18列，main16/comp20/gates5，ATLAS206/206；重放143项误差0。'
            rec['limitations']='已查看外部17类面板，完整训练隔离未证明；不等于官方验证或榜单提升。'
            rec['next_action']='使用已完成EXP032统计；保留缓存与正式0.662基线，不重复推理。'
            rec['timing_note']='守护墙钟10654.553秒；setup523.070、comp3840.104、main6138.553秒。非精确计费GPU时间。'
            add=[rel(p31/'metrics/download-receipt.json'),rel(RESULTS/'result.json'),rel(ANALYSIS/'analysis-manifest.json')]
        else:
            rec['execution'].update(scope='real_cached_postprocessing_comparison',real_comparison_run=True)
            rec['inputs']['content_hashes_verified']=True
            rec['validation'].update(scope='Exploratory external 729 rows/56 prompts/17 covered classes; not official BA or clean OOF',
                covered_17_macro_recall=b['macro_recall'],control_covered_17_macro_recall=a['macro_recall'],
                delta_macro_recall=summary['delta_030_minus_045']['macro_recall'],
                delta_ci95=summary['paired_prompt_bootstrap']['ci95'],
                per_class_recall_path=rel(ANALYSIS/'per_class.csv'))
            rec['conclusion']=f'0.30相对0.45宏召回+{delta:.4f}个百分点，修正5错/新增1错，净多对4条；95%区间[{low:+.4f},{high:+.4f}]个百分点跨0，暂不升级正式基线。'
            rec['next_action']='保留0.30候选，不继续扫参；下一项拟固定轻量基线检验新增Luna/Terra120条，尚未启动。'
            rec['limitations']='已查看17类外部面板、完整训练隔离未知；缺少Gemini3.7真值可能影响平衡参数收益，不代表榜单。'
            add=[rel(ANALYSIS/'summary.json'),rel(ANALYSIS/'analysis-manifest.json'),rel(ANALYSIS/'predictions.csv'),rel(HERE/'metrics/independent-verification.json')]
            mf=read(HERE/'manifest.json');mf.update(real_cache_artifacts_sha256=accepted['artifacts_sha256'],
                real_cache_status='EXP031 V2 COMPLETE_VERIFIED',freeze_status='FROZEN_PROTOCOL_COMPLETED_COMPARISON',
                comparison_result_path=rel(ANALYSIS/'result.json'),completion_observed_at_utc=now)
            save(HERE/'manifest.json',mf)
        rec['evidence']=list(dict.fromkeys(rec['evidence']+add));save(path,rec)
    ideas=read(ROOT/'tracking/ideas.json')
    for x in ideas:
        if x['id']=='IDEA-002':
            x.update(status='evaluated_exploratory',phase='real_cache_verified_small_positive_inconclusive',observed_at=now,
                next_action='EXP031/032完成：0.30净多对4条，差值区间跨0；保留候选和0.662基线，不在已查看面板继续扫描参数。',
                evidence_assessment='实际完整缓存固定对比：17类宏召回+0.4465个百分点，95%区间[-0.0151,+1.0076]个百分点。缺类且训练独立性未知，非榜单收益。',
                planning_scope='真实推理和固定CPU比较已完成；无新训练、提交或重开GPU。')
            x['related_experiments']=list(dict.fromkeys(x['related_experiments']+['EXP-031','EXP-032']))
            x['evidence']=list(dict.fromkeys(x['evidence']+[rel(HERE/'report.md')]))
        if x['id']=='IDEA-005':
            x.update(status='ready_for_controlled_training_design',observed_at=now,
                next_action='复用EXP041已审计数据，固定轻量分类器与题组划分，仅对照加入/不加入Luna/Terra120训练条；Sol/Astra120另臂，60条holdout及78条诊断面板保留。未启动训练。',
                prerequisites='实际loader核验旧新协议/所有渠道/重复答案的全局prompt隔离，保留来源及requested_only标签身份限制；固定来源权重和预算后执行。',
                evidence_assessment='EXP041及本窗口交叉核查378原文/23文件哈希通过。格式改善、小样本渠道差值区间跨0；不是API等价或收益证明。',
                planning_scope='采集结果已读取验收并更新后续设计；没有启动新的采集、训练或提交。')
            x['related_experiments']=list(dict.fromkeys(x['related_experiments']+['EXP-038','EXP-040','EXP-041']))
            x['evidence']=list(dict.fromkeys(x['evidence']+['experiments/EXP-041-aligned-channel-repeat-audit/report.md']))
    save(ROOT/'tracking/ideas.json',ideas)
    print(json.dumps({'status':'REPORTS_UPDATED','control':a,'candidate':b,'delta_pp':delta,'ci_pp':[low,high]},ensure_ascii=False))

if __name__=='__main__':
    main()
