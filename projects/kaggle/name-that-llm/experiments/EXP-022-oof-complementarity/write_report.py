"""Render the project's Markdown research record from completed cached analysis."""
from pathlib import Path
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M = HERE / 'metrics'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result = json.loads((M/'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'COMPLETE' and result['scenarios'] == 16
    summary = pd.read_csv(M/'summary.csv')
    classes = pd.read_csv(M/'per_class.csv')
    folds = pd.read_csv(M/'per_fold.csv')
    prompts = pd.read_csv(M/'per_prompt.csv')
    class_folds = pd.read_csv(M/'per_class_fold.csv')
    dense = result['summaries'][0]
    assert dense['scenario'] == 'dense'
    scenario_rows = []
    for m in (2, 3, 6):
        group = summary[summary.m.eq(m)]
        assert len(group) == 5
        scenario_rows.append({'m': m, 'repeats': 5, 'rows_per_repeat': int(group.support.iloc[0]),
            'stylegram_mean_ba': float(group.stylegram_balanced_accuracy.mean()),
            'svc4_mean_ba': float(group.svc4_balanced_accuracy.mean()),
            'mean_delta_ba': float(group.delta_balanced_accuracy.mean()),
            'delta_min': float(group.delta_balanced_accuracy.min()), 'delta_max': float(group.delta_balanced_accuracy.max()),
            'svc4_ahead_repeats': int((group.delta_balanced_accuracy > 0).sum()),
            'mean_stylegram_centred_fraction': float((group.stylegram_centred_rows/group.support).mean()),
            'mean_svc4_only_correct': float(group.svc4_only_correct.mean()),
            'mean_stylegram_only_correct': float(group.stylegram_only_correct.mean())})
    dclass = classes[classes.scenario.eq('dense')].copy()
    dfold = class_folds[class_folds.scenario.eq('dense')]
    cf = dfold.groupby('model').agg(svc_only_folds=('svc4_only_correct', lambda x: int((x > 0).sum())),
                                  svc_net_positive_folds=('delta_recall', lambda x: int((x > 1e-12).sum())))
    dclass = dclass.join(cf, on='model').sort_values(['svc4_only_correct', 'model'], ascending=[False, True])
    dclass.to_csv(M/'dense_class_stability.csv', index=False, lineterminator='\n')
    dp = prompts[prompts.scenario.eq('dense')]
    top_prompt_counts = dp.svc4_only_correct.sort_values(ascending=False)
    findings = {'scope': 'descriptive existing-member comparison; no deployable fusion evaluated',
        'dense': dense, 'sparse_summaries': scenario_rows,
        'dense_svc_only_prompt_coverage': int((dp.svc4_only_correct > 0).sum()),
        'dense_svc_only_top5_prompt_count': int(top_prompt_counts.head(5).sum()),
        'dense_svc_only_max_one_prompt': int(top_prompt_counts.max()),
        'dense_svc_only_top5_class_count': int(dclass.svc4_only_correct.head(5).sum()),
        'undefined_bootstrap_scenarios': [x['scenario'] for x in result['summaries'] if x['paired_prompt_bootstrap']['status'] != 'OK'],
        'next_experiment_recommendation': 'Predeclare a small, separately validated calibration/blend comparison across dense and sparse settings; do not use true class or true prompt size as a selector.',
        'analytical_sources': {p.name: sha(p) for p in [M/'result.json', M/'summary.csv', M/'per_class.csv', M/'per_fold.csv', M/'per_prompt.csv', M/'per_class_fold.csv']}}
    (M/'findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# EXP-022 · Stylegram 与 svc4 的错误互补性', '',
        '**值得进入下一轮小规模融合验证：两者有分散的独有正确样本，而且强弱会随回答数量改变。当前没有运行融合，也没有得到新的榜单成绩。**', '',
        '对照为EXP-010（Stylegram公共份额50%）与EXP-018（svc4原始公共样本每行权重0.35）。这是两个不同成员/数据配方的比较，不能解释为公共权重大小的因果实验。', '',
        '## 1. 完整回答组：Stylegram整体更强，svc4仍有互补', '',
        '| 结果 | 样本数 |', '|---|---:|',
        f"| 两者都对 | {dense['both_correct']} |",
        f"| 仅Stylegram对 | {dense['stylegram_only_correct']} |",
        f"| 仅svc4对 | {dense['svc4_only_correct']} |",
        f"| 两者都错 | {dense['both_wrong']} |", '',
        f"2699条、150个prompt、18类。Stylegram BA={dense['stylegram_balanced_accuracy']:.6f}，svc4 BA={dense['svc4_balanced_accuracy']:.6f}。svc4相对Stylegram低{-dense['delta_balanced_accuracy']*100:.3f}个百分点，五折均较低，因此现有结果不支持整体用svc4替换Stylegram。", '',
        f"svc4独有的{dense['svc4_only_correct']}条正确预测覆盖{dense['svc4_only_correct_prompt_count']}个prompt、全部18类和全部5折；每折31–44条。最多一个prompt贡献{findings['dense_svc_only_max_one_prompt']}条，贡献最多的5个prompt合计{findings['dense_svc_only_top5_prompt_count']}条（{findings['dense_svc_only_top5_prompt_count']/dense['svc4_only_correct']:.1%}）。这表明互补不只来自一两个题目，但是否能在新样本上选对模型尚未验证。", '',
        '**187条独有正确是已知真值下的诊断，不是预计能增加187条正确，也不是融合提升或可部署的选择规则。**', '',
        '## 2. 稀疏回答组：只剩两条时，svc4领先', '',
        '| 每题保留回答数 | Stylegram平均BA | svc4平均BA | svc4−Stylegram（百分点） | svc4领先抽样数 |',
        '|---|---:|---:|---:|---:|']
    for x in scenario_rows:
        lines.append(f"| {x['m']} | {x['stylegram_mean_ba']:.6f} | {x['svc4_mean_ba']:.6f} | {x['mean_delta_ba']*100:+.3f} | {x['svc4_ahead_repeats']}/5 |")
    lines += ['', '每题2条时svc4五次都更高；3条时接近、方向不一致；6条及完整组仍由Stylegram领先。各次先合并五折再计算18类BA，表中为五次固定抽样的描述均值。五次共享训练模型和prompt，不是五次独立训练，也不能据此声称隐藏测试集一定属于某种场景。', '',
        'Stylegram用了各稀疏场景重新聚类、中心化后保存的预测；svc4逐行预测不依赖同批其他回答，按相同ID选取缓存。中心化路由覆盖随场景变化，可能与强弱变化有关，但本分析没有隔离中心化本身的因果效果。真实prompt组大小在比赛推理时也不直接可知，不能据表格写“m=2就切svc4”的真值路由。', '',
        '## 3. 哪些类别值得继续观察', '',
        '| 真实类别（仅诊断） | svc4独有正确 | Stylegram独有正确 | svc4独有正确出现折数 | svc4净召回更高折数 |',
        '|---|---:|---:|---:|---:|']
    for _, x in dclass.head(5).iterrows():
        lines.append(f"| {x.model} | {int(x.svc4_only_correct)} | {int(x.stylegram_only_correct)} | {int(x.svc_only_folds)}/5 | {int(x.svc_net_positive_folds)}/5 |")
    topwrong = dclass.sort_values('both_wrong', ascending=False).head(3)
    common = '、'.join(f"{x.model}（{int(x.both_wrong)}/{int(x.support)}）" for _, x in topwrong.iterrows())
    lines += ['', 'gemini-3.7-flash、gemini-3.6-flash等类别提供较多互补样本，但净召回优势并非所有折一致；gpt-5.6-luna虽然有独有正确，整体召回仍更低。这些观察用于提出后续假设，不能按真实类别在验证集上挑选模型。', '',
        f'共同错例最多的三类为：{common}。值得优先查看来源与混淆类型；这不证明“只需增加数据”，也不意味着分数融合绝无可能修正两者top-1都错的样本。', '',
        '## 4. 下一轮建议', '',
        '先冻结少量融合/校准候选，针对完整与稀疏场景一起检查，目标是保留Stylegram的完整组表现并减少稀疏组退化。两成员的SVM margin不是概率，不能直接当作概率相加；也不能在这份已用于开发的OOF上反复挑权重后称为独立提升。需要新的独立prompt验证或真正嵌套的分组训练/校准设计；把现有OOF简单再切折不自动提供严格独立性。', '',
        '这一步仅提出下一轮实验，不启动训练或融合、不改变正式EXP-004/0.662。两成员本来就与现有基线有关，当前也不是向完整ATLAS20新增两个模型的证据。', '',
        '## 证据与统计范围', '',
        '- [配置](config.json) · [冻结清单](manifest.json) · [分析代码](analyze.py) · [输入检查](metrics/input-preflight.json)。',
        '- [16场景结果](metrics/result.json) · [场景表](metrics/summary.csv) · [逐类](metrics/per_class.csv) · [逐折](metrics/per_fold.csv) · [逐prompt](metrics/per_prompt.csv) · [类×折](metrics/dense_class_stability.csv)。',
        '- [逐行配对预测](oof/paired_predictions.csv)记录场景、ID、两者预测和推断路由；不含融合预测。',
        '- 103个输入绑定原验收/下载哈希；核对75个Stylegram稀疏NPZ、5个完整折缓存和5个svc4折缓存。未重新拟合或重建模型内部计算；上游官方长度统计、未知语义题目关联等限制仍存在。',
        '- 每个场景另给2000次prompt配对bootstrap区间，条件于这套已拟合开发模型，未校正多重比较。m2-repeat0有1次抽样缺少类别，按预定规则将其区间置空，没有删除或补抽；其原始BA和四格计数仍完整。',
        f"- 缓存计算耗时{result['elapsed_seconds']:.3f}秒，不含实现、复核和报告时间；没有新训练、GPU/API使用或比赛提交。", '']
    (HERE/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    paths = [HERE/'report.md', M/'findings.json', M/'dense_class_stability.csv']
    (M/'report-manifest.json').write_text(json.dumps({'renderer_sha256': sha(Path(__file__)), 'files': [{'path': p.relative_to(HERE).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered report and findings from fixed analysis outputs.')


if __name__ == '__main__':
    main()
