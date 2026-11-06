"""Render the predeclared small blend screen without expanding its grid."""
from pathlib import Path
import hashlib
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
M=HERE/'metrics'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=json.loads((M/'result.json').read_text(encoding='utf-8'))
    config=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    assert result['status']=='COMPLETE' and result['candidate_scenario_count']==48
    assert result['alphas']==[0,.05,.10,.20] and result['primary_alpha']==.10
    table=pd.read_csv(M/'summary.csv');dense=table[table.scenario.eq('dense')]
    primary=table[table.alpha.eq(.10)];primary_dense=next(x for x in result['summaries'] if x['scenario']=='dense' and x['alpha']==.10)
    sparse=[]
    for alpha in (.05,.10,.20):
        for m in (2,3,6):
            g=table[table.alpha.eq(alpha)&table.m.eq(m)]
            assert len(g)==5
            sparse.append({'alpha':alpha,'m':m,'reference_mean_ba':float(g.reference_balanced_accuracy.mean()),
                'blend_mean_ba':float(g.blend_balanced_accuracy.mean()),'mean_delta_ba':float(g.delta_balanced_accuracy.mean()),
                'min_delta_ba':float(g.delta_balanced_accuracy.min()),'max_delta_ba':float(g.delta_balanced_accuracy.max()),
                'positive_repeats':int((g.delta_balanced_accuracy>1e-12).sum()),
                'negative_repeats':int((g.delta_balanced_accuracy< -1e-12).sum()),
                'tied_repeats':int((g.delta_balanced_accuracy.abs()<=1e-12).sum())})
    pd.DataFrame(sparse).to_csv(M/'sparse_summary.csv',index=False,lineterminator='\n')
    findings={'status':'DEVELOPMENT_SCREEN_COMPLETE','primary_alpha':.10,'primary_dense':primary_dense,
        'sparse_summaries':sparse,'screening':result['screening'],
        'recommendation':'Retain predeclared alpha0.10 for independent validation; do not declare stable dense gain or promote formal baseline. Alpha0.20 remains a recorded sensitivity result, not proven inferior.',
        'statistical_scope':'Conditional development estimates; unadjusted intervals; no independent holdout or retraining uncertainty.',
        'candidate_grid_expanded':False,'source_hashes':{p.name:sha(p) for p in [M/'result.json',M/'summary.csv',HERE/'config.json']}}
    (M/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# EXP-024 · Stylegram主体 + svc4小比例融合', '',
        '**主要候选10%通过了事先设定的开发点估计门槛，值得进一步独立验证；完整组只净多对2条，尚无稳定提分证据。**', '',
        '## 固定方案', '',
        '使用EXP-010 Stylegram（公共份额50%）与EXP-018 svc4（raw公共每行权重0.35）的原18维margin。每条样本、每个成员先做跨18类的population z-score，再计算`(1-alpha)×Stylegram + alpha×svc4`。分数先按行maxabs缩放，避免极大数溢出；常数行归零。', '',
        'alpha固定为0、5%、10%、20%；10%是运行前确定的主要候选，另两项是敏感性对照。全部场景用同一权重，没有按真实类别、真实prompt组大小、折或已见结果选权重。alpha0在全部16场景逐ID复现原Stylegram的argmax。', '',
        '**10%指标准化margin的系数，不是校准概率、10%样本改由svc4处理，也不是完整ATLAS20的融合比例。** 标准化会去掉两个成员各自跨类分数幅度信息，这是本轮固定的方案选择；没有拟合温度、校准器或元模型。', '',
        '## 完整组结果', '',
        '| svc4系数 | OOF BA | 相对原Stylegram（百分点） | 答对/2699 | 修正旧错/新增错 |',
        '|---|---:|---:|---:|---:|',
        f"| 0%（对照） | {primary_dense['reference_balanced_accuracy']:.9f} | 0 | {primary_dense['reference_correct']} | — |"]
    for _,r in dense.iterrows():
        lines.append(f"| {r.alpha:.0%} | {r.blend_balanced_accuracy:.9f} | {r.delta_balanced_accuracy*100:+.4f} | {int(r.blend_correct)} | {int(r.blend_only_correct)}/{int(r.reference_only_correct)} |")
    ci=primary_dense['paired_prompt_bootstrap']
    lines += ['',f"主要10%候选修正{primary_dense['blend_only_correct']}错、引入{primary_dense['reference_only_correct']}错，净多对{primary_dense['blend_correct']-primary_dense['reference_correct']}条。BA提升{primary_dense['delta_balanced_accuracy']*100:.4f}个百分点；按prompt配对95%区间[{ci['ci95_lower_delta_ba']*100:+.4f}, {ci['ci95_upper_delta_ba']*100:+.4f}]个百分点，跨过0。完整五折3升2降，不能称为可靠的整体提升。", '',
        '20%与对照答对总数相同，BA的极小差异来自类别召回权重；不能把它解读成20%已被证明更差。三组完整组区间均跨0。', '',
        '## 稀疏场景结果', '',
        '| 每题回答数 | 原Stylegram平均BA | 5%融合平均BA | 10%融合平均BA | 20%融合平均BA |',
        '|---|---:|---:|---:|---:|']
    for m in (2,3,6):
        group={x['alpha']:x for x in sparse if x['m']==m}
        lines.append(f"| {m} | {group[.10]['reference_mean_ba']:.6f} | {group[.05]['blend_mean_ba']:.6f} | {group[.10]['blend_mean_ba']:.6f} | {group[.20]['blend_mean_ba']:.6f} |")
    lines += ['', '每项为5次固定抽样的均值，先拼接五个验证折再计算18类BA；抽样重复共享同一批模型和prompt，不是独立训练重复。', '',
        '| 主要10%候选场景 | 平均变化（百分点） | 单次变化范围（百分点） | 升/降/平次数 |',
        '|---|---:|---:|---:|']
    for x in sparse:
        if x['alpha']==.10:
            lines.append(f"| m={x['m']} | {x['mean_delta_ba']*100:+.4f} | [{x['min_delta_ba']*100:+.4f}, {x['max_delta_ba']*100:+.4f}] | {x['positive_repeats']}/{x['negative_repeats']}/{x['tied_repeats']} |")
    lines += ['', '10%在m=2、m=3的五次抽样全部提高；m=6三升二降，因此“各m平均不降”并不等于每个稀疏场景都不降。20%在稀疏均值上更高，但它没有通过事前严格的完整组不降门槛，当前也没有证明其整体优于10%。', '',
        '## 预定门槛与下一步', '',
        '运行前规定：完整组BA变化≥0，m=2/3/6各自的五次平均变化≥0，且至少一项为正；数值tie容差1e-12。此规则只作开发筛选，不是统计非劣效或等效检验，也不会自动替换正式基线。', '',
        '| 候选 | 是否主要候选 | 点估计门槛 |', '|---|---|---|']
    for x in result['screening']:
        lines.append(f"| {x['alpha']:.0%} | {'是' if x['primary'] else '否'} | {'满足' if x['point_estimate_guardrail_met'] else '未满足'} |")
    lines += ['', '保留预定10%作为进一步验证的候选，保留20%的敏感性结果；不因为看到了结果继续加试更多权重，也不把10%称为全局最优。下一步需要另行设计独立prompt或真正嵌套分组验证，检验收益是否保留。当前OOF已经反复用于开发；固定小网格、重新切分已有OOF或bootstrap都不能把它变成未触碰测试集。', '',
        '本轮只涉及两个已选成员，不能推断完整ATLAS20的收益；原正式EXP-004/0.662保持独立。本轮没有正式提交、GPU使用或新训练。', '',
        '## 证据与限制', '',
        '- [冻结配置](config.json) · [清单](manifest.json) · [运行脚本](run_blend.py) · [输入预检](metrics/input-preflight.json)。',
        '- [48项完整统计](metrics/result.json) · [对比总表](metrics/summary.csv) · [稀疏均值与范围](metrics/sparse_summary.csv) · [逐类](metrics/per_class.csv) · [逐折](metrics/per_fold.csv)。',
        '- [逐行预测](oof/predictions.csv)区分原Stylegram、原svc4与融合预测；每场景NPZ保存4个系数的18维分数，见[oof目录](oof)。',
        '- Stylegram稀疏分数来自75份对应场景NPZ，非完整组分数的直接切片；svc4按同ID取其独立逐行预测。父实验训练数据与上游生成选择偏差等限制保持有效。',
        '- 区间用2000次prompt配对bootstrap、seed42，未做多重比较校正，不包括重训不确定性。m2-repeat0的共同重采样有1次缺类，三候选该场景区间按预定规则置空，无删抽或补抽。',
        f"- 纯缓存计算耗时{result['elapsed_seconds']:.3f}秒，不包含实现、测试、复核和报告时间。", '']
    (HERE/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=[HERE/'report.md',M/'findings.json',M/'sparse_summary.csv']
    (M/'report-manifest.json').write_text(json.dumps({'renderer_sha256':sha(Path(__file__)),'files':[{'path':p.relative_to(HERE).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Report rendered; no candidates added or selected post hoc.')


if __name__=='__main__':
    main()
