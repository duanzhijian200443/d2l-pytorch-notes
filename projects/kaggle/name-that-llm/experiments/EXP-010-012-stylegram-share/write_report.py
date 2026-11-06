"""Render only verified outputs; never infer a result from job exit status."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
def md_table(df):
    out=['| '+' | '.join(df.columns)+' |','|'+'---|'*len(df.columns)]
    out += ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(out)

def main():
    p=HERE/'analysis';v=json.loads((p/'verified.json').read_text(encoding='utf-8'));assert v['status']=='PASS'
    d=pd.DataFrame(v['dense']).sort_values('share');base=float(d.loc[d.share.eq(.5),'balanced_accuracy'].iloc[0])
    d['变化（百分点）']=(d.balanced_accuracy-base).map(lambda x:f'{100*x:+.3f}')
    d['share']=d.share.map(lambda x:f'{x:.0%}');d['balanced_accuracy']=d.balanced_accuracy.map(lambda x:f'{x:.6f}');d['accuracy']=d.accuracy.map(lambda x:f'{x:.6f}')
    d=d.rename(columns={'share':'公共权重份额','balanced_accuracy':'OOF BA','accuracy':'普通准确率','correct':'正确行数'})
    paired=[]
    for x in v['paired']:
        paired.append({'份额':f"{x['share']:.0%}",'BA差（百分点）':f"{100*x['delta_ba']:+.3f}",'按prompt抽样区间（百分点）':f"[{100*x['bootstrap_low']:+.3f}, {100*x['bootstrap_high']:+.3f}]",'修正旧错':x['fixed_errors'],'引入新错':x['introduced_errors'],'提升/下降折数':f"{x['positive_folds']}/{x['negative_folds']}"})
    sparse=pd.read_csv(p/'sparse_summary.csv');sparse['share']=sparse.share.map(lambda x:f'{x:.0%}')
    for c in ['mean','std','min','max']:sparse[c]=sparse[c].map(lambda x:f'{x:.6f}')
    best=max(v['dense'],key=lambda x:x['balanced_accuracy'])
    winning=next((x for x in v['paired'] if x['share']==best['share']),None)
    if best['share']==.5:
        judgement='主指标仍由原0.5份额最高；保留原设置，未找到提高主指标的依据。'
    elif winning['bootstrap_low']<=0<=winning['bootstrap_high']:
        judgement=f"主指标最高的是{best['share']:.0%}，但与0.5的配对区间包含0；暂不据此替换强基线，保留为候选。"
    else:
        judgement=f"主指标最高的是{best['share']:.0%}，在本次拟合模型条件下配对区间未包含0；可进入下一轮稳定性/集成增量验证，不能据此宣称排行榜提升。"
    text=f"""# EXP-010/011/012：stylegram 公共训练权重对照

**{judgement}**

只比较隔离后的 stylegram 成员。0.5 为 EXP-010 对照，0.25 为 EXP-011，0.75 为 EXP-012；完整 ATLAS20 的 0.662 仍保留，本次不是正式提交或整栈 OOF。

## 主结果

{md_table(d)}

{md_table(pd.DataFrame(paired))}

区间使用2000次按prompt成组的配对重采样，条件于本次已拟合模型与开发折；未重训模型、未进行多候选选择校正。它不等于隐藏榜单的不确定性区间。逐折差异保存在 [verified.json](analysis/verified.json)，逐类召回见 [per_class.csv](analysis/per_class.csv)。

## 公平对照与实际范围

- 官方2,699行按固定 prompt-group-v1 五折；每折验证约30个prompt。公开输入在所有对照里统一排除252条已知官方prompt重建回答，留下10,073行/694组。
- 保留原四路TF-IDF特征、float64、SVM C=0.5、seed=42、max_iter=4000；每折词表/IDF只在该折训练官方文本与固定公共文本上拟合，三份额复用同一矩阵。
- 权重逐折、逐类重新计算：官方每行1-share，公共每行share×官方该类训练数/公共该类数；每类总权重保持该折官方计数。
- plain预测所有验证行；仅对无标签文本推断出的2–18行组，替换为centred预测。训练官方和公共文本分别按自己的prompt中心化；公共组允许大上限，与原配方一致。
- 验证批次的无监督内容聚类会读取验证文本，这是原推理流程的一部分；不读取验证真标签或真实prompt分组。已知关联得到隔离，但不能保证识别了所有语义相似或上游数据选择偏差。
- 当前没有断点恢复。此次所有权重在同一CPU运行环境完成，实际依赖版本见 [verified.json](analysis/verified.json)；未声明跨环境逐位复现。

## 稀疏回答压力测试

每个prompt抽2/3/6条，每种固定5次抽样。先选文本，再重新聚类与中心化；每次先拼接五个验证折，再计算18类BA，最后汇总5次抽样。没有直接平均缺类折的BA。

{md_table(sparse)}

这些抽样共享相同训练模型和官方prompt，彼此不是独立的训练重复；std只表示抽样场景变化。m值是压力测试设定，不是已确认的隐藏数据分布。

## 执行与产物

- Kaggle [私有CPU实验](https://www.kaggle.com/code/zhjianduan/name-that-llm-stylegram-share-cv)，实际训练为V2；GPU、TPU和互联网均关闭。
- 训练30个SVM（5折×3份额×plain/centred），GPU/API生成费用为0；主程序耗时 {v['runtime_seconds']:.1f} 秒。没有使用或改动大模型、LoRA权重。
- V1因Kaggle自动解开输入ZIP，在训练开始前因找不到压缩文件退出；V2只修复启动路径，实验算法不变。失败日志保留在 `infra-v1/`。
- 完整环境/代码/输入绑定、225份稀疏缓存与15份完整折缓存已验证。保存的分数是SVM decision_function，不是校准概率；argmax等价于原stylegram的plain/centred路由。
- [准备脚本](prepare.py)、[训练脚本](runner.py)、[结果核验](verify_results.py)、[预检查](preflight.json)、[执行状态](execution-state.json)。

下一步只把通过本地验证的候选带入更严格的稳定性与互补性检查；不要把这次单成员BA与0.662榜单分数直接相减。
"""
    (HERE/'report.md').write_text(text,encoding='utf-8')
    print(judgement)

if __name__=='__main__':main()
