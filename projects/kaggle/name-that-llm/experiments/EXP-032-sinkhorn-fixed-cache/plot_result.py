"""Standalone scientific figure from the accepted fixed-comparison summary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
s=json.loads((root/'analysis-v1/summary.json').read_text())
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
fig,(ax,dx)=plt.subplots(1,2,figsize=(10,3.3),gridspec_kw={'width_ratios':[1,1.35]})
values=[s['baseline_045']['macro_recall']*100,s['candidate_030']['macro_recall']*100]
bars=ax.bar(['0.45 对照','0.30 候选'],values,color=['#7d8793','#257c8b'],width=.48)
ax.set_ylim(0,75);ax.set_ylabel('17 类宏平均召回率（%）')
for bar,value in zip(bars,values):
    ax.text(bar.get_x()+bar.get_width()/2,value+1.4,f'{value:.2f}%',ha='center')
ax.set_title('729 条 / 56 题 · 外部探索面板',fontsize=11)
delta=s['delta_030_minus_045']['macro_recall']*100
lo,hi=[v*100 for v in s['paired_prompt_bootstrap']['ci95']]
dx.axvline(0,color='#777777',lw=1)
dx.errorbar(delta,.5,xerr=[[delta-lo],[hi-delta]],fmt='o',capsize=6,color='#257c8b',lw=2)
dx.set_xlim(-.2,1.2);dx.set_ylim(0,1);dx.set_yticks([])
dx.set_xlabel('0.30 − 0.45：宏召回差值（百分点）')
dx.set_title(f'+{delta:.3f} pp；95%区间 [{lo:+.3f}, {hi:+.3f}]',fontsize=11)
dx.text(delta,.67,'改对 5 条 · 改错 1 条 · 净 +4 条',ha='center',fontsize=10)
dx.text(.5,.24,'区间跨 0；尚不足以认定稳定提升',ha='center',transform=dx.transAxes,fontsize=10)
for a in (ax,dx):
    a.spines[['top','right']].set_visible(False)
    a.grid(axis='x' if a is dx else 'y',alpha=.15);a.set_axisbelow(True)
fig.text(.5,.02,'固定预测、按题组重采样 2000 次；缺少 Gemini 3.7 真值，不是官方18类或榜单成绩。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,1))
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures/fixed-comparison.png',dpi=160,bbox_inches='tight')
print('Saved figures/fixed-comparison.png')
