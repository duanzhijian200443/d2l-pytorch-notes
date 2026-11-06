"""Scientific figure from accepted fixed raw/clean comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'metrics/verified.json').read_text(encoding='utf-8'))
assert r['status']=='PASS_FROZEN_RAW_CLEAN_OOF'
p=r['primary']; delta=p['clean_minus_raw_ba']*100
lo,hi=np.array(r['paired_prompt_bootstrap']['ci95'])*100
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.7),dpi=150,gridspec_kw={'width_ratios':[1,1.25]})
fig.patch.set_facecolor('white')
values=[p['raw120']['balanced_accuracy']*100,p['clean120']['balanced_accuracy']*100]
ax.bar([0,1],values,width=.55,color=['#64748b','#b57851'])
ax.set_ylim(0,60)
ax.set_xticks([0,1],['旧原文120','去格式120'],fontproperties=font)
ax.set_ylabel('18类OOF Balanced Accuracy（%）',fontproperties=font)
for i,x in enumerate(values):ax.text(i,x+1.3,f'{x:.4f}%',ha='center',fontproperties=font,fontsize=10)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
bx.axvline(0,color='#64748b',linestyle='--',linewidth=1)
bx.errorbar(delta,0,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#b57851',capsize=5,linewidth=2)
bx.set_xlim(-.32,.18);bx.set_ylim(-.6,.6);bx.set_yticks([])
bx.set_xlabel('去格式 − 原文（百分点）',fontproperties=font)
bx.set_title('差异很小，95%区间跨过0',fontproperties=font)
bx.text(delta,.19,f'{delta:+.4f}',ha='center',fontproperties=font)
bx.text((lo+hi)/2,-.20,f'95%区间 [{lo:+.4f}, {hi:+.4f}]',ha='center',fontproperties=font,fontsize=9)
bx.grid(axis='x',alpha=.15)
for x in (ax,bx):x.spines[['top','right']].set_visible(False)
fig.suptitle('EXP044：去换行和加粗，未观察到稳定收益',fontproperties=font,fontsize=13)
fig.text(.06,.03,'2699条／150题固定五折；题组配对bootstrap 2000次。单成员开发OOF，非整套方案或榜单分数。',fontproperties=font,fontsize=9,color='#475569')
fig.subplots_adjust(left=.085,right=.98,top=.80,bottom=.22,wspace=.30)
(HERE/'figures').mkdir(exist_ok=True)
fig.savefig(HERE/'figures/format-result.png',bbox_inches='tight')
plt.close(fig)
