"""Static scientific figure for the accepted fixed ablation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'analysis-v1/summary.json').read_text())
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
fig,(a,b)=plt.subplots(1,2,figsize=(10,3.2),gridspec_kw={'width_ratios':[1,1.25]})
vals=[s['dense'][arm]['balanced_accuracy']*100 for arm in ('control','add120')]
bars=a.bar(['不加新数据','加入120条'],vals,width=.5,color=['#687b88','#d48a3f'])
a.set_ylim(0,60);a.set_ylabel('18类 Balanced Accuracy（%）');a.set_title('固定五折 · 2699条官方样本',fontsize=11)
for rect,v in zip(bars,vals):a.text(rect.get_x()+rect.get_width()/2,v+1.1,f'{v:.3f}%',ha='center')
d=s['paired']['delta_ba']*100;lo,hi=[x*100 for x in s['paired']['bootstrap']['ci95']]
b.axvline(0,color='#777777',lw=1)
b.errorbar(d,.5,xerr=[[d-lo],[hi-d]],fmt='o',color='#d48a3f',capsize=6,lw=2)
b.set_xlim(-.45,.35);b.set_ylim(0,1);b.set_yticks([])
b.set_xlabel('加入 − 不加：BA差值（百分点）')
b.set_title(f'{d:+.3f} pp；95%区间 [{lo:+.3f}, {hi:+.3f}]',fontsize=11)
b.text(.5,.73,'改对6条 · 改错8条 · 净少对2条',ha='center',transform=b.transAxes,fontsize=10)
b.text(.5,.23,'区间跨0，尚未看到稳定收益',ha='center',transform=b.transAxes,fontsize=10)
for ax in (a,b):ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
fig.text(.5,.02,'按150个题组成组重采样2000次；单成员开发OOF，非完整ATLAS20或榜单成绩。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,1));(HERE/'figures').mkdir(exist_ok=True)
fig.savefig(HERE/'figures/ablation-result.png',dpi=160,bbox_inches='tight')
print('Saved ablation-result.png')
