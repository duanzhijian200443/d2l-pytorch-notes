"""Plot only accepted dynamic-route score results; never fabricate blocked arms."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / 'outputs/summary.json').read_text(encoding='utf-8'))
comparison = summary['comparisons']['A1-A0']
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
values = [comparison['reference']['macro_recall']*100, comparison['candidate']['macro_recall']*100]
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
fig.patch.set_facecolor('white')
ax.bar([0, 1], values, width=.5, color=['#64748b', '#217a80'])
ax.set_ylim(0, 100)
ax.set_xticks([0, 1], ['原完整方案 A0', '替换公共数据配方 A1'], fontproperties=font)
ax.set_ylabel('17类宏平均召回率（%）', fontproperties=font)
ax.set_title('完整ATLAS20本地缓存对照：最终预测相同', fontproperties=font, pad=16)
for i, value in enumerate(values):
    ax.text(i, value+2, f'{value:.4f}%\n488 / 729 答对', ha='center', va='bottom', fontproperties=font)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.15)
ax.set_axisbelow(True)
fig.text(.08, .05, '已评过的729条外部样本／56题组；非独立OOF或榜单分数。\nA2、A3缺4条ATLAS预测，未计算完整分数。', fontproperties=font, fontsize=9, color='#475569')
fig.subplots_adjust(bottom=.23, top=.85, left=.12, right=.97)
destination = HERE / 'figures'
destination.mkdir(exist_ok=True)
fig.savefig(destination / 'public-data-full-stack.png', bbox_inches='tight')
plt.close(fig)
