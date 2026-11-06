"""Derived eligibility labels, frozen-index duplicate screen and research figure."""
from pathlib import Path
import csv
import json
import hashlib
import unicodedata
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normhash(t):return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',t).casefold().split()).encode()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
d=load(EXP/'metrics/analysis.json')
bpath=ROOT/'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl'
ppath=ROOT/'experiments/EXP-038-paired-cloud-precheck-recovery/collection/combined-records.jsonl'
b=rows(bpath);p=rows(ppath)
reference=ROOT/'experiments/EXP-033-codex-source-contamination-audit/metrics/row-features.csv'
newhash={normhash(r['text']) for r in b+p};seen=Counter();matches=[]
with reference.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        seen[r['source']]+=1
        if r['text_hash_normalized'] in newhash:matches.append({k:r[k] for k in ('source','row_id','model','text_hash_normalized')})
save(EXP/'metrics/frozen-reference-overlap.json',{'scope':'EXP033 frozen four-model row-feature index; not a fresh platform/full-label/semantic overlap audit','reference':str(reference.relative_to(ROOT)),'reference_sha256':sha(reference),'by_source_rows':dict(seen),'new_rows':len(b+p),'overlaps':matches,'normalization':'NFKC + casefold + collapse whitespace'})
elig=[]
for r in b:
    panel_model=r['model_requested'] in ['gpt-5.6-luna','gpt-5.6-terra']
    label='RESERVED_HOLDOUT' if r['split']=='holdout' else ('CONTROLLED_TRAINING_CANDIDATE' if panel_model else 'EXPLORATORY_TRAINING_CANDIDATE_NO_PAIRED_CHANNEL_EVIDENCE')
    elig.append({'job_key':r['job_key'],'prompt_group':r['prompt_group'],'model':r['model_requested'],'split':r['split'],'source_protocol':'codex_subscription_single-paragraph-plain-text-v1','assessment':label,'quality_flags':r['flags'],'attempt_path_project':r['attempt_path_project'],'actual_training_started':False,'formal_pool_accepted':False})
(EXP/'metrics/training-eligibility.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in elig),encoding='utf-8')
save(EXP/'metrics/training-eligibility-summary.json',{'counts':dict(Counter(r['assessment'] for r in elig)),'unchanged_source_sha256':sha(bpath),'paired_panel':'78 answers remain diagnostic panel; no actual pool ingestion','policy':'All old/new protocol versions, models, routes and repeats with the same base prompt must share one split. No training utility claim. Keep raw length outlier, flag rather than silently remove.'})
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white'})
fig,(a,c)=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1,1.25]})
xx=np.arange(2);old=[224/300*100,58/300*100];new=[0,0]
a.bar(xx-.18,old,.34,label='对齐前（同题300条）',color='#788599');a.bar(xx+.18,new,.34,label='对齐后（300条）',color='#197f7a')
for x,y in zip(xx-.18,old):a.text(x,y+1.6,f'{y:.1f}%',ha='center')
for x in xx+.18:a.text(x,1.6,'0%',ha='center',color='#197f7a')
a.set_xticks(xx,['含换行','含加粗']);a.set_ylim(0,92);a.set_ylabel('回答占比（%）');a.set_title('显式格式偏移明显减少',loc='left',fontweight='bold');a.legend(frameon=False,fontsize=9,loc='upper right')
keys=['overall','gpt-5.6-luna','gpt-5.6-terra'];labs=['总体 · 10题组','Luna · 10题组','Terra · 9题组']
for i,k in enumerate(keys):
    r=d['summaries'][k]['metrics']['word_trigram_jaccard_excess'];v=r['mean'];lo,hi=r['ci95'];y=2-i
    c.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color='#197f7a',markersize=7)
    c.text(v,y+.19,f'{v:+.3f}  [{lo:+.3f}, {hi:+.3f}]',ha='center',fontsize=9)
c.axvline(0,ls='--',color='#788599',lw=1);c.set_yticks([2,1,0],labs);c.set_ylim(-.55,2.55);c.set_xlim(-.045,.12)
c.set_title('跨渠道差异 − 同渠道重复波动',loc='left',fontweight='bold');c.set_xlabel('词三元组 Jaccard 距离差（不是污染百分比）')
c.grid(axis='x',alpha=.16)
fig.suptitle('对齐后的渠道差异：未检出明显额外效应，但尚不能证明等价',fontsize=13,fontweight='bold',y=.99)
fig.text(.06,.02,'右图：19个完整题模单元；按原题组 bootstrap 20,000 次的95%区间。小样本探索性诊断，未训练分类器。',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.07,1,.94));(EXP/'figures').mkdir(exist_ok=True)
fig.savefig(EXP/'figures/aligned-comparison.png',dpi=180);plt.close(fig)
print(json.dumps({'eligibility_counts':dict(Counter(r['assessment'] for r in elig)),'reference_overlap_count':len(matches),'figure':'figures/aligned-comparison.png'},ensure_ascii=False))
