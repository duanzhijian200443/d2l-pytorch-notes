"""Prepare per-source intake notes; does not update the canonical registry."""
from pathlib import Path
import json
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
p=BASE/'manifest.json'
manifest=json.loads(p.read_text(encoding='utf-8'))
for row in manifest['entries']:
 folder=ROOT/row['artifact_dir']
 ref=row['ref']; v=row['notebook_version']; owner=ref.split('/')[0]
 row['tags']=['ensemble','version-binding','same-code']
 if owner=='lightningv08':
  row['tags']=['ensemble','postprocessing','ablation','runtime','version-binding']
  row['related_sources']=['SRC-003','SRC-014','SRC-021']
  row['summary']='V43 为未确认得分的新候选：GSQ 7%、ATLAS 10%、关闭边际频率平衡；页面0.662不可归给V43。'
  row['author_claim']='作者明确称本候选没有已确认榜单分；完整119899行公共池+两轮官方数据训练GSQ，保留37-marker分层标点。'
  row['verified']='静态比对SRC-003 V33：GSQ .075→.07、ATLAS .20→.10、balance strength .45→0，并加 strength>0 分支；三项同时改变。'
  row['use_for_us']='作为关闭类频率平衡的消融线索；参考缓存和顺序GPU进程工程。'
  row['test_plan']='固定相同分折预测与GSQ/ATLAS份额，仅比较balancing 0/.30/.45，再单独比较ATLAS与GSQ权重。'
  row['limitations']='未运行、未核查全部附件；作者的候选无已确认分数，不能继承页面0.662或Shiner V2的0.664。'
  body='''## 值得保留的结论

V43 是进一步关闭类频率平衡的候选。与已登记 SRC-003 V33 相比，同时改变 GSQ、ATLAS 份额和频率平衡强度；不能把跨版本的任何分数差单独归因于某一参数。作者正文明确说明这个候选没有已确认榜单成绩。

## 作者声称与实际设置

- 正文称 GSQ 模型用完整 119899 行公共池后接两轮官方数据训练；推理保留完整 token 序列，batch size 1，两张 T4 分担。此处只核查了声明和推理代码，未下载权重或训练日志证明训练过程。
- core/complementary 先按 85/15 混合；被选中行混入 10% ATLAS；随后全部行混入 7% GSQ；边际频率修正关闭。最后保留 37-marker 分层 Bayesian typography，未新增末尾 Hungarian 强制分配。
- 公开三条样例是严格按 ID、字节长度和文本 fingerprint 匹配的 CPU 格式预览；不代表完整隐藏集性能或耗时。
- 数据集附件为数值缓存/权重。代码中的通过标记和文件 inventory 是作者设置，不是我们的运行验收。

## 我们核查了什么

- 静态读了正文与全部代码单元。相对 SRC-003 V33 的 [逐行差分](vs-SRC-003.diff)：`gsq_share` 0.075→0.07；ATLAS 0.20→0.10；`balance_logits` 默认 strength 0.45→0.0，且只有 `strength > 0` 才进入修正分支；执行契约新增 `balance000`，记录强度为 0。
- 与 SRC-014 谱集成旧版本也高度同源，见 [差分](vs-SRC-014.diff)；相近行数比例只用于定位，不证明独立创新或等价。
- 元数据返回固定 Notebook V43；本轮未取得全局 scriptVersionId。源文件与代码 SHA256 见下载凭据。未执行代码。

## 对当前方案的用途与限制

对 IDEA-002（固定预测后比较后处理）有直接参考意义，但 V43 同时变三项，不是可用于归因的单变量实验。源码明确区分 Shiner 的得分 V2（SRC-021，0.664）与本候选，归档保留这个边界。网页列表0.662是 Notebook 页面分数，尚未绑定 V43；本条不登记已复现的收益。

## 待验证想法

固定训练范围可核查的分折预测、相同 GSQ 与 ATLAS 份额，先比较 balancing 0/0.30/0.45；再固定最佳候选单独比较 ATLAS 10%/20% 和 GSQ 7%/7.5%。使用 prompt 分组及稀疏场景，报告错误分歧和逐类召回；未启动这些实验。
'''
 else:
  row['related_sources']=list(dict.fromkeys(row['related_sources']+['SRC-002','SRC-009','SRC-010']))
  row['summary']=f'V{v} 代码逐字同 SRC-002/009/010；保留公开出处，不计为新方法。'
  row['author_claim']='Notebook 正文沿用 alexchilton 的全公开ATLAS20说明和失败实验表；不能把其中第一人称实验当作本页面作者新做的实验。'
  row['verified']='全部代码单元拼接SHA256为d164e570c03550b08337e9934b5f18af1255aeacdf1e0a0d279547bfac693c6f，与SRC-002/009/010完全一致。'
  row['use_for_us']='追踪公开复现出处及附件可用性；不作为独立算法或收益证据。'
  row['test_plan']='若需重跑先核查固定得分版本与附件版本/哈希；无需为同代码副本新增算法实验。'
  row['limitations']='页面0.662尚未绑定固定源码；附件版本未逐一核查，代码相同不保证运行输入与结果相同。'
  extra='本页虽名为 Hungarian，但代码与 SRC-002 完全相同，不能从标题推断其新增了 Hungarian 算法。' if owner=='sujanmajhisuzan' else ('V3 与此前 SRC-009 V1 代码也完全一致，版本变化不代表算法变化。' if owner=='junaid512' else '这是新作者出处，但未发现代码改动。')
  body=f'''## 值得保留的结论

{extra} 保留出处和固定版本，方法上归入已存在的 alexchilton ATLAS20 同代码家族。

## 作者声称与实际设置

正文沿用 alexchilton 全公开版：V31 主体、ATLAS 20%、公开 Llama 路径、8B 双 GPU 与池化优化。文内的失败实验和相同代码分数波动表均为上游已有文字，不能据此声称当前页面作者又独立做了一轮实验。

代码仍使用 complementary 15%、ATLAS 20%、Sinkhorn 0.45、target mix 0.35；正文中的标题或历史0.663说明不覆盖当前页面0.662。公开三条样例快速路径与隐藏评分路径分开。

## 我们核查了什么

- Kaggle API 返回 Notebook V{v}，公开状态为 false isPrivate；原始文件、元数据和哈希随条目保存。全局 scriptVersionId 未取到。
- 全部代码单元按顺序以双换行拼接后 SHA256=`d164e570c03550b08337e9934b5f18af1255aeacdf1e0a0d279547bfac693c6f`，与 SRC-002（alexchilton V8）、SRC-009（junaid V1）、SRC-010（dfenglei V1）完全一致。
- 本次新收集的 junaid V3、sujan V12、dazhijiang V1、gloria V1 同属这一家族。后三者完整 Notebook 源文件 SHA256 也一致；不是四种新方法。
- 未执行下载代码，没有重跑训练或正式提交。

## 对当前方案的用途与限制

用途是保存公开复现出处和潜在附件可用性线索。页面分数0.662由主合并窗口在列表观察，尚未绑定此源码版本；附件引用可能随发布版本变化，代码相同不能证明输入数据/权重完全相同，更不能证明单次成绩可复现。

## 待验证想法

不因同代码副本另开算法实验。若将来需要重跑，先固定源版本对应的得分、所有附件版本与哈希，再复用同一基线验证；若只有挂载变化，作为依赖修复记录。
'''
 notes=f'''# 待合并 · {row['title']} · V{v}

状态：已取得固定源码并作静态对比；未运行；正式 SRC 编号待合并窗口分配。

## 来源与原件

- 作者：{row['author']}（`{owner}`）。[来源页面](https://www.kaggle.com/code/{ref})。
- 采集时间：`{row['observed_at']}`。Notebook 版本：V{v}；全局 scriptVersionId：未取得。
- [源码](source.ipynb) · [API 元数据](metadata.json) · [下载凭据与哈希](receipt.json) · [原正文提取](description.md)。
- 页面分数0.662由主窗口实时网页观察；与本源码版本未绑定，以主窗口保留的页面证据为准。该分数不作为本版本已得分证明。

'''+body+f'''
## 更新记录

- 2026-11-01：首次纳入本批待合并资料。关联旧来源：{', '.join(row['related_sources'])}。本目录属于临时收集包，权威来源信息待合并到正式 SRC JSON。
'''
 (folder/'notes.md').write_text(notes,encoding='utf-8')
 row['notes_path']=(folder/'notes.md').relative_to(ROOT).as_posix()
 row['evidence']=[f.relative_to(ROOT).as_posix() for f in folder.iterdir() if f.name!='code.txt']
manifest['validation']={'each_entry_has_notes':all((ROOT/r['notes_path']).exists() for r in manifest['entries']),'source_count':5,'canonical_registry_modified':False,'ledger_check_render':'PENDING_PARENT_MERGE'}
p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest['validation']))
