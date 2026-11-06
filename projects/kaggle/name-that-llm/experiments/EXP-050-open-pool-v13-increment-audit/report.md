# EXP-050：Open Pool V13 逐行增量审计

**审计完成；未批准入训练、未训练、未生成、未运行GPU、未提交或修改云端任务。** 固定平台 V13 的190,588行/18标签已逐行读取。相对A1实际运行时用数联合，有97,870条文本重合，剩余92,718条是未命中的上界；它们不等于未被现成checkpoint见过的数据，更不等于独立验证数据。

## 固定输入和证据

来源：[Open Pool平台V13](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/13)，作者发行名v11。已有数据卡和清洗说明复用SRC-065。本次只下载固定版本主parquet、prompts、Gemini3.7隔离文件及实际基线缺失的prvs文件；完整原件留在`data/open-pool-v13-audit/`，不改旧原件。

- 主文件96,388,050字节，SHA256 `38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5`。
- 复用官方train、A1的10073条extra、matched/pool-v5/pool-v2、五个CSV来源及EXP008历史风险索引。所有当前比较输入在运行前后重核SHA256；[config.json](config.json)列出完整路径/字节/哈希。
- 主运行参考是EXP031/046已验收的文件字节清单。未把旧EXP004每个历史权重或所有checkpoint原训练语料认证为相同；未读取整个权重文件。

## 查明的基线版本差异

EXP031/046附件请求元数据写prvs平台`/6`，但实际运行manifest中的`passages_clean.csv`是7,519,791字节、hash `502653b8…`。本次固定下载平台V6得到7,314,552字节、hash `6ec42c0b…`，与EXP008旧原件一致，却不匹配实际运行。脚本按预期停止，没有把它当作实际输入。

随后MCP固定V7文件清单和V7单文件下载验证：V7字节/hash精确匹配实际运行清单。因此本审计用**平台V7的5070行**重建实际共享EXTRA；以原loader函数AST做隔离的数据转换复核，得到**10469行/716组**，逐行一致。旧V6为4926行，解释了旧10325投影与10469实际值之间的144行差异。[差异回执](download-version-discrepancy.json)保留失败和修正依据。这个发现不证明所有附件版本选择的通用机制，旧台账未在本子任务修改。

原loader有六条文件glob，已核运行manifest：实际匹配五个CSV（Seyit、Lucas镜像、Abdullah镜像、Dophuchung、prvs），`ntl_generated*.csv`没有匹配。加上dhruvtibarewal的matched/pool数据，覆盖六个公共数据来源；没有拿无匹配glob虚构第六个CSV。

## 文本重合和新增上界

原样匹配用UTF-8正文SHA256；规范化匹配用NFKC、casefold及空白合并后的SHA256。下表为V13行中规范化匹配数，各行重合不可相加。完整原样/规范化对照见[overlap.csv](metrics/overlap.csv)。

| 参照 | 参照行数 | V13命中行数 |
|---|---:|---:|
| 官方train | 2699 | 0 |
| A1 Stylegram固定extra | 10073 | 5762 |
| matched原件 | 24443 | 17252 |
| pool-v5原件 | 119899 | 97598 |
| pool-v2原件（挂载但不作为main选定pool） | 37252 | 29791 |
| main实际去重后的现场训练表 | 123828 | 97845 |
| 实际共享EXTRA | 10469 | 5767 |
| Seyit原CSV | 4998 | 4886 |
| Lucas实际镜像原CSV | 4255 | 3781 |
| Abdullah实际镜像原CSV | 1033 | 997 |
| Dophuchung原CSV | 1029 | 963 |
| prvs实际V7 clean CSV | 5070 | 679 |

main实际表由官方2699、matched24443、去重后pool-v5 96686构成。A1实际运行用数联合还纳入A1 Stylegram、共享EXTRA中GPT/Gemini/GLM gate使用部分，以及Claude gate完整Seyit输入；没有误把原共享EXTRA全量当成替换后的A1 Stylegram。

- A1实际运行用数联合命中**97870**，V13未命中**92718**。
- 全部可得原始公共输入加A1和官方的联合命中**99107**，V13未命中**91481**。
- 既有各参照命中行均没有发现跨标签冲突；主池内原样正文重复、规范化正文重复和同正文不同标签均为0。
- 没有命中官方train正文或EXP008/实际prvs已知train_regen正文，但此项只排除本次可观测的精确/规范化关系，**未重新计算全量语义近似，也不能证明prompt或checkpoint训练隔离**。

## 字段、分组和来源风险

23字段均无null，id无重复、正文/标签/主来源无空值，标签与provider字段一致。词数实算与n_words一致，范围36–275。55主来源的`also_in`集合大小与n_sources一致，主来源均包含在集合中；许可字段是发布者声明，不是我们完成了上游许可审查。

仍有5708条`source_file`为空、157495条`origin_hint`为空，详见[复算记录](verification.json)。这两列的缺失不会由“没有null”自动消失。可连接EXP008历史路线声明的只有919条，没有发现已知identity_mismatch；其余不能因flag为False就认定直连官方API。所有行`independently_verified_served_model=false`。

| prompt层级 | 行数 | 使用边界 |
|---|---:|---|
| `p`，有恢复的跨源prompt标识 | 149022 | 包含14108条centroid推断，仍需确认祖先关系 |
| `s`，仅来源内部标识/id_only | 39641 | 不能保证跨来源同题不被拆折 |
| 空标识/none | 1925 | 独立未知，不能把全部空值当成一个prompt |

非空canonical共31354组；13021条有prompt正文的表能一对一连回主池，组内标签数、passages计数均一致，确有592个18标签完整组。恢复prompt正文在规范化后没有跨canonical重复；已匹配的A1祖先未出现被拆进多个V13 canonical的情况。**这不覆盖无法匹配/推断错误/语义改写的祖先。**

66122条位于重复`(prompt_canon, model)`单元，正文互异，表示同题同模型可能有多份生成，不能把“panel_labels=18”当作恰好18行或把每行视为独立题目。未来选样和划分必须按prompt祖先分组，保留来源及重复版本。

Gemini3.7主池3278条、四个主来源，全部flag_kb_proxy=False；隔离文件10593条均为Gemini3.7，与主池规范化正文交集0。四个主来源及数量：davidpiln2007、alperen704、lxdlxdlxd301、prvs266；这是文件内部与作者Vertex-route声明的一致性，不是独立served-model认证。[source-route-claims.csv](metrics/source-route-claims.csv)保留所有来源/模型/flag/origin分布。

## match_weight与OOF边界

权重有限且正值，范围0.02–50，各类均值约1；有效样本比例约30.3%–78.6%。对某一类极大的权重会降低有效样本量，不能将发布权重直接作为未经试验的loss weight。[per-label.csv](metrics/per-label.csv)有逐类权重与有效样本量。

作者相似度字段有3行恰为0.300（ntlop_250315、ntlop_251082、ntlop_259822），无值大于0.3；由于字段保留的精度有限，本次不能确认未舍入数值是否严格小于0.3，不将卡片“all below0.3”扩写为独立确认。该字段也不是我们重新训练/计算的相似度。

**上游全train依赖仍存在。** SRC065的v11_append_log显示1024个候选按官方train逐标签标点统计删除，264个按官方train内容相似度删除；发布match_weight使用全官方train拟合。此次所有190588行标记`upstream_full_train_selection_exposure=true`及`official_train_fitted_match_weight=true`。仅按折重算权重无法撤销先前删行；要声称严格OOF独立，应从足够原始候选按折重做相关筛选，或使用真正未参与上游筛选/匹配的外部验证。当前只能做清楚标注边界的开发对照。

## 可执行的下一步候选（尚未训练）

已生成只含ID、来源、标签、prompt、正文hash和风险标记的两个候选索引，**approved_for_training均为false**，没有移动或清洗原文。

1. **C1恢复prompt的文本增量**：42143条/8156组，18标签；来自`own_text/own_bank/other_text`，排除source-only/未知/centroid及已知冲突和官方prompt关系。与已审计运行时输入无规范化正文重复，但可为既有prompt新增回答。最小类Qwen491条，可先设计固定每类上限、固定来源配额、保持原样本每行权重的单项CPU对照，不先用发布match_weight。
2. **C2恢复prompt且未命中运行时已知prompt的子集**：29478条/5061组，18标签，最小类Qwen471条。与C1分开检验“新prompt覆盖”假设；未命中仅限可观测canonical/正文桥接，不称为全部语义新题。索引见[C2候选](metrics/C2-new-recovered-prompt-increment-candidate-index.csv)。
3. **Gemini3.7路线与类权重单独对照**：3278主池/10593隔离不混用，优先核对四来源原始route记录；若以后试验，固定其他类、总权重和来源配额，单独比较路线过滤的收益及Gemini3.6/3.8误报代价，不能同时改全池和权重。

这些候选只缩小后续复核与消融范围。公开池许多增量来自旧训练池之外，但现成大模型权重可能已用同源文本；本实验没有训练收益结论。

## 复跑与验收

代码：[src/open_pool_audit.py](../../src/open_pool_audit.py)，10个针对性测试覆盖规范化边界、冲突、空prompt、panel、原loader筛选、main首行选择、输入hash和拒绝覆写。完整审计通过原loader AST逐行比对；另用PyArrow读取及Python集合实现复算文本联合、全部row ID/hash和候选清单，结果一致。测试回执见[test-result.json](test-result.json)，复算见[verification.json](verification.json)。

稳定只读入口（不下载、不写结果）：

```powershell
& 'C:\Users\34331\AppData\Local\uv\cache\environments-v2\audit-e5542b9a7a231d97\Scripts\python.exe' -B -X utf8 src/open_pool_audit.py --verify --config experiments/EXP-050-open-pool-v13-increment-audit/config.json
```

重新审计必须使用新的本EXP子目录，已有输出拒绝覆写：

```powershell
& 'C:\Users\34331\AppData\Local\uv\cache\environments-v2\audit-e5542b9a7a231d97\Scripts\python.exe' -B -X utf8 src/open_pool_audit.py --config experiments/EXP-050-open-pool-v13-increment-audit/config.json --output-dir experiments/EXP-050-open-pool-v13-increment-audit/metrics-rerun-NEW
```

固定版本下载是独立脚本`download_inputs.py`；它复核下载收据后复用原件，错误停止，没有自动重试循环，不保存签名URL或凭据。当前`metrics/summary.json`与`metrics/audit-complete.json`是最终有效结果；`metrics-first-pass/`仅保留首次实现检查的历史产物，不作为第二套当前结论。

本子任务未修改tracking台账、全局README或其他实验，等待根窗口统一登记与check/render。
