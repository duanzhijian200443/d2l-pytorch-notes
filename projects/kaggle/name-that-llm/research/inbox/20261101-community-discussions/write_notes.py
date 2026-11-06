"""Write per-source research notes and pending import proposals, not canonical SRCs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = {
754426: {
 'title':'HALX：轻量保存、输入检查与适配器本地暂存',
 'tags':['runtime','ablation','light-commit','io','ensemble'], 'related_sources':['SRC-026'],
 'summary':'作者将主流程 4884→2866 秒归因于适配器暂存；轻量 commit 成功不能证明隐藏重跑成功。',
 'author_claim':'轻量 commit 约1.8小时→167秒；适配器暂存 /tmp 后主流程4884→2866秒且预测相同。',
 'verified':'已读取 2026-10-31 原帖及当前评论响应；本轮未运行工具 Notebook 或复现计时。',
 'use_for_us':'把输入解析、CUDA 环境和必要输出完整性纳入保存前检查，测量适配器加载耗时。',
 'test_plan':'先固定同一源码/输入/环境，测 mount 与本地暂存的加载与主流程时间及预测哈希；可选支路降级需单独验证输出完整性。',
 'limitations':'作者单一环境自报，4884/2866 是主流程，不是完整隐藏评分；GPU额度观察不是平台政策。',
 'body':'''## 作者声称与实际设置

- 作者在其环境观察到轻量 commit 约 1.8 小时降到 167 秒，完整/轻量两种 commit 的同一管线都得到 0.617。周 GPU 额度仅变化约 175 秒是他的观察；不能登记为平台统一计费规则。
- 轻量路径根据比赛重跑环境标志及公开测试行数跳过重模型。作者记录过路径失配、旧镜像 Torch 无 CUDA：轻量 commit 通过，隐藏重跑约三小时后失败。
- Dataset 挂载上的单个 LoRA 加载约 100 秒，作者上游日志约 24 秒；先花约 34 秒复制到 `/tmp` 后，主流程 4,884→2,866 秒，上游约 2,859 秒。作者说预测相同，但未提供本轮可独立比对的产物哈希。
- 在另一 0.638 blend 上移除 ATLAS 10% 为 0.629，移除 Sinkhorn 为 0.636。移除与加入的效应不可互换，部件之间会替代。

## 我们核查了什么

原帖发于 2026-10-31 05:49:31 UTC；当前接口没有返回评论。原帖关联 [ntl-fork-guard-utility](https://www.kaggle.com/code/mizuno0522/ntl-fork-guard-utility)，本条只归档讨论，工具源码若单独归档应使用 solutions 条目关联。

## 对当前方案的用途与限制

可优先借鉴路径、设备和输出预检；计时分别记录加载、模型主流程、ATLAS 分支及完整隐藏重跑。本条不会证明当前项目已经加速或已通过隐藏测试。作者建议忽略可选支路失败只能在确认为可选且有有效回退结果时评估，不能据此吞掉必要成员或输出失败。

## 待验证想法

固定同一版本，分别测挂载直读与本地暂存；比较输出行数、标签顺序、预测概率及哈希。再单独验证轻量检查能检出错误路径与无 CUDA 环境。正式运行/提交另需实验授权；本次未开跑。
'''
},
754427: {
 'title':'HALX：无 Llama 配方与补充成员的负结果', 'tags':['ablation','llama','ensemble','negative-result'], 'related_sources':['SRC-026'],
 'summary':'0.651→0.642 同时移除 Llama 并改运行保护；ModernBERT/Mistral 未补回差距，不能外推所有集成。',
 'author_claim':'无 Llama 的 v22 为0.642；加入 ModernBERT/Mistral 的多种设置最高仍0.642。',
 'verified':'已读取原帖的七行结果及作者未隔离移除效应的明确限制；未取得对应提交ID与固定版本映射。',
 'use_for_us':'保留低单模OOF但互补的成员；针对当前集成做逐一添加/移除与权重对照。',
 'test_plan':'固定其余条件只将 Llama 权重设0，再比较新成员小权重加入与替换，并记录错误分歧。',
 'limitations':'原表两行均写v26且无scriptVersionId；0.009下降混合了运行保护等变化，不能视为严格Llama消融。',
 'body':'''## 作者声称与实际设置

| 作者版本 | 相对设置 | 作者报告 Public LB |
|---|---|---:|
| 原 nlm-v27 | 含 Llama | 0.651 |
| v22 | 去掉 ll8v5，并加入运行保护 | 0.642 |
| v23 | v22 + ModernBERT-large，权重1.5，单模OOF0.588 | 0.638 |
| v24 | v23 + Mistral-7B两阶段，权重1.0，单模OOF0.578 | 0.640 |
| v25 | v22 + Mistral-7B | 0.641 |
| v26（原文） | v22 + ModernBERT权重0.5 | 0.641 |
| v26（原文） | v22 + ModernBERT权重0.3 | 0.642 |

作者因 Llama 403 无法访问而测试替代。原表重复使用 v26，保留原文，不自行修正为其他版本。所有分数均为帖子自报，没有提交ID或得分源码版本绑定。

## 我们核查了什么

原帖发于 2026-10-31 05:50:13 UTC；当前没有评论。作者明确承认尚未隔离 0.009 下降是否全来自 Llama。其较早在 SRC-026 评论中说 Mistral 加入后的 +0.002，本帖澄清该数是在含 ModernBERT 的 v23 上；在 v22 上则为 -0.001。不得只保留正数。

## 对当前方案的用途与限制

支持检查成员互补性，而不能得出任何集成都不能去 Llama。作者的底座、权重和运行改动与当前 ATLAS20 配方不同；单模OOF不等于集成收益，细小Public差值也不足以证明方向。

## 待验证想法

关联 IDEA-004：在固定训练范围与干净分折缓存上，单独置零 Llama、加入新成员和替换新成员分开比较，并报告成对错误分歧与不同 prompt 组大小表现。
'''
},
754657: {
 'title':'GoPain：TPU v5e-8 的 Qwen3-8B LoRA 吞吐探针', 'tags':['tpu','runtime','lora','bf16','training'], 'related_sources':[],
 'summary':'11月1日新帖报告8芯TPU探针12.5k padded tokens/s；约14.2倍为特定配置吞吐，完整训练收益尚未验证。',
 'author_claim':'Qwen3-8B截断24层、LoRA r16、bf16；8芯SPMD 64×384每步1.97秒，约为2×T4 fp16的14.2倍。',
 'verified':'已读取原帖，作者说在迁移真实训练前测量；未取得其完整probe源码、日志或完成训练结果。',
 'use_for_us':'作为后续LoRA训练的候选硬件路线，先测固定形状、编译、吞吐、收敛与GPU导入兼容。',
 'test_plan':'相同样本顺序与有效token预算比较T4和TPU，统计有效token吞吐、编译/排队时间及相同训练步数OOF。',
 'limitations':'14.2倍为padded-token probe，不是端到端训练或LB提升；0.5小时是按约7小时阶段推算；额度/镜像须另核官方实时信息。',
 'body':'''## 作者声称与实际设置

2026-11-01 03:34:51 UTC 新帖：Qwen3-8B 截断至24层，所有投影 LoRA r16，约30M可训练参数；8层隐藏态均值池化后线性分类头，max_len384。使用 bf16、torch_xla2.8、transformers5.12，每个decoder层采用XLA checkpoint。

| 探针 | batch | step | padded tokens/s | 相对2×T4 fp16 |
|---|---:|---:|---:|---:|
| 单芯 | 8×384 | 1.90秒 | 1.6k | 1.8倍 |
| 8芯SPMD | 64×384 | 1.97秒 | 12.5k | 14.2倍 |

作者称所有loss/梯度范数有限，336个LoRA tensor均有梯度；每个输入形状编译50–80秒。把约7小时阶段折算为约0.5小时是估计，帖子开头说明是在迁移真实训练前先测量。

## 我们核查了什么

已读原帖与评论接口（无评论），未获取完整训练脚本/原始计时日志。作者称旧T4 fp16训练后期大量前向溢出且OOF下降0.10；这是其历史运行自报，不是本轮复现实验。

原帖提供 [序列桶基准 Notebook](https://www.kaggle.com/code/locdunhat/how-many-sequence-buckets-for-tpu-training)，称3–4个优化固定长度桶大约可将单一max_len形状的效率翻倍。完整Qwen probe脚本仅表示愿意分享，未在此帖提供链接。

## 对当前方案的用途与限制

固定形状可减少XLA重编译；作者用128/256/384三个长度与固定token预算、padding行零loss。数据依赖的group数量也需固定；SPMD虚拟设备的memory_info失败不能破坏已完成运行。首次排队约23小时后运行10分钟，吞吐快不等于交付快。作者的周TPU额度和预装库版本都不是官方保证。

## 待验证想法

先小规模测有效token而非padding token吞吐、编译次数、所有梯度有限性、样本覆盖和GPU加载权重兼容。随后才以相同训练步数/数据/seed作收敛与OOF比较。本次未申请TPU、未训练。
'''
},
754463: {
 'title':'0.66到0.70差距提问：方向意见而非高分配方', 'tags':['community','data','negative-result','unverified-claim'], 'related_sources':[],
 'summary':'提问者报告标点、平衡、SVM伪标签和ATLAS覆盖改动未超出重跑波动；回复没有可核验0.70配方。',
 'author_claim':'复现公共0.66配方后，硬标点、更软平衡、n-gram SVM伪标签、更大ATLAS覆盖均在重跑波动内。',
 'verified':'已读原帖与2条评论；评论仅称数据和建模/管线都重要，没有版本、数值或消融证据。',
 'use_for_us':'登记弱证据负结果，避免把泛泛社区回复升级为高分路线证明。',
 'test_plan':'若跟进某个方向需先取得固定源码、参数和对照；不按此帖直接扩大训练。',
 'limitations':'无精确基线、版本绑定、提交ID、数据清单；不能确认0.70方案用何数据/模型。',
 'body':'''## 作者声称与实际设置

提问者 Kratagya Jain 5846 在2026-10-31 07:14:04 UTC称复现公共约0.66方案，试过硬标点规则、较软平衡、n-gram SVM伪标签、扩大ATLAS覆盖，结果未超出其观察的重跑波动。他询问0.70以上差距主要来自数据还是建模。

## 我们核查了什么

读到两条评论：Civitasmass说二者都重要；Carrot Production在10月31日17:01 UTC说数据与pipeline。没有作者给出0.70得分版本、配置或明确消融。本条没有public_score，标题中的分数是讨论目标而非帖子作者已验证成绩。

## 对当前方案的用途与限制

只作为待进一步追踪的经验与负结果线索，不能据此宣称新方法已超0.70，亦不能证明这些改动普遍无效。

## 待验证想法

若取得固定源码/提交/参数后再判断是否可比较；现阶段优先完成已有数据与验证缺口，不为此帖单独开跑。
'''
},
754131: {
 'title':'ATLAS与均匀平滑对照：小幅优势及运行覆盖限制', 'tags':['atlas','calibration','ablation','smoothing','runtime'], 'related_sources':['SRC-001','SRC-026'],
 'summary':'作者将ATLAS预测替换为均匀分布：V31/20%为0.663 vs0.662；归因尚受覆盖量和旧权重恢复限制。',
 'author_claim':'V31/20%真实ATLAS0.663、均匀替换0.662；Teach10%为0.607/0.605；本地两种blend预测近同。',
 'verified':'已读原帖与10月31日评论；原文承认不同运行ATLAS覆盖行数可变，旧对照用恢复checkpoint。',
 'use_for_us':'分开测试去掉ATLAS、均匀平滑替代、真实ATLAS，评估信号与概率平滑的贡献。',
 'test_plan':'冻结同一输入概率、ATLAS处理行集合、权重及后处理，只替换该分支概率，测成对差异。',
 'limitations':'帖子表格分数未绑定提交ID；其他blend的移除效应不能与本表直接相减来量化平滑贡献。',
 'body':'''## 作者声称与实际设置

Joseph Adamski（ATLAS公开权重作者）将ATLAS的18类概率替换为均匀分布，保持混合权重与后处理。

| 配方 | 权重 | 真实ATLAS Public | 均匀替换 Public |
|---|---:|---:|---:|
| V31 / Omer得分v1 | 20% | 0.663 | 0.662 |
| Teach the Model | 10% | 0.607 | 0.605 |

作者另外在640或2210条已知标签文本上测试：两种blend预测约99%以上相同；这些本地集上二者均不优于无blend。表内本地accuracy并非Kaggle分数，不能混记。

## 我们核查了什么

已读10月30日原帖及10月31日03:19 UTC评论。作者明确说各运行可能ATLAS处理行数不同，较旧对照恢复了checkpoint；因此尚未严格分离真实归因信号与去过度自信的作用。

HALX评论：另一0.638底座移除10%ATLAS降0.009；他尚未做均匀替换，只认为平滑解释值得研究。该评论底座不同，不能直接与0.001/0.002相减并当作贡献分解。

## 对当前方案的用途与限制

比简单置零更有区分力的对照是均匀替换：它保留了混合造成的熵变化。但应固定实际覆盖行、概率与后处理，避免时间预算改变计算范围。

## 待验证想法

关联IDEA-002：先固定缓存与覆盖mask，比真实分支/均匀分布/完全去掉三种后处理输入，再检查prompt抽稀和各类变化。本次未运行或提交。
'''
},
754084: {
 'title':'GoPain：用十个LB点挑OOF指标的选择偏差模拟', 'tags':['validation','selection-bias','simulation','oof'], 'related_sources':['SRC-026'],
 'summary':'模拟解释最佳相关OOF的赢家偏差与top5范围收缩；统计数字受高斯模型与独立LB噪声假设限制。',
 'author_claim':'十点中选最高相关proxy常不是真最优；平均观测相关0.65而真实0.47，后续top5相关下降并非必然失效。',
 'verified':'已读原帖，确认该结果是模拟，非新的比赛LB成绩；源码与180000模拟样本本轮未执行。',
 'use_for_us':'避免凭少量Public提交选验证器；预先固定验证协议并检查多个prompt组大小。',
 'test_plan':'固定多个proxy与后续独立候选；做prompt级重采样和固定public子集误差的模拟，而非按已看LB重选指标。',
 'limitations':'高斯无方向偏差、跨提交独立LB噪声与真实固定Public split不同；20点不是通用阈值。',
 'body':'''## 作者声称与实际设置

原帖为2026-10-30。背景延续SRC-026：前10次提交按某OOF切法与LB的Spearman0.82挑指标，后5次仅0.05。作者模拟每配置的潜在质量、多个相关代理、LB相关度以及共享代理误差；第一阶段选最相关代理，第二阶段选其排名前5配置。

当LB相关r=0.7、第一阶段10个点时，接近质量的代理中选中真实最优概率仅28–41%（随机25%）；赢家平均观测相关0.65、真实0.47。被挑出的top5质量接近，相关可降至约-0.02至0.27。若代理误差独立，平均通常有利；强共享误差时可能稀释最强代理。

## 我们核查了什么

已读原帖及无评论响应。原文链接 [模拟Notebook](https://www.kaggle.com/code/locdunhat/which-offline-score-tracks-the-lb) 和 [模拟结果数据](https://www.kaggle.com/datasets/locdunhat/proxy-selection-simulations)，声称18万模拟season、CC0；本条未下载或运行，不能登记为已独立复现。

## 对当前方案的用途与限制

有助解释验证器选择的赢家偏差，以及只看优秀候选时的范围收缩；不能据此量化我们方案的真实相关。作者承认高斯、代理无方向偏差、跨提交独立LB噪声等假设，而真实Public分割固定。约20点的观察仅适用于该网格。

## 待验证想法

沿用固定prompt-group-v1，预先规定m=2/3/6/18等指标与选择准则，按prompt重采样，保留未参与选指标的后续候选；不为获得相关性而机械耗尽提交名额。
'''
},
754026: {
 'title':'Pablo：自适应中心化与收缩边际Sinkhorn的OOF声明', 'tags':['sinkhorn','centering','validation','stylometry','hungarian'], 'related_sources':['SRC-013'],
 'summary':'作者给轻量管线OOF阶梯0.3145→0.4345；无逐行OOF/源码版本绑定，且全prompt Hungarian提升不代表稀疏测试收益。',
 'author_claim':'5折prompt GroupKFold中topic masking0.3686，中心化0.3812，target-shrunk Sinkhorn0.3834，Hungarian0.4345。',
 'verified':'已读原帖，保留其具体参数与表格；该讨论未给得分scriptVersionId、OOF文件和完整验证隔离证据。',
 'use_for_us':'作为IDEA-002的候选校准设置与风格特征方向，必须先检验分折与组大小效应。',
 'test_plan':'固定无泄漏分折缓存，在抽稀后重新聚类，逐项开关中心化/收缩边际/Hungarian并检查label与group泄漏。',
 'limitations':'2699条并不等于文中18×180的完整格；公开测试类别先验不等的解释是作者假设，非官方证明。',
 'body':'''## 作者声称与实际设置

2026-10-30原帖关联 [Starter Notebook](https://www.kaggle.com/code/pablodiegoo/name-that-llm-starter-notebook)。作者在prompt_id的5折GroupKFold给出阶梯：word TF-IDF0.3145；加char/char_wb与20风格比率0.3367；加LinearSVC与top100功能词外掩码0.3686；自适应中心化0.3812；收缩目标边际Sinkhorn0.3834；组内Hungarian0.4345。

参数声明：聚类sim_thr0.15，中心化alpha0.65；Sinkhorn alpha0.45、rho0.35、100轮。另称V40A+V31的供应商家族内残差用logsumexp保持家族质量，但没有独立量化该项收益。

## 我们核查了什么

已读原帖，无评论。帖子给的是作者OOF而非0.664得分配方；没有把表格绑定特定scriptVersionId，也没有逐行OOF或训练/特征构造隔离证据。所链Notebook的已登记SRC应由合并窗口以URL查重；关联旧源码不代表它已支持当前表格。

原文将2699条描述为18模型×180prompt，数目并非完整笛卡尔积，应保留缺失组合而不能按完整组假设实现。作者关于Public/Private类计数不均的解释不是官方统计。

## 对当前方案的用途与限制

全组Hungarian尤其可能受每组类别齐全程度影响，+0.0511的OOF增量不能直接外推隐藏稀疏组；功能词掩码和概率边际是两个不同干预，分别验证。

## 待验证想法

关联IDEA-002：先核对源码与折内fit范围，抽稀后重算聚类、中心化和assignment，再分别打开各组件；报告组大小分层收益和违反单类唯一假设的样本。
'''
},
753974: {
 'title':'Enai：9719条OpenAI生成文本的发布公告', 'tags':['data','provenance','license','openai'], 'related_sources':['SRC-027'],
 'summary':'公告称9719条、四模型且含exact prompts/设置/溯源；数据本体与生成文本许可本轮未独立核验。',
 'author_claim':'9719条=2341 Luna+2338 Sol+2336 Terra+2704 Astra；4311补现有prompt覆盖，5408为新prompt面板。',
 'verified':'已读原公告与1条评论；未读取固定dataset版本文件，公告不能证明逐行溯源、许可或收益。',
 'use_for_us':'可补GPT类数据，但先对接已有prompt键与来源清单，避免将共享prompt算作独立增量。',
 'test_plan':'固定数据版本/哈希，核验exact messages、returned model、settings、去重、官方prompt关联和许可来源，再决定分层对照。',
 'limitations':'SRC-027作者说明未建立生成文本许可所以未纳入v10；存在许可声明不等于生成文本授权已厘清。',
 'body':'''## 作者声称与实际设置

公告发于2026-10-29 23:25 UTC，10月31日01:34 UTC有一条鼓励投票评论，没有补充技术证据。链接 [OpenAI Generated Responses](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses)。

声称总9719条：GPT-5.6 Luna2341、Sol2338、Terra2336、GPT-6 Astra2704；4311条补已有社区prompt缺失类，5408条组成新增prompt面板，使用社区库与部分Dolly指令。称提供CSV/Parquet、prompt JSONL、精确system/user消息、调用设置、来源和许可通知。

## 我们核查了什么

本条读取的是发布公告，不是dataset版本；没有下载文本或模型权重，也没有验证逐行模型返回ID、完整调用日志、prompt授权和去重。若本轮另有dataset条目，应关联它并将详细数据审计留在该条，避免复制主体。

## 对当前方案的用途与限制

SRC-027的Open Pool v10作者明言暂未加入一份新的OpenAI-response数据，因为生成文本许可未建立；应核对是否为本数据及具体数据卡，不把有license notice等同许可已明确。不能把本公告称为新0.664方法或已证明增量收益。

## 待验证想法

先固定版本、文件哈希、上游prompt来源与模型/设置，再查与现有公共池的文本/prompt重叠；有明确许可与足够溯源后才作按prompt隔离的单来源加入对照。
'''
},
753177: {
 'title':'成员OOF与集成收益：稀疏验证及互补性后续核查', 'tags':['ensemble','validation','oof','ablation','negative-result'], 'related_sources':[],
 'summary':'复用SRC-026；原帖6条评论无新实质结果，保留OOF与集成脱节及稀疏验证失效边界。',
 'author_claim':'重训q25 OOF0.569→0.605，替换LB0.651→0.647，旧+新权重0.4为0.650；m=2前十/后五相关差异明显。',
 'verified':'11月1日重读原帖与6条评论，最新仍为10月31日12:09UTC；与旧快照实质一致，删除评论内容仍不可读。',
 'use_for_us':'继续保留旧成员互补性与抽稀后重建组计算的验证约束，不把旧结果当最新增益。',
 'test_plan':'单独比较替换/加入/置零、错误分歧，多个m值抽稀后重建聚类与中心化。',
 'limitations':'社区自报未重训；全局指标相关不能证明接近配置排序能力；评论推测不同数据增益仍未证实。',
 'body':'''## 作者声称与实际设置

此为现有SRC-026的时间更新，应合并回原记录，不新增SRC。原训练q25旧数据约35k，重训用119899条v5公共池后再官方微调；作者报告OOF0.569→0.605，但在nlm-v27替换成员时LB0.651→0.647，保留旧成员再加新成员0.4得到0.650。

## 我们核查了什么

11月1日重读主帖与接口返回6条评论；最新时间仍10月31日12:09 UTC。与旧 [20261031快照](../../../tracking/snapshots/20261031T141753Z/topic-753177.json) 的实质内容一致；一条删除评论无文本，不能称已读删除内容。

GoPain的m=2验证在前10点Pearson/Spearman0.91/0.82，按该指标选后5点为-0.22/0.05，全15点0.79/0.58；去掉易区分离群点后Spearman0.49。HALX的0.638底座同时去两个中心化twin降0.017，单独去各降约0.003–0.004，去ATLAS10%降0.009，去Sinkhorn降0.002。

## 对当前方案的用途与限制

最新独立帖754427补明Mistral的+0.002在含ModernBERT底座上，而在无ModernBERT底座是-0.001，合并时应交叉关联。原帖关于新数据提高多样性的解释仍是假设，HALX明确没有不同数据混合带来增益的证据，也没有测提高旧成员权重。

## 待验证想法

继续关联IDEA-004与IDEA-002：保持分折与训练范围可核验，单成员和加入集成收益分别检查；先抽稀回答后重建所有组相关计算。本次只是来源核查。
'''
},
752766: {
 'title':'Open Pool v10：清洗对照与成对分差的不确定性', 'tags':['data','cleaning','validation','paired-comparison'], 'related_sources':[],
 'summary':'复用SRC-027；新增10月31日作者回复未给新实验，保留193k/198507口径及成对分差依赖错误分歧。',
 'author_claim':'相同DeBERTa配方raw v7约254k得0.585、clean v9约193k得0.587；v10为198507条。',
 'verified':'11月1日读原帖和14条评论；新增10月31日16:53UTC回复仅认同数据清洗，没有新得分或版本。',
 'use_for_us':'记录原池/清洗池与新增来源的独立对照，并量化成对正确性分歧，不能套固定0.01噪声阈值。',
 'test_plan':'固定模型与prompt分组，比较原池/清洗池等量及全量，并用prompt级bootstrap衡量成对指标变化。',
 'limitations':'作者数据/分数未独立重训；sqrt(f/n)假设独立行仅为简化近似，prompt相关误差会放大不确定性。',
 'body':'''## 作者声称与实际设置

此为现有SRC-027的时间更新，应合并回原记录。主帖声称相同DeBERTa-v3-large训练配方中raw v7约254k为0.585，clean v9约193k为0.587；等量控制实验提升0.027另见作者此前清洗帖。v10在v9上新增5581条共198507条，旧行不变，新增ID从ntlop_263484起。

新增来源分布：abdullah2574、zitaoxu1026、carlos973、aswath474、elliot367、lucas167。过滤1396条主要因标点、接近官方文本或截断；训练集标签标点统计参与过滤，需要记录对严格OOF的影响。

## 我们核查了什么

11月1日读取原帖与14条评论。相较旧 [20261031快照](../../../tracking/snapshots/20261031T141753Z/topic-752766.json)，10月31日16:53 UTC新增作者对GoPain的赞同回复；未给新的训练/得分/数据版本结果，不能报告为新提升。

GoPain建议比较两方案正确性不同的行比例f，独立行近似的Public分差标准差约sqrt(f/n_public)。例如以约1200条Public行估算，f从1%到20%时约0.003至0.013。这不是已核实的实际Public行数，也不是我们数据的置信区间；作者也说prompt/topic相关性使估计偏低。关联 [成对分差CPU Notebook](https://www.kaggle.com/code/locdunhat/is-a-0-002-lb-gain-real)，本轮未执行。

## 对当前方案的用途与限制

应分别保留等量控制、全量数据与实际成员增量；“0.002在噪声内”不等于所有0.01以下变化都不可解释。数据卡与文件版本应另在dataset条目核查。本帖不授权生成数据或提交。

## 待验证想法

关联IDEA-003：固定数据清单、prompt隔离和成员配方，比较等量/全量清洗对照，按prompt bootstrap并记录错误分歧；不要把清洗池与它已包含的来源简单相加。
'''
}
}

manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
for rec in manifest['records']:
    d = DATA[rec['topic_id']]
    dest = ROOT / rec['notes_path']
    source_id = rec['existing_src_candidate'] or '__SRC_ID__'
    relative_snapshot = 'snapshots/' + Path(rec['snapshot_path']).name
    header = f'''# {source_id} · {d['title']}

状态：已读取原帖与当前可返回评论；待合并窗口登记/更新正式 SRC。
原始记录：[正式台账](../../../tracking/records/sources/{source_id}.json)（合并时替换占位编号）。

## 值得保留的结论

{d['summary']}

## 来源与原件

- 原帖：[Kaggle 讨论 {rec['topic_id']}]({rec['url']})；作者 `{rec['author']}`。
- 采集时间：{rec['observed_at']}；原帖发布：{rec['post_date']}。
- [原帖与评论快照]({relative_snapshot})；SHA-256 `{rec['snapshot_sha256']}`。这是公开只读 API 响应，不是独立实验验证。

'''
    dest.write_text(header+d['body']+f'''\n## 更新记录

- 2026-11-01：读取原帖和当前评论，保存时间快照。{'复用 '+rec['existing_src_candidate']+'，不新增编号。' if rec['existing_src_candidate'] else '新来源，正式编号由合并窗口分配。'} 本轮未训练、未提交。
''',encoding='utf-8')
    rec['source_record_proposal'] = {k:v for k,v in d.items() if k != 'body'}
    rec['source_record_proposal'].update({'kind':'discussion','category':'discussions','public_score':None,'source_version':{'topic_id':rec['topic_id'],'snapshot_sha256':rec['snapshot_sha256']}})
manifest['import_notes'] = ['Replace __SRC_ID__ in notes during merge.', 'Existing SRC-026 and SRC-027 must be updated, not duplicated; historical evidence stays referenced.', 'Source_record_proposal is an import proposal, not canonical metadata.', 'For SRC-026/SRC-027 historical snapshot links are valid after placement under research/discussions/SRC-*; new notes need canonical links patched by importer.']
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Wrote',len(manifest['records']),'independent notes and pending proposals.')
