# Lucas free-proxy 平台 V12 核查（待合并）

状态：PENDING_IMPORT。此目录是逐条待导入资料，不是正式 SRC 或第二套当前状态台账；相关旧记录为 SRC-034（平台 V7）。

## 来源与时间

- [固定 V12](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations/versions/12)；[V11](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations/versions/11)。
- 官方 Kaggle SDK 的 get_dataset 返回平台 V12，creationDate/lastUpdated = `2026-11-01T14:39:54.000Z`，即北京时间 `2026-11-01 22:39:54`。Z 是 UTC，不是北京时间。
- 当前数据卡、官方版本历史及两个固定版本的 README 和文件清单已保存。固定版本下载通过 `lucasboesen/ntl-free-proxy-generations/12` 和 `/11`；没有下载模型。
- 原始匿名 API URL 请求返回404，web读取也失败；随后官方已认证 SDK 成功获取，未重新登录。

## 作者声称

V12说明为 prompt-bank-v2 的 v2b21 批次，新增192条，总1872 prompts / 7906 passages。协议为 Kaggle Model Proxy 免费额度、单用户消息、单段输出、reasoning low、max_tokens 4096、未设temperature/top_p/seed。prompt-bank-v2 为既有公共题组补缺；许可标 CC BY 4.0。上述生成协议及底层模型身份仍属作者文档，路由字段不能独立证明模型快照。

## 实际核查

| 固定版本 | CSV行数 | 唯一prompt_id | 标签数 |
|---|---:|---:|---:|
| V7（已有原件） | 4255 | 952 | 8 |
| V11 | 7714 | 1824 | 8 |
| V12 | 7906 | 1872 | 8 |

V12对V11全字段多重集差异为新增192、删除0，增加48个prompt_id。全部新增来自 `v2b21`；`gemini-3.6-flash`、`gemini-3.8-flash`、`gpt-5.5`、`gpt-6-astra`各48。新增 `prompt_source` 为 `maniceet openrouter/openai generations`。V12对旧V7新增3651行、删除0、增加920个prompt_id，不能把+192错作相对V7增量。V7原件SHA256与SRC-034相符。

V12的prompt_id/model组合均唯一，完全相同整行和完全相同text均无重复。CSV无独立id列，以(prompt_id, model)作此版本的可验证唯一键。prompt_id数只代表来源命名空间中的题组，不等同全来源语义去重后独立题数。

数据卡/README仍有“每题全8标签”的旧句；实际仅363/1872组全8标签，新48组只有4标签。完整分布见 actual-counts.json。README后部的补缺说明解释了稀疏性，应保留两个证据而不复述旧句。

CSV没有finish_reason字段；因此7906是实际记录数，不能据此声称7906均正常终止、无截断或已通过质量准入。没有运行训练、提交、GPU或生成请求。

## Open Pool重复边界

额外下载了[Open Pool平台V11](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/11)的小文件sources.csv，实际包含来源 `lucasboesen/ntl-free-proxy-generations`、passages=3511、models=8。这是固定来源清单支持的部分收录，不是本轮逐行联表交集结果。已有 SRC-027 原帖快照还保存作者v10更新表的Lucas新增167条声明；167是那次池更新量，3511是来源清单量，不能互相替代，更不能把7906与OpenPool总量直接相加。

本轮未下载OpenPool全量parquet，无法确定当前OpenPool与Lucas V12的精确交集、差集或新192是否全部不在池中。需固定双方版本、规范化prompt身份并做正文/标签/来源键联表，才可确定可用增量。

## 对项目的用途与后续验证

这是有完整版本差分的候选数据更新；先以V7→V12的3651条作为待审计范围，再按固定OpenPool/基线已有用数清单去重。保留prompt_source和原文标点，统一跨源题组；检查空文本、截断、拒答、标签及路由、官方相关性，并用prompt-group隔离验证。未验证分数收益，不把来源更新直接转成正式训练授权或已接纳训练池。

原始CSV放data/community-factcheck-20261102/lucas-v11和lucas-v12；清单记录大小和SHA256。资料原件与分析区分，正式编号与台账导入由合并窗口完成。
