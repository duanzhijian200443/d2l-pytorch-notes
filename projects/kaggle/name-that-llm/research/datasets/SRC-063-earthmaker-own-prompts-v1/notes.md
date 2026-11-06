# SRC-063 · earthmaker 自拟题原文 V1

来源：[固定Kaggle V1](https://www.kaggle.com/datasets/earthmaker/ntl-own-prompt-generations/versions/1)。2026-11-01通过Kaggle MCP读取元数据和固定文件列表，再用官方客户端固定版本下载。此条记录资料可信度与使用范围；我们自己的评测另归EXP-027。

## 作者声称

749条原文来自20道pilot题与36道v22题，题目自行编写，模板只使用user消息、无system。pilot通过Kaggle Benchmarks与OpenRouter覆盖18个请求标签，v22覆盖12个。按与官方回答共享8词片段及少量格式/截断规则删除43条。原题、目标词数、请求模板和路由随文件发布；许可CC BY4.0，模型输出可能受提供方条款约束。

作者明确指出：pilot的gemini-3.7-flash虽然请求和返回字符串相同，该名称未列在代理支持清单中，无法确认实际服务版本。不能把这20条当作可靠的Gemini3.7验证样本。

## 我们核查

已取得V1全部5文件，名称/大小与MCP固定清单一致，原文件保存于`data/public-pools/earthmaker/ntl-own-prompt-generations/v1`。下载接收哈希记录在[EXP027回执](../../../experiments/EXP-027-multiclass-external-validation/download-manifest.json)；作者未提供另行认证的原始服务日志或签名校验和。

输入身份：[api-info.json](api-info.json)、[files-v1.json](files-v1.json)。[EXP027数据门槛](../../../experiments/EXP-027-multiclass-external-validation/metrics/data-gate.json)现已通过：仅排除预定20条Gemini3.7，保留729条、56个规范化题干组、17类；与两成员训练回答、763个已知训练题目祖先、150个已知官方重建题及EXP026归并正文的对应exact交集为0。原题/模板/请求标识联结通过，审计复跑字节一致；另一次独立复核得到同样选集。16组完整覆盖17类，其余覆盖不齐，五个OpenRouter稀缺类仅16～19条。

## 适用条件与用途

预先排除Gemini3.7，最多提供17类外部诊断。先与两轻量成员真实训练输入、已知官方重建题和已用EXP026题目作祖先/规范化正文检查，重叠整题排除；不按模型分数或文风挑选。即使门槛通过，也不等于严格全18类、官方分布或完整ATLAS20验证：发布者已参考官方文本过滤、语义重叠未穷尽，部分现成checkpoint可能见过源数据。

后续模型结果只写EXP027；原数据不改写，不公开资源。
