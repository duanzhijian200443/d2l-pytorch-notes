# 授权 v2 执行完成

原先无法证明$1硬封顶的阻塞已被用户预算变更撤销，未重复以此停止。实际已完成20/20条，Luna10、Terra10，累计平台返回费用 **$0.0214694**。失败0、重试0、截断0、拒答0、长度异常0。首两条完成并核验费用/保存后，才串行推进剩余18条。

使用原Drive文件SHA256 dc7495501a3412775902716a40d1b74d13e840b4b73a4c0c9530e28a6080a4d0，250条完整核对。仅指定10题原始行传至Kaggle并复核选集哈希；没有上传其他240题。

实际账户SDK0.6.2列出的模型键为openai/gpt-5.6-luna、openai/gpt-5.6-terra；20条返回键均匹配。请求reasoning_effort=low、max_completion_tokens=1024，返回未逐项回显生效设置；底层snapshot unknown，temperature/seed省略且实际默认unknown。所有请求接受low字段且正常完成，不宣称精确复现Codex或比赛采样环境。

总输入1042 token，completion3355 token（部分记录有reasoning_tokens细分，缺失不当作零）；最高单次completion323。费用直接使用平台每条input/output纳美元数据，非供应商标价。原文、换行、标点全部保留，不以质量筛除样本。

完整结果保存在responses.jsonl，原始平台响应包含其中；request-journal.jsonl含40条开始/终态记录，逐条即时flush/fsync。未建立每日自动化、未训练提交公开、未充值；Codex采集和项目正式基线保持不动。此批仅作隔离渠道对照，EXP编号/台账登记待本地合并者处理。
