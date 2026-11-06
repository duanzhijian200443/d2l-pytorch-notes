# Kaggle Model Proxy 对照采集：预检停止，零请求

2026-11-01。此目录是唯一命名待交接包，不占用/猜测 EXP 编号。不改写台账和正在运行的 Codex 采集。

## 已完成
- 从 EXP-021 Drive 原文件读取全部 87764 字节，SHA-256 为 `dc7495501a3412775902716a40d1b74d13e840b4b73a4c0c9530e28a6080a4d0`，符合预期；250条 JSONL，字段 prompt_id/prompt/prompt_group/split/category/target_words/source。
- 固定10题全部 train，覆盖10个不同类别。selected-prompts.jsonl 直接筛出原文件行，未改写题目。
- 已阅读项目 AGENTS.md、README、current-state、EXP-021 config；基础指令和low对齐用户要求，但未证明与Codex或比赛采样等价。
- 当前账户界面显示每日免费 $10/$10、每月 $99.99/$100；该余额不等于本任务 $1 硬封顶。

## 停止原因
用户要求每次发送前包含推理 token 的可核实保守费用上界，或有效费用硬上限。官方 SDK 当前公开源码只提取响应 usage.cost 的 input/output 纳美元成本。CLI quota 提供事后已用/剩余额度，说明超额会被显示为零剩余。已检查官方 SDK src 和文档，未取得目标模型 Kaggle 实际扣额费率、推理计费边界或任务/请求美元硬限额。因此不能保证 $1 封顶，按授权在首条生成前停止，不以供应商 API 标价替代、不以费用缺失按零。

## SDK研究（不等于当前账号运行核验）
- LLMChat.prompt 接口默认 seed=0、temperature=0；ModelProxy默认 support_temperature=False，因此temperature不发送。
- proxy_openai 对 openai/gpt-5.6 前缀剔除 seed；reasoning='low'映射reasoning_effort。
- extra_api_params可以向下传输出限制，但未验证目标模型接受的字段、推理token是否受同一上限及实际费用；不凭此认定预算安全。
- 实际可用模型键、模型身份映射、底层snapshot、已安装SDK版本、low实际支持和生效值均未核验。未用展示名称或仓库前缀冒充账户可调用证据。

## 交付与计数
responses.jsonl为空（0条），不是缺失文件。两个目标模型各0条；无失败生成、无重试。request-status.jsonl的20行均为not_sent_preflight_blocked，费用null，绝非20条回答。仅因未发送任何生成请求，本任务生成费用为$0；未声称整个账户总费用为零。

原始输入、筛选原行、请求草案、manifest和状态清单随包提供。未训练、未提交、未公开、未创建日采集或充值、未修改Codex任务。后续需先获得可核实的Kaggle计费上界/硬预算机制，再在当前账号核验目标模型键及low，才能启动两条试采。数据仍按kaggle_model_proxy隔离，不能直接合入训练池。正式台账登记和ledger check/render由本地合并者完成。

## 原始证据
- https://www.kaggle.com/docs/benchmarks （登录账户额度界面，观察2026-11-01T08:49:23Z）
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/src/kaggle_benchmarks/actors/llms.py
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/src/kaggle_benchmarks/actors/proxy_openai.py
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/src/kaggle_benchmarks/kaggle/model_proxy.py
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/src/kaggle_benchmarks/kaggle/models.py
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/user_guide.md
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md

公开源码快照及SHA记录在sources目录；仓库ci为移动分支，不声称其就是账号Notebook已安装版本。
