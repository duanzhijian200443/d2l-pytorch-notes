"""One-time, non-destructive import of the existing project evidence."""
import datetime as dt
import hashlib
import json
import pathlib
import re

TRACK = pathlib.Path(__file__).resolve().parents[1]
ROOT = TRACK.parent
SNAP = TRACK / "snapshots/20261031T141753Z"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    if path.exists():
        raise FileExistsError(f"Import never overwrites: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha(path):
    n = load(path)
    s = "\n\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in n["cells"] if c["cell_type"] == "code")
    return hashlib.sha256(s.encode()).hexdigest()


if any((TRACK / "records/experiments").glob("*.json")):
    raise SystemExit("Records already exist. Update records explicitly; do not re-import.")

snapshot = load(SNAP / "submissions.json")
observed = snapshot["observed_at"]
subs = {x["ref"]: x for x in snapshot["submissions"]}
template = {
    "schema_version": 1, "id": "EXP-000", "title": "待填写", "baseline_id": None,
    "hypothesis": "", "changes": [], "observed_at": None,
    "source": {"notebook_ref": None, "notebook_version": None, "script_version_id": None, "local_path": None, "sha256": None},
    "inputs": {"manifest_path": None, "validation_split_id": None, "content_hashes_verified": False},
    "config": {},
    "environment": {"platform": None, "accelerator": None, "container": None, "seed": None},
    "execution": {"scope": "planned", "status": "PLANNED", "observed_at": None},
    "submission": None,
    "validation": {"balanced_accuracy": None, "scope": None, "per_class_recall_path": None, "oof_path": None, "leakage_audit": "not_audited"},
    "cost": {"gpu_seconds": None, "api_usd": None, "colab_cu": None},
    "timing_note": None, "conclusion": "", "limitations": "", "next_action": "",
    "related_sources": [], "evidence": []
}
save(TRACK / "templates/experiment.json", template)
save(TRACK / "templates/source.json", {
    "schema_version": 1, "id": "SRC-000", "kind": "discussion", "title": "待填写", "url": "",
    "observed_at": None, "source_version": {}, "public_score": None,
    "summary": "一句话说明值得读的原因", "author_claim": "作者的具体结论和适用条件",
    "verified": "我们核查了什么", "use_for_us": "对当前方案有什么用",
    "test_plan": "怎样做一次可证伪的实验", "limitations": "哪些条件或事实尚未确认", "evidence": []
})
save(TRACK / "settings.json", {"project": "Name That LLM", "baseline_experiment_id": "EXP-004", "timezone": "Asia/Shanghai", "automatic_remote_mutations": False})


def experiment(ident, title, submission_id=None, source_file=None, **fields):
    e = json.loads(json.dumps(template))
    e.update(id=ident, title=title, observed_at=observed)
    e.update(fields)
    if submission_id:
        s = subs[submission_id]
        sid = int(re.search(r"scriptVersionId=(\d+)", s["url"])[1])
        e["submission"] = {"id": submission_id, "status": s["status"], "script_version_id": sid,
            "public_score": float(s["publicScore"]) if s.get("publicScore") else None,
            "submitted_at": s["date"], "observed_at": observed, "url": "https://www.kaggle.com" + s["url"]}
        e["source"]["script_version_id"] = sid
        e["evidence"].append(rel(SNAP / "submissions.json"))
    if source_file:
        e["source"].update(local_path=rel(source_file), sha256=sha(source_file))
        e["evidence"].append(rel(source_file))
    save(TRACK / f"records/experiments/{ident}.json", e)


experiment("EXP-001", "无 Llama 的词与字符 TF-IDF / LinearSVC 基线", 57736953,
    hypothesis="建立低成本、按 prompt 分组的基准。", changes=["仅官方训练数据"],
    source={"notebook_ref": "zhjianduan/name-that-llm-no-llama-groupcv", "notebook_version": 4},
    config={"folds": 5, "split": "GroupKFold(prompt_id)"},
    validation={"balanced_accuracy": 0.3122446, "scope": "提交描述记录的五折 OOF BA；本轮未重新计算", "leakage_audit": "历史代码检查；本轮未重训"},
    environment={"platform": "Kaggle CPU", "accelerator": None},
    execution={"scope": "commit", "status": "COMPLETE"},
    conclusion="正式 Public LB 0.308；可作为验证流程参考，不能代表强集成的效果。",
    limitations="不能把该模型的 OOF 改善直接换算为 ATLAS20 的提升。",
    next_action="复用 prompt 分组原则，不重跑整个历史实验。",
    evidence=["cloud/no-llama-groupcv/baseline.py"])
experiment("EXP-002", "Who am I V30 完整复现", 57759707,
    source={"notebook_ref": "zhjianduan/name-that-llm-who-am-i-v30-repro", "notebook_version": 7},
    config={"upstream_notebook_version": 30, "upstream_script_version_id": 362877806},
    inputs={"manifest_path": "cloud/who-am-i-v30/kernel-metadata.json", "content_hashes_verified": True, "validation_split_id": None},
    environment={"platform": "Kaggle", "accelerator": "T4 x2"},
    execution={"scope": "public_sample_full", "status": "COMPLETE", "observed_at": "2026-10-29T10:52:04.052581+00:00"},
    validation={"balanced_accuracy": None, "scope": "公开三条示例预测 3/3 与原 V30 一致；不是隐藏集准确率"},
    timing_note="公开样例全流程 13118.538489206 秒（3 小时 38 分 39 秒）；隐藏评分耗时未知。",
    conclusion="正式 Public LB 0.654；旧 execution-state 中 pending/未验证字段属于历史阶段。",
    limitations="相同显示分数不证明逐条隐藏预测相同；旧状态文件不是实时来源。",
    next_action="保留复现证据；当前优化使用 EXP-004 为对照。",
    evidence=["cloud/who-am-i-v30/execution-state.json", "cloud/who-am-i-v30/outputs/full-v7/local_verification.json"])
experiment("EXP-003", "Who am I V32 Colab A100 尝试", source_file=ROOT / "cloud/who-am-i-v32-colab/upstream-v32.ipynb",
    observed_at="2026-10-29T13:03:55.353090+00:00",
    source={"notebook_ref": "romantamrazov/name-that-llm-who-am-i", "notebook_version": 32, "script_version_id": 363222852},
    config={"watchdog_seconds": 14400, "test_scope": "public_sample"},
    environment={"platform": "Colab", "accelerator": "NVIDIA A100-SXM4-40GB", "driver": "580.82.07"},
    execution={"scope": "public_sample_full", "status": "INTERRUPTED", "observed_at": "2026-10-29T13:03:55.353090+00:00"},
    inputs={"manifest_path": "cloud/who-am-i-v32-colab/execution-state.json", "content_hashes_verified": True, "validation_split_id": None},
    timing_note="仅有部分阶段耗时；没有正常结束记录，不能计算完整吞吐或总耗时。",
    conclusion="互补集成及部分 q14v5 计算完成后后端消失；无 submission.parquet，未得到榜单分数。",
    limitations="真实中断原因未知；不能归因为 OOM、四小时限时或用户手动断开。后续网页启动器的本地测试不代表 GPU 已跑通。",
    next_action="这次尝试已结束；后续重跑另建 EXP，并复用已有缓存。",
    evidence=["cloud/who-am-i-v32-colab/outputs/run-20261029T112726Z/interruption-report.json", "cloud/who-am-i-v32-colab/web-validation.json"])
experiment("EXP-004", "ATLAS20 正式 0.662 基线", 57778357,
    source_file=ROOT / "reviews/own-scheme-readiness-20261030/baseline-v2.ipynb",
    baseline_id="EXP-002", hypothesis="复现 ATLAS20 完整配方作为后续优化对照。",
    changes=["采用 ATLAS20 配方；不是只改一个参数的控制实验"],
    source={"notebook_ref": "zhjianduan/name-that-llm-atlas20-repro", "notebook_version": 2},
    inputs={"manifest_path": "reviews/own-scheme-readiness-20261030/baseline-v2-metadata.json", "content_hashes_verified": False, "validation_split_id": None},
    config={"atlas_w": 0.20, "comp_w": 0.15, "sinkhorn": 0.45, "sinkhorn_target_mix": 0.35, "typography_scale": 1.5},
    environment={"platform": "Kaggle", "accelerator": "T4 x2", "internet": False},
    execution={"scope": "commit", "status": "COMPLETE"},
    conclusion="自己的正式 Public LB 0.662；已绑定提交与历史源码。",
    limitations="尚无该完整集成的干净端到端 OOF；附件内容版本与隐藏数值一致性未全部审计。",
    next_action="保持此版本为固定对照，新增实验从它分支。", related_sources=["SRC-001", "SRC-002"],
    evidence=["reviews/own-scheme-readiness-20261030/baseline-submission-binding.json", "reviews/own-scheme-readiness-20261030/readiness-audit.json"])
experiment("EXP-005", "Who are U 最新 V6 配方的自有副本", 57832487,
    source_file=next((ROOT / "reviews/judith007-0664-20261031/own-fork-v1").glob("*.ipynb")),
    baseline_id="EXP-004", hypothesis="观察更强类别平衡与更弱标点修正的组合。",
    changes=["Sinkhorn 0.45 → 0.60", "标点修正 1.5 → 1.2"],
    source={"notebook_ref": "zhjianduan/0-664-hyperparameter-tuned-who-are-u", "notebook_version": 1},
    config={"atlas_w": 0.20, "comp_w": 0.15, "sinkhorn": 0.60, "sinkhorn_target_mix": 0.35, "typography_scale": 1.2},
    environment={"platform": "Kaggle", "accelerator": "T4 x2", "internet": False},
    execution={"scope": "commit", "status": "COMPLETE"},
    conclusion="目前没有自己的分数；代码哈希与作者 V6 一致。",
    limitations="作者的 0.664 属于 V2，不适用于这个 V6 配方；两项同时变化不能单独归因。",
    next_action="只查询既有提交结果；不重复提交或重启。", related_sources=["SRC-022"],
    evidence=["reviews/judith007-0664-20261031/review-evidence.json"])
who = load(SNAP / "who-are-you-metadata.json")
who_path = SNAP / "who-are-you.ipynb"
v2_path = next((ROOT / "reviews/judith007-0664-20261031/v2-score-0664").glob("*.ipynb"))
same = code_sha(who_path) == code_sha(v2_path)
experiment("EXP-006", "Who Are You 作者得分 V2 配方复现", source_file=who_path,
    baseline_id="EXP-004", hypothesis="减轻全局类别平衡能否改善当前配方。",
    changes=["Sinkhorn 0.45 → 0.30；其余推理参数沿用作者得分 V2"],
    source={"notebook_ref": "zhjianduan/who-are-you", "notebook_version": who["currentVersionNumber"], "script_version_id": None},
    config={"atlas_w": 0.20, "comp_w": 0.15, "sinkhorn": 0.30, "sinkhorn_target_mix": 0.35, "typography_scale": 1.5},
    inputs={"manifest_path": rel(SNAP / "who-are-you-metadata.json"), "content_hashes_verified": False, "validation_split_id": None},
    environment={"platform": "Kaggle", "accelerator": "T4 x2", "container": who.get("dockerImage"), "internet": False},
    execution={"scope": "commit", "status": "UNKNOWN", "observed_at": observed},
    conclusion=f"保存版本代码与作者得分 V2 一致：{same}；尚未关联自己的正式提交。",
    limitations="自己的保存版本编号与作者版本编号不同；公开样例快速保存不能当作正式评分。",
    next_action="刷新保存状态；提交后补登记自己的 submission ID 与 scriptVersionId。", related_sources=["SRC-021"],
    evidence=[rel(SNAP / "who-are-you-metadata.json"), "reviews/judith007-0664-20261031/v2-score-0664/vs-alex-v8.diff"])

catalog = load(ROOT / "reviews/public-top20-20261030/catalog.json")
for n, item in enumerate(catalog, 1):
    directory = "reviews/public-top20-20261030/" + item["directory"]
    note = item["note"].replace("仍是本次有效分数最高项", "2026-10-30 快照中最高项")
    r = {"schema_version": 1, "id": f"SRC-{n:03d}", "kind": "notebook", "title": item["title"],
        "url": item["url"], "observed_at": item["observed_at"],
        "source_version": {"notebook_version": item["version"], "script_version_id": None, "source_sha256": item["source_sha256"], "code_sha256": item["code_sha256"]},
        "family_id": "code-" + item["code_sha256"][:12],
        "public_score": {"value": item["score"], "evidence_type": "platform_page", "bound_to_source": False, "script_version_id": None},
        "summary": note, "author_claim": "正文声明保存在下方 description.md；标题分数不作为独立证据。",
        "verified": "2026-10-30 已观察页面分数并保存源码；增量说明见摘要。",
        "use_for_us": "用于选择对照实验，不直接替换当前基线。", "test_plan": "先核对具体得分版本，再把单项差异变成 IDEA。",
        "limitations": "页面分数与下载源码版本未绑定；排名只代表当日快照。同代码的副本不是独立算法。",
        "evidence": [directory + "/source.ipynb", directory + "/metadata.json", directory + "/description.md", "reviews/public-top20-20261030/ui-scores.json"]}
    save(TRACK / f"records/sources/{r['id']}.json", r)

for ident, version, val, params, folder in [("SRC-021", 2, 0.664, "Sinkhorn=0.30；标点修正=1.5", "v2-score-0664"), ("SRC-022", 6, None, "Sinkhorn=0.60；标点修正=1.2", ".")]:
    base = ROOT / "reviews/judith007-0664-20261031" / folder
    r = {"schema_version": 1, "id": ident, "kind": "notebook", "title": f"Who are U 作者 V{version}：{params}",
        "url": "https://www.kaggle.com/code/judith007/0-664-hyperparameter-tuned-who-are-u" + ("?scriptVersionId=363834272" if version == 2 else ""),
        "observed_at": "2026-10-31T14:01:22.370879+00:00",
        "source_version": {"notebook_version": version, "script_version_id": 363834272 if version == 2 else None, "code_sha256": code_sha(next(base.glob('*.ipynb')))},
        "public_score": {"value": val, "evidence_type": "platform_page" if val else "not_verified", "bound_to_source": version == 2, "script_version_id": 363834272 if version == 2 else None},
        "summary": params, "author_claim": "Notebook 标题沿用 0.664。",
        "verified": "V2 页面 0.664 已绑定 scriptVersionId；相对 alexchilton V8 仅改一行 Sinkhorn。" if version == 2 else "最新 V6 改两项参数；我们的旧副本 V1 代码与它完全一致。",
        "use_for_us": "作为后处理单参数实验候选。" if version == 2 else "保留为两参数组合实验，不给它登记作者 V2 的成绩。",
        "test_plan": "固定预测与验证划分比较；完整评分只验证少量候选。", "limitations": "单次 0.002 差异不能直接归为稳定提升；附件内容版本未独立逐一核查。",
        "evidence": [rel(next(base.glob('*.ipynb'))), "reviews/judith007-0664-20261031/review-evidence.json"]}
    save(TRACK / f"records/sources/{ident}.json", r)

discussions = [
    (671648, "Benchmarks 额度用途", "工作人员解释额度用于 Benchmarks 模型推理。", "可用模型和确切版本要在调用前查询；平台额度不等于特定比赛用途已获确认。", "先列出候选模型可用性，保留来源与计费信息。"),
    (750038, "比赛内生成数据的规则问题", "帖子询问免费额度能否生成训练数据；现有肯定回复来自参赛者。", "不能把参赛者回复写成主办方批准。", "追踪主办方正式澄清；在未确认前不预记生成额度为已授权支出。"),
    (751011, "API 数据成本、公开和复现义务", "作者集中提出六个规则问题；现有回复继续请求澄清。", "尚未从这个线程得到成本上限、公开期限等明确主办方答复。", "保留生成成本、许可证和版本记录，正式采用前核对规则。"),
    (753177, "单模型 OOF 提升却损害集成；稀疏验证的后续更新", "作者报告成员 OOF 0.569→0.605，替换后集成 LB 0.651→0.647；保留旧成员再加新成员得 0.650。GoPain 更新：m=2 在前 10 次相关 0.91，后 5 次为 -0.22，全部 15 次为 0.79。", "均为社区自报，未独立重训。不能只摘取 0.91；稀疏验证主要用于排除明显差配置，未证明能排序微小差异。", "比较单成员和加入集成的增量；先抽稀回答再重建聚类与中心化，保留多种组大小。"),
    (752766, "Open Pool v10 清洗与配对比较", "作者报告 raw v7 254k / LB 0.585，clean v9 193k / LB 0.587；v10 为 198507 条。评论补充：微小分差的不确定性与两方案错误分歧有关。", "作者自报，未独立复现；不能把 0.01 当成通用噪声阈值。按标点过滤可能强化已有偏差，需对照验证。", "固定分类器做原池/清洗池的等量与全量对照，记录各来源和类别权重。"),
    (751370, "Gemini 版本与生成数据标签风险", "发帖者怀疑旧端点转发；评论者的检测未发现官方标签明确受污染，删除某来源的 3.7 行也未提升整体分数。", "弱分类器混淆不能证明接口实际返回了哪个版本；讨论不是模型提供商的正式路由说明。", "保存请求模型与返回模型标识，不能凭风格检测器自行重标标签。")
]
for n, (topic_id, short, claim, caveat, test) in enumerate(discussions, 23):
    snap = load(SNAP / f"topic-{topic_id}.json")
    r = {"schema_version": 1, "id": f"SRC-{n:03d}", "kind": "discussion", "title": short,
        "url": "https://www.kaggle.com" + snap['topic']['url'], "observed_at": snap['observed_at'],
        "source_version": {"topic_id": topic_id, "snapshot_sha256": sha(SNAP / f"topic-{topic_id}.json")}, "public_score": None,
        "summary": short, "author_claim": claim, "verified": "本轮重新读取了帖子及返回的全部评论页；这是来源核查，不是实验复现。",
        "use_for_us": test, "test_plan": test, "limitations": caveat,
        "evidence": [rel(SNAP / f"topic-{topic_id}.json")]}
    save(TRACK / f"records/sources/{r['id']}.json", r)

save(TRACK / "records/sources/SRC-029.json", {
    "schema_version": 1, "id": "SRC-029", "kind": "local_audit", "title": "现有验证与 pool-centred SVC 的代码检查",
    "url": "../reviews/own-scheme-readiness-20261030/readiness-audit.json", "observed_at": "2026-10-30T15:40:20.795855+00:00",
    "source_version": {}, "public_score": None, "summary": "--sim 跳过最终标点步骤；全数据适配器不能后切成干净 OOF；svc4c 存在分组 ID/索引混用。",
    "author_claim": "本项目的静态检查与最小反例。", "verified": "原组 ID [0,0,2,2] 的筛选反例已核查；没有运行新训练。",
    "use_for_us": "先修验证方法，再评价数据和参数。", "test_plan": "隔离验证折；原始独立特征可缓存，组相关计算须在抽样之后重算。",
    "limitations": "这是源码问题定位，不是修复后收益证明。", "evidence": ["reviews/own-scheme-readiness-20261030/readiness-audit.json"]})

save(TRACK / "ideas.json", [
    {"id": "IDEA-001", "priority": 0, "title": "建立固定、无泄漏的 prompt 分组验证", "status": "planned", "sources": ["SRC-026", "SRC-029"], "next_action": "固定样本 ID、标签顺序、折划分和数据哈希，先跑轻量基线并输出逐类召回与混淆。", "acceptance": "同一 prompt 不跨折；所有拟合隔离验证；重复结果可追溯；不同验证范围分别报告。", "budget": "先 CPU；不启动完整多大模型训练。"},
    {"id": "IDEA-002", "priority": 1, "title": "缓存预测后比较 Sinkhorn 与标点修正", "status": "planned", "sources": ["SRC-021", "SRC-022"], "next_action": "保留后处理之前的概率，先固定 1.5 比较 Sinkhorn 0.30/0.45；再单独比较标点系数。", "acceptance": "同一批预测、同一验证划分；记录逐行改判及分差；少量候选进入完整评测。", "budget": "缓存后的参数比较使用 CPU；GPU 完整运行单独登记。"},
    {"id": "IDEA-003", "priority": 1, "title": "固定分类器比较公共池质量", "status": "planned", "sources": ["SRC-027"], "next_action": "记录来源、去重、长度、类别和过滤规则；比较原池、清洗池、等量抽样池。", "acceptance": "分组验证与来源权重固定；检查过滤是否删除了真实风格差异；小模型收益需主集成确认。", "budget": "先 CPU 与小规模数据，筛选后再训练 GPU 模型。"},
    {"id": "IDEA-004", "priority": 2, "title": "评估新成员的互补性", "status": "planned", "sources": ["SRC-026"], "next_action": "记录成员错误重叠、去掉/加入成员的变化，比较替换与小权重加入。", "acceptance": "同时报告单模型 OOF 和完整集成增量，不能仅凭单模型 OOF 决定替换。", "budget": "优先复用缓存；新训练另定上限。"},
    {"id": "IDEA-005", "priority": 2, "title": "Benchmarks 生成数据的小试准备", "status": "planned", "sources": ["SRC-023", "SRC-024", "SRC-025", "SRC-028"], "next_action": "先核实规则、实际可用候选版本和每条成本，再拟定共同提示词的小批量方案。", "acceptance": "模型版本可追溯、原文不润色、生成与验证 prompt 隔离，少量加入有效才扩大。", "budget": "尚未授权支出；此前 $5 小试/$100 月额度只是规划，不是已消费金额。"}
])
print("Imported 6 experiments, 29 sources and 5 ideas. Source comparison for Who Are You V2:", same)
