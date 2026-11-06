"""Local experiment ledger. Rendering is offline; sync-kaggle performs READS only."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
from urllib.parse import parse_qs, urlparse

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TZ = dt.timezone(dt.timedelta(hours=8))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def records(kind):
    return [read(p) for p in sorted((HERE / "records" / kind).glob("*.json"))]


def project_path(value):
    """Resolve a project-relative path without allowing traversal or drive paths."""
    if not isinstance(value, str) or not value or pathlib.PureWindowsPath(value).anchor:
        raise ValueError("path must be relative to the project root")
    relative = pathlib.PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError("path must stay inside the project root")
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("path resolves outside the project root")
    return target


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def create_experiment(experiments, settings, title, slug=None, config_path="configs/experiment-defaults.json"):
    """Create a planned record and its local draft workspace; never execute it."""
    if not title or not title.strip():
        raise ValueError("--title is required")
    if slug is not None and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        raise ValueError("--slug must contain lowercase letters/digits separated by hyphens")
    template_path = project_path(config_path)
    template_bytes = template_path.read_bytes()
    config = json.loads(template_bytes.decode("utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("--config must contain a JSON object")
    # Reject non-JSON NaN/Infinity values before creating any output.
    json_bytes(config)

    ident = f"EXP-{1 + max([int(x['id'].split('-')[1]) for x in experiments] or [0]):03d}"
    dirname = ident + (f"-{slug}" if slug is not None else "")
    workspace_root = project_path("experiments")
    workspace = project_path(f"experiments/{dirname}")
    record_path = project_path(f"tracking/records/experiments/{ident}.json")
    if record_path.exists() or record_path.is_symlink():
        raise FileExistsError(record_path)
    # Also reject orphan directories with this ID and a different slug.
    if workspace_root.exists():
        for candidate in workspace_root.iterdir():
            name = candidate.name.casefold()
            if name == ident.casefold() or name.startswith(ident.casefold() + "-"):
                raise FileExistsError(candidate)

    relative_workspace = workspace.relative_to(ROOT.resolve()).as_posix()
    relative_record = record_path.relative_to(ROOT.resolve()).as_posix()
    relative_config = relative_workspace + "/config.json"
    relative_manifest = relative_workspace + "/manifest.json"
    relative_report = relative_workspace + "/report.md"
    created_at = now()
    item = read(HERE / "templates" / "experiment.json")
    item.update(id=ident, title=title.strip(), observed_at=created_at,
                baseline_id=settings["baseline_experiment_id"], config_path=relative_config)
    item["config"] = {"note": "仅作台账索引；完整参数以 config_path 指向的 config.json 为准。"}
    item["execution"].update(scope="planned", status="PLANNED", observed_at=created_at)
    item["inputs"].update(manifest_path=relative_manifest, content_hashes_verified=False)
    item["evidence"] = [relative_config, relative_manifest, relative_report]
    item["next_action"] = "填写假设与主要改动，核验代码、输入、验证划分、环境和预算，冻结配置后再运行。"
    manifest = {
        "schema_version": 1,
        "experiment_id": ident,
        "created_at": created_at,
        "freeze_status": "DRAFT",
        "note": "仅记录创建时读取的配置文件哈希；代码、输入、分折与环境尚未核验或冻结。修改配置后须重新核验哈希。",
        "config": {"path": relative_config, "sha256": hashlib.sha256(template_bytes).hexdigest()},
        "config_template": {
            "path": template_path.relative_to(ROOT.resolve()).as_posix(),
            "sha256": hashlib.sha256(template_bytes).hexdigest(),
        },
        "inputs": [],
        "code": [],
    }
    report_title = " ".join(item["title"].splitlines())
    report = "\n".join([
        f"# {ident} · {report_title}", "",
        "状态：**PLANNED（未运行）**。当前仅建立本地计划与目录。", "",
        f"- 台账原始记录：[{ident}](../../{relative_record})",
        f"- 实验总览：[实验台账](../../tracking/实验台账.md#{ident.lower()})",
        f"- 对照基线：`{item['baseline_id']}`；登记时间：`{created_at}`。",
        "- 参数来源：[config.json](config.json)；文件清单：[manifest.json](manifest.json)。", "",
        "## 假设与主要改动", "", "待填写：可证伪的假设、对照与唯一主要改动。", "",
        "## 运行前核验", "",
        "清单为 **DRAFT**；仅复制了配置快照。代码与输入哈希、分折、标签顺序、环境和预算待补齐并核验。",
        "config.json 是本次参数来源；模板的后续修改不会自动改变此快照。", "",
        "## 执行与结果", "",
        "未运行。运行 ID、时间、实际环境、OOF、正式成绩及成本均待真实证据，未知值留空。",
        "OOF 存入 `oof/`，指标存入 `metrics/`，日志存入 `logs/`。", "",
        "## 结论、限制与下一步", "",
        "待填写；尚无实验结论。完成核验并获得相应运行授权后，才可开始执行。",
        "结果回填台账，再运行 `python -X utf8 tracking/ledger.py check` 与 `render`。", "",
    ]).encode("utf-8")
    outputs = [
        (workspace / "config.json", template_bytes),
        (workspace / "manifest.json", json_bytes(manifest)),
        (workspace / "report.md", report),
        (record_path, json_bytes(item)),
    ]

    created_files, created_dirs = [], []
    try:
        if not workspace_root.exists():
            workspace_root.mkdir()
            created_dirs.append(workspace_root)
        workspace.mkdir()
        created_dirs.append(workspace)
        for name in ("oof", "metrics", "logs"):
            directory = workspace / name
            directory.mkdir()
            created_dirs.append(directory)
        for path, contents in outputs:
            with path.open("xb") as stream:
                created_files.append(path)
                stream.write(contents)
    except BaseException:
        # Remove only entries created by this attempt, with no recursive deletion.
        for path in reversed(created_files):
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError(f"Refusing cleanup outside project root: {path}")
            path.unlink()
        for path in reversed(created_dirs):
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError(f"Refusing cleanup outside project root: {path}")
            path.rmdir()
        raise
    return item


def stamp(value):
    if not value:
        return "未记录"
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(TZ).strftime("%Y-%m-%d %H:%M 北京时间")


def cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def score(value):
    return "—" if value is None else f"{value:.6f}".rstrip("0").rstrip(".")


def local_link(path, label=None):
    return f"[{label or pathlib.PurePosixPath(path).name}](../{path})"


def state(experiment):
    sub = experiment.get("submission") or {}
    run = experiment.get("execution") or {}
    if sub.get("id"):
        status = sub.get("status", "UNKNOWN")
        return {"COMPLETE": "正式评分完成", "PENDING": "正式评分待完成", "ERROR": "正式评分失败"}.get(status, "正式提交：" + status)
    status = run.get("status", "PLANNED")
    if run.get("scope") == "commit" and status == "COMPLETE":
        return "保存完成；未关联正式提交"
    return {"PLANNED": "计划", "QUEUED": "排队", "RUNNING": "运行中", "COMPLETE": "运行完成（非榜单成绩）", "INTERRUPTED": "中断", "ERROR": "运行失败", "PREPARED": "准备完成；未运行", "UNKNOWN": "状态待核对"}.get(status, status)


def needs_attention(experiment):
    sub = experiment.get("submission") or {}
    run = experiment.get("execution") or {}
    if sub.get("id"):
        return sub.get("status") != "COMPLETE"
    return run.get("scope") == "commit" or run.get("status") != "COMPLETE"


def validate(experiments, sources, ideas, settings):
    errors = []
    exp_ids = {x["id"] for x in experiments}
    src_ids = {x["id"] for x in sources}
    for kind, rows, pattern in [("experiments", experiments, r"EXP-\d{3,}"), ("sources", sources, r"SRC-\d{3,}")]:
        ids = [x.get("id") for x in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{kind}: duplicate IDs")
        for row in rows:
            ident = row.get("id", "")
            if not re.fullmatch(pattern, ident) or not row.get("title"):
                errors.append(f"{ident}: invalid ID/title")
            if not (HERE / "records" / kind / f"{ident}.json").exists():
                errors.append(f"{ident}: record filename does not match ID")
            for path in row.get("evidence", []):
                target = (ROOT / path).resolve()
                if not target.is_relative_to(ROOT.resolve()) or not target.exists():
                    errors.append(f"{ident}: missing/outside evidence {path}")
            for date_key in ("observed_at",):
                if row.get(date_key):
                    try:
                        if dt.datetime.fromisoformat(row[date_key].replace("Z", "+00:00")).tzinfo is None:
                            raise ValueError("timezone required")
                    except ValueError:
                        errors.append(f"{ident}: invalid timestamp")
    for row in experiments:
        ident = row["id"]
        if row.get("baseline_id") and row["baseline_id"] not in exp_ids:
            errors.append(f"{ident}: unknown baseline")
        for source in row.get("related_sources", []):
            if source not in src_ids:
                errors.append(f"{ident}: unknown source {source}")
        sub = row.get("submission") or {}
        cv = (row.get("validation") or {}).get("balanced_accuracy")
        if cv is not None and not 0 <= cv <= 1:
            errors.append(f"{ident}: validation score outside [0,1]")
        if sub.get("public_score") is not None:
            if sub.get("status") != "COMPLETE" or not sub.get("id") or not sub.get("script_version_id"):
                errors.append(f"{ident}: score without complete version-bound submission")
            if not 0 <= sub["public_score"] <= 1:
                errors.append(f"{ident}: public score outside [0,1]")
        source = row.get("source") or {}
        if source.get("local_path") and source.get("sha256"):
            f = ROOT / source["local_path"]
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != source["sha256"]:
                errors.append(f"{ident}: source hash mismatch")
        if sub.get("script_version_id") and source.get("script_version_id") and sub["script_version_id"] != source["script_version_id"]:
            errors.append(f"{ident}: source/submission version mismatch")
    for row in sources:
        metric = row.get("public_score") or {}
        if metric.get("value") is not None and not 0 <= metric["value"] <= 1:
            errors.append(f"{row['id']}: public score outside [0,1]")
        if metric.get("bound_to_source") and not (metric.get("script_version_id") and row.get("source_version", {}).get("script_version_id") == metric["script_version_id"]):
            errors.append(f"{row['id']}: claimed score/source binding has no matching version")
    for idea in ideas:
        for source in idea.get("sources", []):
            if source not in src_ids:
                errors.append(f"{idea['id']}: unknown source {source}")
        for experiment in idea.get("related_experiments", []):
            if experiment not in exp_ids:
                errors.append(f"{idea['id']}: unknown experiment {experiment}")
        for path in idea.get("evidence", []):
            try:
                target = project_path(path)
            except ValueError:
                errors.append(f"{idea['id']}: missing/outside evidence {path}")
            else:
                if not target.is_file():
                    errors.append(f"{idea['id']}: missing/outside evidence {path}")
    if settings.get("baseline_experiment_id") not in exp_ids:
        errors.append("Unknown configured baseline")
    return errors


def render(experiments, sources, ideas, settings):
    generated = stamp(now())
    footer = ["", "---", f"视图生成时间：{generated}。数据更新时间见每条记录；生成视图不代表重新查询远端。", "原始记录位于 `records/`，请修改记录后运行 `python tracking/ledger.py render`，不要手改本页。", ""]
    exps = {x["id"]: x for x in experiments}
    baseline = exps[settings["baseline_experiment_id"]]
    finished = [x for x in experiments if (x.get("submission") or {}).get("status") == "COMPLETE" and x["submission"].get("public_score") is not None]
    best = max(finished, key=lambda x: x["submission"]["public_score"]) if finished else None
    current = ["# 当前状态", "", "[项目入口](../README.md) · [实验台账](实验台账.md) · [研究索引](研究索引.md) · [待验证想法](待验证想法.md)", "", "本页是带时间的状态快照，不是实时监控。对旧 RUNNING/PENDING 状态，先刷新再决定操作。", "", f"- 固定对照基线：**{baseline['id']} · {baseline['title']}**，Public LB **{score((baseline.get('submission') or {}).get('public_score'))}**。", f"- 已记录的自己最好正式成绩：**{score(best['submission']['public_score'])}**（{best['id']}）。" if best else "- 尚无已确认正式成绩。", "- 公开 Notebook 的成绩只进入研究索引，不参与‘自己的最好成绩’统计。", "", "## 最近状态与需要处理的事项", "", "| 实验 | 状态 | 最后核查 | 下一步 |", "|---|---|---|---|"]
    for e in experiments:
        sub = e.get("submission") or {}
        run = e.get("execution") or {}
        if needs_attention(e):
            current.append(f"| [{e['id']}](实验台账.md#{e['id'].lower()}) {cell(e['title'])} | {state(e)} | {stamp(e.get('observed_at'))} | {cell(e.get('next_action'))} |")
    unregistered_path = HERE / "unregistered-submissions.json"
    if unregistered_path.exists():
        unregistered = read(unregistered_path)
        if unregistered["submissions"]:
            current += ["", "## 有正式提交尚未关联 EXP", "", f"核查于 {stamp(unregistered['observed_at'])}。按提交 ID 和 scriptVersionId 核对后登记，不能按标题猜测。", ""]
            current += [f"- 提交 `{s['ref']}`：{s.get('status')}；页面分数 {s.get('publicScore') or '未返回'}。" for s in unregistered["submissions"]]
    current += ["", "## 最近核查的资料", ""]
    for r in sorted(sources, key=lambda x: x.get("observed_at") or "", reverse=True)[:6]:
        current.append(f"- [{r['id']} {r['title']}](研究索引.md#{r['id'].lower()})：{r['summary']}（核查于 {stamp(r.get('observed_at'))}）")
    current += ["", "## 记录中的关键边界", "", "- **0.664 绑定作者 judith007 的 V2 / scriptVersionId 363834272**：Sinkhorn=0.30、标点修正=1.5。2026-10-31 读取的 V6 是 0.60/1.2，不能继承 V2 的得分。见 SRC-021 / SRC-022。", "- 已有 Top20 是 **2026-10-30 的资料快照**，不是实时排行榜。多数页面分数尚未绑定到下载版本。", "- 先建立 prompt 分组验证；正式榜单、OOF、公开三条样例和本地模拟测试分别记录。", "- V32 旧运行记录有残留 RUNNING 字段，已知那次尝试的结果是中断，原因未知；网页启动器准备完成不等于后续 GPU 已成功运行。", "", "## 下一批工作", ""]
    current += [f"- **{i['id']} {i['title']}**：{i['next_action']}" for i in sorted(ideas, key=lambda x: (x['priority'], x['id'])) if i['status'] == 'planned'][:3]
    (HERE / "当前状态.md").write_text("\n".join(current + footer), encoding="utf-8")

    lines = ["# 自己的实验台账", "", "[当前状态](当前状态.md) · [研究索引](研究索引.md)", "", "一条 EXP 对应一个固定配置的一次独立尝试。重跑另建 EXP，并通过 baseline_id 关联；保存阶段与正式评分阶段分开。", "", "| 编号 | 实验 | 关键改动 | 状态 | OOF BA | 正式 Public LB | 提交 / 版本 |", "|---|---|---|---|---:|---:|---|"]
    for e in experiments:
        s = e.get("submission") or {}
        v = e.get("validation") or {}
        lines.append(f"| [{e['id']}](#{e['id'].lower()}) | {cell(e['title'])} | {cell('; '.join(e.get('changes', [])))} | {state(e)} | {score(v.get('balanced_accuracy'))} | {score(s.get('public_score'))} | {s.get('id') or '—'} / {s.get('script_version_id') or '—'} |")
    for e in experiments:
        lines += ["", f"## {e['id']}", "", f"**{e['title']}**", "", f"- 假设：{e.get('hypothesis') or '历史复现；未记录独立假设。'}", f"- 对照：{e.get('baseline_id') or '无'}；改动：{'；'.join(e.get('changes', [])) or '无'}。", f"- 状态：{state(e)}；核查于 {stamp(e.get('observed_at'))}。"]
        src = e.get("source") or {}
        if src:
            lines.append(f"- 源码：`{src.get('notebook_ref') or '本地/Colab'}`；保存版本 `{src.get('notebook_version')}`；scriptVersionId `{src.get('script_version_id')}`。")
        environment, cost = e.get("environment") or {}, e.get("cost") or {}
        lines += [f"- 环境：{environment.get('platform') or '未记录'}；硬件：{environment.get('accelerator') or '未记录/不适用'}。", f"- 成本：GPU 秒数 {cell(cost.get('gpu_seconds'))}；API 美元 {cell(cost.get('api_usd'))}；Colab CU {cell(cost.get('colab_cu'))}。空值表示未核实。"]
        if e.get("config_path"):
            lines.append("- 参数快照（本次参数来源）：" + local_link(e["config_path"]))
        lines += [f"- 配置：`{json.dumps(e.get('config', {}), ensure_ascii=False)}`", f"- 验证范围：{(e.get('validation') or {}).get('scope') or '尚无干净的本地验证成绩'}。", f"- 耗时：{e.get('timing_note') or '未核实；不以提交到查询的时间差代替执行时长。'}", f"- 结论：{e.get('conclusion') or '待结果。'}", f"- 局限：{e.get('limitations') or '待补充。'}", f"- 下一步：{e.get('next_action') or '无'}", "- 证据：" + " · ".join(local_link(x) for x in e.get("evidence", [])), f"- 可编辑记录：[records/experiments/{e['id']}.json](records/experiments/{e['id']}.json)"]
    (HERE / "实验台账.md").write_text("\n".join(lines + footer), encoding="utf-8")

    lines = ["# 社区资料与源码研究索引", "", "[当前状态](当前状态.md) · [实验台账](实验台账.md) · [待验证想法](待验证想法.md)", "", "社区结论用于提出假设。作者报告、平台页面分数、源码检查和我们的实验结论互不替代。日期是最后核查时间；未绑定分数的源码不能称为该分数配方。", "", "| 编号 | 资料 | 页面分数 | 下载 / 分数版本关系 | 要点 |", "|---|---|---:|---|---|"]
    for r in sources:
        m = r.get("public_score") or {}
        binding = "已绑定" if m.get("bound_to_source") else "未绑定" if m.get("value") is not None else "无榜单成绩"
        ver = r.get("source_version") or {}
        lines.append(f"| [{r['id']}](#{r['id'].lower()}) | [{cell(r['title'])}]({r['url']}) | {score(m.get('value'))} | 源码 v{ver.get('notebook_version') or '—'}；{binding} | {cell(r['summary'])} |")
    for r in sources:
        lines += ["", f"## {r['id']}", "", f"**[{r['title']}]({r['url']})** · {r['kind']} · 核查于 {stamp(r.get('observed_at'))}", "", f"- 来源声明：{r.get('author_claim') or '未单独提取。'}", f"- 已核查：{r.get('verified') or '仅收录，未独立验证。'}", f"- 对我们的用途：{r.get('use_for_us') or '候选参考；需要实验。'}", f"- 待验证：{r.get('test_plan') or '尚未安排。'}", f"- 局限：{r.get('limitations') or '不能将社区观测直接当作自己的效果。'}", "- 证据：" + " · ".join(local_link(x) for x in r.get("evidence", [])), f"- 可编辑记录：[records/sources/{r['id']}.json](records/sources/{r['id']}.json)"]
    (HERE / "研究索引.md").write_text("\n".join(lines + footer), encoding="utf-8")

    lines = ["# 待验证想法", "", "[当前状态](当前状态.md) · [实验台账](实验台账.md) · [研究索引](研究索引.md)", "", "这里记录可证伪的假设，不把想法当成已实现功能。排期不等于启动 GPU、生成 API 数据或提交比赛的授权。", "", "| 编号 | 优先级 | 假设 / 工作 | 状态 | 来源 |", "|---|---|---|---|---|"]
    for i in sorted(ideas, key=lambda x: (x['priority'], x['id'])):
        lines.append(f"| {i['id']} | P{i['priority']} | {i['title']} | {i['status']} | {', '.join(i.get('sources', []))} |")
    for i in sorted(ideas, key=lambda x: (x['priority'], x['id'])):
        lines += ["", f"### {i['id']} {i['title']}", "", f"- 下一步：{i['next_action']}", f"- 判断标准：{i['acceptance']}", f"- 成本边界：{i['budget']}"]
        for field, label in (
            ("baseline_delta", "相对正式基线的增量"),
            ("evidence_assessment", "证据与限制"),
            ("expected_benefit", "预期收益"),
            ("prerequisites", "前置条件"),
            ("stop_rule", "停止/降级条件"),
        ):
            if i.get(field):
                lines.append(f"- {label}：{i[field]}")
        if i.get("related_experiments"):
            lines.append("- 关联实验：" + " · ".join(
                f"[{ident}](实验台账.md#{ident.lower()})" for ident in i["related_experiments"]))
        if i.get("evidence"):
            lines.append("- 证据：" + " · ".join(local_link(path) for path in i["evidence"]))
        lines.append("")
    (HERE / "待验证想法.md").write_text("\n".join(lines + footer), encoding="utf-8")


def sync_kaggle(experiments):
    # No push, submit, start, stop, publish or credential-creation calls exist here.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    observed = now()
    rows = []
    for row in api.competition_submissions("name-that-llm"):
        d = row.to_dict()
        d["status"] = str(row.status).split(".")[-1]
        rows.append(d)
    dest = HERE / "snapshots" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write(dest / "submissions.json", {"observed_at": observed, "source": "official Kaggle API", "submissions": rows})
    evidence = (dest / "submissions.json").relative_to(ROOT).as_posix()
    by_id = {str(x["ref"]): x for x in rows}
    known_ids = {str((e.get("submission") or {}).get("id")) for e in experiments}
    write(HERE / "unregistered-submissions.json", {"observed_at": observed, "evidence": evidence, "submissions": [r for r in rows if str(r["ref"]) not in known_ids]})
    for e in experiments:
        sub = e.get("submission") or {}
        if sub.get("id"):
            row = by_id.get(str(sub["id"]))
            if row:
                old_status = sub.get("status")
                sid = parse_qs(urlparse(row.get("url", "")).query).get("scriptVersionId", [None])[0]
                if sid and sub.get("script_version_id") and int(sid) != sub["script_version_id"]:
                    raise ValueError(f"{e['id']}: remote submission binding changed")
                sub.update(status=row["status"], public_score=float(row["publicScore"]) if row["status"] == "COMPLETE" and row.get("publicScore") not in (None, "") else None, observed_at=observed)
                e["observed_at"] = observed
                if row["status"] == "COMPLETE" and old_status != "COMPLETE":
                    # Preserve the researcher's interpretation. Metrics are displayed separately.
                    e["next_action"] = "记录结果并比较固定验证；不自动重跑或更换基线。"
                e["evidence"] = list(dict.fromkeys(e.get("evidence", []) + [evidence]))
        elif (e.get("source") or {}).get("notebook_ref") and (e.get("execution") or {}).get("scope") == "commit":
            # Poll the registered version, not a later edited notebook version.
            src = e["source"]
            if src.get("notebook_version"):
                response = api.kernels_status(f"{src['notebook_ref']}/{src['notebook_version']}")
                status = str(response.status).split(".")[-1]
                sp = dest / f"{e['id']}-commit-status.json"
                write(sp, {"observed_at": observed, "notebook_ref": src["notebook_ref"], "notebook_version": src["notebook_version"], "status": status})
                e["execution"].update(status=status, observed_at=observed)
                e["observed_at"] = observed
                e["evidence"] = list(dict.fromkeys(e.get("evidence", []) + [sp.relative_to(ROOT).as_posix()]))
                if status == "COMPLETE":
                    e["next_action"] = "保存完成；如要正式评分，由用户确认对应版本后提交，并登记提交 ID。"
        write(HERE / "records" / "experiments" / f"{e['id']}.json", e)
    print("Read-only Kaggle refresh saved:", dest.relative_to(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "render", "sync-kaggle", "new-experiment"])
    parser.add_argument("--title")
    parser.add_argument("--slug", help="optional lowercase letters/digits separated by hyphens")
    parser.add_argument("--config", default="configs/experiment-defaults.json", help="project-relative JSON configuration template")
    args = parser.parse_args(argv)
    experiments, sources = records("experiments"), records("sources")
    ideas, settings = read(HERE / "ideas.json"), read(HERE / "settings.json")
    if args.command == "new-experiment":
        existing_errors = validate(experiments, sources, ideas, settings)
        if existing_errors:
            for error in existing_errors:
                print("ERROR:", error)
            return 1
        try:
            item = create_experiment(experiments, settings, args.title, args.slug, args.config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        experiments.append(item)
        print("Created planned record:", f"tracking/records/experiments/{item['id']}.json")
        print("Created draft workspace:", pathlib.PurePosixPath(item["config_path"]).parent)
    if args.command == "sync-kaggle":
        sync_kaggle(experiments)
    errors = validate(experiments, sources, ideas, settings)
    if errors:
        for error in errors:
            print("ERROR:", error)
        return 1
    print(f"PASS: {len(experiments)} experiments, {len(sources)} sources, {len(ideas)} ideas")
    if args.command != "check":
        render(experiments, sources, ideas, settings)
        print("Rendered 4 Markdown views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
