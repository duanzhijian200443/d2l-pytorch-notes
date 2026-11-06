"""Read-only source scan of 1,000 saved Codex answers; no inference or training.

Writes only metrics/direct-traces.json. Candidate matches are screening evidence,
not automatic contamination verdicts. Inspect samples in the companion review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
DATA = ROOT / "experiments/EXP-023-codex-subscription-pilot-r1"
RECORDS = DATA / "collection/records.jsonl"
INSTRUCTIONS = ROOT / "experiments/EXP-021-codex-subscription-pilot/model-instructions.txt"
PREFLIGHT = ROOT / "experiments/EXP-021-codex-subscription-pilot/runtime-preflight.json"
PROMPTS = ROOT / "experiments/EXP-021-codex-subscription-pilot/prompts.jsonl"

STRONG_PATTERNS = {
    "codex_or_context_file": r"\bCodex\b|\bAGENTS\.md\b|\bMEMORY(?:_SUMMARY)?\.md\b|\bMEMORY_SUMMARY\b|\bSKILL\.md\b|\bCODEX_HOME\b|\.codex(?:[/\\]|\b)",
    "instruction_configuration": r"\b(?:model_instructions_file|developer_instructions|approval_policy|sandbox_mode|skip_host_skill_discovery|code_mode_host|include_environment_context)\b",
    "role_or_context_delimiter": r"<\|(?:im_start|im_end|start|end|assistant|system|developer)\|>|</?(?:system|developer|environment_context|skills_instructions|permissions|tool_call|tool_result)(?:\s|>)|\[/?INST\]",
    "runtime_tool_namespace": r"\bfunctions\.(?:exec|wait)\b|\bmcp__\w+|\b(?:recipient|to)=(?:functions|tools|python|web)\b|<tool_call>|<tool_result>",
    "workspace_path": r"\b[A-Z]:[/\\](?:Users|计算机学习|Windows|agent-config)(?:[/\\]|\b)|/(?:home|root|workspace|mnt/data)/|name-that-llm|shiroha_quiz|长期记忆策略|OpenAI 接入开发文档",
    "base_instruction_literal": r"Answer the user's request\.",
    "explicit_identity": r"\b(?:I(?:'m| am)|as)\s+(?:an?\s+)?(?:Codex|ChatGPT|OpenAI(?:'s)?\s+(?:AI\s+)?(?:assistant|model))\b",
}
CONTEXT_PATTERNS = {
    "instruction_discussion": r"\b(?:system prompt|system instructions?|developer messages?|developer instructions?|hidden instructions?|instruction hierarchy|my instructions?|as instructed|as requested)\b",
    "ai_self_reference": r"\b(?:as an? (?:AI|language model)|I(?:'m| am) an? (?:AI|language model)|my (?:training data|knowledge cutoff))\b",
    "generic_assistant_term": r"\b(?:assistant|ChatGPT|OpenAI|language model|system message|prompt injection)\b",
    "tool_or_execution_claim": r"\b(?:I(?:'ve| have) (?:updated|modified|edited|run|executed|tested)|I(?:'ll| will) (?:inspect|run|execute|edit|modify|read the files)|tests? passed|running command|tool calls?|sandbox|repository|terminal command|command output)\b",
    "access_disclaimer": r"\bI (?:cannot|can't|don't have access to|do not have access to) (?:browse|access|run|execute|inspect)\b",
    "answer_wrapper": r"\A\s*(?:Sure[!,.—:]|Certainly[!,.—:]|Here(?:'s| is| are)\b|Of course[!,.—:])",
    "politeness_discourse": r"\bOf course,",
    "cjk_text": r"[\u3400-\u9fff]",
}
TOOL_TYPES = {
    "command_execution", "mcp_tool_call", "web_search", "file_change",
    "collab_tool_call", "tool_call", "function_call", "computer_call",
}
NOTICE_PREFIXES = {
    "experimental_feature_notice": "Under-development features enabled: skip_host_skill_discovery.",
    "disabled_code_mode_notice": "Code Mode is unavailable because code-mode host is disabled.",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def excerpt(text, start, end, radius=110):
    return text[max(0, start-radius):min(len(text), end+radius)]


def scan():
    raw = RECORDS.read_bytes()
    rows = read_jsonl(RECORDS)
    prompts = {p["prompt_id"]: p for p in read_jsonl(PROMPTS)}
    patterns = {**STRONG_PATTERNS, **CONTEXT_PATTERNS}
    compiled = {name: re.compile(pattern, re.I | re.M) for name, pattern in patterns.items()}
    hits = defaultdict(list)
    model_counts = Counter()
    item_types = Counter()
    event_types = Counter()
    notice_counts = Counter()
    consistency_errors = []
    tool_event_rows = []
    failure_event_rows = []
    unknown_error_items = []
    prompt_echo_rows = []
    completed_counts = Counter()
    agent_message_counts = Counter()
    input_tokens = defaultdict(list)
    for row in rows:
        key, model, text = row["job_key"], row["model_requested"], row["text"]
        model_counts[model] += 1
        prompt = row["prompt"]
        if prompts[row["prompt_id"]]["prompt"] != prompt:
            consistency_errors.append({"job_key": key, "reason": "original_prompt_mismatch"})
        attempt = (DATA / row["attempt_path"]).resolve()
        if not attempt.is_relative_to((DATA / "collection/attempts").resolve()):
            raise ValueError("Attempt outside authorized saved collection")
        final_raw = (attempt / "final.txt").read_bytes()
        events = read_jsonl(attempt / "raw_stdout.jsonl")
        final_text = final_raw.decode("utf-8")
        if (final_text != text or sha(final_raw) != row["raw_final_sha256"]
                or sha(text.encode("utf-8")) != row["text_sha256"]):
            consistency_errors.append({"job_key": key, "reason": "final_or_hash_mismatch"})
        completed = [e for e in events if e.get("type") == "turn.completed"]
        completed_counts[len(completed)] += 1
        messages = []
        for event in events:
            kind = event.get("type", "")
            event_types[kind] += 1
            item = event.get("item") or {}
            itype = item.get("type", "")
            if itype:
                item_types[itype] += 1
            if kind == "item.completed" and itype == "agent_message":
                messages.append(item.get("text", ""))
            if itype in TOOL_TYPES or itype.endswith("_tool_call"):
                tool_event_rows.append({"job_key": key, "item_type": itype})
            if kind in {"error", "turn.failed"}:
                failure_event_rows.append({"job_key": key, "type": kind})
            if itype == "error":
                message = item.get("message", "")
                known = next((name for name, prefix in NOTICE_PREFIXES.items() if message.startswith(prefix)), None)
                if known:
                    notice_counts[known] += 1
                    if message in text:
                        consistency_errors.append({"job_key": key, "reason": "runtime_notice_in_final"})
                else:
                    unknown_error_items.append({"job_key": key, "message": message[:400]})
        agent_message_counts[len(messages)] += 1
        if len(messages) != 1 or messages[0] != text:
            consistency_errors.append({"job_key": key, "reason": "agent_message_final_mismatch"})
        for event in completed:
            value = (event.get("usage") or {}).get("input_tokens")
            if value is not None:
                input_tokens[model].append(value)
        for name, regex in compiled.items():
            match = regex.search(text)
            if match:
                hits[name].append({"job_key": key, "prompt_id": row["prompt_id"], "model": model,
                                  "category": row["category"], "matched": match.group(),
                                  "in_prompt": bool(regex.search(prompt)),
                                  "excerpt": excerpt(text, match.start(), match.end()),
                                  "prompt": prompt})
        norm_prompt, norm_text = normalize(prompt), normalize(text)
        if norm_prompt in norm_text:
            prompt_echo_rows.append({"job_key": key, "prompt_id": row["prompt_id"], "model": model})
    strong_rows = {r["job_key"] for name in STRONG_PATTERNS for r in hits[name]}
    context_rows = {r["job_key"] for name in CONTEXT_PATTERNS for r in hits[name]}
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "Direct output trace screening, not causal system-prompt equivalence test",
        "source": {"path": RECORDS.relative_to(ROOT).as_posix(), "sha256": sha(raw), "rows": len(rows)},
        "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p.read_bytes())}
                   for p in (INSTRUCTIONS, PROMPTS, PREFLIGHT, DATA / "config.json")],
        "by_model": dict(model_counts),
        "strong_pattern_rows": len(strong_rows),
        "context_screen_rows": len(context_rows),
        "pattern_counts": {name: len(hits[name]) for name in patterns},
        "patterns": {"strong_candidates": STRONG_PATTERNS, "context_only_candidates": CONTEXT_PATTERNS},
        "hits": dict(hits),
        "entire_prompt_echo_rows": prompt_echo_rows,
        "event_checks": {"turn_completed_count_distribution": dict(completed_counts),
                         "agent_message_count_distribution": dict(agent_message_counts),
                         "event_types": dict(event_types), "item_types": dict(item_types),
                         "tool_event_rows": tool_event_rows, "failure_event_rows": failure_event_rows,
                         "informational_error_item_counts": dict(notice_counts),
                         "unknown_error_items": unknown_error_items},
        "consistency_errors": consistency_errors,
        "input_token_ranges": {model: {"n": len(values), "min": min(values), "max": max(values)}
                               for model, values in input_tokens.items()},
        "model_identity_scope": dict(Counter(r["model_observation_scope"] for r in rows)),
        "runtime_preflight_evidence": {
            "operation": preflight["operation"], "limitations": preflight["limitations"],
            "disabled_environment_probe": preflight["environment_context_disabled_probe"],
            "minimal_instruction": INSTRUCTIONS.read_text(encoding="utf-8"),
        },
        "limitations": [
            "No matched intervention removing runtime context; absent literal traces do not exclude indirect style or content conditioning.",
            "Actual full server-side prompt and backend model snapshots are not exposed by the saved events.",
            "debug prompt-input had different user-config handling from actual exec --ignore-user-config; do not equate its content with all generated requests.",
            "Regex matches require context review; generic task words and answer wrappers are not intrinsic contamination.",
            "Hash/event checks cover all 1,000 saved answers; manual semantic review covers candidate hits, not every sentence of the full corpus.",
        ],
    }
    destination = EXP / "metrics/direct-traces.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "strong_pattern_rows": len(strong_rows),
                      "context_screen_rows": len(context_rows), "pattern_counts": result["pattern_counts"],
                      "consistency_errors": len(consistency_errors), "tool_events": len(tool_event_rows),
                      "failure_events": len(failure_event_rows), "notices": dict(notice_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    scan()
