"""Replay captured ATLAS20 arithmetic on CPU, without models or evaluation labels.

Inputs are ordered, decoded observer events, including their first ``data``
event. Cache byte/hash verification belongs to ``read_cache``. A passing replay
establishes arithmetic consistency conditional on the captured member outputs;
it does not authenticate model execution, checkpoint provenance, or clean OOF.
The equations and operation order follow the frozen EXP-009 consumer scripts.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.special import logsumexp, softmax


LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
MARKS = [" — ", "—", "–", "’", "‘", "“", "”", "«", "»", '"', ";", ":", "(", ")",
         "[", "]", "**", "`", "…", "•", "\n", "\n\n", "e.g.", "i.e.", "However,", "Overall,",
         "In conclusion", "Therefore,", "Moreover,", "\t", " - ", " = ", "\n- ", "\n1.", "###", "```", "> "]
GATE_ORDER = ("claude", "claude_lora", "gpt", "gemini", "glm")
COMP_REQUIRED = {"stylegram", "stylometry_lgb", "debv3l", "electl", "desk", "ultrabert"}
GROUP_REQUIRED = {"q3_8b_twin", "q25_7b_twin", "q3_4b_twin", "glm9b_twin", "debv3l_twin",
                  "gauss_q3_8b_L22", "gauss_glm9b_L8", "gauss_q3_4b_L4", "gauss_debv3l_L24"}


class ReplayError(ValueError):
    """Malformed, incomplete or numerically inconsistent captured events."""


def _require(condition, message):
    if not condition:
        raise ReplayError(message)


def _get(payload, key, where):
    _require(key in payload, f"{where}: missing {key}")
    return payload[key]


def _array(value, shape, where, kind="f"):
    _require(isinstance(value, np.ndarray), where + ": expected a decoded numpy array")
    _require(value.shape == tuple(shape), where + f": shape {value.shape}, expected {tuple(shape)}")
    _require(value.dtype.kind in kind and np.isfinite(value).all(), where + ": invalid dtype/nonfinite values")
    if kind == "f":
        _require(value.dtype in (np.dtype("float32"), np.dtype("float64")), where + ": expected float32/float64")
    return value


def _scalar(value, where, low=None, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
             and isinstance(value, (int, float, np.integer, np.floating)), where + ": expected numeric scalar")
    value = float(value)
    _require(np.isfinite(value) and (low is None or value >= low) and (high is None or value <= high),
             where + ": invalid scalar range")
    return value


def _integer(value, where, low=0):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
             and int(value) >= low, where + ": expected integer")
    return int(value)


def _prob(value, rows, where, columns=18):
    value = _array(value, (rows, columns), where)
    _require(np.all(value >= 0) and np.all(value <= 1 + 2e-6), where + ": invalid probability range")
    _require(np.allclose(value.sum(1), 1, atol=2e-6, rtol=2e-6), where + ": probability rows do not sum to one")
    return value


def _groups(value, rows, where):
    value = _array(value, (rows,), where, "iu")
    _require(np.all(value >= 0), where + ": negative group")
    _, inverse, counts = np.unique(value, return_inverse=True, return_counts=True)
    return value, counts[inverse]


def _row_indices(value, rows, where):
    _require(isinstance(value, np.ndarray) and value.ndim == 1, where + ": expected row vector")
    if value.dtype.kind == "b":
        _require(value.shape == (rows,), where + ": boolean mask length mismatch")
        return np.flatnonzero(value)
    _array(value, value.shape, where, "iu")
    _require(np.all((value >= 0) & (value < rows)) and len(np.unique(value)) == len(value),
             where + ": invalid/duplicate row indices")
    return value


def _softmax_original(z):
    shifted = z - z.max(1, keepdims=True)
    result = np.exp(shifted)
    result /= result.sum(1, keepdims=True)
    return result


def _logsoftmax64(z):
    z = np.asarray(z, dtype=np.float64)
    return z - logsumexp(z, axis=1, keepdims=True)


class _Checks:
    def __init__(self):
        self.stages = []

    def compare(self, name, actual, expected, *, scores=False, exact=False):
        _require(isinstance(expected, np.ndarray) and np.isfinite(expected).all(), name + ": nonfinite recomputation")
        actual = _array(actual, expected.shape, name, expected.dtype.kind)
        _require(actual.dtype == expected.dtype, name + f": dtype changed ({actual.dtype} vs {expected.dtype})")
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0))
        tolerance = 2e-6 if expected.dtype == np.dtype("float32") else 2e-10
        close = np.array_equal(actual, expected) if exact or expected.dtype.kind != "f" else np.allclose(
            actual, expected, atol=tolerance, rtol=tolerance)
        argmax_equal = bool(np.array_equal(actual.argmax(1), expected.argmax(1))) if scores and len(expected) else True
        self.stages.append({"stage": name, "max_abs_error": error,
                            "argmax_equal": argmax_equal if scores else None, "dtype": str(expected.dtype),
                            "atol": 0.0 if exact else tolerance, "rtol": 0.0 if exact else tolerance})
        _require(close, name + f": numerical mismatch (max_abs_error={error})")
        _require(argmax_equal, name + ": argmax mismatch, including near ties")


class _Events:
    def __init__(self, events, component):
        _require(isinstance(events, (list, tuple)) and len(events) >= 2, component + ": missing event stream")
        self.items, self.by_name = [], {}
        for i, item in enumerate(events):
            _require(isinstance(item, (list, tuple)) and len(item) == 2, component + ": invalid event tuple")
            name, payload = item
            _require(isinstance(name, str) and isinstance(payload, dict), component + ": invalid event payload")
            self.items.append((name, payload))
            self.by_name.setdefault(name, []).append((i, payload))
        _require(self.items[0][0] == "data" and self.items[-1][0] == "component_end",
                 component + ": first data and final component_end are required")
        self.component = component

    def one(self, name, required=True):
        entries = self.by_name.get(name, [])
        _require(len(entries) <= 1 and (bool(entries) or not required), self.component + ": missing/duplicate " + name)
        return entries[0][1] if entries else None

    def order(self, ranks):
        previous = -1
        for name, _ in self.items:
            _require(name in ranks and ranks[name] >= previous, self.component + ": unexpected/out-of-order " + name)
            previous = ranks[name]


def _identity(events):
    data = events.one("data")
    labels = _get(data, "labels", "data")
    ids = _get(data, "ids", "data")
    hashes = _get(data, "text_sha256", "data")
    _require(isinstance(labels, (list, tuple)) and tuple(labels) == LABELS, "data: fixed 18-label order differs")
    _require(isinstance(ids, (list, tuple)) and len(ids) > 0
             and all(isinstance(x, str) and x.strip() for x in ids) and len(ids) == len(set(ids)), "data: invalid/duplicate IDs")
    _require(isinstance(hashes, (list, tuple)) and len(hashes) == len(ids)
             and all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in hashes), "data: invalid text hashes")
    _require(data.get("component") == events.component, "data: component differs")
    _require(data.get("scope") in {"synthetic_math", "public_sample_engineering", "exploratory_external"},
             "data: unsupported/unverified scientific scope")
    return {"ids": list(ids), "text_sha256": list(hashes), "labels": list(labels), "scope": data["scope"]}


def _inventory(observed, expected, where, defaults=None):
    _require(len(observed) > 0 and len(set(observed)) == len(observed), where + ": empty/duplicate members")
    if expected is not None:
        _require(isinstance(expected, (list, tuple, set, frozenset))
                 and all(isinstance(x, str) and x for x in expected) and len(set(expected)) == len(expected),
                 where + ": invalid expected member declaration")
        _require(set(observed) == set(expected), where + ": missing/extra declared members")
    elif defaults is not None:
        _require(set(defaults).issubset(observed), where + ": missing mandatory baseline member")


def _replay_comp(events, n, expected_members, checks):
    events.order({"data": 0, "style_margin_context": 1, "style_margin": 1, "style_routed": 1,
                  "member": 1, "comp_before_gates": 2, "family_gate_applied": 3, "component_end": 4})
    before, end = events.one("comp_before_gates"), events.one("component_end")
    groups, sizes = _groups(_get(before, "groups", "comp_before_gates"), n, "comp groups")
    style_groups, style_sizes = _groups(_get(before, "style_groups", "comp_before_gates"), n, "style groups")
    mask = _array(_get(before, "group_mask", "comp_before_gates"), (n,), "comp group_mask", "b")
    style_mask = _array(_get(before, "style_mask", "comp_before_gates"), (n,), "style mask", "b")
    checks.compare("comp.group_mask", mask, (sizes >= 2) & (sizes <= 18), exact=True)
    checks.compare("comp.style_mask", style_mask, (style_sizes >= 2) & (style_sizes <= 18), exact=True)
    logit = np.zeros((n, 18))
    members, style_probability = [], None
    for _, payload in events.by_name.get("member", []):
        key = _get(payload, "key", "member")
        _require(isinstance(key, str) and key and key not in members, "comp: invalid/duplicate member")
        members.append(key)
        row_value = _get(payload, "rows", "member")
        selected = np.arange(n) if row_value is None else _row_indices(row_value, n, key + ".rows")
        _require(len(selected) > 0, key + ": empty contribution")
        probability = _prob(_get(payload, "probability", "member"), len(selected), key + ".probability")
        lp = np.log(np.clip(probability, 1e-9, 1.0))
        checks.compare("comp.member." + key + ".log", _get(payload, "log_probability", "member"), lp, scores=True)
        weight = _scalar(_get(payload, "weight", "member"), key + ".weight", 0)
        if row_value is None:
            logit[:] += weight * lp
        else:
            logit[row_value] += weight * lp
        if key.endswith("_twin") or key.startswith("gauss_"):
            _require(row_value is not None and np.array_equal(selected, np.flatnonzero(mask)), key + ": grouped rows differ")
        if key == "stylegram":
            _require(row_value is None, "stylegram must cover all rows")
            style_probability = probability
    defaults = COMP_REQUIRED | (GROUP_REQUIRED if mask.any() else set())
    _inventory(members, expected_members, "comp", defaults)
    checks.compare("comp.before_gates", _get(before, "logit", "comp_before_gates"), logit, scores=True)
    # Optional raw SVC observations are checked when supplied; member probabilities
    # remain the mathematical boundary for members without raw score captures.
    contexts = events.by_name.get("style_margin_context", [])
    margin_entries = events.by_name.get("style_margin", [])
    routed = events.one("style_routed", required=False)
    if contexts or margin_entries or routed is not None:
        _require(style_probability is not None and routed is not None and len(contexts) == len(margin_entries),
                 "style routing: incomplete raw-margin events")
        parts = {}
        for position, context in contexts:
            _require(position + 1 < len(events.items) and events.items[position + 1][0] == "style_margin",
                     "style margin must immediately follow context")
            role = _get(context, "role", "style_margin_context")
            _require(role in {"plain", "centred"} and role not in parts and context.get("evaluation") is True,
                     "style margin: invalid/duplicate role")
            rows = _row_indices(_get(context, "rows", "style_margin_context"), n, "style margin rows")
            expected_rows = np.arange(n) if role == "plain" else np.flatnonzero(style_mask)
            _require(np.array_equal(rows, expected_rows) and len(rows) > 0, "style margin rows differ")
            classes = _array(_get(context, "classes", "style_margin_context"), (18,), "style classes", "iu")
            _require(set(classes.tolist()) == set(range(18)), "style class permutation differs")
            margin = _array(_get(events.items[position + 1][1], "value", "style_margin"), (len(rows), 18), "style margin")
            part = np.zeros((len(rows), 18))
            part[:, classes] = _softmax_original(margin)
            parts[role] = part
        _require(set(parts) == ({"plain", "centred"} if style_mask.any() else {"plain"}), "style margin role missing")
        style = parts["plain"].copy()
        if style_mask.any():
            style[style_mask] = parts["centred"]
        checks.compare("comp.style_routed", _get(routed, "probability", "style_routed"), style, scores=True)
        checks.compare("comp.style_member", style_probability, style, scores=True)
        checks.compare("comp.style_groups", _get(routed, "groups", "style_routed"), style_groups, exact=True)
        checks.compare("comp.style_route_mask", _get(routed, "mask", "style_routed"), style_mask, exact=True)
    gates = _get(before, "gates", "comp_before_gates")
    _require(isinstance(gates, dict) and set(gates).issubset(GATE_ORDER), "comp: unknown family gate")
    expected_gates = [key for key in GATE_ORDER if gates.get(key) is not None]
    applied = [payload for _, payload in events.by_name.get("family_gate_applied", [])]
    _require([p.get("key") for p in applied] == expected_gates, "comp: missing/extra/out-of-order family gate")
    for payload, key in zip(applied, expected_gates):
        family = "claude" if key == "claude_lora" else key
        idx = np.array([i for i, label in enumerate(LABELS) if label.startswith(family)])
        _require(payload.get("family") == family, "family gate family differs")
        captured_idx = _array(_get(payload, "indices", key), (len(idx),), key + ".indices", "iu")
        _require(np.array_equal(captured_idx, idx), "family gate indices differ")
        raw = _array(gates[key], (n, len(idx)), key + ".raw")
        checks.compare("comp.gate." + key + ".raw", _get(payload, "raw_score", key), raw, scores=True, exact=True)
        score = np.asarray(raw, dtype=np.float64)
        if key != "claude_lora":
            score = score - score.mean(1, keepdims=True)
            scale = np.sqrt(np.mean(score * score, axis=1, keepdims=True))
            score = score / np.maximum(scale, 0.25)
        gate_lp = _logsoftmax64(score)
        base = logit[:, idx].copy()
        base_lp = _logsoftmax64(base)
        ordered = np.sort(np.exp(base_lp), axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        confidence = np.clip((0.45 - margin) / 0.45, 0, 1)
        lam = 1.5 if key == "claude" else 1.0 if key == "claude_lora" else 0.25
        strength = min(0.45, lam / (1 + lam)) * confidence
        mixed_lp = _logsoftmax64((1 - strength[:, None]) * base_lp + strength[:, None] * gate_lp)
        mass = logsumexp(base, axis=1)
        logit[:, idx] = mass[:, None] + mixed_lp
        for field, value, scores in (("gate_log_probability", gate_lp, True), ("base", base, True),
                                     ("strength", strength, False), ("family_mass", mass, False), ("logit", logit, True)):
            checks.compare("comp.gate." + key + "." + field, _get(payload, field, key), value, scores=scores)
    checks.compare("comp.end", _get(end, "logit", "comp end"), logit, scores=True)
    serialized = logit.astype(np.float32)
    checks.compare("comp.float32_boundary", _get(end, "serialized_logit", "comp end"), serialized, scores=True, exact=True)
    return serialized, members, expected_gates


def _main_inventory(payload, args, n, checks, expected_members):
    members = _get(payload, "members", "main_members")
    _require(isinstance(members, dict) and all(isinstance(k, str) and k for k in members), "main: invalid member mapping")
    _inventory(list(members), expected_members, "main")
    specs = _get(payload, "specs", "main_members")
    _require(isinstance(specs, list) and all(isinstance(s, dict) and isinstance(s.get("name"), str) for s in specs),
             "main: invalid specs")
    names = [s["name"] for s in specs]
    _require(len(names) == len(set(names)), "main: duplicate spec names")
    needed = {"tfidf"}
    if _scalar(_get(args, "svc_w", "args"), "svc_w", 0) > 0:
        needed.add("svc4")
    if _scalar(_get(args, "distort_w", "args"), "distort_w", 0) > 0:
        needed.add("distort")
    adapt = _scalar(_get(args, "adapt_w", "args"), "adapt_w", 0)
    for spec in specs:
        needed.add(spec["name"])
        if spec.get("type") == "l24" and _scalar(spec.get("gauss", 0), "gauss weight", 0) > 0:
            needed.add(spec["name"] + "_gauss")
        if adapt > 0 and (spec.get("type") == "l24" or spec.get("type") not in {"enc", "head", "lora"}):
            needed.add(spec["name"] + "_twin")
    _require(set(members) == needed, "main: member inventory differs from captured specs/args")
    groups, sizes = _groups(_get(payload, "groups", "main_members"), n, "main groups")
    for key, value in members.items():
        _array(value, (n, 18), "main member " + key)
        if key.endswith("_gauss"):
            _require(np.all(value[(sizes < 2) | (sizes > 18)] == 0), key + ": ungrouped Gaussian contribution")
        elif key.endswith("_twin"):
            _require(np.all(value[sizes < 2] == 0), key + ": ungrouped twin contribution")
    total = sum(members.values())  # Preserve captured mapping/reduction order and dtype.
    checks.compare("main.member_total", _get(payload, "total", "main_members"), total, scores=True)
    return members, total


def _replay_typography(payload, raw, y, checks):
    n, _ = raw.shape
    _require(payload.get("marks") == MARKS, "typography: marker order differs")
    m = len(MARKS)
    rows = _integer(_get(payload, "train_rows", "typography"), "typography train_rows", 1)
    _require(rows == len(y), "typography: train row count differs")
    counts = np.bincount(y, minlength=18).astype(np.float64)
    checks.compare("typography.train_class_counts", _get(payload, "train_class_counts", "typography"), counts, exact=True)
    hits = _array(_get(payload, "train_marker_hits", "typography"), (18, m), "typography hits")
    _require(hits.dtype == np.float64 and np.all(hits >= 0) and np.all(hits <= counts[:, None])
             and np.all(hits == np.floor(hits)), "typography: impossible marker counts")
    x = _array(_get(payload, "test_marker_matrix", "typography"), (n, m), "typography test markers")
    _require(x.dtype == np.float32 and np.all((x == 0) | (x == 1)), "typography: expected float32 binary markers")
    kappa = _scalar(_get(payload, "kappa", "typography"), "typography kappa")
    _require(kappa == 24, "typography kappa changed")
    # Original Xtr is float32, so its marker sums and global smoothing are float32.
    global_rate = (hits.sum(0).astype(np.float32) + 2.0) / (rows + 4.0)
    rate = (hits + kappa * global_rate[None, :]) / (counts[:, None] + kappa)
    rate = np.clip(rate, 0.01, 0.99)
    global_rate = np.clip(global_rate, 0.01, 0.99)
    signal = np.sqrt(np.mean((rate - global_rate[None, :]) ** 2, axis=0))
    weights = signal / max(float(np.median(signal[signal > 0])) if np.any(signal > 0) else 1.0, 1e-6)
    weights = np.clip(weights, 0.25, 2.0)
    evidence = (x * weights[None, :]) @ np.log(rate).T + ((1.0 - x) * weights[None, :]) @ np.log1p(-rate).T
    evidence -= ((x * weights[None, :]) @ np.log(global_rate)
                 + ((1.0 - x) * weights[None, :]) @ np.log1p(-global_rate))[:, None]
    evidence -= evidence.mean(1, keepdims=True)
    evidence = np.clip(1.5 * evidence / max(float(weights.sum()), 1e-6), -0.30, 0.30)
    probabilities = softmax(raw, axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    confidence = np.clip((0.60 - margin) / 0.60, 0.0, 1.0)
    words = _array(_get(payload, "word_count", "typography"), (n,), "typography word_count")
    _require(words.dtype == np.float32 and np.all(words >= 0) and np.all(words == np.floor(words)),
             "typography: invalid captured regex word counts")
    length = np.clip(np.sqrt(np.maximum(words, 1.0) / 80.0), 0.40, 1.0)
    active = x.sum(1)
    coverage = np.clip(np.sqrt((active + 1.0) / 4.0), 0.45, 1.0)
    gate = confidence * length * coverage
    final = raw + gate[:, None] * evidence
    for field, value, score in (("global_rate", global_rate, False), ("rates", rate, False),
                                ("feature_weights", weights, False), ("evidence", evidence, True),
                                ("active_markers", active, False), ("confidence_gate", confidence, False),
                                ("length_gate", length, False), ("coverage_gate", coverage, False),
                                ("gate", gate, False), ("logits_before", raw, True), ("logits_after", final, True)):
        checks.compare("typography." + field, _get(payload, field, "typography"), value, scores=score)
    return final


def _replay_main(events, identity, comp_serialized, expected_members, checks):
    events.order({"data": 0, "main_svc4_margin": 1, "main_members": 2, "comp_loaded": 3, "comp_status": 4,
                  "main_probability": 5, "pre_atlas": 6, "atlas_requested": 7, "atlas_batch": 8,
                  "atlas_returned": 8, "pre_sinkhorn": 9, "sinkhorn_applied": 10,
                  "postprocessing_base": 11, "typography": 12, "component_end": 13})
    n = len(identity["ids"])
    payload = events.one("main_members")
    args = _get(payload, "args", "main_members")
    _require(isinstance(args, dict), "main args must be a mapping")
    _require(args.get("perpassage") == 1 and args.get("post") == 0 and not args.get("sim"),
             "Replay supports the frozen perpassage=1, post=0, non-simulation baseline only")
    members, total = _main_inventory(payload, args, n, checks, expected_members)
    y = _get(payload, "train_targets", "main_members")
    _require(isinstance(y, np.ndarray) and y.ndim == 1 and len(y) > 0, "main: missing training targets")
    _array(y, y.shape, "main train_targets", "iu")
    _require(np.all((y >= 0) & (y < 18)), "main: invalid training class index")
    margin_event = events.one("main_svc4_margin", required=False)
    if margin_event is not None:
        _require("svc4" in members, "SVC margins without main SVC member")
        classes = _array(_get(margin_event, "classes", "main_svc4_margin"), (18,), "main SVC classes", "iu")
        _require(np.array_equal(classes, np.arange(18)), "main SVC class order differs")
        d = _array(_get(margin_event, "margins", "main_svc4_margin"), (n, 18), "main SVC margins")
        svc = (d - d.max(1, keepdims=True) - np.log(np.exp(d - d.max(1, keepdims=True)).sum(1, keepdims=True))) * args["svc_w"]
        checks.compare("main.svc4_margin", members["svc4"], svc, scores=True)
    comp_w = _scalar(_get(args, "comp_w", "args"), "comp_w", 0, 1)
    status = events.one("comp_status")
    _require(_scalar(_get(status, "requested_weight", "comp_status"), "comp_status weight") == comp_w,
             "comp requested weight differs")
    loaded = events.one("comp_loaded", required=False)
    comp_probability = None
    if comp_w > 0:
        _require(loaded is not None and status.get("probability") is not None, "Required complementary member was skipped")
        meta = _get(loaded, "comp_meta", "comp_loaded")
        _require(isinstance(meta, dict) and meta.get("ids") == identity["ids"] and meta.get("labels") == list(LABELS),
                 "comp loaded metadata IDs/labels differ")
        _require(loaded.get("path") == status.get("path") == args.get("comp_path") and bool(args.get("comp_path")),
                 "comp source path differs")
        checks.compare("main.comp_loaded_float32", _get(loaded, "logit", "comp_loaded"), comp_serialized, scores=True, exact=True)
        # The original np.load(... )[:, permutation] uses advanced column
        # indexing even for an identity permutation. It produces an F-layout
        # matrix, whose float32 row reduction can differ from a C-layout copy.
        # decode_events copies arrays, so reproduce the operation rather than
        # trusting the captured array's current memory order.
        loaded_columns = comp_serialized[:, [meta["labels"].index(label) for label in LABELS]]
        comp_probability = _softmax_original(loaded_columns)
        checks.compare("main.comp_probability", _get(loaded, "probability", "comp_loaded"), comp_probability, scores=True)
        checks.compare("main.comp_status_probability", status["probability"], comp_probability, scores=True)
    else:
        _require(loaded is None and status.get("probability") is None, "Unexpected complementary probability")
    _require(isinstance(args.get("atlas_dir"), str), "args: missing atlas_dir")
    bundles = [x for x in args["atlas_dir"].split(",") if x]
    _require(len(bundles) == len(set(bundles)), "Duplicate ATLAS bundles")
    try:
        weights = [float(x) for x in str(_get(args, "atlas_w", "args")).split(",")] if bundles else []
    except ValueError as error:
        raise ReplayError("Invalid ATLAS weights") from error
    if bundles and len(weights) == 1:
        weights = weights * len(bundles)
    _require(len(weights) == len(bundles) and all(np.isfinite(x) and x >= 0 for x in weights), "ATLAS bundle/weight mismatch")
    sinkhorn = _scalar(_get(args, "sinkhorn", "args"), "sinkhorn", 0, 1)
    _require(args.get("sk_div") in (0, 1), "args: invalid sk_div")
    combine = (sinkhorn > 0 and n >= 18) or bool(bundles) or comp_probability is not None
    coverage, partial = [], False
    if combine:
        mp = events.one("main_probability")
        divisor = sum(abs(value).max() > 0 for value in members.values())
        _require(_integer(_get(mp, "divisor", "main_probability"), "main divisor") == divisor, "main divisor differs")
        lpn = total / max(1, divisor) if args["sk_div"] else total
        p = _softmax_original(lpn)
        checks.compare("main.normalized_logits", _get(mp, "normalized_logits", "main_probability"), lpn, scores=True)
        checks.compare("main.probability", _get(mp, "probability", "main_probability"), p, scores=True)
        if comp_probability is not None:
            p = (1 - comp_w) * p + comp_w * comp_probability
            p /= p.sum(1, keepdims=True)
        checks.compare("main.pre_atlas", _get(events.one("pre_atlas"), "probability", "pre_atlas"), p, scores=True)
        request = events.one("atlas_requested", required=False)
        returned = [value for _, value in events.by_name.get("atlas_returned", [])]
        batches = [value for _, value in events.by_name.get("atlas_batch", [])]
        if bundles and sum(weights) > 0:
            _require(request is not None, "Missing ATLAS request event")
            margin_limit = _scalar(_get(args, "atlas_margin", "args"), "atlas_margin")
            ordered = np.sort(p, 1)
            margin = ordered[:, -1] - ordered[:, -2]
            rows = np.where(margin < margin_limit)[0] if margin_limit > 0 and n >= 50 else np.arange(n)
            requested = _row_indices(_get(request, "rows", "atlas_requested"), n, "ATLAS requested rows")
            _require(np.array_equal(requested, rows), "ATLAS requested rows differ from recomputed routing")
            _require(request.get("bundles") == bundles and request.get("weights") == weights, "ATLAS requested bundles/weights differ")
            _require(_scalar(_get(request, "budget", "atlas_requested"), "atlas budget", 0)
                     == _scalar(_get(args, "atlas_budget", "args"), "args atlas budget", 0), "ATLAS budget differs")
            _require(_integer(_get(request, "probe", "atlas_requested"), "atlas probe")
                     == _integer(_get(args, "atlas_probe", "args"), "args atlas probe"), "ATLAS probe differs")
            checks.compare("atlas.route_margin", _get(request, "margin", "atlas_requested"), margin)
            checks.compare("atlas.request_probability", _get(request, "probability", "atlas_requested"), p, scores=True)
            returned_keys = [value.get("bundle") for value in returned]
            _require(len(returned_keys) == len(set(returned_keys)) and set(returned_keys).issubset(bundles), "Unexpected/duplicate ATLAS return")
            _require(returned_keys == [b for b in bundles if b in returned_keys], "ATLAS return order differs")
            _require(all(value.get("bundle") in bundles for value in batches), "Unexpected ATLAS batch bundle")
            by_bundle = {value["bundle"]: value for value in returned}
            mixed = p.copy()
            for bundle, weight in zip(bundles, weights):
                result = by_bundle.get(bundle)
                if result is None:
                    mask = np.zeros(len(rows), dtype=bool)
                    _require(len(rows) > 0, "Missing ATLAS return even for empty request")
                else:
                    _require(_scalar(_get(result, "weight", "atlas_returned"), "atlas returned weight") == weight,
                             "ATLAS return weight differs")
                    _require(np.array_equal(_row_indices(_get(result, "requested_rows", "atlas_returned"), n,
                                                        "ATLAS return rows"), rows), "ATLAS returned request row order differs")
                    mask = _array(_get(result, "completed_mask", "atlas_returned"), (len(rows),), "ATLAS completed mask", "b")
                    completed = _row_indices(_get(result, "completed_rows", "atlas_returned"), n, "ATLAS completed rows")
                    _require(np.array_equal(completed, rows[mask]), "ATLAS completed row mapping differs")
                    a = _array(_get(result, "probabilities", "atlas_returned"), (len(rows), 18), "ATLAS probabilities")
                    _require(a.dtype == np.float32, "ATLAS returned array lost original float32 boundary")
                    _prob(a[mask], int(mask.sum()), "ATLAS completed probabilities")
                    _require(np.all(a[~mask] == 0), "ATLAS uncompleted placeholders differ; they must not be used as scores")
                    bundle_batches = [b for b in batches if b["bundle"] == bundle]
                    if bundle_batches:
                        assembled, seen = np.zeros_like(a), np.zeros(len(rows), dtype=bool)
                        for batch in bundle_batches:
                            _require(np.array_equal(_row_indices(_get(batch, "requested_rows", "atlas_batch"), n,
                                                                "ATLAS batch request"), rows), "ATLAS batch request differs")
                            local = _row_indices(_get(batch, "local_rows", "atlas_batch"), len(rows), "ATLAS batch local rows")
                            _require(len(local) > 0 and not seen[local].any(), "Duplicate/empty ATLAS batch rows")
                            _require(np.array_equal(_row_indices(_get(batch, "global_rows", "atlas_batch"), n,
                                                                "ATLAS global rows"), rows[local]), "ATLAS batch global mapping differs")
                            labels = _get(batch, "native_labels", "atlas_batch")
                            _require(isinstance(labels, (list, tuple)) and len(labels) == 18 and set(labels) == set(LABELS),
                                     "ATLAS native labels differ")
                            native = _prob(_get(batch, "probability_native", "atlas_batch"), len(local), "ATLAS native probability")
                            assembled[local] = native[:, [labels.index(label) for label in LABELS]]
                            seen[local] = True
                            checks.compare("atlas.batch_mask." + bundle, _get(batch, "completed_mask", "atlas_batch"), seen, exact=True)
                        checks.compare("atlas.batch_completion." + bundle, mask, seen, exact=True)
                        checks.compare("atlas.batch_return." + bundle, a, assembled, scores=True, exact=True)
                    completed = rows[mask]
                    if len(completed):
                        mixed[completed] += weight * (a[mask] - p[completed])
                missing = rows[~mask]
                partial |= len(missing) > 0
                coverage.append({"bundle": bundle, "weight": weight, "requested_rows": int(len(rows)),
                                 "completed_rows": int(mask.sum()), "missing_row_indices": missing.tolist(),
                                 "missing_ids": [identity["ids"][i] for i in missing]})
            p = mixed / mixed.sum(1, keepdims=True)
        else:
            _require(request is None and not returned and not batches, "Unexpected ATLAS events when disabled")
        _prob(p, n, "recomputed pre-Sinkhorn probability")
        checks.compare("main.pre_sinkhorn", _get(events.one("pre_sinkhorn"), "probability", "pre_sinkhorn"), p, scores=True)
        if sinkhorn > 0 and n >= 18:
            sk = events.one("sinkhorn_applied")
            q = p.copy()
            rho = float(np.clip(_scalar(_get(args, "sinkhorn_target_mix", "args"), "target_mix"), 0, 1))
            predicted = np.clip(p.mean(0), 1e-6, None)
            predicted /= predicted.sum()
            prior = np.bincount(y, minlength=18).astype(np.float64) + 1.0
            prior /= prior.sum()
            target = (1 - rho) * predicted + rho * prior
            target = np.clip(target, 1e-6, None)
            target /= target.sum()
            for _ in range(100):
                q *= (len(q) * target[None, :]) / np.clip(q.sum(0, keepdims=True), 1e-12, None)
                q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
            total = (1 - sinkhorn) * np.log(p + 1e-12) + sinkhorn * np.log(q + 1e-12)
            _require(_scalar(_get(sk, "target_mix", "sinkhorn_applied"), "captured target_mix") == rho
                     and _scalar(_get(sk, "weight", "sinkhorn_applied"), "captured sinkhorn weight") == sinkhorn,
                     "Sinkhorn coefficients differ")
            for field, value, score in (("probability", p, True), ("balanced_probability", q, True), ("logits", total, True),
                                        ("predicted_marginal", predicted, False), ("train_prior", prior, False), ("target", target, False)):
                checks.compare("sinkhorn." + field, _get(sk, field, "sinkhorn_applied"), value, scores=score)
        else:
            _require(events.one("sinkhorn_applied", required=False) is None, "Unexpected Sinkhorn event")
            total = np.log(p + 1e-12)
    else:
        _require(not any(name in events.by_name for name in ("main_probability", "pre_atlas", "atlas_requested", "atlas_returned",
                                                            "atlas_batch", "pre_sinkhorn", "sinkhorn_applied")), "Unexpected combine stages")
    base = events.one("postprocessing_base")
    _require(base.get("post") == 0 and base.get("sim") == args.get("sim"), "Postprocessing mode differs")
    checks.compare("main.postprocessing_base", _get(base, "logits", "postprocessing_base"), total, scores=True)
    final = _replay_typography(events.one("typography"), total.copy(), y, checks)
    end = events.one("component_end")
    _require(end.get("typography_applied") is True and isinstance(end.get("output_path"), str) and end["output_path"],
             "main: missing final typography/output completion")
    return list(members), coverage, partial, final.dtype.name


def verify_replay(main_events, comp_events, expected_main_members=None, expected_comp_members=None):
    """Return a JSON-compatible cached-arithmetic audit, or raise ``ReplayError``.

    Partial ATLAS coverage returns ``NEEDS_ADDITIONAL_ATLAS_SCORES`` with
    ``accepted=False``. Missing rows are never filled with zero scores. Inventory
    arguments are declarations of the requested capture, not proof that those
    models ran. Without a comp declaration, the fixed mandatory runtime members
    and eligible group heads are required; optional checkpoints remain unknown.
    This function neither loads weights nor calculates accuracy/validation gain.
    """
    main, comp = _Events(main_events, "main"), _Events(comp_events, "comp")
    main_identity, comp_identity = _identity(main), _identity(comp)
    _require(main_identity == comp_identity, "Main/comp data identity, order or scope differs")
    checks = _Checks()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            serialized, comp_members, gates = _replay_comp(comp, len(main_identity["ids"]), expected_comp_members, checks)
            main_members, atlas, partial, dtype = _replay_main(main, main_identity, serialized, expected_main_members, checks)
    except (FloatingPointError, OverflowError) as error:
        raise ReplayError("Nonfinite or overflowing arithmetic during independent replay") from error
    return {
        "schema_version": 1,
        "status": "NEEDS_ADDITIONAL_ATLAS_SCORES" if partial else "PASS_CACHED_BASELINE_MATH",
        "accepted": not partial,
        "scope": "engineering_cached_math",
        "input_scope": main_identity["scope"], "rows": len(main_identity["ids"]), "labels": list(LABELS),
        "main_members": main_members, "comp_members": comp_members, "family_gates": gates,
        "expected_main_inventory_supplied": expected_main_members is not None,
        "expected_comp_inventory_supplied": expected_comp_members is not None,
        "stage_checks": checks.stages, "maximum_absolute_error": max(x["max_abs_error"] for x in checks.stages),
        "all_checked_argmax_equal": all(x["argmax_equal"] is not False for x in checks.stages),
        "comp_serialization_dtype": "float32", "final_dtype": dtype, "atlas_coverage": atlas,
        "models_loaded": False, "accuracy_scored": False, "real_gpu_execution_verified": False,
        "clean_oof_verified": False, "model_inventory_authenticated": False,
        "limits": [
            "Member probabilities, weighted main LP, grouping and raw ATLAS probabilities are captured inputs; replay does not recompute model inference or authenticate checkpoint identity.",
            "Typography is recomputed from captured counts and binary marker matrices; regex word counts and marker extraction cannot be reconstructed from text hashes alone.",
            "Declared inventories establish expected event membership only, not scientific independence or actual full-stack model execution.",
            "ATLAS coverage applies only to the recomputed original request set. A candidate requesting any additional row needs actual additional scores; absent rows cannot be imputed as zero.",
            "Cache file integrity, source/checkpoint hashes and final persisted submission bytes require separate verification by the caller.",
        ],
    }
