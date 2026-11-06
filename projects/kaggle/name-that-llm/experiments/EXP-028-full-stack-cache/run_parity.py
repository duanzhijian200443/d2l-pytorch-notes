"""CPU synthetic parity of pinned original and observer-instrumented math.

No model definition, checkpoint loading, feature extraction, classifier fitting,
GPU operation, or remote call is executed. Parquet calls are intercepted as
DataFrame snapshots because this environment has scipy but no parquet engine.
All generated files stay beneath this experiment's fixtures/<case> directory.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
HELPER_PATHS = {"full_stack_cache": ROOT / "src/full_stack_cache.py",
                "full_stack_observer": ROOT / "src/full_stack_observer.py",
                "full_stack_replay": ROOT / "src/full_stack_replay.py",
                "cache_instrument": ROOT / "cloud/full-stack-cache/instrument.py"}
HELPER_BYTES = {name: path.read_bytes() for name, path in HELPER_PATHS.items()}
OPEN_HASHES = {HELPER_PATHS[name]: hashlib.sha256(content).hexdigest() for name, content in HELPER_BYTES.items()}
OPEN_HASHES[Path(__file__).resolve()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Compile the captured bytes themselves: no stale timestamp-based pyc can stand
# in for the helper source whose opening hash is recorded in the fixture.
for _module_name, _source_bytes in HELPER_BYTES.items():
    _module_spec = importlib.util.spec_from_file_location(_module_name, HELPER_PATHS[_module_name])
    _module = importlib.util.module_from_spec(_module_spec)
    sys.modules[_module_name] = _module
    exec(compile(_source_bytes, str(HELPER_PATHS[_module_name]), "exec"), _module.__dict__)
for _helper_path, _opening_hash in OPEN_HASHES.items():
    if hashlib.sha256(_helper_path.read_bytes()).hexdigest() != _opening_hash:
        raise RuntimeError("Helper source changed during frozen import: " + str(_helper_path))
from full_stack_cache import LABELS, read_cache
from full_stack_observer import decode_events
from full_stack_replay import verify_replay

INSTRUMENT = sys.modules["cache_instrument"]
CASES = {"n60_full": (60, False), "n3_full": (3, False), "n60_partial": (60, True)}
REQUIRED_EVENTS = {"main": ["main_members", "postprocessing_base", "typography", "component_end"],
                   "comp": ["member", "comp_before_gates", "component_end"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_helper_hashes():
    for path, digest in OPEN_HASHES.items():
        require(sha(path) == digest, "Helper source changed during parity execution: " + str(path))


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


def capture(node, event):
    return (isinstance(node, ast.Expr) and INSTRUMENT.observer_call(node.value, "capture")
            and node.value.args and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == event)


def singleton(nodes, predicate, description):
    result = [node for node in nodes if predicate(node)]
    require(len(result) == 1, "Expected one source anchor: " + description)
    return result[0]


def select_math(tree, component, observed):
    """Select source AST nodes, without rewriting any mathematical operation."""
    body = tree.body
    setup = []
    if observed:
        setup.append(singleton(body, lambda n: isinstance(n, ast.ImportFrom) and n.module == "full_stack_observer", "observer import"))
        setup.append(singleton(body, lambda n: assigned(n, "_cache"), "from_env initializer"))
    if component == "comp":
        names = {"LABELS", "K", "LAB2I", "FAMILY_IDX", "W", "TWIN_W", "LORA_TWIN_W", "GATE_LAM", "GAUSS_HEADS", "LOGIT"}
        functions = [singleton(body, lambda node, key=key: isinstance(node, ast.FunctionDef) and node.name == key, key)
                     for key in ("add_member", "_np_logsoftmax", "_apply_conditional_family_gate")]
        constants = [node for node in body[:body.index(functions[0])] if any(assigned(node, name) for name in names)]
        require(len(constants) == len(names), "Complementary constants changed")
        tail_start = body.index(functions[-1]) + 1
        prefix = constants + setup + functions
        tail = body[tail_start:]
    else:
        guards = singleton(body, lambda node: assigned(node, "GUARDS"), "GUARDS")
        total = singleton(body, lambda node: assigned(node, "tot"), "LP sum")
        prefix = setup + [guards, total]
        if observed:
            event = singleton(body, lambda node: capture(node, "main_members"), "main_members")
            require(body.index(event) == body.index(total) + 1, "main_members moved away from original LP sum")
            prefix.append(event)
        start = singleton(body, lambda node: assigned(node, "COMP_P"), "COMP_P = None")
        require(isinstance(start.value, ast.Constant) and start.value.value is None, "COMP_P anchor changed")
        tail = body[body.index(start):]
    return ast.Module(body=copy.deepcopy(prefix), type_ignores=[]), ast.Module(body=copy.deepcopy(tail), type_ignores=[])


def strip_equal(original, generated):
    stripped = INSTRUMENT.strip_observer_calls(generated)
    require(INSTRUMENT.ast_identity(original) == INSTRUMENT.ast_identity(stripped), "Selected math differs after observer removal")


def redirect_comp_paths(tree, directory):
    redirects = []

    class Redirect(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/kaggle/working/"):
                relative = node.value.removeprefix("/kaggle/working/")
                require(relative in {"comp_logit.npy", "comp_meta.json"}, "Unexpected source output path")
                target = str(directory / relative)
                redirects.append({"original": node.value, "fixture": target})
                return ast.copy_location(ast.Constant(value=target), node)
            return node

    return Redirect().visit(copy.deepcopy(tree)), redirects


def execute(tree, namespace, path):
    ast.fix_missing_locations(tree)
    code = "# Harness-selected original math; model paths are excluded.\n" + ast.unparse(tree) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(code)
    exec(compile(code, str(path), "exec"), namespace)


def probability(random, n):
    logits = random.normal(0, .12, size=(n, 18))
    rows = np.arange(max(1, n // 3))
    logits[rows, rows % 18] += 7.
    values = np.exp(logits - logits.max(1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def make_inputs(n, case):
    random = np.random.default_rng(28000 + n)
    training = []
    marks = [" — ", "’", '"', "“", "–", ";", "###", "However,", "Overall,", "\n- "]
    for label_index, label in enumerate(LABELS):
        for row in range(4 + label_index % 5):
            text = "Synthetic training passage " + "word " * (30 + row) + marks[label_index % len(marks)] * (row % 3 + 1)
            training.append({"id": f"train-{label_index}-{row}", "prompt_id": f"train-prompt-{row}", "model": label, "text": text})
    train = pd.DataFrame(training)
    texts = ["Synthetic evaluation passage " + "token " * (20 + row % 80) + marks[row % len(marks)] + marks[(row + 3) % len(marks)] for row in range(n)]
    test = pd.DataFrame({"id": [f"{case}-row-{row:03d}" for row in range(n)], "text": texts})
    groups = np.r_[np.zeros(18, dtype=np.int64), np.ones(12, dtype=np.int64), np.arange(2, n - 28)] if n >= 50 else np.array([0, 0, 1])
    require(len(groups) == n, "Synthetic group construction changed")
    sizes = np.bincount(groups)[groups]
    group_mask = (sizes >= 2) & (sizes <= 18)
    style_groups = np.arange(n, dtype=np.int64) // 3
    style_sizes = np.bincount(style_groups)[style_groups]
    style_mask = (style_sizes >= 2) & (style_sizes <= 18)
    specs = [
        {"name": "q14v5", "type": "l24", "w": 3., "twin": .35, "gauss": .70},
        {"name": "q8v5", "type": "l24", "w": 2., "twin": .35, "gauss": .44},
        {"name": "ll8v5", "type": "l24", "w": 1., "twin": .35},
        {"name": "n14b", "type": "l24", "w": 2.5, "twin": .35},
        {"name": "q25_14", "type": "l24", "w": 1., "twin": .35},
        {"name": "deb", "type": "enc", "w": 1.},
    ]
    members = {"tfidf": np.log(probability(random, n)) * .25,
               "svc4": np.log(probability(random, n)) * 1.5,
               "distort": np.log(probability(random, n)) * .10}
    for spec in specs:
        if spec.get("gauss", 0):
            members[spec["name"] + "_gauss"] = np.where(group_mask[:, None], np.log(probability(random, n)), 0.) * spec["gauss"]
        members[spec["name"]] = np.log(probability(random, n)) * spec["w"]
        if spec["type"] == "l24":
            members[spec["name"] + "_twin"] = np.where((sizes >= 2)[:, None], np.log(probability(random, n)), 0.) * spec["w"] * spec["twin"]
    gates = {family: random.normal(0, .3, (n, sum(label.startswith(family) for label in LABELS)))
             for family in ("claude", "gpt", "gemini", "glm")}
    gates["claude_lora"] = np.log(random.dirichlet(np.ones(5), n))
    atlas = probability(random, n).astype(np.float32)
    return {"train": train, "test": test, "y": train.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy(dtype=np.int64),
            "groups": groups, "group_mask": group_mask, "style_groups": style_groups, "style_mask": style_mask,
            "members": members, "specs": specs, "gates": gates, "atlas": atlas}


def comp_plan(namespace, n, mask):
    random = np.random.default_rng(28100 + n)
    plan = []

    def add(key, weight=None, grouped=False):
        scores = probability(random, n)
        plan.append((key, scores[mask].copy() if grouped else scores, mask.copy() if grouped else None, weight))

    add("stylegram"); add("stylometry_lgb")
    for key in ("glm9b", "q3_8b", "q25_7b", "q3_4b", "debv3l", "electl", "desk"):
        if namespace["W"][key] > 0:
            add(key)
        if key in namespace["TWIN_W"]:
            add(key + "_twin", namespace["TWIN_W"][key], True)
        for name, spec in namespace["GAUSS_HEADS"].items():
            if spec["src"] == key:
                add(name, spec["w"], True)
    for key in ("lora24_public", "lora24_seed7", "lora24_claude", "lora20_q25_instyle"):
        add(key)
        if key in namespace["LORA_TWIN_W"]:
            add(key + "_twin", namespace["LORA_TWIN_W"][key], True)
    for key in ("tokattn_q35", "tokattn_deb", "ultrabert"):
        add(key)
    return plan


class FixedClock:
    @staticmethod
    def time():
        return 1000.


def run_variant(case_dir, inputs, selections, observed, partial):
    variant = case_dir / ("instrumented" if observed else "original")
    variant.mkdir()
    comp_dir, main_dir = variant / "comp", variant / "main"
    comp_dir.mkdir(); main_dir.mkdir()
    environment = {"NTL_CACHE_EXPORT": str(case_dir / "cache") if observed else "",
                   "NTL_CACHE_CONTEXT": str(case_dir / "cache-context.json"), "NTL_DROP_Q35": "0", "NTL_FAKE_TEST": ""}
    comp = {"np": np, "pd": pd, "os": os, "json": json, "time": FixedClock, "T0": 1000.,
            "train": inputs["train"].copy(), "test": inputs["test"].copy(), "TE": inputs["test"].text.tolist(),
            "DATA_DIR": case_dir / "inputs", "SMOKE": False, "TINY": len(inputs["test"]) < 50,
            "G_TE": inputs["groups"].copy(), "GROUP_ROWS": inputs["group_mask"].copy(),
            "G_style": inputs["style_groups"].copy(), "rows_style": inputs["style_mask"].copy(),
            "GATES": {key: value.copy() for key, value in inputs["gates"].items()}}
    frames, atlas_calls = [], []
    with patch.dict(os.environ, environment), (variant / "execution.log").open("x", encoding="utf-8") as log_stream, contextlib.redirect_stdout(log_stream):
        execute(copy.deepcopy(selections["comp"][0]), comp, comp_dir / "prefix.py")
        plan = comp_plan(comp, len(inputs["test"]), inputs["group_mask"])
        for key, value, rows, weight in plan:
            comp["add_member"](key, value, rows=rows, weight=weight)
        pre_gate = comp["LOGIT"].copy()
        tail, redirects = redirect_comp_paths(selections["comp"][1], comp_dir)
        execute(tail, comp, comp_dir / "tail.py")
        args = SimpleNamespace(data=str(case_dir / "inputs"), sim="", fake_test=0, perpassage=1, post=0,
            models=json.dumps(inputs["specs"]), comp_path=str(comp_dir / "comp_logit.npy"), comp_w=.15,
            atlas_dir="synthetic-atlas-bundle", atlas_base="SYNTHETIC_NO_MODEL", atlas_w="0.20", atlas_margin=.3,
            atlas_budget=3600., atlas_probe=0, atlas_bs=4, atlas_chunk=3,
            sinkhorn=.45, sinkhorn_target_mix=.35, sk_div=1, tfidf_w=.25, distort_w=.10, svc_w=1.5,
            adapt_w=1., adapt_thr=.15, extra_w=.5, out=str(main_dir / "submission.parquet"),
            cache=str(main_dir), save_lp=str(main_dir / "normalized_lp.npy"))

        def atlas_scores(bundle, rows, budget):
            require(bundle == args.atlas_dir and budget == args.atlas_budget, "Unexpected synthetic ATLAS request")
            mask = np.ones(len(rows), dtype=bool)
            if partial:
                mask[1::2] = False
            output = np.zeros((len(rows), 18), dtype=np.float32)
            output[mask] = inputs["atlas"][rows[mask]]
            atlas_calls.append({"bundle": bundle, "rows": rows.copy(), "mask": mask.copy(), "probabilities": output.copy()})
            return output, mask

        def capture_parquet(frame, path, *positional, **kwargs):
            require(not positional and kwargs == {"index": False}, "Unexpected to_parquet signature")
            target = Path(path).resolve()
            require(target.is_relative_to(main_dir.resolve()) and target.name == "submission.parquet", "Output escaped fixture")
            frames.append(frame.copy(deep=True))
            # Deliberately do not write a .parquet file or claim Parquet I/O coverage.

        main = {"np": np, "pd": pd, "os": os, "json": json, "time": FixedClock, "T0": 1000.,
                "a": args, "LABELS": list(LABELS), "L2I": {label: i for i, label in enumerate(LABELS)},
                "train": inputs["train"].copy(), "test": inputs["test"].copy(), "y": inputs["y"].copy(),
                "Gte": inputs["groups"].copy(), "LP": {key: value.copy() for key, value in inputs["members"].items()},
                "specs": copy.deepcopy(inputs["specs"]), "atlas_scores": atlas_scores,
                "log": lambda *values: print(*values), "gpu_mem": lambda: "CPU synthetic stub; no GPU inspected",
                "linear_sum_assignment": linear_sum_assignment, "re": __import__("re")}
        with patch.object(pd.DataFrame, "to_parquet", capture_parquet):
            execute(copy.deepcopy(selections["main"][0]), main, main_dir / "prefix.py")
            execute(copy.deepcopy(selections["main"][1]), main, main_dir / "tail.py")
    require(len(frames) == 2, "Expected pre-typography and final source to_parquet calls")
    for index, frame in enumerate(frames, 1):
        frame.to_csv(main_dir / f"parquet-call-{index:02d}.csv", index=False)
    arrays = {"comp_pre_gate": pre_gate, "comp_post_gate": comp["LOGIT"],
              "comp_serialized": np.load(comp_dir / "comp_logit.npy", allow_pickle=False),
              "main_total_before_hard": main["_v38_before_hard"], "main_final_logits": main["_safe_logits"],
              "main_final_argmax": main["_safe_logits"].argmax(1), "main_pre_typography_argmax": main["pred"],
              "main_post_atlas_probability": main["P"], "main_balanced_probability": main["Q"],
              "main_typography_evidence": main["_evidence"], "main_typography_gate": main["_gate"],
              "main_normalized_lp": np.load(main_dir / "normalized_lp.npy", allow_pickle=False)}
    for name, value in arrays.items():
        require(np.isfinite(value).all(), "Nonfinite synthetic result: " + name)
    np.savez_compressed(variant / "math-output.npz", **arrays)
    return {"arrays": arrays, "frames": frames, "atlas_calls": atlas_calls, "redirects": redirects,
            "comp_members": [item[0] for item in plan], "main_members": list(inputs["members"]),
            "sinkhorn_iterations": 100 if main.get("_") == 99 else 0}


def run_case(case, source_trees, generated_trees, receipts, run_tag):
    check_helper_hashes()
    n, partial = CASES[case]
    case_dir = HERE / "fixtures" / (case + "-" + run_tag)
    require(not case_dir.exists(), "Refusing an existing parity fixture: " + str(case_dir))
    case_dir.mkdir(parents=True)
    (case_dir / "inputs").mkdir()
    inputs = make_inputs(n, case)
    inputs["train"].to_csv(case_dir / "inputs/train.csv", index=False)
    inputs["test"].to_csv(case_dir / "inputs/test.csv", index=False)
    save_json(case_dir / "cache-context.json", {"schema_version": 1, "scope": "synthetic_math", "run_id": "EXP-028-" + case + "-" + run_tag,
        "required_events": REQUIRED_EVENTS, "models_executed": False, "checkpoint_identity_verified": False,
        "fixture_policy": "Deterministic synthetic model outputs; actual extracted source math only"})
    originals, generated = {}, {}
    for component in ("comp", "main"):
        originals[component] = select_math(source_trees[component], component, False)
        generated[component] = select_math(generated_trees[component], component, True)
        for left, right in zip(originals[component], generated[component]):
            strip_equal(left, right)
    original = run_variant(case_dir, inputs, originals, False, partial)
    instrumented = run_variant(case_dir, inputs, generated, True, partial)
    parity = []
    for name, left in original["arrays"].items():
        right = instrumented["arrays"][name]
        require(left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(), "Numerical parity failed: " + name)
        parity.append({"array": name, "shape": list(left.shape), "dtype": left.dtype.str, "bitwise_equal": True,
                       "maximum_absolute_error": 0.})
    for left, right in zip(original["frames"], instrumented["frames"]):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    require(len(original["atlas_calls"]) == len(instrumented["atlas_calls"]) == 1, "ATLAS call count changed")
    left, right = original["atlas_calls"][0], instrumented["atlas_calls"][0]
    for key in ("rows", "mask", "probabilities"):
        require(np.array_equal(left[key], right[key]), "ATLAS stub boundary differs: " + key)
    requested = len(right["rows"])
    require((0 < requested < n) if n >= 50 else requested == n, "ATLAS margin/short-panel branch was not exercised")
    require(original["sinkhorn_iterations"] == instrumented["sinkhorn_iterations"] == (100 if n >= 18 else 0), "Sinkhorn branch was not exercised")
    require(bool(np.any(instrumented["arrays"]["main_typography_evidence"] != 0)), "Typography was degenerate")
    ids, texts = inputs["test"].id.tolist(), inputs["test"].text.tolist()
    snapshots = {component: read_cache(case_dir / "cache", component, ids, texts, LABELS) for component in ("main", "comp")}
    events = {component: decode_events(snapshot) for component, snapshot in snapshots.items()}
    replay = verify_replay(events["main"], events["comp"], expected_main_members=instrumented["main_members"],
                           expected_comp_members=instrumented["comp_members"])
    expected_status = "NEEDS_ADDITIONAL_ATLAS_SCORES" if partial else "PASS_CACHED_BASELINE_MATH"
    require(replay["status"] == expected_status and replay["accepted"] is (not partial), "Replay coverage result differs")
    check_helper_hashes()
    save_json(case_dir / "replay-review.json", replay)
    provenance = {}
    for component in ("main", "comp"):
        selected = [*originals[component][0].body, *originals[component][1].body]
        provenance[component] = {"source_sha256": receipts[component]["source_sha256"],
            "generated_sha256": receipts[component]["generated_sha256"],
            "selected_original_nodes": [{"node": type(node).__name__, "name": getattr(node, "name", None),
                "line": node.lineno, "end_line": node.end_lineno} for node in selected],
            "selected_algorithm_ast_equal_after_removing_observers": True}
    report = {"schema_version": 1, "status": "PASS_SYNTHETIC_SOURCE_PARITY", "experiment_id": "EXP-028", "case": case,
        "scope": "synthetic_math", "rows": n, "labels": list(LABELS), "fixtures_only": True,
        "harness_sha256": OPEN_HASHES[Path(__file__).resolve()], "instrument_sha256": OPEN_HASHES[HELPER_PATHS["cache_instrument"]],
        "observer_sha256": OPEN_HASHES[HELPER_PATHS["full_stack_observer"]], "cache_writer_sha256": OPEN_HASHES[HELPER_PATHS["full_stack_cache"]],
        "replayer_sha256": OPEN_HASHES[HELPER_PATHS["full_stack_replay"]], "source_execution": provenance,
        "helper_open_and_close_hashes_equal": True,
        "helper_import_policy": "Execute exactly the bytes captured and hashed before import; verify unchanged after import and at case close",
        "main_members": instrumented["main_members"], "comp_members": instrumented["comp_members"],
        "array_parity": parity, "source_output_table_calls_verified": 2, "final_argmax_equal": True,
        "real_parquet_io_verified": False, "parquet_policy": "Controlled DataFrame.to_parquet snapshots compared exactly and exported as CSV; no Parquet file claimed",
        "atlas": {"requested_rows": right["rows"].tolist(), "completed_rows": right["rows"][right["mask"]].tolist(),
                  "probability_dtype": str(right["probabilities"].dtype), "partial": partial,
                  "request_branch": "top-two margin < .3 with N>=50" if n >= 50 else "all rows because N<50"},
        "sinkhorn_iterations_executed": instrumented["sinkhorn_iterations"], "typography_executed": True,
        "replay_status": replay["status"], "replay_accepted": replay["accepted"],
        "manifest_sha256": {component: snapshot.manifest_sha256 for component, snapshot in snapshots.items()},
        "event_names": {component: [name for name, _ in values] for component, values in events.items()},
        "harness_only_path_redirects": {"original": original["redirects"], "instrumented": instrumented["redirects"]},
        "stubs": ["All component/classifier probabilities, weighted main LP arrays, group assignments and family-gate scores are deterministic synthetic inputs.",
                  "atlas_scores returns fixed synthetic float32 probabilities and a predeclared completion mask; its heavy source function definition/body is excluded.",
                  "time.time is a fixed fixture clock; log and gpu_mem report CPU-stub messages. Real wall-clock/ATLAS-budget effects are untested.",
                  "DataFrame.to_parquet is intercepted; both original calls and final table values are checked, not the Parquet storage engine."],
        "limits": ["No checkpoint, model definition, feature extraction, classifier fit, GPU work, network request, or accuracy scoring occurred.",
                   "Selected source AST extraction and /kaggle/working output redirection are harness-only and applied symmetrically; original/instrument modules are not edited.",
                   "Original diagnostic JSON retains its source wording; this enclosing synthetic_math receipt controls the fixture's scope.",
                   "COMMITTED component caches establish byte integrity only; this is not full real-model or clean-OOF evidence."],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    save_json(case_dir / "fixture.json", report)
    check_helper_hashes()
    return {"case": case, "status": report["status"], "fixture": str(case_dir.relative_to(ROOT)),
            "rows": n, "requested_atlas": requested, "completed_atlas": int(right["mask"].sum()),
            "sinkhorn_iterations": report["sinkhorn_iterations_executed"], "replay_status": replay["status"],
            "fixture_sha256": sha(case_dir / "fixture.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--run-tag", required=True, help="A new run suffix; existing fixture directories are never overwritten")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_tag) is not None, "Unsafe run tag")
    check_helper_hashes()
    source_trees, generated_trees, receipts = {}, {}, {}
    original_hashes = {}
    for component, (filename, _) in INSTRUMENT.EXPECTED.items():
        path = INSTRUMENT.SOURCE_DIR / filename
        content = path.read_bytes(); original_hashes[path] = hashlib.sha256(content).hexdigest()
        generated, receipts[component] = INSTRUMENT.instrument_source(content, component)
        source_trees[component] = ast.parse(content, filename=filename)
        generated_trees[component] = ast.parse(generated, filename="instrumented-" + filename)
    require("torch" not in sys.modules, "A heavy model runtime was unexpectedly imported")
    results = [run_case(case, source_trees, generated_trees, receipts, args.run_tag) for case in (CASES if args.case == "all" else [args.case])]
    for path, digest in original_hashes.items():
        require(sha(path) == digest, "Original source changed during parity execution")
    require("torch" not in sys.modules, "A heavy model runtime was unexpectedly imported")
    check_helper_hashes()
    print(json.dumps({"status": "PASS_SYNTHETIC_SOURCE_PARITY", "cases": results,
                      "models_loaded": False, "real_parquet_io_verified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
