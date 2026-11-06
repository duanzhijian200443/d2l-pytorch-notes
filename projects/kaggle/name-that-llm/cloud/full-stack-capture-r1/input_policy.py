"""Pure CPU/GPU path policy for the fixed EXP-031 retry inputs.

No filesystem, model, network or numerical operation is performed here.
This policy resolves the original optional branches against the complete list
of mounted files. Content hashes and actual attachment versions remain the
caller's responsibility. A 20-member target requires an eligible comp group.
"""
from fnmatch import fnmatchcase
from functools import lru_cache
import hashlib
import json
from pathlib import PurePosixPath
import re


EXPECTED_COMP_MEMBERS = (
    "stylegram", "stylometry_lgb", "glm9b_twin", "gauss_glm9b_L8", "q3_8b_twin",
    "gauss_q3_8b_L22", "q25_7b_twin", "q3_4b_twin", "gauss_q3_4b_L4", "debv3l",
    "debv3l_twin", "gauss_debv3l_L24", "electl", "desk", "lora24_seed7", "lora24_claude",
    "lora24_claude_twin", "lora20_q25_instyle", "tokattn_deb", "ultrabert",
)
ALLOWED_OMISSIONS = ("lora24_public", "lora24_public_twin", "tokattn_q35")
PUBLIC_MARKER = "**/lora24/final/COMPLETE.json"
SOURCE_COMP_SHA256 = "b6b6a7b8adbe50580c2d0de5a41d0764ae752fe729e1471872a9197ac956f222"

MANDATORY_GLOBS = {
    "lora24_seed7": "**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json",
    "lora24_claude": "**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json",
    "lora20_q25_instyle": "**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json",
    "tokattn_deb": "**/tokattn_debl23_final.pt",
    "ultrabert": "**/ultrabert-380m-ntl/fold*/config.json",
    "claude_gate": "**/ntl_extra_passages_claude.csv",
    "backbone_deberta": "**/deberta-v3-large/config.json",
    "backbone_electra": "**/electra-large-discriminator/config.json",
    "backbone_desklib": "**/desklib-detector/config.json",
    "backbone_glm": "**/glm4_9b_trunc25/config.json",
    "backbone_qwen3_4b": "**/qwen-3/transformers/4b-base/1/config.json",
    "backbone_qwen3_8b": "**/qwen-3/transformers/8b-base/1/config.json",
    "backbone_qwen3_14b": "**/qwen-3/transformers/14b-base/1/config.json",
    "setup_qwen3_8b_instruct": "**/qwen-3/transformers/8b/1/config.json",
    "setup_qwen3_14b_instruct": "**/qwen-3/transformers/14b/1/config.json",
    "backbone_qwen25_7b": "**/qwen2.5/transformers/7b/1/config.json",
    "backbone_qwen25_14b": "**/qwen2.5/transformers/14b/1/config.json",
    "backbone_llama": "**/llama31-8b-base-verified/config.json",
    "main_deberta": "**/deberta_v3_large/encoder/config.json",
    "main_deberta_meta": "**/deberta_v3_large/meta.json",
    "main_n14_meta": "**/lora_qwen3_14b/meta.json",
    "main_q25_meta": "**/lora_qwen25_14b/meta.json",
    "setup_n8_meta": "**/lora8b_v1/meta.json",
    "setup_n14_meta": "**/lora14b_v1/meta.json",
    "setup_heads": "**/heads/q3_8bx.npz",
    "atlas_bundle": "**/atlas_bundle.json",
    "matched_public": "**/nlm_matched_clean.parquet",
    "pool_public": "**/nlm_pool_v5.parquet",
    "official_train": "**/name-that-llm/train.csv",
}
# The original cell2 resolves these even when a head is not selected by MODELS.
_HEAD_NAMES = (
    ["ts_q3b8"] + [f"ts_q3b8_f{i}" for i in range(5)]
    + ["ts_q3b14"] + [f"ts_q3b14_f{i}" for i in (0, 2, 4)]
    + ["ts_q25_7b", "ts_llama8"] + [f"ts_llama8_f{i}" for i in (0, 2, 4)]
    + ["ts8_s7"] + [f"ts8_s7_f{i}" for i in (0, 2, 4)]
    + [name + suffix for name in ("q8v5", "q14v5", "ll8v5") for suffix in ("", "_f0", "_f1", "_f2", "_f3", "_f4")]
)
MANDATORY_GLOBS.update({"head_" + name: "**/" + name + "/heads.pt" for name in _HEAD_NAMES})
_LORA_NORMALS = {"lora24_seed7": "norms.safetensors", "lora24_claude": "heads.safetensors",
                 "lora20_q25_instyle": "norms.safetensors"}
_LORA_ANCHORS = {"lora24_seed7": "valid_features.npz", "lora24_claude": "anchor_features.npz",
                 "lora20_q25_instyle": "valid_features.npz"}


class InputPolicyError(ValueError):
    """Input availability differs from the fixed original-branch retry policy."""


def _require(value, message):
    if not value:
        raise InputPolicyError(message)


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as error:
        raise InputPolicyError("Input policy must be finite JSON data") from error


def _path(value):
    _require(isinstance(value, str) and value and not any(ord(c) < 32 for c in value)
             and "\\" not in value and ":" not in value, "Invalid input file path")
    parsed = PurePosixPath(value)
    _require(parsed.parts[:3] == ("/", "kaggle", "input") and len(parsed.parts) > 3
             and not {".", ".."}.intersection(value.split("/")) and parsed.as_posix() == value
             and not value.endswith("/"), "Input file must be a canonical absolute /kaggle/input path")
    return str(value)


def _paths(values):
    _require(not isinstance(values, (str, bytes, dict)), "Input paths must be a sequence, not one path")
    try:
        paths = [_path(value) for value in values]
    except TypeError as error:
        raise InputPolicyError("Input paths must be iterable strings") from error
    _require(bool(paths), "Input file inventory is empty")
    _require(len(set(paths)) == len(paths) and len({value.casefold() for value in paths}) == len(paths),
             "Duplicate or case-colliding input file paths")
    return sorted(paths)


def match_input_glob(path, pattern):
    """Match a relative input glob, with ** spanning zero or more directories."""
    actual = PurePosixPath(_path(path)).parts[3:]
    _require(isinstance(pattern, str) and pattern and not pattern.startswith("/")
             and "\\" not in pattern and ":" not in pattern
             and not {"", ".", ".."}.intersection(pattern.split("/")), "Unsafe input glob")
    wanted = tuple(pattern.split("/"))

    @lru_cache(None)
    def visit(i, j):
        if j == len(wanted):
            return i == len(actual)
        if wanted[j] == "**":
            return visit(i, j + 1) or (i < len(actual) and visit(i + 1, j))
        return i < len(actual) and fnmatchcase(actual[i], wanted[j]) and visit(i + 1, j + 1)

    return visit(0, 0)


def resolve_input_policy(paths):
    """Resolve the complete mounted-file list, without reading any file bytes.

    Public LoRA absence is fixed for this retry. Its unexpected presence would
    change original model composition, so it is rejected rather than silently
    enabled. Call assert_input_policy_equal on a GPU resolution and the CPU
    frozen policy; the complete inventory digest catches all added/lost paths.
    """
    paths = _paths(paths)
    known = set(paths)
    public = [path for path in paths if match_input_glob(path, PUBLIC_MARKER)]
    _require(not public, "Unexpected lora24_public marker changes the fixed20-member policy")
    q35 = [path for path in paths if any(re.search(r"qwen(?:-|_)?3(?:[._-])5", part, re.IGNORECASE)
                                        for part in PurePosixPath(path).parts)]
    _require(not q35, "Unexpected named Q35 input changes the frozen omission policy")
    resolved = {"lora24_public": []}
    mandatory = dict(MANDATORY_GLOBS)
    for key, pattern in mandatory.items():
        matches = [path for path in paths if match_input_glob(path, pattern)]
        if key == "ultrabert":
            _require(len(matches) == 5 and {PurePosixPath(p).parent.name for p in matches} == {f"fold{i}" for i in range(5)}
                     and len({PurePosixPath(p).parent.parent for p in matches}) == 1,
                     "Expected exactly five distinct UltraBERT folds0..4 from one directory")
        else:
            _require(len(matches) == 1, f"Required input missing or ambiguous: {key} ({len(matches)} matches)")
        resolved[key] = matches

    def require_siblings(key, files):
        parent = PurePosixPath(resolved[key][0]).parent
        for filename in files:
            path = (parent / filename).as_posix()
            _require(path in known, "Missing required sibling for " + key + ": " + filename)
            resolved[key + "/" + filename] = [path]

    for key, norm in _LORA_NORMALS.items():
        require_siblings(key, [norm, "readouts.npz", "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                               "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"])
        anchor = (PurePosixPath(resolved[key][0]).parent / _LORA_ANCHORS[key]).as_posix()
        resolved[key + "/optional_anchor"] = [anchor] if anchor in known else []

    for config in resolved["ultrabert"]:
        parent = PurePosixPath(config).parent
        for filename in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            path = (parent / filename).as_posix()
            _require(path in known, "Missing UltraBERT fold sibling: " + path)
            resolved["ultrabert/" + parent.name + "/" + filename] = [path]

    for key in [name for name in mandatory if name.startswith("backbone_") or name == "main_deberta"]:
        parent = PurePosixPath(resolved[key][0]).parent
        weights = [path for path in paths if PurePosixPath(path).parent == parent and path.endswith(".safetensors")]
        _require(bool(weights), "No backbone weight files: " + key)
        resolved[key + "/weights"] = weights
        require_siblings(key, ["tokenizer.json", "tokenizer_config.json"])

    return {
        "schema_version": 1, "policy_id": "exp031-original-optional-inputs-v1", "status": "RESOLVED_PATH_AVAILABILITY",
        "source_comp_sha256": SOURCE_COMP_SHA256,
        "expected_comp_members": list(EXPECTED_COMP_MEMBERS), "allowed_omissions": list(ALLOWED_OMISSIONS),
        "mandatory_globs": mandatory, "resolved_paths": resolved,
        "input_file_count": len(paths), "input_paths_sha256": hashlib.sha256(_canonical(paths)).hexdigest(),
        "omission_reasons": {"lora24_public": "No original marker; original loop skips this checkpoint",
                              "lora24_public_twin": "Parent checkpoint absent; no twin can contribute",
                              "tokattn_q35": "No Q35 backbone among the frozen inputs; the head alone is insufficient"},
        "group_coverage": {"source": "comp G_TE / GROUP_ROWS", "eligible_size_min": 2, "eligible_size_max": 18,
                           "requires_at_least_one_eligible_group": True,
                           "if_unsatisfied": "CONDITIONAL_COVERAGE_UNSATISFIED; do not fabricate groups or remove planned keys"},
        "weights_changed": False,
        "limits": ["This pure resolver checks paths, not file contents, model_type JSON, tensor compatibility or model execution.",
                   "The caller must also verify the CPU manifest content hashes and actual immutable attachment version bindings.",
                   "The GPU must resolve its actual complete file list and compare to the frozen CPU policy before model work."],
    }


def assert_input_policy_equal(actual, frozen):
    """Require exact typed JSON equality with the externally hash-bound policy."""
    for value in (actual, frozen):
        _require(isinstance(value, dict) and value.get("schema_version") == 1
                 and value.get("policy_id") == "exp031-original-optional-inputs-v1"
                 and value.get("status") == "RESOLVED_PATH_AVAILABILITY", "Invalid input policy object")
    _require(_canonical(actual) == _canonical(frozen), "Input availability differs from the frozen CPU policy")
