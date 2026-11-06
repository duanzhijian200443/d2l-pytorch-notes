"""EXP046 mounted-input preflight. No installation, model loading or training.

Only successful checks write a receipt. Large frozen noncompetition files are
checked for exact size, not rehashed; immutable attachment versions plus the
historical full hash inventory remain their provenance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

BASE_MANIFEST_SHA256 = "de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5"
EXTRA_MANIFEST_SHA256 = "b7bf1347c6185c14d2480975873989f58115c02de65dc8bbd12ab09697faa687"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SMALL_LIMIT = 16 * 1024**2
EXTRA_DIRS = ("datasets/zhjianduan/ntl-stylegram-cv-inputs-v1", "ntl-stylegram-cv-inputs-v1")
DATA_DIRS = ("competitions/name-that-llm", "name-that-llm")
EXTRA_NAMES = {"extra.csv", "folds.csv", "protocol.json", "runner.py", "stylegram_core.py"}
COMP_PREFIX = "/kaggle/input/competitions/name-that-llm/"
EXPECTED_COUNTS = {"baseline_noncompetition_files":528,"baseline_small_files_hashed":398,
                   "baseline_large_files_size_only":130,"extra_files_hashed":5}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def relative_file(value):
    require(isinstance(value,str) and value and "\\" not in value and ":" not in value, "Unsafe relative input path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and not {".",".."}.intersection(path.parts), "Unsafe input traversal")
    return value

def validate_spec(entry):
    require(set(entry) == {"path","bytes","sha256"}, "Unexpected input specification")
    require(type(entry["bytes"]) is int and entry["bytes"] >= 0, "Invalid declared file size")
    digest = entry["sha256"]
    require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest), "Invalid input digest")

def check_file(path, entry, full_hash):
    require(path.is_file(), "Missing input file: " + str(path))
    require(path.stat().st_size == entry["bytes"], "Input size mismatch: " + str(path))
    if full_hash:
        require(sha(path) == entry["sha256"], "Input SHA256 mismatch: " + str(path))

def unique_directory(input_root, candidates, role):
    present = [input_root / relative_file(name) for name in candidates if (input_root / name).exists()]
    require(len(present)==1 and present[0].is_dir(), role + " directory missing or ambiguous")
    return present[0]

def validate_test(path):
    """Hidden row count/bytes/content are deliberately not compared to old samples."""
    require(path.is_file(), "Official test.csv missing")
    ids = set()
    with path.open("r",encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["id","text"], "Official test.csv must have exactly id,text columns")
        for row in reader:
            require(set(row)=={"id","text"} and all(isinstance(row[k],str) and row[k].strip() for k in ("id","text")),
                    "Unreadable or empty official test row")
            require(row["id"] not in ids, "Duplicate official test ID")
            ids.add(row["id"])
    require(bool(ids), "Official test.csv is empty")
    return len(ids)

def validate_inputs(manifest, extra_manifest, input_root, small_limit=SMALL_LIMIT, progress=None):
    """Pure validation core; small synthetic inventories are supported for tests."""
    input_root = Path(input_root)
    require(input_root.is_dir(), "Input mount root missing")
    require(type(small_limit) is int and small_limit >= 0, "Invalid small-file threshold")
    files = manifest.get("files")
    require(isinstance(files,list) and files, "Empty baseline inventory")
    require(manifest.get("status")=="PASS", "Baseline manifest was not accepted")
    seen = set()
    noncompetition = []
    competition = {}
    for entry in files:
        validate_spec(entry)
        name = entry["path"]
        require(name.startswith("/kaggle/input/"), "Unexpected manifest mount root")
        relative_file(name.removeprefix("/kaggle/input/"))
        require(name not in seen, "Duplicate baseline inventory path")
        seen.add(name)
        if name.startswith(COMP_PREFIX):
            short = name.removeprefix(COMP_PREFIX)
            require(short in {"train.csv","test.csv","sample_submission.parquet"}, "Unexpected historical competition file")
            competition[short] = entry
        else:
            noncompetition.append(entry)
    require(set(competition)=={"train.csv","test.csv","sample_submission.parquet"}, "Historical official file inventory changed")
    small_count = large_count = small_bytes = large_bytes = 0
    if progress:
        progress("Checking frozen baseline paths and sizes; hashing only files <=16MiB.")
    for index, entry in enumerate(noncompetition,1):
        path = input_root / entry["path"].removeprefix("/kaggle/input/")
        full_hash = entry["bytes"] <= small_limit
        check_file(path,entry,full_hash)
        if full_hash:
            small_count += 1
            small_bytes += entry["bytes"]
        else:
            large_count += 1
            large_bytes += entry["bytes"]
        if progress and index % 100 == 0:
            progress(f"Checked {index}/{len(noncompetition)} baseline files; large weights are size-only.")
    data_dir = unique_directory(input_root,DATA_DIRS,"Official competition")
    check_file(data_dir/"train.csv",competition["train.csv"],True)
    test_rows = validate_test(data_dir/"test.csv")
    require(extra_manifest.get("dataset_ref")=="zhjianduan/ntl-stylegram-cv-inputs-v1"
            and extra_manifest.get("dataset_version")==1, "Wrong added dataset/version")
    require(extra_manifest.get("candidate_relative_directories")==list(EXTRA_DIRS), "Extra mount candidate paths changed")
    extra_entries = extra_manifest.get("files")
    require(isinstance(extra_entries,list) and len(extra_entries)==5
            and {e.get("path") for e in extra_entries}==EXTRA_NAMES, "Added dataset file set differs")
    extra_dir = unique_directory(input_root,EXTRA_DIRS,"A1 Stylegram")
    require({p.name for p in extra_dir.iterdir()} == EXTRA_NAMES, "Unexpected file/directory in added dataset")
    for entry in extra_entries:
        validate_spec(entry)
        require(PurePosixPath(entry["path"]).name == entry["path"], "Added dataset must contain five flat files")
        check_file(extra_dir/entry["path"],entry,True)
    return {"status":"PASS_EXP046_INPUT_PREFLIGHT",
            "data_dir":str(data_dir),"extra_dir":str(extra_dir),"extra_path":str(extra_dir/"extra.csv"),
            "stylegram_core_path":str(extra_dir/"stylegram_core.py"),"test_rows":test_rows,
            "counts":{"baseline_noncompetition_files":len(noncompetition),"baseline_small_files_hashed":small_count,
                      "baseline_large_files_size_only":large_count,"extra_files_hashed":5},
            "bytes":{"baseline_small_hashed":small_bytes,"baseline_large_size_only":large_bytes},
            "official_train_sha256":competition["train.csv"]["sha256"],"small_hash_threshold_bytes":small_limit,
            "dynamic_competition_policy":{"test":"Only id,text schema and nonempty unique IDs/text checked; no old row count, size or hash imposed.",
                                          "sample_submission":"Historical public sample size/hash deliberately ignored; no placeholder generated."},
            "limits":["Large weights were NOT rehashed in this run. Only exact sizes checked, with fixed attachment versions and historical full-hash provenance.",
                      "No additional filetree recursion or model loading; unrelated files outside fixed paths are not asserted absent.",
                      "Input compatibility is not a successful hidden inference, submission, or leaderboard result."],
            "training_started":False,"installation_performed":False}

def run_preflight(manifest_path, extra_manifest_path, receipt_path=None,
                  input_root=Path("/kaggle/input")):
    """Validate frozen manifests/mounts and return exact paths for Notebook setup."""
    manifest_path, extra_manifest_path = Path(manifest_path),Path(extra_manifest_path)
    require(sha(manifest_path)==BASE_MANIFEST_SHA256, "Frozen531 manifest SHA256 mismatch")
    require(sha(extra_manifest_path)==EXTRA_MANIFEST_SHA256, "Added dataset manifest SHA256 mismatch")
    manifest, extra = read_json(manifest_path),read_json(extra_manifest_path)
    require(len(manifest["files"])==531, "Frozen baseline inventory count")
    train = next(e for e in manifest["files"] if e["path"]==COMP_PREFIX+"train.csv")
    require(train["sha256"]==TRAIN_SHA256, "Official training identity changed")
    started = time.monotonic()
    result = validate_inputs(manifest,extra,input_root,
                            progress=lambda message:print("INPUT_PREFLIGHT "+message,flush=True))
    require(result["counts"]==EXPECTED_COUNTS, "Frozen inventory category counts changed")
    require(sha(manifest_path)==BASE_MANIFEST_SHA256 and sha(extra_manifest_path)==EXTRA_MANIFEST_SHA256,
            "Input manifest changed during preflight")
    result.update(manifest_sha256=BASE_MANIFEST_SHA256,extra_manifest_sha256=EXTRA_MANIFEST_SHA256,
                  verifier_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,
                  timestamp_unix=time.time())
    if receipt_path is not None:
        destination = Path(receipt_path)
        require(not destination.resolve().is_relative_to(Path(input_root).resolve()), "Cannot write receipt into input mount")
        require(not destination.exists(), "Refusing to overwrite an existing preflight receipt")
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    print(f"INPUT_PREFLIGHT PASS: {result['test_rows']} test rows; no old hidden-test size/hash constraint.",flush=True)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--extra-manifest",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,default=Path("/kaggle/working/input_preflight.json"))
    parser.add_argument("--input-root",type=Path,default=Path("/kaggle/input"))
    args = parser.parse_args(argv)
    result = run_preflight(args.manifest,args.extra_manifest,args.receipt,args.input_root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__ == "__main__":
    main()
