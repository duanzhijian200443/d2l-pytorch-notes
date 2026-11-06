"""Local fixtures only: never mount or read actual model files."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("exp046_preflight",Path(__file__).with_name("input_preflight.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

class InputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="exp046-input-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/"input"
        self.root.mkdir()
        self.entries = []
        def add(relative,data):
            p = self.root/relative
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(data)
            e = {"path":"/kaggle/input/"+relative,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
            self.entries.append(e)
            return e
        add("datasets/owner/model/config.json",b'{"model_type":"fixture"}')
        add("datasets/owner/model/weights.bin",b"fixed-large-fixture")
        add("competitions/name-that-llm/train.csv",b"id,model,text\ntr1,fixture,official\n")
        add("competitions/name-that-llm/test.csv",b"id,text\npub1,public sample\n")
        add("competitions/name-that-llm/sample_submission.parquet",b"public-sample")
        self.manifest={"status":"PASS","files":self.entries}
        self.extra={"dataset_ref":"zhjianduan/ntl-stylegram-cv-inputs-v1","dataset_version":1,
                    "candidate_relative_directories":list(M.EXTRA_DIRS),"files":[]}
        self.extra_dir=self.root/M.EXTRA_DIRS[0]
        self.extra_dir.mkdir(parents=True)
        for name in sorted(M.EXTRA_NAMES):
            data=("fixture-"+name).encode()
            (self.extra_dir/name).write_bytes(data)
            self.extra["files"].append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    def check(self,**kwargs):
        return M.validate_inputs(self.manifest,self.extra,self.root,**kwargs)
    def test_hidden_4000_rows_ignores_public_sizes_hashes(self):
        p=self.root/"competitions/name-that-llm/test.csv"
        with p.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["id","text"])
            w.writerows((f"hidden{i}","real hidden response\nsecond line") for i in range(4000))
        (p.parent/"sample_submission.parquet").write_bytes(b"changed arbitrarily and intentionally")
        r=self.check()
        self.assertEqual(r["test_rows"],4000)
        self.assertEqual(r["extra_path"],str(self.extra_dir/"extra.csv"))
    def test_train_same_size_tamper_rejected(self):
        p=self.root/"competitions/name-that-llm/train.csv"
        p.write_bytes(p.read_bytes().replace(b"official",b"tampered"))
        with self.assertRaisesRegex(ValueError,"SHA256"):self.check()
    def test_small_file_tamper_rejected(self):
        p=self.root/"datasets/owner/model/config.json";b=p.read_bytes()
        p.write_bytes(b[:-1]+b" ")
        with self.assertRaisesRegex(ValueError,"SHA256"):self.check()
    def test_extra_same_size_tamper_rejected(self):
        p=self.extra_dir/"extra.csv";b=p.read_bytes();p.write_bytes(b"X"+b[1:])
        with self.assertRaisesRegex(ValueError,"SHA256"):self.check()
    def test_duplicate_exact_extra_mount_rejected(self):
        shutil.copytree(self.extra_dir,self.root/M.EXTRA_DIRS[1])
        with self.assertRaisesRegex(ValueError,"ambiguous"):self.check()
    def test_short_extra_mount_supported_and_unrelated_extra_ignored(self):
        short=self.root/M.EXTRA_DIRS[1];self.extra_dir.rename(short)
        unrelated=self.root/"datasets/someone/other/extra.csv";unrelated.parent.mkdir(parents=True);unrelated.write_text("unrelated")
        self.assertEqual(self.check()["extra_path"],str(short/"extra.csv"))
    def test_extra_unknown_file_rejected(self):
        (self.extra_dir/"train.csv").write_text("not permitted")
        with self.assertRaisesRegex(ValueError,"Unexpected"):self.check()
    def test_hidden_duplicate_or_blank_rejected(self):
        p=self.root/"competitions/name-that-llm/test.csv"
        for content in ["id,text\na,one\na,two\n","id,text\na, \n","id,text\n,one\n","id,text\n","id,text,label\na,one,X\n"]:
            p.write_text(content)
            with self.assertRaises(ValueError):self.check()
    def test_large_size_only_is_explicit_and_wrong_size_rejected(self):
        p=self.root/"datasets/owner/model/weights.bin"
        original=p.read_bytes();p.write_bytes(b"Z"*len(original))
        r=self.check(small_limit=0)
        self.assertEqual(r["counts"]["baseline_large_files_size_only"],2)
        p.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError,"size"):self.check(small_limit=0)
    def test_duplicate_or_traversal_spec_rejected(self):
        self.manifest["files"].append(copy.deepcopy(self.entries[0]))
        with self.assertRaisesRegex(ValueError,"Duplicate"):self.check()
        self.manifest["files"].pop()
        self.entries[0]["path"]="/kaggle/input/../outside"
        with self.assertRaises(ValueError):self.check()
    def test_wrapper_rejects_unfrozen_manifest_without_receipt(self):
        m=Path(self.tmp.name)/"manifest.json";e=Path(self.tmp.name)/"extra.json";out=Path(self.tmp.name)/"receipt.json"
        m.write_text(json.dumps(self.manifest));e.write_text(json.dumps(self.extra))
        with self.assertRaisesRegex(ValueError,"manifest SHA256"):
            M.run_preflight(m,e,out,self.root)
        self.assertFalse(out.exists())

if __name__=="__main__":
    unittest.main()
