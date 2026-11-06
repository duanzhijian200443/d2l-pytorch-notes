"""Exercise the generated execution cell locally with frozen attachment files; no heavy fits."""
from pathlib import Path
from unittest.mock import patch
import fnmatch
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

import generate as gen

FILES=json.loads((gen.ROOT/"experiments/EXP-053-apex-stylegram-volume/inputs/input-receipt.json").read_text(encoding="utf-8"))["loader_files"]
OBSERVED={}

def attachment_glob(pattern, recursive=True):
    return [str(gen.ROOT/r["path"]) for r in FILES if fnmatch.fnmatchcase(r["name"],pattern.split("/")[-1])]

def run_cell(arm, directory, fast, on_run):
    kernel,nb,_,_,body=gen.generate(arm)
    del kernel
    cell=json.loads(nb)["cells"][14]["source"]
    # Redirect only the Kaggle working directory during the local simulation.
    cell=gen.replace_once(cell,"_v_pathlib.Path('/kaggle/working')",f"_v_pathlib.Path({str(directory)!r})")
    (directory/"comp_stack.py").write_bytes(body)
    with patch("glob.glob",side_effect=attachment_glob), patch("subprocess.run",side_effect=on_run):
        exec(compile(cell,"<actual generated execution cell>","exec"),{"FAST_COMMIT":fast,"DATA":str(gen.ROOT)})

class GeneratedTests(unittest.TestCase):
    def test_materialized_public_inputs_both_arms(self):
        def unexpected(*args,**kwargs):
            self.fail("Public path launched a training subprocess")
        for arm,(rows,_,csv_sha) in gen.ARM_CONFIG.items():
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                run_cell(arm,directory,True,unexpected)
                pre=json.loads((directory/"volume_preflight.json").read_text())
                self.assertEqual((pre["status"],pre["rows"],pre["groups"],pre["fit_count"]),("PASS",rows,rows//4,0))
                self.assertEqual((pre["shared_rows"],pre["shared_groups"],pre["shared_sha256"]),(10743,738,gen.SHARED_SHA))
                self.assertEqual(pre["source_csv_sha256"],csv_sha)
                check=gen.verify_materialized(arm,directory/"comp_stack.py")
                self.assertEqual(check["comp_stack_sha256"],pre["comp_stack_sha256"])
                self.assertEqual(check["arm"],arm)
                OBSERVED[arm]={"public_cell14":pre,"materialized_module":check}

    def test_hidden_rejects_failed_subprocess_and_missing_receipt(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                def simulate(cmd, **kwargs):
                    self.assertTrue(kwargs["check"])
                    if not missing:
                        raise subprocess.CalledProcessError(23,cmd)
                    return subprocess.CompletedProcess(cmd,0)
                with self.assertRaises(FileNotFoundError if missing else subprocess.CalledProcessError):
                    run_cell("four120",directory,False,simulate)
                self.assertFalse((directory/"volume_comp_execution.json").exists())

    def test_corrupted_materialized_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"comp_stack.py"
            p.write_bytes(gen.generate("four120")[-1]+b"# unexpected mutation")
            with self.assertRaisesRegex(ValueError,"Materialized"):
                gen.verify_materialized("four120",p)

    def test_hidden_binds_nonce_and_rejects_invalid_seconds(self):
        labels=sorted(pd.read_csv(gen.ROOT/"train.csv").model.unique())
        ids=pd.read_csv(gen.ROOT/"test.csv").id.astype(str).tolist()
        for seconds in (2.0, "not-a-number"):
            with self.subTest(seconds=seconds),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                def simulate(cmd,**kwargs):
                    self.assertTrue(kwargs["check"])
                    receipt=json.loads((directory/"volume_preflight.json").read_text())
                    receipt.update(status="MEMBER_ADDED",run_nonce=kwargs["env"]["NTL_VOLUME_RUN_NONCE"],
                                   fit_count=2,test_rows=len(ids),output_classes=18,member_weight=1.516,
                                   official_rows=2699,full_training_rows=2699+10743+120)
                    (directory/"stylegram_volume_receipt.json").write_text(json.dumps(receipt))
                    (directory/"comp_meta.json").write_text(json.dumps({"ids":ids,"labels":labels,"seconds":seconds,
                                                                       "family_gates_applied":True}))
                    np.save(directory/"comp_logit.npy",np.zeros((len(ids),18),np.float32))
                    return subprocess.CompletedProcess(cmd,0)
                if isinstance(seconds,str):
                    with self.assertRaises(AssertionError):
                        run_cell("four120",directory,False,simulate)
                else:
                    run_cell("four120",directory,False,simulate)
                    self.assertEqual(json.loads((directory/"volume_comp_execution.json").read_text())["status"],"PASS")

if __name__=="__main__":
    if sys.argv[1:] == ["--record"]:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeneratedTests)
        result=unittest.TextTestRunner().run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        report={"status":"PASS_LOCAL_PREFLIGHT","test_count":result.testsRun,"observed":OBSERVED,
                "test_script_sha256":gen.sha(Path(__file__).read_bytes()),
                "manifest_sha256":gen.sha((gen.HERE/"manifest.json").read_bytes()),
                "runtime_scope":"Generated public cell and archived local attachments. Hidden subprocess outcomes simulated for guard checks. No formal model fit."}
        (gen.HERE/"local-preflight.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    else:
        unittest.main()
