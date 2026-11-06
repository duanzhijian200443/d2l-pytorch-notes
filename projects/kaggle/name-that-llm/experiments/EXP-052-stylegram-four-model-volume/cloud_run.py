"""Prepare or operate the one registered private Kaggle CPU comparison."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

EX = Path(__file__).resolve().parent
ROOT = EX.parents[1]
CLOUD = ROOT / "cloud/stylegram-volume-052"
REFERENCE = ROOT / "cloud/luna-terra120-042/payload"
REF = "zhjianduan/name-that-llm-stylegram-volume-052"
STATE = CLOUD / "execution-state.json"
KERNEL = CLOUD / "kernel"
RUN_ID = "exp052-stylegram-four-volume-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


BOOTSTRAP = """import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
directory = pathlib.Path('/tmp/exp052-fixed-inputs')
directory.mkdir(exist_ok=False)
raw = base64.b64decode(ENCODED_PAYLOAD, validate=True)
assert hashlib.sha256(raw).hexdigest() == EXPECTED_BUNDLE
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        assert pathlib.PurePosixPath(name).name == name
        (directory/name).write_bytes(archive.read(name))
cfg = json.loads((directory/'config.json').read_text())
inputs = pathlib.Path('/kaggle/input')
for name in ('train.csv','folds.csv','extra.csv'):
    matches = [p for p in inputs.rglob(name) if hashlib.sha256(p.read_bytes()).hexdigest() == cfg['input_hashes'][name]]
    assert len(matches) == 1, ('Frozen input mismatch',name,len(matches))
    (directory/name).write_bytes(matches[0].read_bytes())
for name, expected in cfg['input_hashes'].items():
    assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
output = pathlib.Path('/kaggle/working/results')
print('EXP052 fixed CPU run, starting the two four-model training arms', flush=True)
try:
    subprocess.run([sys.executable,'-u',str(directory/'train_volume.py'),'--payload',str(directory),'--output',str(output)],check=True,timeout=6660)
finally:
    if output.is_dir():
        with zipfile.ZipFile('/kaggle/working/exp052-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    archive.write(path,path.relative_to(output).as_posix())
"""


def prepare():
    require(not STATE.exists(), "Preparation already recorded; inspect before retrying")
    gate_path = EX / "metrics/data-gate.json"
    gate = read(gate_path)
    require(gate["status"] == "PASS_FROZEN_BEFORE_EXP052_OOF", "Training data checks failed")
    old = read(REFERENCE / "config.json")
    files = {name: REFERENCE / name for name in ("train.csv", "folds.csv", "extra.csv", "stylegram_core.py")}
    files.update({"legacy_runner.py": REFERENCE / "runner.py",
                  "format_runner.py": ROOT / "cloud/format120-044/payload/runner.py",
                  "train_volume.py": EX / "train_volume.py", "volume_gate.json": gate_path})
    for name in ("four120.csv", "four240.csv"):
        files[name] = EX / "data" / name
        require(digest(files[name]) == gate["outputs"][name]["sha256"], "Training input changed")
    for name in ("train.csv", "folds.csv", "extra.csv", "stylegram_core.py", "runner.py"):
        require(digest(REFERENCE/name) == old["input_hashes"][name], f"Original recipe changed: {name}")
    old44 = read(ROOT/"cloud/format120-044/payload/config.json")
    require(digest(files["format_runner.py"]) == old44["input_hashes"]["runner.py"], "EXP044 worker implementation changed")
    cfg = {key: old[key] for key in ("labels","validation_scheme","C","dual","max_iter","seed","share","sparse_per_prompt","sparse_repeats","threads","holdout_prompt_group_hashes","environment","budget")}
    cfg.update(schema_version=1, experiment_id="EXP-052", run_id=RUN_ID, baseline_id="EXP-042/control = EXP-010",
               arms=["four120","four240"], parallel_workers=2,
               new_labels=["gpt-5.6-luna","gpt-5.6-terra","gpt-5.6-sol","gpt-6-astra"],
               input_hashes={name:digest(path) for name,path in files.items()},
               weight_policy=old["weight_policy"].replace("adding120", "adding samples"),
               evaluation={"primary":"four240 minus four120 pooled 18-class official OOF balanced accuracy",
                           "secondary":["four240 minus reused control","four120 minus reused control"],
                           "paired_unit":"official prompt_id","bootstrap_draws":2000,"bootstrap_seed":42,
                           "inspect":["prediction changes","recall and false positives by class","per fold","sparse stress diagnostic"]},
               baseline_refit=False, new_holdout_used=False)
    write(EX / "config.json", cfg)
    files["config.json"] = EX / "config.json"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):
            if name not in ("train.csv","folds.csv","extra.csv"):
                archive.writestr(name,path.read_bytes())
    raw=buffer.getvalue()
    body=BOOTSTRAP.replace("ENCODED_PAYLOAD",repr(base64.b64encode(raw).decode("ascii"))).replace("EXPECTED_BUNDLE",repr(hashlib.sha256(raw).hexdigest()))
    compile(body, "<kaggle-notebook>", "exec")
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "cells":[{"id":"volume052","cell_type":"code","metadata":{},"execution_count":None,"source":body.splitlines(True),"outputs":[]}]}
    write(KERNEL / "experiment.ipynb",nb)
    meta=read(ROOT/"cloud/format120-044/kernel/kernel-metadata.json")
    meta.update(id=REF,title="Name That LLM Stylegram Volume 052")
    write(KERNEL/"kernel-metadata.json", meta)
    control_dir=ROOT/"cloud/luna-terra120-042/outputs-v2/results"
    control=control_dir/"oof/control.npz"
    baseline_file=next(f for f in read(control_dir/"artifacts.json")["files"] if f["path"]=="oof/control.npz")
    require(digest(control)==baseline_file["sha256"], "Control did not match the accepted EXP042 artifact")
    paths=list(files.values())+[EX/"prepare_data.py",Path(__file__),control,control_dir/"artifacts.json",KERNEL/"experiment.ipynb",KERNEL/"kernel-metadata.json"]
    write(EX/"manifest.json",{"experiment_id":"EXP-052","freeze_status":"FROZEN_BEFORE_TRAINING","observed_at":now(),"run_id":RUN_ID,
                              "files":[entry(path) for path in paths],"baseline_prediction":entry(control),"no_new_control_fits":True})
    write(STATE,{"status":"PREPARED","observed_at":now(),"ref":REF,"run_id":RUN_ID,"run_requests":0,"manifest_sha256":digest(EX/"manifest.json")})


def client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    proxy=read(ROOT/"configs/experiment-workbench.json").get("proxy")
    if proxy:
        os.environ["HTTP_PROXY"]=os.environ["HTTPS_PROXY"]=proxy
    api=KaggleApi()
    api.authenticate()
    return api


def readback(api,version):
    folder=CLOUD/f"readback-v{version}"
    api.kernels_pull(f"{REF}/{version}",str(folder),metadata=True,quiet=True)
    meta=read(folder/"kernel-metadata.json")
    expected=read(KERNEL/"kernel-metadata.json")
    remote=read(folder/meta["code_file"])
    local=read(KERNEL/"experiment.ipynb")
    def sources(nb):
        return [(c["cell_type"],"".join(c["source"])) for c in nb["cells"]]
    require(sources(remote)==sources(local),"Remote source mismatch")
    require(meta["is_private"] and not any(meta[k] for k in ("enable_gpu","enable_tpu","enable_internet")), "Remote settings mismatch")
    require(meta["dataset_sources"]==["zhjianduan/ntl-stylegram-cv-inputs-v1"] and meta["competition_sources"]==["name-that-llm"],"Input mismatch")
    require(not meta["kernel_sources"] and not meta["model_sources"] and meta["docker_image"]==expected["docker_image"],"Container/input mismatch")
    write(folder/"verified.json",{"status":"PASS","ref":REF,"version":version,"private":True,"gpu":False,"internet":False,"image":meta["docker_image"],"source_equal":True,"observed_at":now()})


def start():
    state=read(STATE)
    require(state["status"]=="PREPARED" and state["run_requests"]==0,"Prior dispatch exists; inspect before proceeding")
    require(digest(EX/"manifest.json")==state["manifest_sha256"],"Manifest changed")
    for item in read(EX/"manifest.json")["files"]:
        require(digest(ROOT/item["path"])==item["sha256"],"Frozen file changed: "+item["path"])
    api=client()
    seen=set()
    for page in range(1,101):
        items=api.kernels_list(mine=True,page=page,page_size=100)
        require(isinstance(items,list) and all(x is not None for x in items),"Inventory unavailable")
        refs={str(x.ref).removeprefix("/code/").strip("/") for x in items}
        require(REF not in refs and not seen.intersection(refs),"Remote target exists or inventory repeated")
        seen.update(refs)
        if len(items)<100:break
    else:raise RuntimeError("Inventory incomplete")
    for version,dry in [(1,True),(2,False)]:
        state.update(status="SAVE_REQUEST_IN_FLIGHT" if dry else "RUN_REQUEST_IN_FLIGHT",run_requests=0 if dry else 1,observed_at=now())
        write(STATE,state)
        response=api.kernels_push(str(KERNEL),timeout=7200,no_run=dry)
        ref=response.ref.removeprefix("/code/").strip("/")
        require(ref==REF and response.version_number==version and not response.error,"Unexpected push response")
        require(not (response.invalid_dataset_sources or response.invalid_kernel_sources or response.invalid_model_sources),"Rejected input sources")
        write(CLOUD/f"v{version}-ack.json",{"ref":ref,"version":version,"no_run":dry,"observed_at":now()})
        readback(api,version)
        state.update(status="SAVED_VERIFIED" if dry else "DISPATCHED_VERIFIED",version=version,observed_at=now())
        write(STATE,state)
    status(api)


def status(api):
    state=read(STATE)
    require(state.get("version")==2,"No confirmed run version")
    result=api.kernels_status(f"{REF}/2")
    state.update(status=str(result.status).split(".")[-1],observed_at=now())
    write(STATE,state)
    print(json.dumps(state))
    return state["status"]


def collect():
    api=client()
    terminal=status(api)
    require(terminal in {"COMPLETE","ERROR","CANCELLED","FAILED"},"Still active")
    dest=CLOUD/"outputs-v2"
    api.kernels_output(f"{REF}/2",str(dest),file_pattern=r"^exp052-results\.zip$",page_size=100,quiet=True)
    bundle=dest/"exp052-results.zip"
    if not bundle.is_file():
        require(terminal!="COMPLETE","Completed run has no result bundle")
        return
    extracted=dest/"results"
    with zipfile.ZipFile(bundle) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)) and sum(f.file_size for f in archive.infolist())<256*1024**2,"Invalid artifact list/size")
        for name in names:
            key=PurePosixPath(name)
            require(not key.is_absolute() and ".." not in key.parts and "\\" not in name and ":" not in name,"Unsafe artifact path")
            path=extracted/key
            path.parent.mkdir(parents=True,exist_ok=True)
            content=archive.read(name)
            if path.exists():require(path.read_bytes()==content,"Recovered file changed")
            else:path.write_bytes(content)
    print(json.dumps({"status":terminal,"bundle_sha256":digest(bundle),"files":len(names)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","start","status","collect"])
    cmd=parser.parse_args().action
    if cmd=="prepare":prepare()
    elif cmd=="start":start()
    elif cmd=="status":status(client())
    else:collect()
