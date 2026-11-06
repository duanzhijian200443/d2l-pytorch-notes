"""Package, run, observe and recover the two fixed, private CPU jobs for EXP053."""
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
BASE = ROOT / "cloud/luna-terra120-042/payload"
CLOUD = ROOT / "cloud/apex-stylegram-053"
SOURCE = next((ROOT / "research/solutions").glob("SRC-067-*/source.ipynb"))
VERSIONS = json.loads((BASE/"config.json").read_text())["environment"]["required_versions"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def save(p, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(p)


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def ref(which):
    return "zhjianduan/ntl-apex-volume-053-"+which


BOOT = """import base64, hashlib, io, json, os, pathlib, subprocess, sys, time, zipfile, psutil
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
target=pathlib.Path('/tmp/exp053-inputs')
target.mkdir(exist_ok=False)
raw=base64.b64decode(BUNDLE,validate=True)
assert hashlib.sha256(raw).hexdigest()==BUNDLE_HASH
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        assert pathlib.PurePosixPath(name).name==name and ':' not in name
        (target/name).write_bytes(z.read(name))
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((target/'experiment-config.json').read_text())
assert cfg['job']==JOB and cfg['experiment_id']=='EXP-053'
assert not pathlib.Path('/dev/nvidia0').exists()
for name in cfg['mounted']:
    matches=[p for p in pathlib.Path('/kaggle/input').rglob(name) if digest(p)==cfg['input_hashes'][name]]
    assert matches, ('Missing expected input bytes',name)
    (target/name).write_bytes(matches[0].read_bytes())
if JOB=='style':
    import ast, fnmatch, re, types
    import pandas as pd
    sources={}
    for name, expected in cfg['loader_csv'].items():
        matches=[p for p in pathlib.Path('/kaggle/input').rglob(name) if digest(p)==expected]
        assert len(matches)==1, ('Raw loader input differs', name)
        sources[name]=str(matches[0])
    def list_source_files(pattern,recursive=False):
        return sorted(p for n,p in sources.items() if fnmatch.fnmatch(n,pattern.split('/')[-1]))
    cells=json.loads((target/'source.ipynb').read_text())['cells']
    code=next(''.join(c['source']).split(chr(10),1)[1] for c in cells if ''.join(c['source']).startswith('%%writefile comp_stack.py'))
    definition=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='load_extra_passages']
    assert len(definition)==1
    ns={'pd':pd,'re':re,'glob':types.SimpleNamespace(glob=list_source_files),'LABELS':sorted(pd.read_csv(target/'train.csv').model.unique()),'EXTRA_MIN_LABELS':12}
    exec(compile(ast.Module(body=definition,type_ignores=[]),'<frozen author loader>','exec'),ns)
    (target/'shared.csv').write_bytes(ns['load_extra_passages']().to_csv(index=False,lineterminator=chr(10)).encode())
for name,d in cfg['input_hashes'].items():
    assert digest(target/name)==d,name
import numpy,pandas,scipy,sklearn
actual={'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
assert actual==cfg['required_versions'], (actual,cfg['required_versions'])
out=pathlib.Path('/kaggle/working/results')
out.mkdir(exist_ok=False)
start=time.monotonic()
peak=0
child=subprocess.Popen([sys.executable,'-u',str(target/SCRIPT),'--input',str(target),'--output',str(out)])
try:
    process=psutil.Process(child.pid)
    next_print=0
    while child.poll() is None:
        rss=0
        try:
            for p in [process]+process.children(recursive=True):
                try:rss+=p.memory_info().rss
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
        peak=max(rss,peak)
        elapsed=time.monotonic()-start
        if elapsed>=6600 or rss>=24*(1024**3):
            raise RuntimeError('CPU job budget exceeded')
        if elapsed>=next_print:
            print('cpu_job_seconds',round(elapsed),'rss_GiB',round(rss/1024**3,2),flush=True)
            next_print=elapsed+60
        time.sleep(2)
    if child.returncode!=0:raise RuntimeError('CPU subprocess failed: '+str(child.returncode))
finally:
    if child.poll() is None:
        for p in process.children(recursive=True):
            try:p.kill()
            except psutil.NoSuchProcess:pass
        child.kill()
        child.wait()
    receipt={'job':JOB,'versions':actual,'returncode':child.returncode,'seconds':time.monotonic()-start,'peak_rss_bytes':peak,'config_sha256':digest(target/'experiment-config.json')}
    (out/'cloud-receipt.json').write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile('/kaggle/working/exp053-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(out).as_posix())
"""


def prepare(which):
    home=CLOUD/which
    check(not (home/"state.json").exists(),"Already prepared")
    old=read(BASE/"config.json")
    pool=read(next((ROOT/"experiments").glob("EXP-050-*/config.json")))["inputs"]
    files={"source.ipynb":SOURCE,"panel.csv":ROOT/"experiments/EXP-029-full-stack-capture/panel/test.csv","train.csv":BASE/"train.csv"}
    mounted=["train.csv"]
    attachments=[]
    if which=="main":
        script="main_cpu.py"
        files.update({"nlm_matched_clean.parquet":ROOT/pool["matched"]["path"],"ntl_open_pool.parquet":ROOT/pool["open_pool"]["path"]})
        mounted+=["nlm_matched_clean.parquet","ntl_open_pool.parquet"]
        attachments=["dhruvtibarewal/nlm-matched-clean/3","crtproduction/name-that-llm-open-pool/13"]
    else:
        script="style_cpu.py"
        fixed=ROOT/"experiments/EXP-052-stylegram-four-model-volume/data"
        files.update({"four120.csv":fixed/"four120.csv","four240.csv":fixed/"four240.csv",
                      "folds.csv":BASE/"folds.csv","shared.csv":EX/"inputs/shared.csv",
                      "regen-fold-map.csv":EX/"inputs/regen-fold-map.csv"})
        loader=read(EX/"inputs/input-receipt.json")["loader_files"]
        attachments=[row["ref"]+"/"+str(row["version"]) for row in loader]
    files[script]=EX/script
    hashes={n:sha(p) for n,p in files.items()}
    check(hashes["source.ipynb"]=="ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd","Source changed")
    check(hashes["train.csv"]==old["input_hashes"]["train.csv"],"Official data changed")
    if which=="main":
        check(hashes["nlm_matched_clean.parquet"]==pool["matched"]["sha256"] and hashes["ntl_open_pool.parquet"]==pool["open_pool"]["sha256"],"Wrong V13/matched bytes")
    else:
        gate=read(ROOT/"experiments/EXP-052-stylegram-four-model-volume/metrics/data-gate.json")
        check(all(hashes[n]==gate["outputs"][n]["sha256"] for n in ("four120.csv","four240.csv")),"EXP052 arms drifted")
    cfg={"experiment_id":"EXP-053","job":which,"input_hashes":hashes,"mounted":mounted,"required_versions":VERSIONS,
         "author_ref":"synthreaper/ntl-sovereign-gold-apex","author_version":1,"author_script_version_id":364684268,
         "cpu_budget_seconds":6600,"residual_scope":"Conditional analysis on old cached heavy members",
         "reference_scope":"Official five-fold end point is isolated Stylegram. Main panel exposure precludes generalization claims."}
    if which=="style":
        cfg["loader_csv"]={row["name"]:row["sha256"] for row in loader}
    save(home/"experiment-config.json",cfg)
    files["experiment-config.json"]=home/"experiment-config.json"
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for name,p in sorted(files.items()):
            if name not in mounted and name!="shared.csv":z.writestr(name,p.read_bytes())
    raw=buf.getvalue()
    body=BOOT.replace("SCRIPT",repr(script)).replace("JOB",repr(which)).replace("BUNDLE_HASH",repr(hashlib.sha256(raw).hexdigest())).replace("BUNDLE",repr(base64.b64encode(raw).decode()))
    compile(body,"cpu-bootstrap","exec")
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},
              "cells":[{"id":"exp053","cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":body.splitlines(True)}]}
    kernel=home/"kernel"
    save(kernel/"experiment.ipynb",notebook)
    meta=read(ROOT/"cloud/format120-044/kernel/kernel-metadata.json")
    meta.update(id=ref(which),title="NTL Apex Volume 053 "+which.title(),dataset_sources=attachments)
    save(kernel/"kernel-metadata.json",meta)
    save(home/"state.json",{"status":"PREPARED","ref":ref(which),"run_requests":0,"prepared_at":stamp(),
         "notebook_sha256":sha(kernel/"experiment.ipynb"),"metadata_sha256":sha(kernel/"kernel-metadata.json"),
         "config_sha256":sha(home/"experiment-config.json")})


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    proxy=read(ROOT/"configs/experiment-workbench.json")["proxy"]
    os.environ["HTTPS_PROXY"]=os.environ["HTTP_PROXY"]=proxy
    client=KaggleApi()
    client.authenticate()
    return client


def start(which):
    home=CLOUD/which
    state=read(home/"state.json")
    kernel=home/"kernel"
    check(state["status"]=="PREPARED" and state["run_requests"]==0,"Inspect the existing attempt")
    check(sha(kernel/"experiment.ipynb")==state["notebook_sha256"] and sha(kernel/"kernel-metadata.json")==state["metadata_sha256"],"Frozen kernel changed")
    client=api()
    for page in range(1,51):
        items=client.kernels_list(mine=True,page=page,page_size=100)
        check(isinstance(items,list) and all(i and str(i.ref).removeprefix("/code/").strip("/")!=ref(which) for i in items),"Target not clear")
        if len(items)<100:break
    else:raise RuntimeError("Incomplete inventory")
    state.update(status="RUN_REQUEST_IN_FLIGHT",run_requests=1,observed_at=stamp())
    save(home/"state.json",state)
    response=client.kernels_push(str(kernel),timeout=7200,no_run=False)
    check(response.ref.removeprefix("/code/").strip("/")==ref(which) and response.version_number==1 and not response.error,"Unexpected acknowledgement")
    check(not (response.invalid_dataset_sources or response.invalid_model_sources or response.invalid_kernel_sources),"Invalid attachments")
    save(home/"ack.json",{"ref":ref(which),"version":1,"observed_at":stamp()})
    dest=home/"readback"
    client.kernels_pull(ref(which)+"/1",str(dest),metadata=True,quiet=True)
    meta=read(dest/"kernel-metadata.json")
    expected=read(kernel/"kernel-metadata.json")
    cells=lambda nb:[(c["cell_type"],"".join(c["source"])) for c in read(nb)["cells"]]
    check(cells(dest/meta["code_file"])==cells(kernel/"experiment.ipynb"),"Remote code differs")
    check(meta["is_private"] and not any(meta[k] for k in ("enable_gpu","enable_tpu","enable_internet")) and meta["docker_image"]==expected["docker_image"],"Remote settings differ")
    check(set(meta["dataset_sources"])=={s.rsplit("/",1)[0] for s in expected["dataset_sources"]} and meta["competition_sources"]==["name-that-llm"] and not (meta["kernel_sources"] or meta["model_sources"]),"Remote inputs differ")
    state.update(version=1,readback_verified=True,status=str(client.kernels_status(ref(which)+"/1").status).split(".")[-1],observed_at=stamp())
    save(home/"state.json",state)
    print(state["status"],ref(which),flush=True)


def finish(which,download):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    import re
    home=CLOUD/which
    state=read(home/"state.json")
    client=api()
    st=str(client.kernels_status(ref(which)+"/1").status).split(".")[-1]
    state.update(status=st,observed_at=stamp())
    save(home/"state.json",state)
    print(st,flush=True)
    if not download:return
    check(st in {"COMPLETE","ERROR","CANCELLED","FAILED"},"Job still active")
    dest=home/"outputs"
    client.kernels_output(ref(which)+"/1",str(dest),file_pattern=r"^exp053-results\.zip$",page_size=100,quiet=True)
    bundle=dest/"exp053-results.zip"
    if not bundle.exists():
        check(st!="COMPLETE","Missing completed archive")
        return
    with zipfile.ZipFile(bundle) as z:
        check(len(z.namelist())==len(set(z.namelist())) and sum(f.file_size for f in z.infolist())<300*1024**2,"Output size/name check failed")
        for name in z.namelist():
            path=PurePosixPath(name)
            check(not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name,"Invalid output path")
            fp=dest/"results"/path
            fp.parent.mkdir(exist_ok=True,parents=True)
            data=z.read(name)
            if fp.exists():check(fp.read_bytes()==data,"Changed output")
            else:fp.write_bytes(data)
    q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=ref(which).split("/");q.version_label="v1";q.page_size=100
    with client.build_kaggle_client() as c:listing=c.kernels.kernels_api_client.list_kernel_session_output(q)
    f=next(x for x in listing.files if x.file_name=="exp053-results.zip")
    script_id=int(re.search(r"/kf/(\d+)/",f.url).group(1))
    save(dest/"download-receipt.json",{"status":st,"scriptVersionId":script_id,"observed_at":stamp(),"ref":ref(which),"version":1,"archive_sha256":sha(bundle),"config_sha256":state["config_sha256"]})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","start","status","collect"])
    parser.add_argument("job",choices=["main","style"])
    args=parser.parse_args()
    if args.action=="prepare":prepare(args.job)
    elif args.action=="start":start(args.job)
    else:finish(args.job,args.action=="collect")
