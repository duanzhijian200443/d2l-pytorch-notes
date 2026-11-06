"""Build and check the two EXP054-derived local Kaggle packages. No remote operations."""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "cloud/apex-zero-054/kernel"
DATA = ROOT / "experiments/EXP-052-stylegram-four-model-volume/data"
SOURCE_SHA = "ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd"
META_SHA = "048535fc89265728c52dbd42089edf8dfebfbcd331b1dcfe26141efdc27f0763"
COMP_SHA = "4a4b800a5380094d78da66d140d224f8a3974d1e3012086783d5bf81b1264514"
SHARED_SHA = "b98c246578afbc38b9bf3c0be1e666a41cc15d051c64d03e875ab0c85da1bd3b"
ARM_CONFIG = {
    "four120": (120, 56, "a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d"),
    "four240": (240, 57, "694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01"),
}
UNCHANGED_MODULES = {
    "gauss.py":"908b57113f1a7b8ebdbf204c6be87cbc385f765ba50ac07b4e270c64b7cf1000",
    "infer.py":"63112c88d43f9d69b3132243315e0db28c40b4bf0f42bac651f73ef07a67f9bb",
    "sibling_style.py":"8a7f3b3dad7cbe1755b66e8455a157894cac92b8983fc805f5ed6286ea915e66",
}
META_INPUTS = ("competition_sources","dataset_sources","model_sources","kernel_sources")
SIZE_CAP = 512 * 1024  # local conservative bound; the API's actual limit is not inferred here

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def replace_once(text, original, replacement):
    need(text.count(original) == 1, "Expected exactly one original source anchor")
    return text.replace(original, replacement, 1)

def arm_input(arm):
    n, _, binding = ARM_CONFIG[arm]
    raw = (DATA / (arm + ".csv")).read_bytes()
    need(sha(raw) == binding, "EXP052 input drift")
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    need(len(records) == n and all(r["split"] == "train" for r in records), "Non-training input or wrong count")
    need(len({r["row_id"] for r in records}) == len({r["job_key"] for r in records}) == n, "Duplicate source ID")
    need(all(sha(r["text"].encode()) == r["text_sha256"] for r in records), "Raw response hash mismatch")
    groups = {}
    for row in records:
        groups.setdefault(row["prompt_id"], set()).add(row["model"])
    models = {"gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-6-astra"}
    need(len(groups) == n//4 and all(group == models for group in groups.values()), "Incomplete four-answer prompt")
    return raw

def generate(arm):
    rows, number, binding = ARM_CONFIG[arm]
    source, meta_raw = (BASE/"apex-zero.ipynb").read_bytes(), (BASE/"kernel-metadata.json").read_bytes()
    need(sha(source) == SOURCE_SHA and sha(meta_raw) == META_SHA, "Frozen EXP054 input changed")
    raw = arm_input(arm)
    payload = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    # Adjacent short Python literals compile into one AST constant.
    chunks = "\n".join("    " + repr(payload[i:i+120]) for i in range(0,len(payload),120))
    base_nb, meta = json.loads(source), json.loads(meta_raw)
    parent = base_nb["cells"][12]["source"]
    magic, parent_body = parent.split("\n",1)
    need(magic == "%%writefile comp_stack.py" and sha(parent_body.encode()) == COMP_SHA, "Wrong parent module")
    start = parent_body.index("grams = StyleGrams().fit(TR + EX_T)")
    stop = parent_body.index("G_style = normalised_content_groups(TE)",start)
    body = parent_body
    params = {"__ARM__":arm,"__ROWS__":str(rows),"__CSV_SHA__":binding,"__SHARED_SHA__":SHARED_SHA}
    block = (HERE/"volume_block.py.in").read_text(encoding="utf-8").replace("__B64__","\n"+chunks+"\n")
    for token,value in params.items():
        block = block.replace(token,value)
    body = replace_once(body,parent_body[start:stop],block.rstrip()+"\n")
    anchor = "add_member('stylegram', P_style)\n"
    receipt = (HERE/"member_receipt.py.in").read_text(encoding="utf-8")
    body = replace_once(body,anchor,anchor+receipt.rstrip()+"\n")
    compile(body, "comp_stack.py", "exec")
    # Assert that definitions and all statements outside the intended member are untouched.
    prefix = parent_body[:start]
    suffix = parent_body[parent_body.index("del X_tr_c, svc_plain, svc_centred, P_style"):]
    need(body.startswith(prefix) and body.endswith(suffix), "Out-of-scope comp_stack change")
    route = parent_body[stop:parent_body.index(anchor)+len(anchor)]
    need(body.count(route)==1, "Original inference routing or member addition changed")
    fit_def = "def fit_svc(X, labels, C=0.5, **kw):\n    return LinearSVC(C=C, random_state=42, max_iter=4000).fit(X, labels, **kw)"
    need(fit_def in parent_body and fit_def in body, "Author estimator changed")
    changed_comp = magic+"\n"+body
    comp_sha = sha(body.encode("utf-8"))
    run = (HERE/"run_cell.py.in").read_text(encoding="utf-8")
    for token,value in dict(params, __COMP_SHA__=comp_sha).items():
        run = run.replace(token,value)
    compile(run,"candidate-cell14","exec")
    # Replace the two existing JSON string literals directly; all other bytes
    # remain exactly those of the archived author notebook.
    text = source.decode("utf-8")
    for index,value in ((12,changed_comp),(14,run)):
        text = replace_once(text,json.dumps(base_nb["cells"][index]["source"],ensure_ascii=False),
                           json.dumps(value,ensure_ascii=False))
    notebook = text.encode("utf-8")
    need(len(notebook) <= SIZE_CAP, "Exceeded conservative Notebook size cap")
    parsed = json.loads(notebook)
    changed = [i for i,(old,new) in enumerate(zip(base_nb["cells"],parsed["cells"])) if old != new]
    need(changed == [12,14], "Unexpected changed cells")
    for i in changed:
        parsed["cells"][i]["source"] = base_nb["cells"][i]["source"]
    need(parsed == base_nb, "Other notebook fields changed")
    input_count = sum(len(meta[k]) for k in META_INPUTS if k != "competition_sources")
    need(input_count==31 and meta["competition_sources"]==["name-that-llm"]
         and meta["is_private"] is True and meta["enable_internet"] is False, "Original metadata drift")
    slug = f"ntl-apex-codex{rows}-{number:03}"
    meta.update(id="zhjianduan/"+slug,title=f"NTL Apex Codex{rows} {number:03}",code_file="apex.ipynb")
    metadata = (json.dumps(meta,indent=2,ensure_ascii=False)+"\n").encode()
    directory = ROOT / f"cloud/apex-codex{rows}-{number:03}/kernel"
    report = {
        "arm":arm, "ref":meta["id"], "kernel_dir":directory.relative_to(ROOT).as_posix(),
        "notebook_sha256":sha(notebook),"notebook_bytes":len(notebook),"metadata_sha256":sha(metadata),
        "comp_stack_sha256":comp_sha,"comp_stack_bytes":len(body.encode()),"added_csv_sha256":binding,
        "rows":rows,"groups":rows//4,"shared_sha256":SHARED_SHA,"shared_rows":10743,"shared_groups":738,
        "input_count":input_count,"competition_input_count":1,"encoded_payload_bytes":len(payload),"changed_cells":changed,
        "unchanged_module_hashes":UNCHANGED_MODULES, "original_comp_stack_sha256":COMP_SHA,
        "unchanged_comp_prefix_sha256":sha(prefix.encode()),"unchanged_comp_suffix_sha256":sha(suffix.encode()),
        "unchanged_route_sha256":sha(route.encode()),
        "metadata_changes":["id","title","code_file"],
        "public_preflight":"volume_preflight.json","hidden_member_receipt":"stylegram_volume_receipt.json",
        "hidden_execution_receipt":"volume_comp_execution.json",
    }
    return directory, notebook, metadata, report, body.encode()

def verify_materialized(arm, path):
    _,_,_,report,expected = generate(arm)
    data = path.read_bytes()
    need(data == expected, "Materialized comp_stack differs from the exact candidate")
    assignment, = [n for n in ast.parse(data).body if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="_volume_b64"]
    actual = zlib.decompress(base64.b64decode(ast.literal_eval(assignment.value),validate=True))
    need(actual == arm_input(arm), "Decoded materialized input differs from the frozen CSV")
    return {"status":"PASS_MATERIALIZED","arm":arm,"comp_stack_sha256":sha(data),"rows":report["rows"],
            "groups":report["groups"],"source_csv_sha256":sha(actual),"shared_sha256_required":SHARED_SHA}

def build(verify=False):
    arms={}
    for arm in ARM_CONFIG:
        directory,notebook,metadata,arms[arm],_ = generate(arm)
        for filename,content in (("apex.ipynb",notebook),("kernel-metadata.json",metadata)):
            p=directory/filename
            if verify or p.exists():
                need(p.read_bytes()==content,"Existing candidate differs: "+str(p))
            else:
                directory.mkdir(parents=True,exist_ok=True)
                with p.open("xb") as f:
                    f.write(content)
    manifest={"source_notebook_sha256":SOURCE_SHA,"source_metadata_sha256":META_SHA,"local_size_cap_bytes":SIZE_CAP,
              "scope":"Only Stylegram fits and their execution checks differ from EXP054, with identical input requests.",
              "build_inputs":{name:sha((HERE/name).read_bytes()) for name in
                              ("generate.py","volume_block.py.in","member_receipt.py.in","run_cell.py.in")},
              "arms":arms}
    p=HERE/"manifest.json"
    encoded=(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n").encode()
    if verify or p.exists():
        need(p.read_bytes()==encoded,"Manifest mismatch")
    else:
        with p.open("xb") as f:
            f.write(encoded)
    print(json.dumps(arms,ensure_ascii=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("build");sub.add_parser("verify")
    material=sub.add_parser("verify-materialized")
    material.add_argument("arm",choices=ARM_CONFIG)
    material.add_argument("path",type=Path)
    args=parser.parse_args()
    if args.action=="verify-materialized":
        print(json.dumps(verify_materialized(args.arm,args.path)))
    else:
        build(verify=args.action=="verify")
