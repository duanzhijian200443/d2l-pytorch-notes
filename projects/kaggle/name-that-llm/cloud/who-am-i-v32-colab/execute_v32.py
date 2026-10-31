"""Execute the frozen notebook with only Colab paths and subprocess plumbing adapted."""
import ast
import os
import sys
import json
import time
import hashlib
import platform
import subprocess
import importlib.metadata as md
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path('/kaggle/working')
INPUT = os.environ['V32_INPUT_ROOT']
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
import torch
assert torch.cuda.device_count() == 1, 'Expected one A100 GPU'
assert 'A100' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
environment = {'python':platform.python_version(), 'gpu_name':torch.cuda.get_device_name(0),
    'gpu_count':torch.cuda.device_count(), 'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
    'packages':{n:md.version(n) for n in ['torch','transformers','peft','numpy','pandas','scipy','scikit-learn','lightgbm','safetensors','pyarrow']},
    'platform':'Colab A100', 'same_container_as_upstream':False}
(WORK/'reproduction_environment.json').write_text(json.dumps(environment,indent=2))
print('ENVIRONMENT',json.dumps(environment),flush=True)
nb=json.loads((ROOT/'upstream-v32.ipynb').read_text())
source=lambda i: ''.join(nb['cells'][i]['source'])
adapt=lambda s:s.replace('/kaggle/input',INPUT)
namespace={'__name__':'__main__'}
paths=adapt(source(1))
replacement="BL = str(Path(INPUT) / 'datasets/zhjianduan/llama31-8b-base-verified')"
assert paths.count("BL = mdir('/llama-3.1/', '/8b/')")==1
paths=paths.replace("BL = mdir('/llama-3.1/', '/8b/')",replacement)
namespace.update({'Path':Path,'INPUT':INPUT})
exec(compile(paths,'v32_paths.py','exec'),namespace)
hashes={}
for i in (2,3,4):
    cell=source(i)
    first,body=cell.split('\n',1)
    filename=first.split()[-1]
    assert filename in ['gauss.py','infer.py','comp_stack.py']
    patched=adapt(body)
    ast.parse(patched)
    (WORK/filename).write_text(patched)
    hashes[filename]={'upstream_sha256':hashlib.sha256(body.encode()).hexdigest(),
                      'colab_sha256':hashlib.sha256(patched.encode()).hexdigest(),
                      'changes':'input root path only'}
(WORK/'module_provenance.json').write_text(json.dumps(hashes,indent=2))
os.environ['NTL_DROP_Q35']='0'
start=time.time()
subprocess.run([sys.executable,'-u','comp_stack.py'],check=True)
namespace['COMP_SEC']=time.time()-start
print('COMP STACK DONE',namespace['COMP_SEC'],flush=True)
exec(compile(adapt(source(6)),'v32_main.py','exec'),namespace)
import pandas as pd
submission=pd.read_parquet(WORK/'submission.parquet')
test=pd.read_csv(Path(namespace['DATA'])/'test.csv')
train=pd.read_csv(Path(namespace['DATA'])/'train.csv')
assert submission.id.tolist()==test.id.tolist()
assert submission.model.isin(train.model.unique()).all()
reference=pd.read_parquet(ROOT/'reference-submission.parquet')
comparison=submission.merge(reference,on='id',suffixes=('_actual','_reference'),validate='one_to_one')
assert len(comparison)==len(test)
report={'status':'COMPLETE','upstream_version':32,'upstream_script_version_id':363222852,
        'test_rows':len(test),'sample_predictions_match':bool((comparison.model_actual==comparison.model_reference).all()),
        'sample_comparison':comparison.to_dict(orient='records'),'public_score_verified':False,
        'elapsed_inference_seconds':time.time()-start,'comp_seconds':namespace['COMP_SEC'],
        'hardware_changed':'T4 x2 to A100; original algorithm and ensemble parameters retained'}
(WORK/'reproduction_result.json').write_text(json.dumps(report,indent=2))
print('REPRODUCTION RESULT',json.dumps(report),flush=True)
