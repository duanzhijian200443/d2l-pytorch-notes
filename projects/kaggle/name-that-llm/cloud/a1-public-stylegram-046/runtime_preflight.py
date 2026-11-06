"""Checks that execute in both quick commit and hidden rerun before heavy work."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

def run_runtime_preflight(payload):
    from input_preflight import run_preflight
    from input_policy import resolve_input_policy, EXPECTED_COMP_MEMBERS, ALLOWED_OMISSIONS
    from a1_stylegram import self_test
    began=time.monotonic()
    receipt=run_preflight(manifest_path=payload/'preflight_manifest.json',
        extra_manifest_path=payload/'extra-inputs.json',
        receipt_path=Path('/kaggle/working/input_preflight.json'),input_root=Path('/kaggle/input'))
    paths=sorted(str(p) for p in Path('/kaggle/input').rglob('*') if p.is_file())
    policy=resolve_input_policy(paths)
    assert policy['expected_comp_members']==list(EXPECTED_COMP_MEMBERS)
    assert policy['allowed_omissions']==list(ALLOWED_OMISSIONS)
    Path('/kaggle/working/availability.json').write_text(json.dumps(policy,indent=2))
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','transformers','peft','accelerate']}
    assert versions['scikit-learn']=='1.6.1',versions
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip().splitlines()
    assert len(gpus)==2 and all('T4' in x for x in gpus),gpus
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==2
    for i in range(2):
        assert (torch.ones(2,device=f'cuda:{i}')+1).sum().item()==4
    synthetic=self_test(core_path=payload/'stylegram_core.py')
    Path('/kaggle/working/a1_synthetic_parity.json').write_text(json.dumps(synthetic,indent=2))
    Path('/kaggle/working/reproduction_environment.json').write_text(json.dumps({
        'status':'PASS','python':platform.python_version(),'versions':versions,'gpus':gpus,
        'cuda_test':'PASS','synthetic_parity':synthetic,'preflight_seconds':time.monotonic()-began},indent=2))
    print('EXP046 INPUT/ENVIRONMENT/SYNTHETIC PARITY PASS',flush=True)
    return receipt
