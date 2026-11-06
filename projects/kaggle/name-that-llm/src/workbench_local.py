"""Non-writing package verification adapter for the already frozen EXP049."""
import importlib.util
import sys
from workbench_core import *

def verify_package(p):
    require(p['id']=='EXP-049','PACKAGE_VERIFIER_NOT_SUPPORTED')
    for path,digest in p['local_code_hashes'].items():require(sha(inside(path))==digest,'LOCAL_JOB_CODE_CHANGED')
    source=inside('cloud/a1-sinkhorn030-049/prepare_verify.py')
    spec=importlib.util.spec_from_file_location('workbench_exp049_preparer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old,metadata=module.source_gate()
    m=read(module.EXP/'manifest.json');require(m['status']=='READY_NOT_LAUNCHED','CANDIDATE_NOT_READY')
    for item in [m['config']]+m['code']+m['source_bindings']+m['candidate_files']:
        require(sha(inside(item['path']))==item['sha256'],'CANDIDATE_BINDING_CHANGED')
    audit=module.validate_candidate(old,metadata,read(module.HERE/'kernel/a1.ipynb'),read(module.HERE/'kernel/kernel-metadata.json'))
    return {'status':'PASS_READY_NOT_LAUNCHED','readonly':True,'writes':0,'audit':audit}

if __name__=='__main__':
    require(len(sys.argv)==3 and sys.argv[1]=='verify-package','UNSUPPORTED_LOCAL_COMMAND')
    _,p,_=profile(sys.argv[2]);print(json.dumps(verify_package(p),ensure_ascii=False))
