import hashlib,json,pathlib,runpy,sys,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
matches=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()=='8cada9716782903f6e799374c8bfc7aa7f6beb6f39c74cf0e1d941cac68faaa2']
assert len(matches)==1,'Expected one pinned protocol'
target=matches[0].parent
protocol=json.loads(matches[0].read_text())
for name,digest in protocol['input_hashes'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'guard.py'),run_name='__main__')
