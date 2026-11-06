"""CPU-only inventory of the 31 pinned attachments before a GPU capture.

Reads and hashes mounted input bytes. No model is imported or fitted.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def main():
    start=time.monotonic(); deadline=start+3450
    root=Path('/kaggle/input'); output=Path('/kaggle/working/preflight'); output.mkdir(exist_ok=True)
    config=json.loads((Path(__file__).parent/'preflight_config.json').read_text())
    bindings=config['attachment_bindings'];assert len(bindings)==31
    state={'schema_version':1,'experiment_id':'EXP-029','stage':'CPU_INPUT_HASHING','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'new_model_execution':False,'gpu_requested':False,'files_completed':0,'bytes_completed':0}
    save(output/'state.json',state)
    try:
        paths=sorted(p for p in root.rglob('*') if p.is_file())
        assert paths,'No mounted input files'
        for pattern in config['required_globs']:
            assert list(root.glob(pattern)),('Required input missing',pattern)
        for suffix in config['known_input_sha256']:
            assert sum(str(p).endswith('/'+suffix) for p in paths)==1,('Pinned file missing/ambiguous',suffix)
        total=sum(p.stat().st_size for p in paths)
        state.update(files_total=len(paths),bytes_total=total)
        save(output/'state.json',state)
        print(f'INPUT INVENTORY: {len(paths)} files, {total} bytes',flush=True)
        def digest(path):
            sha=hashlib.sha256();before=path.stat()
            with path.open('rb') as stream:
                while chunk:=stream.read(16*1024*1024):
                    if time.monotonic()>deadline:raise TimeoutError('CPU input hashing exceeded 3450 seconds')
                    sha.update(chunk)
            after=path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Input changed during read'
            return {'path':str(path),'bytes':after.st_size,'sha256':sha.hexdigest()}
        records=[]
        with (output/'files.jsonl').open('x',encoding='utf-8') as journal, ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(digest,p) for p in paths]
            for future in as_completed(futures):
                item=future.result();records.append(item);journal.write(json.dumps(item)+'\n');journal.flush()
                state['files_completed']+=1;state['bytes_completed']+=item['bytes']
                state['elapsed_seconds']=round(time.monotonic()-start,3)
                save(output/'state.json',state)
                print(f"HASH {len(records)}/{len(paths)} {state['bytes_completed']}/{total} bytes {Path(item['path']).name}",flush=True)
        for suffix,expected in config['known_input_sha256'].items():
            matches=[x for x in records if x['path'].endswith('/'+suffix)]
            assert len(matches)==1,('Expected exactly one pinned file',suffix,len(matches))
            assert matches[0]['sha256']==expected,('Pinned hash differs',suffix)
        train=[x for x in records if x['path'].endswith('/name-that-llm/train.csv')]
        assert len(train)==1 and train[0]['sha256']==config['train_sha256'],'Official train mismatch'
        versions={}
        for name in ('numpy','pandas','scipy','scikit-learn','torch','transformers','peft','pyarrow','lightgbm'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]=None
        manifest={'schema_version':1,'status':'PASS','experiment_id':'EXP-029','scope':'mounted_input_bytes_only',
                  'attachment_bindings':bindings,'files':sorted(records,key=lambda x:x['path']),
                  'environment':{'python':platform.python_version(),'versions':versions,'platform':platform.platform()},
                  'bytes_total':total,'elapsed_seconds':time.monotonic()-start,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
                  'preflight_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'config_sha256':hashlib.sha256((Path(__file__).parent/'preflight_config.json').read_bytes()).hexdigest(),
                  'new_model_execution':False,'training_exposure_certified':False,
                  'note':'Version bindings are the preselected request; host must verify saved attachment readback. Hashes bind mounted bytes, not historical score provenance.'}
        save(output/'preflight_manifest.json',manifest)
        state.update(stage='CPU_PREFLIGHT_COMPLETE',status='PASS',elapsed_seconds=time.monotonic()-start)
        save(output/'state.json',state);print('CPU PREFLIGHT PASS',flush=True)
    except BaseException as error:
        state.update(stage='CPU_PREFLIGHT_FAILED',status='FAILED',error=repr(error),elapsed_seconds=time.monotonic()-start)
        save(output/'state.json',state)
        (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':main()
