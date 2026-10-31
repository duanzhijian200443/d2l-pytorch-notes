import json
import subprocess
import sys
from pathlib import Path

root=Path('/content/drive/MyDrive/Kaggle/name-that-llm/v32-colab')
wheels=root/'cli-wheels'
wheels.mkdir(exist_ok=True)
packages=['torch==2.11.0','transformers==5.16.1','peft==0.20.0',
          'numpy==2.1.3','pandas==2.3.3','scipy==1.16.3','scikit-learn==1.6.1',
          'lightgbm==4.6.0','safetensors==0.8.0','pyarrow==23.0.1',
          'sentencepiece','protobuf','accelerate']
job_code='''import json,subprocess,sys,time
from pathlib import Path
root=Path(ROOT_LITERAL)
started=time.time()
with (root/'cli-wheels.log').open('w') as stream:
    result=subprocess.run([sys.executable,'-m','pip','download','--only-binary=:all:','--dest',str(root/'cli-wheels')]+PACKAGES_LITERAL,stdout=stream,stderr=subprocess.STDOUT)
(root/'cli_wheels_result.json').write_text(json.dumps({'status':'READY' if result.returncode==0 else 'ERROR','exit_code':result.returncode,'elapsed_seconds':time.time()-started,'python':sys.version,'packages':PACKAGES_LITERAL},indent=2))
'''.replace('ROOT_LITERAL',repr(str(root))).replace('PACKAGES_LITERAL',repr(packages))
script=Path('/content/v32_cache_wheels.py')
script.write_text(job_code)
proc=subprocess.Popen([sys.executable,str(script)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'wheel_cache_pid':proc.pid,'destination':str(wheels)}),flush=True)
