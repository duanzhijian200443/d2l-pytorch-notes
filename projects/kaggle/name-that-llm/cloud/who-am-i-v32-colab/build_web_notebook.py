"""Package the browser launcher without allocating or executing any cloud runtime."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
FILES = ['web_runner.py', 'web_execute.py', 'web_checkpoint.py', 'web_instrument.py', 'web_watchdog.py',
         'upstream-v32.ipynb', 'input-lock.json', 'reference-submission.parquet']
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in FILES:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8'), filename=name)
        archive.writestr(name, (ROOT / name).read_bytes())
payload = buffer.getvalue()
digest = hashlib.sha256(payload).hexdigest()
code = '''# 复用现有 Drive 缓存；自动恢复已完成阶段。此单元格不会自行申请 GPU。
import base64, hashlib, io, zipfile, sys
from pathlib import Path

_payload = base64.b64decode(PAYLOAD_LITERAL)
assert hashlib.sha256(_payload).hexdigest() == DIGEST_LITERAL
_root = Path('/content/v32_web_payload')
_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(_payload)) as _archive:
    assert set(_archive.namelist()) == FILES_LITERAL
    assert all(Path(name).name == name for name in _archive.namelist())
    _archive.extractall(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
exec(compile((_root / 'web_runner.py').read_text(), 'web_runner.py', 'exec'),
     {'__name__': '__main__', '__file__': str(_root / 'web_runner.py')})
'''.replace('PAYLOAD_LITERAL', repr(base64.b64encode(payload).decode())).replace('DIGEST_LITERAL', repr(digest)).replace('FILES_LITERAL', repr(set(FILES)))
ast.parse(code)
intro = '''# Who am I V32 · 缓存与断点续跑版

使用已准备好的 **31 个固定版本输入**，保留 [原版 V32](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i?scriptVersionId=363222852) 的模型、集成参数和数学计算。这里只运行公开样例，不会提交比赛，也不能据此确认 0.660 榜单分数。

### 从网页启动
1. 选择 **A100**，运行下面的单元格，并完成 Google Drive 挂载授权。
2. 程序检查 `MyDrive/Kaggle/name-that-llm/v32-colab` 中的现有缓存和依赖；缺少文件时会报错停止，不会在 GPU 上重新下载整套权重。
3. 正常完成、报错、手动中断时都会尝试释放 GPU；另有独立的 **4 小时**保护。后台若直接回收虚拟机，最终保存不一定有机会执行，已经提交到 Drive 的阶段断点仍可保留。

### 断点如何恢复
- 完成的互补集成、TF-IDF、SVC、文本变形分类器、各主模型阶段，以及已完成的 LoRA 调用和 ATLAS 调用会保存。
- 再次连接 A100 并运行同一个单元格，会读取匹配的断点；**尚未完成的阶段会重做**，并非每个 batch 都可恢复。
- 代码、数据、模型版本或关键运行环境改变时会使用新的断点目录。损坏或未写完整的断点不会被使用。
- 上一次 CLI 中断没有这些计算断点，因此第一次运行此新版仍需重新计算；已有权重与依赖缓存可以复用。
- 同一份任务不要在多个运行时同时运行。

### 查看记录
实时进度会显示在单元格下方。完整日志、环境和结果保存在 `v32-colab/outputs/web-日期-编号/`；最近状态在 `v32-colab/web_run_state.json`，计算断点在 `v32-colab/web-checkpoints/`。

**验证范围：**本地会校验 Notebook 格式、断点写入/恢复/损坏处理及中断释放逻辑；新版完整 A100 推理尚未执行，由你从网页启动。
'''
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(code)],
    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
              'colab': {'name': 'Who_am_I_V32_A100.ipynb'}, 'v32_launcher': {'version': 1, 'payload_sha256': digest, 'manual_start': True}})
nbformat.validate(nb)
nbformat.write(nb, ROOT / 'Who_am_I_V32_A100.ipynb')
(ROOT / 'v32_web_payload.zip').write_bytes(payload)
print(json.dumps({'notebook': 'Who_am_I_V32_A100.ipynb', 'payload_sha256': digest,
                  'files': FILES, 'notebook_bytes': (ROOT / 'Who_am_I_V32_A100.ipynb').stat().st_size}))
