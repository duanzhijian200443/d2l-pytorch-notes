"""Create the Colab transfer notebook from pinned public metadata and local source."""

import ast
import json
import urllib.request
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
REPO = 'NousResearch/Meta-Llama-3.1-8B'
REVISION = '1f47e50cdbe801ad8a5174156ec3a0655108fb9f'
OFFICIAL_REVISION = 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
NAMES = [
    'LICENSE', 'USE_POLICY.md', 'README.md', 'config.json', 'generation_config.json',
    'model.safetensors.index.json', 'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.json',
    *[f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)],
]


def metadata(repo, revision):
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=30) as response:
        payload = json.load(response)
    assert payload['sha'] == revision
    return {item['rfilename']: item for item in payload['siblings']}


public = metadata(REPO, REVISION)
official = metadata('meta-llama/Llama-3.1-8B', OFFICIAL_REVISION)
files = []
for name in NAMES:
    item = public[name]
    if name not in ['LICENSE', 'USE_POLICY.md', 'README.md']:
        assert item['blobId'] == official[name]['blobId'], name
    files.append({
        'name': name, 'size': item['size'],
        'sha256': item.get('lfs', {}).get('sha256'), 'git_blob_sha1': item['blobId'],
    })
manifest = {'repo': REPO, 'revision': REVISION, 'official_revision': OFFICIAL_REVISION, 'files': files}
(ROOT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell
helper_source = (ROOT / 'transfer_helpers.py').read_text(encoding='utf-8')
cells = [
    md('# Llama 3.1 8B：Google Drive → Kaggle\n\n使用 **CPU** 将固定版本的公开基础模型保存到 Google Drive，校验后上传为 `zhjianduan` 的私有 Kaggle Dataset。约 16.07 GB；不会加载模型或训练。\n\n首次运行需要授权挂载 Drive。Kaggle 凭据只放在 Colab Secrets 或隐藏输入框中，不写入代码或 Drive。\n\n来源：[NousResearch/Meta-Llama-3.1-8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B)。Built with Llama；保留原许可证和使用政策。'),
    md('## 1. 挂载 Google Drive\n\n文件保存在下面的专用目录；下载中断后可以重跑，已校验的完整文件会跳过。'),
    code("from google.colab import drive\nfrom pathlib import Path\ndrive.mount('/content/drive')\nassert Path('/content/drive/MyDrive').is_dir(), 'Drive is not mounted'\nMODEL_DIR = Path('/content/drive/MyDrive/Kaggle/name-that-llm/models/llama-3.1-8b-base-1f47e50')\nMODEL_DIR.mkdir(parents=True, exist_ok=True)\nprint('Model directory:', MODEL_DIR)"),
    md('## 2. 固定文件清单与校验函数\n\n四个权重分片使用官方仓库元数据中的 SHA-256 校验；其余文件使用固定版本的 Git blob hash 校验。仅下载 Transformers 所需文件，不重复下载 `original/`。'),
    code('import json\nMANIFEST = json.loads(' + repr(json.dumps(manifest)) + ')'),
    code(helper_source),
    md('## 3. 下载并逐文件校验\n\n每约 20 秒显示下载进度。校验失败时停止，完整文件不会被覆盖。'),
    code('verification = download_verified(MODEL_DIR, MANIFEST)\ndataset_metadata = prepare_kaggle_dataset(MODEL_DIR, MANIFEST)\nprint({"status": verification["status"], "total_GB": round(verification["bytes"] / 1e9, 3), "dataset": dataset_metadata["id"]})'),
    md('## 4. 登录 Kaggle\n\n优先读取 Colab Secrets 中名为 `KAGGLE_API_TOKEN` 的密钥；没有时会出现隐藏输入框。可在 [Kaggle 设置](https://www.kaggle.com/settings) 的 API 区域生成 token。不要把 token 粘贴到代码或聊天中。'),
    code("import os, sys, subprocess, getpass\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle==2.2.5'], check=True)\nfrom google.colab import userdata\ntry:\n    kaggle_token = userdata.get('KAGGLE_API_TOKEN')\nexcept (userdata.SecretNotFoundError, userdata.NotebookAccessError):\n    kaggle_token = getpass.getpass('Kaggle API token (hidden): ')\nif not kaggle_token or not kaggle_token.strip():\n    raise RuntimeError('Kaggle token is required for upload; downloaded files are safe in Drive.')\nos.environ['KAGGLE_API_TOKEN'] = kaggle_token.strip()\ndel kaggle_token\nfrom kaggle.api.kaggle_api_extended import KaggleApi\napi = KaggleApi()\napi.authenticate()\nusername = api.config_values.get('username')\nassert username == 'zhjianduan', f'Wrong Kaggle account: {username}'\nprint('Authenticated Kaggle account:', username)"),
    md('## 5. 上传私有 Dataset\n\n仅上传已校验的模型目录。保留 Llama 3.1 许可，默认私有；若同名 Dataset 已存在，停止创建并保留原版本。'),
    code((ROOT / 'upload_to_kaggle.py').read_text(encoding='utf-8')),
    md('## 6. 完成后\n\n确认 Kaggle Dataset 处理完成、文件齐全后，将它挂载到比赛 Notebook 并修改模型路径。这一步完成的是权重传输；比赛推理、分数验证需要单独运行。'),
]
for cell in cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'colab': {'name': 'Llama31_Drive_to_Kaggle.ipynb', 'provenance': []}}
nbformat.validate(nb)
nbformat.write(nb, ROOT / 'Llama31_Drive_to_Kaggle.ipynb')
print(json.dumps({'notebook': str(ROOT / 'Llama31_Drive_to_Kaggle.ipynb'), 'cells': len(cells), 'files': len(files), 'bytes': sum(f['size'] for f in files)}))
