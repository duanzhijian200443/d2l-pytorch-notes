"""Read-only source intake; fetched notebook cells are never executed."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest
base = Path(__file__).resolve().parent
api = KaggleApi(); api.authenticate()
rows = []
for label, prefix in [('v11', ''), (None, 'latest-')]:
    q = ApiGetKernelRequest(); q.user_name = 'mercurydust'; q.kernel_slug = 'name-that-llm-submission'
    if label: q.version_label = label
    with api.build_kaggle_client() as client: response = client.kernels.kernels_api_client.get_kernel(q)
    metadata = response.metadata.to_dict()
    assert not metadata.get('isPrivate'), 'Unexpected private source'
    raw = response.blob.source
    nb = json.loads(raw)
    def extract(kind):
        return '\n\n'.join(''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells'] if c['cell_type'] == kind)
    code = extract('code')
    for name, content in [('source.ipynb',raw),('code.txt',code),('description.md',extract('markdown')),('metadata.json',json.dumps(metadata,ensure_ascii=False,indent=2))]:
        (base/(prefix+name)).write_bytes(content.encode('utf-8'))
    row = {'requested_version':label or 'latest','observed_at':datetime.now(timezone.utc).isoformat(),'metadata':metadata,'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'code_sha256':hashlib.sha256(code.encode()).hexdigest()}
    rows.append(row)
(base/'fetch-receipt.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
