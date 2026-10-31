import json
import zipfile
from pathlib import Path

root=Path('/content/v32_payload')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/v32_cli_payload.zip') as archive:
    assert all(Path(n).name==n for n in archive.namelist())
    archive.extractall(root)
print(json.dumps({'payload_files':len(list(root.iterdir())),'root':str(root)}),flush=True)
