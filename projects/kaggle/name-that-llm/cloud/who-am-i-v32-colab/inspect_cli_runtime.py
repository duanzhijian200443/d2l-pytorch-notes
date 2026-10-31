import json
import os
import shutil
from pathlib import Path

root = Path('/content/drive/MyDrive/Kaggle/name-that-llm')
report = {
    'drive_mounted': (root.parent).is_dir(),
    'llama_manifest': (root/'models/llama-3.1-8b-base-1f47e50/transfer_manifest.json').is_file(),
    'disk_free_gib': round(shutil.disk_usage('/content').free / 1024**3, 1),
    'v32_directory_exists': (root/'v32-colab').is_dir(),
    'completed_cached_inputs': len(list((root/'v32-colab/api-cache').rglob('.v32-download-complete.json'))),
    'tbe_release_endpoint_present': bool(os.environ.get('TBE_RUNTIME_ADDR')),
}
try:
    from google.colab import userdata
    token = userdata.get('KAGGLE_API_TOKEN')
    report['colab_kaggle_secret_available'] = bool(token)
    del token
except Exception as exc:
    report['colab_kaggle_secret_available'] = False
    report['secret_error_type'] = type(exc).__name__
print(json.dumps(report), flush=True)
