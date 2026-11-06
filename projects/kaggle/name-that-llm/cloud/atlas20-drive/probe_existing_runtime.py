from pathlib import Path
import json
import sys

print(json.dumps({
    'drive_mounted': Path('/content/drive/MyDrive').is_dir(),
    'model_library_present': Path('/content/drive/MyDrive/Kaggle/Model-Library').is_dir(),
    'python': sys.version.split()[0],
    'scope': 'read-only mount probe; no GPU allocation or model loading',
}))
