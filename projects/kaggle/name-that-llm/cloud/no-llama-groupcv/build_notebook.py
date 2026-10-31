"""Build and validate the Kaggle notebook from the reviewable percent-format source."""
from pathlib import Path

import jupytext
import nbformat

root = Path(__file__).resolve().parent
notebook = jupytext.read(root / "baseline.py")
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
notebook.metadata["language_info"] = {"name": "python"}
nbformat.validate(notebook)
target = root / "baseline.ipynb"
nbformat.write(notebook, target)
print(f"Validated {len(notebook.cells)} cells: {target}")
