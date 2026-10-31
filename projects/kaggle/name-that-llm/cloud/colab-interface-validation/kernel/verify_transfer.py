import hashlib
import json
import pathlib
import platform

expected = "38b1059fe3fe4a52df37c7e633312af2c5f9b55db5ea50bc8481651d35dd68a1"
files = list(pathlib.Path("/kaggle/input").rglob("ntl_cli_probe.txt"))
assert len(files) == 1, f"Expected exactly one probe file, found {len(files)}"
data = files[0].read_bytes()
actual = hashlib.sha256(data).hexdigest()
assert len(data) == 53, f"Unexpected size: {len(data)}"
assert actual == expected, f"SHA256 mismatch: {actual}"
assert data == b"Name That LLM: Colab CLI CPU verification 2026-10-29\n"
result = {
    "status": "PASS",
    "bytes": len(data),
    "sha256": actual,
    "file": str(files[0]),
    "python": platform.python_version(),
    "competition_submission": False,
    "transport": "Colab CPU -> local staging -> private Kaggle Dataset -> offline CPU Notebook",
}
pathlib.Path("/kaggle/working/validation_result.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result), flush=True)
