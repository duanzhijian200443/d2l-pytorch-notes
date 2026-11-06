"""Download six version-pinned public datasets with the official Kaggle client.

Run using the existing Kaggle CLI Python environment. No secrets or signed URLs
are printed or stored. Existing matching files are preserved.
"""
import concurrent.futures
import hashlib
import importlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

k = importlib.import_module("kaggle.api.kaggle_api_extended")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data/public-pools"
SPEC = json.loads((HERE / "input-metadata.json").read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(directory, name):
    target = (directory / name).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError("Unsafe archive path")
    return target


def stream_response(response, destination):
    try:
        response.raise_for_status()
        partial = destination.with_suffix(destination.suffix + ".part")
        with partial.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        partial.replace(destination)
    finally:
        response.close()


def download(entry):
    ref, version = entry["ref"], entry["version"]
    directory = RAW / ref / f"v{version}"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f["name"]: int(f["total_bytes"]) for f in entry["file_listing"]["dataset_files"]}
    missing = [name for name, size in expected.items()
               if not (directory / name).is_file() or (directory / name).stat().st_size != size]
    api = k.KaggleApi()
    api.authenticate()
    if missing:
        query = k.ApiDownloadDatasetRequest()
        query.owner_slug, query.dataset_slug = ref.split("/")
        query.dataset_version_number = version
        archive = directory / "_download.zip"
        archive_ok = False
        try:
            with api.build_kaggle_client() as client:
                stream_response(client.datasets.dataset_api_client.download_dataset(query), archive)
            archive_ok = True
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 404:
                raise RuntimeError(f"{ref} v{version}: archive download failed ({type(exc).__name__}; status={status})") from None
            print(f"{ref} v{version}: archive unavailable, using fixed-version individual files", flush=True)
        if archive_ok:
            with zipfile.ZipFile(archive) as z:
                names = {x.filename: x for x in z.infolist() if not x.is_dir()}
                if set(names) != set(expected):
                    raise RuntimeError(f"{ref}: archive listing differs from pinned MCP listing")
                for name in missing:
                    if names[name].file_size != expected[name]:
                        raise RuntimeError(f"{ref}: archive member size differs: {name}")
                    target = safe_target(directory, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as source, target.open("wb") as target_file:
                        shutil.copyfileobj(source, target_file)
        else:
            for name in missing:
                query.file_name, query.raw = name, True
                target = safe_target(directory, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with api.build_kaggle_client() as client:
                    try:
                        stream_response(client.datasets.dataset_api_client.download_dataset(query), target)
                    except Exception as exc:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        raise RuntimeError(f"{ref}: {name} download failed ({type(exc).__name__}; status={status})") from None
                print(f"{ref}: {name} downloaded", flush=True)
                time.sleep(0.8)
    files = []
    for name, size in expected.items():
        target = safe_target(directory, name)
        if target.stat().st_size != size:
            raise RuntimeError(f"{ref}: pinned file size mismatch: {name}")
        files.append({"name": name, "bytes": size, "sha256": sha(target),
                      "local_path": target.relative_to(ROOT).as_posix()})
    print(f"READY {ref} v{version}: {len(files)} files, {sum(x['bytes'] for x in files):,} bytes", flush=True)
    return {"ref": ref, "version": version, "files": files, "status": "READY"}


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, SPEC["sources"]))
    manifest = {"observed_at": datetime.now(timezone.utc).isoformat(), "datasets": results,
                "listing_source": "Kaggle MCP pinned version", "download_source": "official Kaggle Python client; explicit dataset_version_number"}
    (HERE / "download-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ALL SIX DATASETS READY", flush=True)
