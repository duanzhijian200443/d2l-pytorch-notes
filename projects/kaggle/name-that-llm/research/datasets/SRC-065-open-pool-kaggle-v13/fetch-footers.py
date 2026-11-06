from kaggle.api.kaggle_api_extended import KaggleApi
import json,requests,pathlib,datetime,hashlib
p=pathlib.Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
log=[]
for version,name in [(12,"ntl_open_pool.parquet"),(13,"ntl_open_pool.parquet"),(13,"gemini37_unverified.parquet"),(13,"prompts.parquet")]:
    with api.build_kaggle_client() as k:
        f=k.datasets.dataset_api_client.download_dataset
        req=f.__globals__["ApiDownloadDatasetRequest"]()
        req.owner_slug="crtproduction";req.dataset_slug="name-that-llm-open-pool";req.dataset_version_number=version;req.file_name=name
        r=f(req);url=r.request.url;r.close()
    r=requests.get(url,headers={"Range":"bytes=-65536"},stream=True,timeout=45)
    if r.status_code!=206: r.close();raise RuntimeError("Range request not honored")
    b=r.raw.read(65536);content_range=r.headers.get("Content-Range");r.close()
    if b[-4:]!=b"PAR1":raise RuntimeError("Not a parquet footer")
    out=p/f"v{version}-{name}-tail.bin";out.write_bytes(b)
    log.append({"source_url":f"https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/{version}?select={name}","method":"authenticated official Kaggle SDK file download response followed by HTTP byte range; signed URL deliberately not persisted","observed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"file":out.name,"bytes":len(b),"content_range":content_range,"sha256":hashlib.sha256(b).hexdigest()})
(p/"range-fetch-log.json").write_text(json.dumps(log,indent=2),encoding="utf8")
print(json.dumps(log,indent=2))
