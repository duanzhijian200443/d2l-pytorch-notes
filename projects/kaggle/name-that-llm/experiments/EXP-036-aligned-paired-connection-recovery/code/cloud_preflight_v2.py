import os,json,importlib.util
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime,timezone
r={"observed_at":datetime.now(timezone.utc).isoformat(),"generation_requests":0,"model_proxy_url_present":bool(os.environ.get("MODEL_PROXY_URL")),"model_proxy_key_present":bool(os.environ.get("MODEL_PROXY_API_KEY")),"kbench_installed":importlib.util.find_spec("kaggle_benchmarks") is not None,"openai_installed":importlib.util.find_spec("openai") is not None}
u=urlsplit(os.environ.get("MODEL_PROXY_URL",""))
r.update(base_host=u.hostname,base_path=u.path)
r["target_registry"]={k:k in os.environ.get("LLMS_AVAILABLE","").split(",") for k in ["openai/gpt-5.6-luna","openai/gpt-5.6-terra"]}
Path("/kaggle/working/proxy-preflight.json").write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
