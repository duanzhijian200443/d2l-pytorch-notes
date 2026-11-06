import json, importlib.util
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime, timezone
result = {"observed_at": datetime.now(timezone.utc).isoformat(), "generation_requests": 0, "gpu_requested": False}
try:
 import kaggle_benchmarks as kbench
 result["kbench_present"] = True
 result["targets"] = {}
 for key in ("openai/gpt-5.6-luna", "openai/gpt-5.6-terra"):
  try:
   llm = kbench.llms[key]
   u = urlsplit(str(llm.client.base_url))
   result["targets"][key] = {"registered": True, "model": llm.model, "base_host": u.hostname, "base_path": u.path}
  except Exception as e:
   result["targets"][key] = {"registered": False, "error_type": type(e).__name__}
except Exception as e:
 result["error_type"] = type(e).__name__
Path("/kaggle/working/proxy-preflight.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
