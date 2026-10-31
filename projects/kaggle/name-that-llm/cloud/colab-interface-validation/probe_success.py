import hashlib, json, pathlib, platform
payload = b"Name That LLM: Colab CLI CPU verification 2026-10-29\n"
p = pathlib.Path("/content/ntl_cli_probe.txt")
p.write_bytes(payload)
print(json.dumps({"status":"PASS", "bytes":len(payload), "sha256":hashlib.sha256(p.read_bytes()).hexdigest(), "python":platform.python_version(), "file":str(p)}), flush=True)
