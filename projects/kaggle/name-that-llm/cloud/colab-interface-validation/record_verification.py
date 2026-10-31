"""Check the downloaded artifacts and saved CLI event logs; record observed results."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected = json.loads((root / "expected.json").read_text(encoding="utf-8-sig"))
data = (root / "ntl_cli_probe.txt").read_bytes()
assert len(data) == expected["bytes"]
assert hashlib.sha256(data).hexdigest() == expected["sha256"]
result = json.loads((root / "kaggle-output/validation_result.json").read_text())
assert result["status"] == "PASS"
assert result["bytes"] == expected["bytes"]
assert result["sha256"] == expected["sha256"]

sessions = []
for name in ("ntl-cli-smoke-261029", "ntl-cli-auto-ok-261029", "ntl-cli-auto-error-261029"):
    events = [json.loads(line) for line in
              (root / "colab-logs" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    created = next(e for e in events if e["event_type"] == "session_created")
    terminated = next(e for e in events if e["event_type"] == "session_terminated")
    assert created["accelerator"] == "NONE"
    execution = next(e for e in events if e["event_type"] == "execution")
    if "error" in name:
        assert any(o.get("evalue") == "EXPECTED_COLAB_CLEANUP_PROBE" for o in execution["outputs"])
        assert terminated["reason"] == "run_failed"
    sessions.append({"name": name, "accelerator": created["accelerator"],
                     "created_at": created["timestamp"],
                     "terminated_at": terminated["timestamp"],
                     "termination_reason": terminated.get("reason", "explicit_stop")})

report = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "PASS",
    "scope": "CPU integration smoke test, not V32 inference or GPU benchmark",
    "cli": {"package": "google-colab-cli", "version": "0.7.5", "platform": "WSL Ubuntu",
            "auth": "OAuth2", "repeat_command_without_browser_login": True,
            "long_term_token_refresh_tested": False},
    "network": {"wsl_direct_oauth_endpoint": "12-second connection timeout",
                "existing_windows_proxy": "127.0.0.1:7890",
                "local_only_transport_wrapper": "colab_with_proxy.py",
                "persistent_network_settings_changed": False,
                "temporary_fixed_bridge_stopped": True},
    "sessions": sessions,
    "colab_final_backend_message": "No active sessions found on server.",
    "colab_success_run_exit_code": 0,
    "colab_expected_error_run_exit_code": 1,
    "file": expected,
    "kaggle": {"dataset": "zhjianduan/ntl-colab-cli-probe-20261029",
               "dataset_version": 1, "dataset_status": "READY", "dataset_private": True,
               "notebook": "zhjianduan/ntl-colab-cli-offline-check",
               "notebook_version": 1, "notebook_status": "COMPLETE",
               "notebook_private": True, "enable_gpu": False, "enable_internet": False,
               "attached_inputs": ["zhjianduan/ntl-colab-cli-probe-20261029"],
               "downloaded_result": result},
    "gpu_started": False,
    "competition_submitted": False,
    "limits": ["Small file used local staging between Colab and Kaggle.",
               "No A100 throughput or complete V32 model loading was tested.",
               "Cleanup after local machine power loss/network loss was not tested."]
}
(root / "verification_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": report["status"], "file_sha256": expected["sha256"],
                  "cpu_sessions_terminated": len(sessions), "kaggle_offline_result": result["status"]}))
