# Colab CLI CPU integration verification — 2026-10-29

The official `google-colab-cli==0.7.5` runs in WSL Ubuntu. OAuth2 login and repeated authenticated CLI commands were tested. Credentials remain in the user's WSL CLI configuration, outside this project and every upload.

## Verified

- Created a named CPU runtime, executed local Python remotely, and downloaded a 53-byte generated file.
- Verified SHA-256 `38b1059fe3fe4a52df37c7e633312af2c5f9b55db5ea50bc8481651d35dd68a1` locally and again inside Kaggle.
- Explicit stop, normal `colab run` termination, and intentional Python-error termination all released the CPU runtime. The backend subsequently reported no active sessions.
- Uploaded only the synthetic file to private Dataset `zhjianduan/ntl-colab-cli-probe-20261029`, version 1.
- Private Notebook `zhjianduan/ntl-colab-cli-offline-check`, version 1, completed using CPU with internet disabled and produced `kaggle-output/validation_result.json` with `PASS`.
- No A100 or full V32 inference was started; no competition submission was made.

The small-file route was **Colab -> local staging -> private Kaggle Dataset -> offline Kaggle Notebook**. This test does not establish large-file transfer performance or A100 throughput. It also does not establish cleanup after local computer shutdown or lost connectivity.

## Reusable PowerShell command

WSL cannot directly reach the Google OAuth endpoint on this machine. The wrapper below temporarily forwards opaque traffic to the user's existing Windows loopback proxy, without changing global network settings or opening a LAN listener. It launches the unmodified official CLI and closes its local listener when the command finishes.

```powershell
wsl -d Ubuntu -- python3 -u '/mnt/c/计算机学习/深度学习/projects/kaggle/name-that-llm/cloud/colab-interface-validation/colab_with_proxy.py' --auth=oauth2 sessions
```

Replace `sessions` with the official CLI command and arguments. Windows proxy `127.0.0.1:7890` must be available. CLI: `/home/zhijian/.local/share/codex-colab-cli/bin/colab`.

For full experiments, persist results before releasing the VM. `colab run` releases the VM on normal completion and a Python exception unless `--keep` is passed. A remote watchdog is still needed for a paid long run to cover local-machine or connection failure. Long-term OAuth refresh was not tested by this short verification.

## Evidence

- `verification_result.json`: combined observed results, checked against saved artifacts.
- `colab-logs/`: official CLI event logs for the three CPU sessions.
- `kaggle-output/`: downloaded Kaggle verification result and log.
- `record_verification.py`: rechecks the saved file, Kaggle result and Colab lifecycle logs.
- Official CLI: https://github.com/googlecolab/google-colab-cli

The private synthetic Dataset and Notebook are retained as inspectable evidence. They have no running GPU and contain no credentials or competition data.
