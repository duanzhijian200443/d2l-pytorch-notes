# Kaggle official local transport notes

Prepared 2026-11-01 for EXP-035. The transport implementation and its offline checks do not generate model output. Root owns live preflight, dispatch, durable accounting, raw-result storage, and the authorized stop conditions.

## Installed sources and official route

- Inspected local `kaggle==2.2.5` and `kagglesdk==0.1.38` under `C:\Users\34331\AppData\Roaming\uv\tools\kaggle\Lib\site-packages`.
- CLI `kaggle benchmarks tasks models` calls `KaggleApi._fetch_all_benchmark_models`, which calls `kaggle.benchmarks.benchmarks_api_client.list_benchmark_models(ApiListBenchmarkModelsRequest())`. SDK endpoint: `/api/v1/benchmarks/models`. The transport calls the service directly, follows pagination, and does not use the CLI retry wrapper. The eight default models written by `benchmarks init` are not the current server catalog.
- `KaggleApi.benchmarks_quota()` calls official `get_model_proxy_quotas(ApiGetModelProxyQuotasRequest())`, endpoint `/api/v1/models/proxy/quota`. The result exposes daily/monthly used and total USD and refill timestamps.
- Official `create_default_model_proxy_token(ApiCreateDefaultModelProxyTokenRequest())`, endpoint `/api/v1/models/proxy/token`, returns a local-scope token, base URI and expiry. The implementation resolves existing authentication through `KaggleApi.authenticate()` and keeps the returned proxy token in memory. It does not invoke CLI `benchmarks auth`, write `.env`, print token tails, or copy credential material to reports.
- Installed SDK comments describe a short-lived token for `kaggle-benchmarks-local` and a verified-account requirement. Receiving a token and seeing a model in the benchmark catalog do not prove that token can perform inference for that model.

## Runtime and interface

Use one environment containing `kaggle==2.2.5`, `kagglesdk==0.1.38`, `openai==1.109.1`, and `six==1.17.0`. The root installed the combined environment; the explicit `six` dependency fixes the observed Kaggle import failure. A reproducible invocation is:

```powershell
uv run --offline --no-project --with kaggle==2.2.5 --with kagglesdk==0.1.38 --with openai==1.109.1 --with six==1.17.0 python <runner.py>
```

- `make_client() -> KaggleProxyContext`: authenticates and requests an official short-lived proxy token in memory. Uses `OpenAI(max_retries=0, timeout=180)`, HTTP redirects disabled, and only an HTTPS Kaggle-owned origin returned by the official SDK. It ignores `MODEL_PROXY_*` and `OPENAI_API_KEY` overrides. No inference happens here.
- `preflight(context, minimum_free_usd='0.25') -> dict`: reads quota, official catalog and local OpenAI-compatible `models.list`. Only `ready is True` permits this context to dispatch. Returns sanitized quota, token expiry, expected/listed/missing models, listing scope, optional local-list result, library versions, and error class/status only. Missing free quota, catalog omissions, auth/rate/network failures, and incomplete pagination fail closed.
- Metadata matching accepts only the two exact known provider-free catalog slugs (`gpt-5.6-luna`, `gpt-5.6-terra`) or their exact requested `openai/` IDs. It retains raw `listed_models` / local `models`, records `model_listing_matches` with the observed spelling, and exposes normalized `matched_requested_models`. This narrow allowlist is shared by both listings; it does not strip arbitrary provider prefixes, accept snapshots or suffixes, perform fuzzy matching, or alter generation payload IDs.
- A local `models.list` 404 or 405 means unsupported endpoint; the benchmark catalog can establish listing without proving local inference access. `local_inference_access` remains `unknown_until_first_generation` even when local listing succeeds. The first already-authorized generation verifies actual access. No separate generation probe is sent.
- `generate(context, payload) -> dict`: performs one Chat Completions request with the exact supplied payload and returns the unmodified `response.model_dump(mode='json')`. It never retries, substitutes models, or falls back to another service. A provider exception deactivates the context and reports only sanitized stage/type/status. An exception is outcome-uncertain; the caller must stop dispatch rather than automatically resend.
- `context.close()` closes the HTTP client and clears its API-key reference. Context repr is redacted, and serialization is blocked.

The module rejects payloads outside this EXP contract: either exact target model ID (`openai/gpt-5.6-luna`, `openai/gpt-5.6-terra`), exactly one minimal system message (`Answer the user's request.`) and one unchanged user message, `reasoning_effort='low'`, `max_completion_tokens=1024`, `stream=False`, `n=1`. It adds no message or sampling parameter. The OpenAI request count is at most 40 per context; the runner must additionally enforce durable experiment-wide request and spend limits across all contexts/restarts. The module does not itself enforce a dollar budget or inspect response cost.

## Evidence and limitations

Offline fake-client checks passed: module compilation; complete benchmark catalog plus unsupported-local-list 404 permits the authorized first request; 403 stops; exact caller payload reaches the fake client; raw response including unknown fields remains unchanged; malformed message is rejected before a request. No real authentication, quota/model request or model inference was performed by this transport subtask. Live evidence is recorded by the root separately.

The catalog-spelling correction additionally passed offline checks for both exact bare slugs, both exact provider-qualified IDs, mixed spelling evidence, one missing target, and rejection of different provider prefixes, case changes and extra snapshot/latest suffixes. A fake preflight using bare IDs in both listings succeeds while retaining their original spelling; a fake catalog with a different provider fails closed. These checks import the module without authentication and make no network calls.

The first call must be evaluated for returned model ID, cost availability, finish reason and request acceptance before dispatching the next request. Metadata listing does not prove the provider applied `low` identically across channels, establish the hidden backend snapshot, certify a hard per-request dollar bound, or make the daily free allowance an experiment budget cap. The runner owns the confirmed cumulative spend stop at USD 0.10, authorized total/request limits, and the no-top-up/no-paid-fallback constraint. Existing quota and metadata are observations at a particular time.

Official source references inspected:

- [Kaggle Benchmarks model proxy](https://raw.githubusercontent.com/Kaggle/kaggle-benchmarks/ci/src/kaggle_benchmarks/kaggle/model_proxy.py): base URL normalization and OpenAI-compatible client construction.
- [Kaggle Benchmarks OpenAI actor](https://raw.githubusercontent.com/Kaggle/kaggle-benchmarks/ci/src/kaggle_benchmarks/actors/proxy_openai.py): Chat Completions transport; higher-level actor output processing is deliberately bypassed to preserve raw JSON.
- [Kaggle models registry](https://raw.githubusercontent.com/Kaggle/kaggle-benchmarks/ci/src/kaggle_benchmarks/kaggle/models.py): environment-derived registry is not dynamic account-access evidence.
- [OpenAI Python library error types](https://developers.openai.com/api/docs/guides/error-codes#python-library-error-types), read through OpenAI Developer Docs MCP: transport/auth/permission/rate-limit errors remain distinct; this experiment explicitly disables automatic retries.
