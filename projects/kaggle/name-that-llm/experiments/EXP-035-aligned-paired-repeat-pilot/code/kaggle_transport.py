"""Official Kaggle Model Proxy transport; credentials remain in memory.

Importing this module does not authenticate, list models or generate text.
The caller owns durable request accounting, budget stops and raw-result storage.
No .env file is read or written by this module. Existing Kaggle authentication is
resolved by the official KaggleApi, and an official short-lived local proxy token
is requested in memory. No credentials, response headers or exception bodies are
printed or included in the sanitized preflight result.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version
from io import StringIO
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit


EXPECTED_MODELS = ('openai/gpt-5.6-luna', 'openai/gpt-5.6-terra')
# The official benchmark catalog uses these exact provider-free slugs. This
# allowlist applies to metadata only; outbound request IDs remain unchanged.
MODEL_LISTING_IDS = {
    'openai/gpt-5.6-luna': frozenset({'openai/gpt-5.6-luna', 'gpt-5.6-luna'}),
    'openai/gpt-5.6-terra': frozenset({'openai/gpt-5.6-terra', 'gpt-5.6-terra'}),
}
BASE_INSTRUCTION = "Answer the user's request."
TIMEOUT_SECONDS = 180.0


class KaggleTransportError(RuntimeError):
    """Safe failure metadata; provider response bodies are deliberately omitted."""
    def __init__(self, stage: str, error_type: str, http_status: int | None = None):
        self.stage = stage
        self.error_type = error_type
        self.http_status = http_status
        super().__init__(f'{stage}: {error_type}' + (f' (HTTP {http_status})' if http_status else ''))

    def sanitized(self):
        return {'stage': self.stage, 'error_type': self.error_type, 'http_status': self.http_status}


def _failure(stage, exc):
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return KaggleTransportError(stage, type(exc).__name__, status if isinstance(status, int) else None)


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _normalize_url(value):
    parsed = urlsplit(value)
    host = (parsed.hostname or '').lower()
    allowed_host = host in {'kaggle.net', 'kaggle.com'} or host.endswith(('.kaggle.net', '.kaggle.com'))
    if (parsed.scheme != 'https' or not allowed_host or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.port not in (None, 443)):
        raise KaggleTransportError('token', 'UnexpectedProxyOrigin')
    path = parsed.path.rstrip('/')
    for suffix in ('/openapi', '/genai'):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path+'/openapi', '', '')), host


class KaggleProxyContext:
    def __init__(self, api, client, expires_at, host):
        self._api = api
        self._client = client
        self._expires_at = expires_at
        self._host = host
        self._ready = False
        self._allowed_models = frozenset()
        self._requests_sent = 0

    def __repr__(self):
        return '<KaggleProxyContext credentials=memory-only>'

    def __getstate__(self):
        raise TypeError('Do not serialize a credential-bearing transport context')

    def close(self):
        self._ready = False
        self._client.close()
        self._client.api_key = ''


def make_client() -> KaggleProxyContext:
    """Authenticate and create an official short-lived proxy client, in memory.

    This performs an authenticated token request, not model inference. It does
    not use MODEL_PROXY_URL/MODEL_PROXY_API_KEY or OPENAI_API_KEY overrides.
    """
    for name in ('openai', 'httpx', 'httpcore', 'kaggle', 'kagglesdk'):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        # The package initializer may print help if authentication is absent.
        # Suppress it rather than forwarding arbitrary auth-library output.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.models.types.model_proxy_api_service import ApiCreateDefaultModelProxyTokenRequest
            from openai import OpenAI, DefaultHttpxClient
            api = KaggleApi()
            api.authenticate()
            with api.build_kaggle_client() as kaggle:
                response = kaggle.models.model_proxy_api_client.create_default_model_proxy_token(
                    ApiCreateDefaultModelProxyTokenRequest())
        if not response.token or response.expiry_time is None:
            raise KaggleTransportError('token', 'MissingTokenOrExpiry')
        base_url, host = _normalize_url(response.base_uri)
        expiry = _utc(response.expiry_time)
        if (expiry-datetime.now(timezone.utc)).total_seconds() < 60:
            raise KaggleTransportError('token', 'TokenExpiredOrExpiring')
        client = OpenAI(api_key=response.token, base_url=base_url,
                        organization='', project='', max_retries=0,
                        timeout=TIMEOUT_SECONDS,
                        http_client=DefaultHttpxClient(timeout=TIMEOUT_SECONDS, follow_redirects=False))
        return KaggleProxyContext(api, client, expiry, host)
    except KaggleTransportError:
        raise
    except (Exception, SystemExit) as exc:
        raise _failure('make_client', exc) from None


def _quota(api):
    response = api.benchmarks_quota()
    balances = {}
    for balance in response.quota_balances or []:
        name = getattr(balance.refill_period, 'name', '')
        if name not in ('DAILY', 'MONTHLY'):
            continue
        used, total = Decimal(str(balance.quota_used)), Decimal(str(balance.total_quota_allowed))
        if not used.is_finite() or not total.is_finite() or used < 0 or total < 0:
            raise KaggleTransportError('quota', 'InvalidQuotaValue')
        balances[name.lower()] = {'used_usd': str(used), 'total_usd': str(total),
                                 'remaining_usd': str(max(Decimal(0), total-used)),
                                 'refill_at': _utc(balance.refill_time).isoformat() if balance.refill_time else None}
    if set(balances) != {'daily','monthly'}:
        raise KaggleTransportError('quota', 'MissingDailyOrMonthlyQuota')
    return balances


def _benchmark_catalog(api):
    """Official CLI tasks-models service, without its automatic retry wrapper."""
    from kagglesdk.benchmarks.types.benchmarks_api_service import ApiListBenchmarkModelsRequest
    ids, page_count, seen_tokens = set(), 0, set()
    token = ''
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListBenchmarkModelsRequest()
            if token:
                request.page_token = token
            response = kaggle.benchmarks.benchmarks_api_client.list_benchmark_models(request)
            page_count += 1
            for model in response.benchmark_models or []:
                slug = getattr(getattr(model, 'version', None), 'slug', None)
                if isinstance(slug, str) and slug:
                    ids.add(slug)
            token = response.next_page_token
            if not token:
                break
            if page_count >= 10 or token in seen_tokens:
                raise KaggleTransportError('benchmark_catalog', 'ModelListingIncomplete')
            seen_tokens.add(token)
    return ids, page_count


def _model_listing_matches(ids):
    """Return exact allowlisted metadata matches, without fuzzy aliasing."""
    return {requested: sorted(allowed.intersection(ids))
            for requested, allowed in MODEL_LISTING_IDS.items()
            if allowed.intersection(ids)}


def preflight(context: KaggleProxyContext, minimum_free_usd: str = '0.25') -> dict[str, Any]:
    """Read quota and model metadata only. Never generates a probe answer.

    The official benchmark catalog establishes listing, not local-token access.
    An unsupported local models endpoint (404/405) can use this catalog, while
    actual inference access and reasoning acceptance await the first authorized
    generation. Authentication, quota and all other metadata errors fail closed.
    """
    context._ready = False
    result = {'observed_at': datetime.now(timezone.utc).isoformat(),
              'proxy_host': context._host, 'token_expiry': context._expires_at.isoformat(),
              'credential_storage': 'memory_only', 'max_retries': 0, 'timeout_seconds': TIMEOUT_SECONDS,
              'expected_models': list(EXPECTED_MODELS), 'model_requests_sent': context._requests_sent,
              'versions': {name: version(name) for name in ('kaggle','kagglesdk','openai')},
              'local_inference_access': 'unknown_until_first_generation',
              'ready': False}
    try:
        required = Decimal(str(minimum_free_usd))
        if not required.is_finite() or required < 0:
            raise KaggleTransportError('quota', 'InvalidRequiredQuota')
        if (context._expires_at-datetime.now(timezone.utc)).total_seconds() < 60:
            raise KaggleTransportError('token', 'TokenExpiredOrExpiring')
        balances = _quota(context._api)
        result['quota'] = balances
        result['minimum_required_usd'] = str(required)
        result['quota_sufficient'] = all(Decimal(b['remaining_usd']) >= required for b in balances.values())
        if not result['quota_sufficient']:
            result['stop_reason'] = 'INSUFFICIENT_FREE_QUOTA'
            return result
        ids, page_count = _benchmark_catalog(context._api)
        result['listed_models'] = sorted(ids)
        result['model_listing_pages'] = page_count
        result['listing_scope'] = 'official_benchmark_catalog; does_not_prove_local_proxy_inference_access'
        result['listing_service'] = 'benchmarks.benchmarks_api_client.list_benchmark_models'
        result['model_listing_matches'] = _model_listing_matches(ids)
        result['matched_requested_models'] = sorted(result['model_listing_matches'])
        result['missing_models'] = sorted(set(EXPECTED_MODELS)-set(result['matched_requested_models']))
        result['low_reasoning_validation'] = 'Requested value fixed to low; actual acceptance/effectiveness not established by read-only model listing.'
        if result['missing_models']:
            result['stop_reason'] = 'TARGET_MODEL_NOT_LISTED'
            return result
        try:
            local_ids, local_pages = set(), 0
            page = context._client.models.list()
            while True:
                local_pages += 1
                local_ids.update(model.id for model in page.data if isinstance(model.id, str))
                if not page.has_next_page():
                    break
                if local_pages >= 10:
                    raise KaggleTransportError('proxy_model_listing', 'ModelListingIncomplete')
                page = page.get_next_page()
            local_matches = _model_listing_matches(local_ids)
            result['local_model_listing'] = {'status': 'supported', 'models': sorted(local_ids),
                                              'pages': local_pages,
                                              'model_listing_matches': local_matches,
                                              'matched_requested_models': sorted(local_matches),
                                              'missing_models': sorted(set(EXPECTED_MODELS)-set(local_matches))}
            if result['local_model_listing']['missing_models']:
                result['stop_reason'] = 'TARGET_MODEL_NOT_LISTED_BY_LOCAL_PROXY'
                return result
        except (Exception, SystemExit) as exc:
            error = exc if isinstance(exc, KaggleTransportError) else _failure('proxy_model_listing', exc)
            if error.http_status not in (404, 405):
                raise error from None
            result['local_model_listing'] = {'status': 'unsupported_endpoint', 'error': error.sanitized()}
        context._allowed_models = frozenset(EXPECTED_MODELS)
        context._ready = True
        result['ready'] = True
        return result
    except KaggleTransportError as exc:
        result['error'] = exc.sanitized()
    except (Exception, SystemExit) as exc:
        result['error'] = _failure('preflight', exc).sanitized()
    result['stop_reason'] = 'PREFLIGHT_NOT_ESTABLISHED'
    return result


def generate(context: KaggleProxyContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Send exactly one approved caller-supplied payload and return raw SDK JSON.

    No retries, aliases, fallback providers, output rewriting or hidden test call.
    Caller must persist intent before this call and inspect cost/model/finish after
    it. Any exception is outcome-uncertain and must stop the caller's dispatch.
    """
    if not context._ready:
        raise KaggleTransportError('generate', 'PreflightNotReady')
    if context._requests_sent >= 40:
        raise KaggleTransportError('generate', 'KaggleRequestLimitReached')
    if (context._expires_at-datetime.now(timezone.utc)).total_seconds() < 30:
        context._ready = False
        raise KaggleTransportError('generate', 'TokenExpiredOrExpiring')
    allowed_keys = {'model','messages','reasoning_effort','max_completion_tokens','stream','n'}
    messages = payload.get('messages')
    if (set(payload) != allowed_keys or payload.get('model') not in context._allowed_models
            or payload.get('reasoning_effort') != 'low' or payload.get('max_completion_tokens') != 1024
            or payload.get('stream') is not False or payload.get('n') != 1
            or not isinstance(messages,list) or len(messages) != 2
            or messages[0] != {'role':'system','content':BASE_INSTRUCTION}
            or not isinstance(messages[1], dict) or set(messages[1]) != {'role','content'} or messages[1].get('role') != 'user'
            or not isinstance(messages[1].get('content'),str) or not messages[1]['content'].strip()):
        raise KaggleTransportError('generate', 'PayloadDiffersFromApprovedContract')
    context._requests_sent += 1
    try:
        response = context._client.chat.completions.create(**payload)
        return response.model_dump(mode='json')
    except (Exception, SystemExit) as exc:
        context._ready = False
        raise _failure('generate', exc) from None
