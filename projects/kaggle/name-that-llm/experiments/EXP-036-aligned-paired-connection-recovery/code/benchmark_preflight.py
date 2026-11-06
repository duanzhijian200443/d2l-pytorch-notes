# %%
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import kaggle_benchmarks as kbench

# %%
@kbench.task(name='ntl-exp036-proxy-metadata')
def proxy_metadata(llm):
    """Inspect injected route metadata without invoking an LLM or judge."""
    result = {'observed_at': datetime.now(timezone.utc).isoformat(),
              'generation_requests': 0, 'targets': {},
              'model_proxy_key_present': bool(os.environ.get('MODEL_PROXY_API_KEY'))}
    for key in ('openai/gpt-5.6-luna', 'openai/gpt-5.6-terra'):
        try:
            actor = kbench.llms[key]
            parsed = urlsplit(str(actor.client.base_url))
            result['targets'][key] = {'registered': True, 'model': actor.model,
                                      'base_host': parsed.hostname, 'base_path': parsed.path}
        except Exception as exc:
            result['targets'][key] = {'registered': False, 'error_type': type(exc).__name__}
    Path('/kaggle/working/proxy-preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return True

# %%
proxy_metadata.run(kbench.llm)
