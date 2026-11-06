# %
import os
os.environ['ENABLE_LOCAL_CACHING'] = 'false'
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import kaggle_benchmarks as kbench
from openai import OpenAI, DefaultHttpxClient
PAYLOAD = json.loads('{"max_completion_tokens":1024,"messages":[{"content":"Answer the user\'s request.","role":"system"},{"content":"Make this paragraph easier for a general audience to understand: \'The room booking procedure comprises submission of an intended usage interval, subsequent availability verification, and issuance of an acknowledgement. An acknowledgement of receipt does not constitute approval of the requested interval.\' Aim for about 80 words.\\n\\nWrite the requested response as one self-contained paragraph of plain text. Do not use a title, headings, bullet points, numbered lists, Markdown formatting, code fences, or line breaks. Include only the requested content, without any introductory or closing commentary from the assistant. Keep the approximate word target given in the task. If the task calls for an email, dialogue, steps, or calculations, express that content within the same paragraph.","role":"user"}],"model":"openai/gpt-5.6-luna","n":1,"reasoning_effort":"low","stream":false}')
JOB_ID = 'd9d0a0b704cb612c2ad24248d1e7b6661082d90740912b8721ba5eb0c92fdd82'
PAYLOAD_SHA256 = 'fa90bff0f10141d5282c259cd095946d9c3d80978ff03d22cea9f2a062e8b08c'

# %
@kbench.task(name='ntl-exp038-d9d0a0b704cb612c2ad24248d1e7b666')
def collect_one(llm):
    root = Path('/kaggle/working')
    marker = root / ('dispatch-' + JOB_ID + '.started')
    try:
        with marker.open('x') as f:
            f.write(JOB_ID)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        return None
    result = {'job_id':JOB_ID, 'payload_sha256':PAYLOAD_SHA256,
              'generation_requests':0, 'started_at':datetime.now(timezone.utc).isoformat(),
              'status':'STARTED', 'model_requested':PAYLOAD['model'],
              'sdk_max_retries':0, 'response_cache':False}
    client = None
    try:
        actor = kbench.llms[PAYLOAD['model']]
        client = OpenAI(api_key=actor.client.api_key, base_url=actor.client.base_url,
                        organization='', project='', max_retries=0, timeout=180,
                        http_client=DefaultHttpxClient(timeout=180, follow_redirects=False))
        # Keep unrelated OpenAI account routing headers absent, including empty
        # values; constructor blanks prevent ambient env values being inherited.
        client.organization = None
        client.project = None
        result['generation_requests'] = 1
        (root/'cloud-dispatch.json').write_text(json.dumps(result,indent=2))
        response = client.chat.completions.create(**PAYLOAD)
        raw = response.model_dump(mode='json')
        data = (json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
        with (root/'raw-response.json').open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        result.update(status='RECEIVED', raw_response_sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        result.update(status='FAILED_OR_UNKNOWN', error_type=type(exc).__name__)
        status = getattr(exc,'status_code',None)
        if type(status) is int: result['http_status'] = status
        chain, seen, current = [], set(), exc
        while current is not None and id(current) not in seen and len(chain)<8:
            seen.add(id(current))
            item = {'type':type(current).__name__}
            for name in ('errno','verify_code'):
                value = getattr(current,name,None)
                if type(value) is int: item[name] = value
            chain.append(item)
            current = current.__cause__ or current.__context__
        result['cause_chain'] = chain
    finally:
        result['ended_at'] = datetime.now(timezone.utc).isoformat()
        with (root/'cloud-result.json').open('x') as f:
            json.dump(result,f,indent=2)
            f.flush()
            os.fsync(f.fileno())
        if client is not None: client.close()
    return None

# %
collect_one.run(kbench.llm)
