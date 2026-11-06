"""Read public Kaggle discussions; save source evidence only; never execute posts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://www.kaggle.com/api/i/discussions.DiscussionsService/'
TOPICS = {
    754426: 'halx-fork-guards',
    754427: 'halx-no-llama-ablations',
    754657: 'gopain-tpu-qwen-lora',
    754463: 'public-066-to-070-gap',
    754131: 'atlas-vs-uniform-smoothing',
    754084: 'ten-lb-points-simulation',
    754026: 'target-shrunk-sinkhorn',
    753974: 'openai-provenance-data-announcement',
    753177: 'member-oof-and-sparse-validation',
    752766: 'open-pool-v10-updates',
}

def call(method, payload):
    r = requests.post(BASE + method, json=payload, timeout=45)
    r.raise_for_status()
    return r.json()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    path.write_text(raw, encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def walk(comments):
    for c in comments:
        yield c
        yield from walk(c.get('replies', []))

def fetch(pair):
    tid, slug = pair
    start = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = ROOT / 'discussions' / f'{tid}-{slug}'
    topic = call('GetForumTopicById', {'forumTopicId': tid, 'includeComments': True})
    comments = call('GetForumMessagesInTopic', {'topicId': tid, 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': False})
    partials = [c['id'] for c in walk(comments.get('comments', [])) if c.get('isPartial') or not c.get('author')]
    hydrations = []
    if partials:
        hydrations.append(call('BatchGetForumMessages', {'topicId': tid, 'messageIds': partials}))
    evidence = {'observed_at': start, 'retrieval': {'method': 'read-only POST', 'topic_endpoint': BASE + 'GetForumTopicById', 'comments_endpoint': BASE + 'GetForumMessagesInTopic'}, 'topic_response': topic, 'comments_response': comments, 'hydrated_message_responses': hydrations}
    path = dest / 'snapshots' / f'{stamp}.json'
    sha = save(path, evidence)
    t = topic['forumTopic']
    record = {'topic_id': tid, 'slug': slug, 'category': 'discussions', 'url': 'https://www.kaggle.com' + t['url'], 'title': t['name'], 'author': t.get('authorUserName'), 'observed_at': start, 'post_date': t['postDate'], 'notes_path': (dest/'notes.md').relative_to(ROOT).as_posix(), 'snapshot_path': path.relative_to(ROOT).as_posix(), 'snapshot_sha256':sha, 'existing_src_candidate': {753177:'SRC-026',752766:'SRC-027'}.get(tid), 'status':'PENDING_IMPORT'}
    print(json.dumps({'id':tid,'title':t['name'],'comments':len(list(walk(comments.get('comments', [])))),'partial_ids':partials}, ensure_ascii=False),flush=True)
    return record

if __name__ == '__main__':
    lists = []
    for page in [1, 2]:
        lists.append({'request':{'forumId':13688374,'sortBy':'new','page':page},'response':call('GetTopicListByForumId',{'forumId':13688374,'sortBy':'new','page':page})})
    save(ROOT/'discovery-snapshot.json', {'observed_at':datetime.now(timezone.utc).isoformat(), 'role':'Discovery evidence, not canonical source ledger', 'lists':lists})
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, TOPICS.items()))
    save(ROOT/'manifest.json', {'status':'PENDING_IMPORT', 'role':'Temporary import manifest; formal SRC numbers assigned only by merge window', 'scope':'New forum topics 2026-10-30 through 2026-11-01 and relevant updates to existing discussions; new-sort pages 1-2 inspected', 'records':records})
