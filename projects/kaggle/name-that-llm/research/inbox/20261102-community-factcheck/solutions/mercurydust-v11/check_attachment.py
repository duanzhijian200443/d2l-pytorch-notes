"""Read only the small metadata/file listing of the notebook's required A/B dataset."""
from pathlib import Path
from datetime import datetime,timezone
import json,os
os.environ['HTTP_PROXY']=os.environ['HTTPS_PROXY']='http://127.0.0.1:7890'
from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetDatasetMetadataRequest
api=KaggleApi();api.authenticate();base=Path(__file__).resolve().parent
result={'observed_at':datetime.now(timezone.utc).isoformat(),'ref':'mercurydust/ntllm-chatgpt-a-b','files_downloaded':False}
try:
    q=ApiGetDatasetMetadataRequest();q.owner_slug='mercurydust';q.dataset_slug='ntllm-chatgpt-a-b'
    with api.build_kaggle_client() as c:r=c.datasets.dataset_api_client.get_dataset_metadata(q)
    result['metadata']=r.to_dict()
except Exception as e:
    result['metadata_error']={'type':type(e).__name__,'message':str(e).split('Response headers:')[0][:1500]}
try: result['file_listing']=api.dataset_list_files(result['ref'],page_size=20).to_dict()
except Exception as e: result['listing_error']={'type':type(e).__name__,'message':str(e).split('Response headers:')[0][:1500]}
if 'file_listing' in result: result['file_listing'].pop('nextPageToken',None)
try: result['public_search']=[x.to_dict() for x in api.dataset_list(user='mercurydust',search='NTLLM-ChatGPT-A_B')]
except Exception as e: result['search_error']={'type':type(e).__name__,'message':str(e).split('Response headers:')[0][:1500]}
(base/'attachment-metadata-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
