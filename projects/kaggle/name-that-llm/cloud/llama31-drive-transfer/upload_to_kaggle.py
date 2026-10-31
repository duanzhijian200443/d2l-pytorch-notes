"""Create once, preserve a receipt, and report a status-read failure separately."""
from requests.exceptions import HTTPError

dataset_metadata = prepare_kaggle_dataset(MODEL_DIR, MANIFEST)
dataset_ref = dataset_metadata['id']
receipt_path = MODEL_DIR.parent / (MODEL_DIR.name + '-kaggle-receipt.json')
if receipt_path.exists():
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['dataset_ref'] == dataset_ref
    assert receipt['source_revision'] == MANIFEST['revision']
    assert receipt['creation_accepted'] is True
    print('Using the existing upload receipt:', receipt['dataset_url'])
else:
    # Reuse the accepted result when only the following status lookup failed.
    if 'result' not in globals() or not getattr(result, 'url', None):
        result = api.dataset_create_new(
            str(MODEL_DIR), public=False, quiet=False,
            convert_to_csv=False, dir_mode='skip',
        )
    if getattr(result, 'error', None):
        raise RuntimeError(result.error)
    expected_url = 'https://www.kaggle.com/datasets/' + dataset_ref
    assert getattr(result, 'url', None) == expected_url
    receipt = {
        'dataset_ref': dataset_ref,
        'dataset_url': result.url,
        'source_revision': MANIFEST['revision'],
        'creation_accepted': True,
        'private_requested': True,
        'expected_files': len(MANIFEST['files']) + 2,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Dataset creation accepted:', receipt['dataset_url'])

try:
    status = api.dataset_status(dataset_ref)
    print('Kaggle processing status:', status)
    receipt['api_status_read'] = 'SUCCESS'
    receipt['api_status'] = status if isinstance(status, (dict, str, list)) else str(status)
except HTTPError as exc:
    if exc.response is None or exc.response.status_code != 403:
        raise
    receipt['api_status_read'] = '403_FORBIDDEN'
    print('Upload was accepted, but the API status lookup returned 403.')
    print('Check the private dataset page for version, file count and readiness:', receipt['dataset_url'])
finally:
    _ = os.environ.pop('KAGGLE_API_TOKEN', None)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print('Upload receipt saved in Drive:', receipt_path)
