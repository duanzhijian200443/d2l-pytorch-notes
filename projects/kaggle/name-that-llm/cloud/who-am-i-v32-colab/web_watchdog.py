"""Independent, bounded release guard; record its own release reason when possible."""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

parent, seconds, config_file = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
assert 0 < seconds <= 14400
deadline = time.monotonic() + seconds
reason = 'TIME_LIMIT'
while time.monotonic() < deadline:
    try:
        os.kill(parent, 0)
    except ProcessLookupError:
        reason = 'CONTROLLER_PROCESS_EXITED'
        break
    time.sleep(min(10, max(0.1, deadline - time.monotonic())))
event = {'status': 'INTERRUPTED', 'reason': reason, 'observed_at': time.time(), 'release_requested': True}
try:
    config = json.loads(config_file.read_text())
    destination = Path(config['output'])
    if destination.is_dir():
        (destination / 'watchdog_event.json').write_text(json.dumps(event, indent=2))
        os.sync()
except Exception:
    pass
address = os.environ.get('TBE_RUNTIME_ADDR')
if address:
    for attempt in range(4):
        try:
            urllib.request.urlopen(urllib.request.Request('http://' + address + '/unassign', method='POST'), timeout=20).close()
            break
        except Exception:
            time.sleep(3)
