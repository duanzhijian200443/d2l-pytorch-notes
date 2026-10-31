"""Release a paid Colab runtime if the controlling kernel dies or the time limit expires."""
import os
import sys
import time
import urllib.request

parent = int(sys.argv[1])
deadline = time.monotonic() + int(sys.argv[2])
while time.monotonic() < deadline:
    try:
        os.kill(parent, 0)
    except ProcessLookupError:
        break
    time.sleep(20)
address = os.environ.get('TBE_RUNTIME_ADDR')
if address:
    for _ in range(4):
        try:
            request = urllib.request.Request(f'http://{address}/unassign', method='POST')
            urllib.request.urlopen(request, timeout=20).close()
            break
        except Exception:
            time.sleep(5)
